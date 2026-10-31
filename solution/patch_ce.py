"""Plugs the fused head + loss (fused_ce.py) into Soup's run.

Two class patches, both looked up at call time:
- `Qwen2ForCausalLM.forward`: while armed (below), a labelled training forward runs the decoder,
  then `fused_lm_head_loss` on its last hidden states instead of `lm_head` + `ForCausalLMLoss`,
  and returns the loss with `logits=None`;
- `trl.SFTTrainer.compute_loss`: arms the forward, calls transformers' `Trainer.compute_loss`
  (what TRL's own `compute_loss` delegates to), then logs TRL's `entropy`, `num_tokens` and
  `mean_token_accuracy` by TRL's formulas from the per-position values the kernels produced, and
  leaves the per-position losses where Soup's rewind recorder reads them when a step has no
  logits (`trainer._soup_chunked_step.token_loss`, see soup_cli/trainer/rewind_hf.py). Soup's
  own `stream_loss_chunk` mode stays off; only its recorder's input is used.

Any case these do not cover (evaluation, no labels, another model, Liger, label smoothing, a
custom loss, prompt tuning, context parallelism, no fp16/bf16 autocast) takes the original path
untouched.
"""

from types import SimpleNamespace

import torch

from .fused_ce import LOGITS_BK, fused_lm_head_loss

# Where Soup's rewind recorder looks for per-position losses when `outputs.logits` is None.
REWIND_STEP_ATTR = "_soup_chunked_step"


class _State:
    armed = False  # set by compute_loss around its own forward only
    step = None  # what the last fused forward produced for the metrics


def _head_ok(causal_lm):
    head = getattr(causal_lm, "lm_head", None)
    config = getattr(causal_lm, "config", None)
    return (type(head) is torch.nn.Linear and head.bias is None
            and not head.weight.requires_grad and head.weight.is_cuda
            and getattr(config, "final_logit_softcapping", None) is None
            and head.weight.shape[1] % LOGITS_BK == 0)


def _autocast_dtype():
    if torch.is_autocast_enabled("cuda"):
        dtype = torch.get_autocast_dtype("cuda")
        if dtype in (torch.float16, torch.bfloat16):
            return dtype
    return None


def patch_qwen2_causal_lm():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2ForCausalLM.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        dtype = _autocast_dtype()
        if not (_State.armed and labels is not None and self.training and dtype is not None
                and isinstance(logits_to_keep, int) and logits_to_keep == 0
                and kwargs.get("shift_labels") is None and _head_ok(self)):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        loss, entropy, token_loss, correct, labelled = fused_lm_head_loss(
            outputs.last_hidden_state, self.lm_head.weight, labels, attention_mask,
            num_items_in_batch=kwargs.get("num_items_in_batch"), compute_dtype=dtype)
        _State.step = SimpleNamespace(token_entropy=entropy, token_loss=token_loss,
                                      correct=correct, labelled=labelled)
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    modeling_qwen2.Qwen2ForCausalLM.forward = forward


def _causal_lm(trainer, model):
    unwrap = getattr(getattr(trainer, "accelerator", None), "unwrap_model", None)
    model = unwrap(model) if callable(unwrap) else model
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


def _can_fuse(trainer, model, inputs):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    args = getattr(trainer, "args", None)
    if not (isinstance(model, torch.nn.Module) and model.training and isinstance(inputs, dict)
            and "labels" in inputs and "attention_mask" in inputs
            and "shift_labels" not in inputs
            and not getattr(args, "use_liger_kernel", False)
            and getattr(args, "loss_type", "nll") == "nll"
            and getattr(trainer, "label_smoother", None) is None
            and getattr(trainer, "compute_loss_func", None) is None
            and not getattr(trainer, "aux_loss_enabled", False)
            and getattr(trainer, "num_virtual_tokens", 0) == 0
            and hasattr(trainer, "_metrics")):
        return False
    causal_lm = _causal_lm(trainer, model)
    return isinstance(causal_lm, Qwen2ForCausalLM) and _head_ok(causal_lm)


def _log_metrics(trainer, mode, inputs, token_entropy, correct, labelled):
    """TRL's SFTTrainer.compute_loss metrics (trl 0.29.1), from per-position values."""
    gather = trainer.accelerator.gather_for_metrics
    attention_mask = inputs["attention_mask"]
    entropy = torch.sum(token_entropy * attention_mask) / attention_mask.sum()
    trainer._metrics[mode]["entropy"].append(gather(entropy).mean().item())
    if mode == "train":
        trainer._total_train_tokens += gather(attention_mask.sum()).sum().item()
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    correct, labelled = gather(correct), gather(labelled)
    total = labelled.sum()
    accuracy = (correct.sum() / total).item() if total > 0 else 0.0
    trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)


def _metrics_from_logits(logits, labels):
    """The same values from logits, for a forward that could not take the fused path."""
    from trl.trainer.utils import entropy_from_logits

    with torch.no_grad():
        token_entropy = entropy_from_logits(logits)
        shift_labels = labels[..., 1:]
        mask = shift_labels != -100
        correct = ((logits[..., :-1, :].argmax(dim=-1) == shift_labels) & mask).sum()
    return token_entropy, correct, mask.sum()


def patch_sft_trainer():
    from transformers import Trainer
    from trl.trainer import sft_trainer

    original = sft_trainer.SFTTrainer.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not _can_fuse(self, model, inputs):
            setattr(self, REWIND_STEP_ATTR, None)
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train" if self.model.training else "eval"
        # As TRL's compute_loss does before it delegates to transformers'.
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        labels = inputs["labels"]
        _State.armed, _State.step = True, None
        try:
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
        finally:
            _State.armed = False
        step, _State.step = _State.step, None
        if step is not None:
            _log_metrics(self, mode, inputs, step.token_entropy, step.correct, step.labelled)
            setattr(self, REWIND_STEP_ATTR, SimpleNamespace(token_loss=step.token_loss))
        else:
            _log_metrics(self, mode, inputs, *_metrics_from_logits(outputs.logits, labels))
            setattr(self, REWIND_STEP_ATTR, None)
        return (loss, outputs) if return_outputs else loss

    sft_trainer.SFTTrainer.compute_loss = compute_loss
