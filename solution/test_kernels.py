"""Correctness of your kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    ref = torch.nn.functional.silu(gate.float()) * up.float()
    ref.backward(grad.float())
    ref_dgate, ref_dup = gate.grad.float(), up.grad.float()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup, rtol=tol, atol=tol)


def reference_head_loss(hidden, weight, labels, attention_mask, num_items_in_batch=None):
    """What the base run computes under fp16 autocast: fp16 logits, fp32 cross-entropy
    (ForCausalLMLoss), TRL's entropy and accuracy, Soup's per-position losses."""
    logits = (hidden.half() @ weight.half().t())
    target = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
    flat = logits.float().view(-1, logits.shape[-1])
    total = torch.nn.functional.cross_entropy(flat, target.reshape(-1), ignore_index=-100,
                                              reduction="sum")
    labelled = (target != -100).sum()
    loss = total / (labelled if num_items_in_batch is None else num_items_in_batch)
    with torch.no_grad():
        logp = torch.log_softmax(logits.float(), dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)
        token_loss = torch.nn.functional.cross_entropy(
            flat, target.reshape(-1), ignore_index=-100, reduction="none").view_as(target)
        correct = ((logits.argmax(-1) == target) & (target != -100)).sum()
    return loss, entropy * attention_mask, token_loss, correct, labelled


def make_batch(batch, seq, width, vocab, seed=0):
    torch.manual_seed(seed)
    hidden = (torch.randn(batch, seq, width, device="cuda") * 2).requires_grad_(True)
    weight = torch.randn(vocab, width, device="cuda") * 0.05
    lengths = torch.randint(seq // 3, seq + 1, (batch,), device="cuda")
    lengths[0] = seq
    positions = torch.arange(seq, device="cuda")[None, :]
    attention_mask = (positions < lengths[:, None]).long()
    labels = torch.randint(0, vocab, (batch, seq), device="cuda")
    prompt = (lengths // 3)[:, None]
    labels = torch.where((positions >= prompt) & (attention_mask == 1), labels, -100)
    return hidden, weight, labels, attention_mask


@pytest.mark.parametrize("batch,seq,width,vocab,chunk,items", [
    (2, 64, 256, 1000, 1024, None),      # one chunk
    (3, 97, 256, 4099, 64, None),        # many chunks, odd vocabulary, chunk boundary in labels
    (2, 40, 128, 777, 16, 33),           # num_items_in_batch given
    (2, 300, 1536, 151936, 256, None),   # the real head size
])
def test_fused_head_loss_matches_pytorch(batch, seq, width, vocab, chunk, items):
    from solution.fused_ce import fused_lm_head_loss

    hidden, weight, labels, mask = make_batch(batch, seq, width, vocab)
    ref = reference_head_loss(hidden, weight, labels, mask, items)
    ref[0].backward()
    ref_grad, hidden.grad = hidden.grad.clone(), None

    loss, entropy, token_loss, correct, labelled = fused_lm_head_loss(
        hidden, weight, labels, mask, num_items_in_batch=items, chunk=chunk)
    loss.backward()

    torch.testing.assert_close(loss, ref[0], rtol=2e-4, atol=2e-4)
    # entropy sums 151,936 terms; on a T4 single positions differ from the reference by up to ~1%
    # (TRL logs the mean over positions, checked within 3%), so check the mean tightly
    torch.testing.assert_close(entropy, ref[1].float(), rtol=2e-2, atol=5e-2)
    assert abs(float(entropy.sum() - ref[1].float().sum())) <= 1e-3 * float(ref[1].float().sum())
    torch.testing.assert_close(token_loss, ref[2], rtol=1e-3, atol=2e-3)
    assert int(labelled) == int(ref[4])
    # argmax over fp16 logits from a different summation order may flip on near-ties
    assert abs(int(correct) - int(ref[3])) <= max(1, int(ref[4]) // 200)
    scale = ref_grad.abs().max()
    torch.testing.assert_close(hidden.grad / scale, ref_grad / scale, rtol=0, atol=2e-3)
