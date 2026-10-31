"""Our solution: a fused output head + cross-entropy in Triton (fused_ce.py), plugged into Soup
by patch_ce.py, and an in-place SwiGLU for the MLP (mlp_act.py).

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer.
"""

from .patch_ce import patch_qwen2_causal_lm, patch_sft_trainer
from .mlp_act import patch_qwen2_mlp_inplace


def apply():
    patch_qwen2_mlp_inplace()  # our in-place SwiGLU (replaces the shipped example)
    patch_qwen2_causal_lm()  # lm_head + loss without logits: our kernels
    patch_sft_trainer()  # TRL's metrics and Soup's rewind log from the fused values
