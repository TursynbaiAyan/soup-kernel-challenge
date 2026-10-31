# Submission: <team name>

## Links
- Repository, final commit: https://github.com/<you>/<repo>/commit/<sha>
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  README.md):
  https://www.kaggle.com/code/<you>/<notebook>?scriptVersionId=<version>
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `lm_head_logits_kernel` | `solution/fused_ce.py` | autocast fp32→fp16 cast of `lm_head` + cuBLAS logits matmul, per chunk of non-pad positions | <…> % |
| `ce_stats_grad_kernel` | `solution/fused_ce.py` | fp32 upcast, `cross_entropy` fwd/bwd, TRL entropy/argmax, Soup per-row loss | <…> % |
| `lm_head_dgrad_kernel` | `solution/fused_ce.py` | backward matmul of `lm_head` (grad on hidden states), split-K | <…> % |
| together | | | <…> % (≥ 5%) |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- chunked fused linear + cross-entropy: Liger-Kernel `fused_linear_cross_entropy`
  (https://github.com/linkedin/Liger-Kernel), Cut Cross-Entropy (https://arxiv.org/abs/2411.09009);
- online softmax (https://arxiv.org/abs/1805.02867);
- tile ordering and split-K: Triton matmul tutorial
  (https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | | | speedup x |
| Peak GPU memory, MiB | | | base/ours x |
| Eval loss (public rows) | | | % vs base |

- [ ] All 10 checks of `harness/check_rules.py` passed (if not: which failed and why)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations you sped up and why, how the kernels work, results against the base (speed,
memory, quality), limits and next steps.

## Limitations
Only the output head and the loss are fused; RMSNorm, RoPE, LoRA and NF4 dequantization run as
in the base. Padding is dropped in the head only. Tiles were tuned on an Ampere GPU (with
headroom for the T4's 64 KB of shared memory); the Kaggle T4 numbers above are the reference.
Next: fused RMSNorm + fp16 cast, fused LoRA add in q/v, NF4 dequant inside the matmul.
