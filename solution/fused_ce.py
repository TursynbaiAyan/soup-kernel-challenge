"""Fused output head + causal-LM cross-entropy for a frozen, tied fp32 `lm_head`, in Triton.

What the base run does per step (fp16 autocast on a T4): cast the fp32 `lm_head` weight
(151,936 x 1,536) to fp16, build `[batch, seq, vocab]` fp16 logits for every position (pads
included), upcast them to fp32 for the loss, keep fp32 softmax/gradient buffers of the same shape,
then read the logits again for TRL's entropy and accuracy and once more for Soup's rewind log.

Here, in chunks of real (non-pad) positions:
  1. `lm_head_logits_kernel`: logits = h @ W^T with W read in fp32 and rounded to fp16 in
     registers (the same rounding autocast does, without a 467 MB fp16 copy of W);
  2. `ce_stats_grad_kernel`: one program per position, online over the vocabulary: log-sum-exp,
     entropy, argmax, the label's cross-entropy; for labelled positions a second sweep writes
     (softmax - onehot) * GRAD_SCALE over the logits in place;
  3. `lm_head_dgrad_kernel`: dh = G @ W, again with W read in fp32, scaled back in the epilogue
     and stored in fp32.
The gradient on the hidden states is taken in the forward pass (the head is frozen, so nothing
else needs the logits), and no `[batch, seq, vocab]` tensor exists at any point.

Ideas (not code): chunked fused linear + cross-entropy as in Liger-Kernel's
`fused_linear_cross_entropy` (github.com/linkedin/Liger-Kernel) and Apple's Cut Cross-Entropy
(arXiv:2411.09009); online softmax from Milakov & Gimelshein (arXiv:1805.02867); the grouped tile
order of the Triton matmul tutorial (triton-lang.org, "Matrix Multiplication").
"""

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
# Positions per chunk: the only vocab-sized buffer is CHUNK x vocab fp16 (about 311 MB at 1,024).
CHUNK = 1024
# The gradient rows are stored in fp16 multiplied by this; 256 keeps |g| <= 256 (fp16 max 65,504)
# and lifts small probabilities above fp16's subnormal floor. Divided out in fp32 afterwards.
GRAD_SCALE = 256.0

# Tile sizes, fixed (no autotuning: compile time counts in the measured run); picked from a sweep
# at the real head size (1,024 x 1,536 x 151,936). Both matmuls use 24-32 KB of shared memory,
# well under the T4's 64 KB per block.
LOGITS_BM, LOGITS_BN, LOGITS_BK, LOGITS_WARPS, LOGITS_STAGES = 128, 128, 32, 4, 2
DGRAD_BM, DGRAD_BN, DGRAD_BK, DGRAD_WARPS, DGRAD_STAGES = 256, 128, 32, 8, 2
# Split-K for the gradient matmul: enough programs to fill a T4 (40 SMs) a few times over.
DGRAD_TARGET_PROGRAMS, DGRAD_MAX_SPLIT = 160, 32
CE_BLOCK, CE_WARPS = 2048, 8


@triton.jit(do_not_specialize=["M"])
def lm_head_logits_kernel(x_ptr, w_ptr, y_ptr, M, N, K,
                          stride_xm, stride_wn, stride_ym,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """y[M, N] = x[M, K] @ w[N, K]^T; x fp16/bf16, w fp32 rounded to x's dtype on load, fp32
    accumulation, y in x's dtype. Row tiles vary fastest, so the programs that share a weight
    tile run together and the tile is read from DRAM once and from L2 by the rest."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    pid_m = pid % num_m
    pid_n = pid // num_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    x_ptrs = x_ptr + rm[:, None] * stride_xm + rk[None, :]
    w_ptrs = w_ptr + rn[None, :] * stride_wn + rk[:, None]
    m_ok = rm[:, None] < M
    n_ok = rn[None, :] < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):  # K (hidden size) is a multiple of BK, checked by the caller
        x = tl.load(x_ptrs, mask=m_ok, other=0.0)
        w = tl.load(w_ptrs, mask=n_ok, other=0.0).to(x.dtype)
        acc = tl.dot(x, w, acc)
        x_ptrs += BK
        w_ptrs += BK
    y_ptrs = y_ptr + rm[:, None] * stride_ym + rn[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=m_ok & n_ok)


@triton.jit(do_not_specialize=["n_labelled"])
def ce_stats_grad_kernel(logits_ptr, target_ptr, loss_ptr, entropy_ptr, correct_ptr,
                         N, stride_row, n_labelled,
                         GRAD_SCALE: tl.constexpr, BLOCK: tl.constexpr):
    """One program per position (row of logits). Rows [0, n_labelled) carry a label; for them
    the row is overwritten with (softmax - onehot(label)) * GRAD_SCALE in the logits' dtype.

    Online over the vocabulary in fp32: running max m, s = sum exp(x - m),
    t = sum exp(x - m) * x; then lse = m + log s and entropy = lse - t / s. The argmax keeps the
    first index of the maximum, as torch.argmax does."""
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    cols = tl.arange(0, BLOCK)
    m = -float("inf")
    s = 0.0
    t = 0.0
    best = -float("inf")
    best_idx = 0
    for start in range(0, N, BLOCK):
        idx = start + cols
        ok = idx < N
        x = tl.load(base + idx, mask=ok, other=-float("inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        block_arg = tl.argmax(x, axis=0)
        new_m = tl.maximum(m, block_max)
        alpha = tl.exp(m - new_m)
        e = tl.exp(x - new_m)
        s = s * alpha + tl.sum(e, axis=0)
        t = t * alpha + tl.sum(tl.where(ok, e * x, 0.0), axis=0)
        m = new_m
        better = block_max > best
        best_idx = tl.where(better, start + block_arg, best_idx)
        best = tl.where(better, block_max, best)
    lse = m + tl.log(s)
    tl.store(entropy_ptr + row, lse - t / s)
    labelled = row < n_labelled
    target = tl.load(target_ptr + row)
    if labelled:
        x_target = tl.load(base + target).to(tl.float32)
        tl.store(loss_ptr + row, lse - x_target)
        tl.store(correct_ptr + row, (best_idx == target).to(tl.int32))
        for start in range(0, N, BLOCK):
            idx = start + cols
            ok = idx < N
            x = tl.load(base + idx, mask=ok, other=0.0).to(tl.float32)
            g = tl.exp(x - lse) - (idx == target).to(tl.float32)
            tl.store(base + idx, (g * GRAD_SCALE).to(logits_ptr.dtype.element_ty), mask=ok)
    else:
        tl.store(loss_ptr + row, 0.0)
        tl.store(correct_ptr + row, 0)


@triton.jit(do_not_specialize=["M", "K_PER_SPLIT"])
def lm_head_dgrad_kernel(g_ptr, w_ptr, out_ptr, scale_ptr, M, N, K, K_PER_SPLIT,
                         stride_gm, stride_wk, stride_om,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                         EVEN_K: tl.constexpr):
    """out[M, N] += (g[M, K] @ w[K, N]) * scale over one K slice, fp32 out (zeroed by the
    caller); g fp16/bf16 (the scaled gradient of the logits), w the fp32 head weight
    [vocab, hidden] rounded to g's dtype on load. K is the vocabulary, so it is split across
    programs (split-K, partial sums added atomically). Row tiles vary fastest: the programs that
    stream the same slice of the 0.93 GB fp32 weight run together and share it through L2."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    pid_m = pid % num_m
    pid_n = (pid // num_m) % num_n
    pid_s = pid // (num_m * num_n)
    k_start = pid_s * K_PER_SPLIT
    k_stop = tl.minimum(k_start + K_PER_SPLIT, K)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = k_start + tl.arange(0, BK)
    g_ptrs = g_ptr + rm[:, None].to(tl.int64) * stride_gm + rk[None, :]
    w_ptrs = w_ptr + rk[:, None].to(tl.int64) * stride_wk + rn[None, :]
    m_ok = rm[:, None] < M
    n_ok = rn[None, :] < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(k_start, k_stop, BK):
        if EVEN_K:
            g = tl.load(g_ptrs, mask=m_ok, other=0.0)
            w = tl.load(w_ptrs, mask=n_ok, other=0.0).to(g.dtype)
        else:
            k_ok = (k + tl.arange(0, BK)) < k_stop
            g = tl.load(g_ptrs, mask=m_ok & k_ok[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=n_ok & k_ok[:, None], other=0.0).to(g.dtype)
        acc = tl.dot(g, w, acc)
        g_ptrs += BK
        w_ptrs += BK * stride_wk
    acc = acc * tl.load(scale_ptr)
    out_ptrs = out_ptr + rm[:, None] * stride_om + rn[None, :]
    tl.atomic_add(out_ptrs, acc, mask=m_ok & n_ok)


def _logits(x, weight, out):
    M, K = x.shape
    N = weight.shape[0]
    grid = (triton.cdiv(M, LOGITS_BM) * triton.cdiv(N, LOGITS_BN),)
    lm_head_logits_kernel[grid](x, weight, out, M, N, K, x.stride(0), weight.stride(0),
                                out.stride(0), BM=LOGITS_BM, BN=LOGITS_BN, BK=LOGITS_BK,
                                num_warps=LOGITS_WARPS, num_stages=LOGITS_STAGES)


def _dgrad(g, weight, out, scale):
    """out += g @ weight * scale; `out` must start at zero."""
    M, K = g.shape
    N = weight.shape[1]
    tiles = triton.cdiv(M, DGRAD_BM) * triton.cdiv(N, DGRAD_BN)
    splits = max(1, min(DGRAD_MAX_SPLIT, triton.cdiv(DGRAD_TARGET_PROGRAMS, tiles)))
    k_per_split = triton.cdiv(triton.cdiv(K, splits), DGRAD_BK) * DGRAD_BK
    splits = triton.cdiv(K, k_per_split)
    lm_head_dgrad_kernel[(tiles * splits,)](
        g, weight, out, scale, M, N, K, k_per_split, g.stride(0), weight.stride(0),
        out.stride(0), BM=DGRAD_BM, BN=DGRAD_BN, BK=DGRAD_BK, EVEN_K=K % DGRAD_BK == 0,
        num_warps=DGRAD_WARPS, num_stages=DGRAD_STAGES)


def use_triton_matmul(device):
    """Our matmul kernels on Ampere and newer; cuBLAS on Turing (T4), where tl.dot is slow."""
    return torch.cuda.get_device_capability(device) >= (8, 0)


class FusedHeadLoss(torch.autograd.Function):
    """Loss, per-position metrics and the gradient on the hidden states in one forward."""

    @staticmethod
    def forward(ctx, hidden, weight, target, keep, denominator, compute_dtype, chunk):
        batch, seq, width = hidden.shape
        vocab = weight.shape[0]
        flat = hidden.reshape(-1, width)
        target = target.reshape(-1)
        labelled = target != IGNORE_INDEX
        # Labelled positions first: only they need the second sweep and the gradient matmul.
        lab_rows = labelled.nonzero().squeeze(1)
        rest_rows = (keep.reshape(-1) & ~labelled).nonzero().squeeze(1)
        rows = torch.cat([lab_rows, rest_rows])
        n_lab = lab_rows.numel()
        n_rows = rows.numel()
        x = flat.index_select(0, rows).to(compute_dtype)
        row_target = target.index_select(0, rows).clamp(min=0).to(torch.int32).contiguous()

        with_grad = ctx.needs_input_grad[0]
        loss_rows = torch.empty(n_rows, device=hidden.device, dtype=torch.float32)
        entropy_rows = torch.empty_like(loss_rows)
        correct_rows = torch.empty(n_rows, device=hidden.device, dtype=torch.int32)
        dh_rows = torch.zeros((n_lab, width), device=hidden.device, dtype=torch.float32) \
            if with_grad else None
        if torch.is_tensor(denominator):
            denominator = denominator.to(device=hidden.device, dtype=torch.float32)
        else:
            denominator = torch.tensor(float(denominator), device=hidden.device)
        grad_scale = (1.0 / (GRAD_SCALE * denominator)).reshape(1).contiguous()

        buffer = torch.empty((min(chunk, max(n_rows, 1)), vocab), device=hidden.device,
                             dtype=compute_dtype)
        # On Turing (T4, sm75) Triton's tl.dot is far slower than cuBLAS, so there the two
        # matmuls go to cuBLAS on a per-call fp16 copy of the head (what autocast makes anyway)
        # and Triton keeps the vocabulary pass; on Ampere and newer our matmul kernels run.
        triton_mm = use_triton_matmul(hidden.device)
        w_c = None if triton_mm else weight.to(compute_dtype)
        for start in range(0, n_rows, chunk):
            stop = min(start + chunk, n_rows)
            logits = buffer[: stop - start]
            if triton_mm:
                _logits(x[start:stop], weight, logits)
            else:
                torch.matmul(x[start:stop], w_c.t(), out=logits)
            chunk_lab = max(0, min(n_lab, stop) - start)
            ce_stats_grad_kernel[(stop - start,)](
                logits, row_target[start:stop], loss_rows[start:stop],
                entropy_rows[start:stop], correct_rows[start:stop], vocab, logits.stride(0),
                chunk_lab, GRAD_SCALE=GRAD_SCALE, BLOCK=CE_BLOCK, num_warps=CE_WARPS)
            if with_grad and chunk_lab > 0:
                if triton_mm:
                    _dgrad(logits[:chunk_lab], weight, dh_rows[start:start + chunk_lab],
                           grad_scale)
                else:
                    dh_rows[start:start + chunk_lab] = \
                        torch.matmul(logits[:chunk_lab], w_c).float() * grad_scale
        del buffer, w_c

        loss = loss_rows[:n_lab].sum() / denominator
        token_entropy = torch.zeros(batch * seq, device=hidden.device, dtype=torch.float32)
        token_entropy.index_copy_(0, rows, entropy_rows)
        token_loss = torch.zeros_like(token_entropy)
        token_loss.index_copy_(0, lab_rows, loss_rows[:n_lab])
        correct = correct_rows[:n_lab].sum()

        ctx.hidden_shape, ctx.hidden_dtype = hidden.shape, hidden.dtype
        if with_grad:
            ctx.save_for_backward(lab_rows, dh_rows)
        ctx.mark_non_differentiable(token_entropy, token_loss, correct)
        return (loss, token_entropy.view(batch, seq), token_loss.view(batch, seq), correct)

    @staticmethod
    def backward(ctx, grad_loss, _grad_entropy, _grad_token_loss, _grad_correct):
        lab_rows, dh_rows = ctx.saved_tensors
        batch, seq, width = ctx.hidden_shape
        grad = torch.zeros((batch * seq, width), device=dh_rows.device, dtype=torch.float32)
        grad.index_copy_(0, lab_rows, dh_rows * grad_loss)
        return grad.view(batch, seq, width).to(ctx.hidden_dtype), None, None, None, None, None, \
            None


def fused_lm_head_loss(hidden, weight, labels, attention_mask=None, num_items_in_batch=None,
                       compute_dtype=torch.float16, chunk=CHUNK):
    """`ForCausalLMLoss(lm_head(hidden), labels)` without logits, plus what TRL and Soup read
    from the logits. Returns (loss, token_entropy [B, T], token_loss [B, T], correct, labelled):
    token_loss[:, t] is the cross-entropy of labels[:, t + 1] (0 where it is -100), the layout
    Soup's rewind recorder reads; token_entropy is 0 at positions outside `attention_mask`.

    The loss is the sum over labelled positions divided by `num_items_in_batch` when given,
    else their mean, as transformers' `ForCausalLMLoss` does."""
    if hidden.shape[-1] % LOGITS_BK:
        raise ValueError(f"hidden size {hidden.shape[-1]} is not a multiple of {LOGITS_BK}")
    weight = weight.contiguous()
    target = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]
    target = target.to(hidden.device)
    labelled_mask = target != IGNORE_INDEX
    if attention_mask is None:
        keep = torch.ones_like(labelled_mask)
    else:
        keep = attention_mask.to(hidden.device) != 0
    labelled = labelled_mask.sum()
    denominator = labelled if num_items_in_batch is None else num_items_in_batch
    loss, token_entropy, token_loss, correct = FusedHeadLoss.apply(
        hidden, weight, target, keep, denominator, compute_dtype, chunk)
    return loss, token_entropy, token_loss, correct, labelled
