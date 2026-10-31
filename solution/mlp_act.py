"""In-place SwiGLU for the Qwen2 MLP: silu(gate) * up forward, and a backward that writes
d(gate) and d(up) over the saved gate and up tensors instead of allocating two new ones (each is
[batch, seq, 8 960] fp16, 2 x ~37 MB per layer per step).

Our own kernel, replacing the shipped example: rows of the [tokens, 8 960] matrices are processed
as 2-D tiles, the sigmoid is computed once as 1 / (1 + e^-g), and the backward reuses its
inputs' storage.
"""

import torch
import triton
import triton.language as tl

ROWS, COLS = 4, 1024


@triton.jit
def swiglu_rows_fwd(g_ptr, u_ptr, y_ptr, n_rows, n_cols, stride,
                    ROWS: tl.constexpr, COLS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)[:, None]
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)[None, :]
    inside = (r < n_rows) & (c < n_cols)
    at = r.to(tl.int64) * stride + c
    g = tl.load(g_ptr + at, mask=inside, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + at, mask=inside, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    tl.store(y_ptr + at, (g * sig * u).to(y_ptr.dtype.element_ty), mask=inside)


@triton.jit
def swiglu_rows_bwd_inplace(dy_ptr, g_ptr, u_ptr, n_rows, n_cols, stride,
                            ROWS: tl.constexpr, COLS: tl.constexpr):
    """Overwrites g with d(gate) and u with d(up)."""
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)[:, None]
    c = tl.program_id(1) * COLS + tl.arange(0, COLS)[None, :]
    inside = (r < n_rows) & (c < n_cols)
    at = r.to(tl.int64) * stride + c
    dy = tl.load(dy_ptr + at, mask=inside, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + at, mask=inside, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + at, mask=inside, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-g))
    silu = g * sig
    d_gate = dy * u * (sig + silu * (1.0 - sig))
    tl.store(g_ptr + at, d_gate.to(g_ptr.dtype.element_ty), mask=inside)
    tl.store(u_ptr + at, (dy * silu).to(u_ptr.dtype.element_ty), mask=inside)


class InplaceSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        shape = gate.shape
        g = gate.reshape(-1, shape[-1]).contiguous()
        u = up.to(gate.dtype).reshape(-1, shape[-1]).contiguous()
        y = torch.empty_like(g)
        grid = (triton.cdiv(g.shape[0], ROWS), triton.cdiv(g.shape[1], COLS))
        swiglu_rows_fwd[grid](g, u, y, g.shape[0], g.shape[1], g.stride(0), ROWS=ROWS, COLS=COLS)
        ctx.save_for_backward(g, u)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        g, u = ctx.saved_tensors
        dy = dy.reshape(g.shape).to(g.dtype).contiguous()
        grid = (triton.cdiv(g.shape[0], ROWS), triton.cdiv(g.shape[1], COLS))
        swiglu_rows_bwd_inplace[grid](dy, g, u, g.shape[0], g.shape[1], g.stride(0),
                                      ROWS=ROWS, COLS=COLS)
        return g.view(ctx.shape), u.view(ctx.shape)


def patch_qwen2_mlp_inplace():
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not x.is_cuda:
            return original(self, x)
        return self.down_proj(InplaceSwiGLU.apply(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = forward
