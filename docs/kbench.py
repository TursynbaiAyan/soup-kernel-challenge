"""Per-kernel time and shared memory of solution/fused_ce.py at the real head size."""
import sys
import time

import torch

sys.path.insert(0, sys.argv[1])
import solution.fused_ce as f  # noqa: E402

V, H = 151936, 1536
M = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
w = torch.randn(V, H, device="cuda") * 0.02
x = torch.randn(M, H, device="cuda", dtype=torch.half)
y = torch.empty(M, V, device="cuda", dtype=torch.half)
t = torch.randint(0, V, (M,), device="cuda", dtype=torch.int32)
loss, ent = torch.empty(M, device="cuda"), torch.empty(M, device="cuda")
cor = torch.empty(M, device="cuda", dtype=torch.int32)
dh = torch.zeros(M, H, device="cuda")
sc = torch.ones(1, device="cuda")


def tm(fn, n=5):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def ce():
    f.ce_stats_grad_kernel[(M,)](y, t, loss, ent, cor, V, y.stride(0), M, GRAD_SCALE=256.0,
                                 BLOCK=f.CE_BLOCK, num_warps=f.CE_WARPS)


a = tm(lambda: f._logits(x, w, y))
b = tm(ce)
d = tm(lambda: f._dgrad(y, w, dh, sc))
w16 = w.half()
r1, r2 = tm(lambda: x @ w16.t()), tm(lambda: y @ w16)
tf = 2 * M * H * V / 1e9
print(f"M={M}: logits {a:.1f} ms ({tf / a:.1f} TF/s), ce {b:.1f} ms, dgrad {d:.1f} ms "
      f"({tf / d:.1f} TF/s) | cuBLAS fp16 fwd {r1:.1f}, dgrad {r2:.1f}")
for kern in (f.lm_head_logits_kernel, f.ce_stats_grad_kernel, f.lm_head_dgrad_kernel):
    for dev_cache in kern.device_caches.values():
        for compiled in dev_cache[0].values():
            print(f"  {kern.fn.__name__}: shared {compiled.metadata.shared} B, "
                  f"regs {getattr(compiled, 'n_regs', '?')}")
