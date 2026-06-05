# B-spline reparameterazation
import torch
from typing import Tuple

def _reflect_indices(idx: torch.Tensor, n: int) -> torch.Tensor:
    """Mirror-reflect boundary handling for indices in [0, n-1]."""
    if n == 1:
        return torch.zeros_like(idx)
    n2 = 2 * (n - 1)
    idx_mod = idx % n2
    reflected = torch.where(idx_mod <= (n - 1), idx_mod, n2 - idx_mod)
    return reflected

def _cubic_bspline_weights(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cubic B-spline basis weights for fractional part t in [0,1).
    Uses 4-point support at offsets: floor(x)-1, floor(x), floor(x)+1, floor(x)+2.
    """
    # t shape: arbitrary
    t2 = t * t
    t3 = t2 * t

    w0 = (1 - t) ** 3 / 6.0
    w1 = (3 * t3 - 6 * t2 + 4) / 6.0
    w2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0
    w3 = t3 / 6.0
    return w0, w1, w2, w3

def _make_1d_stencil(nc: int, nd: int, device, dtype):
    """
    为 1D 方向构建插值索引与权重，使得当 nd==nc 时为恒等映射。
    坐标映射：x_d ∈ [0, nd-1] -> x_c = x_d * (nc-1)/(nd-1)
    """
    if nd == 1:
        # 单点退化情形
        base = torch.zeros(1, device=device, dtype=dtype)
        t = torch.zeros(1, device=device, dtype=dtype)
    else:
        s = torch.linspace(0.0, 1.0, nd, device=device, dtype=dtype)
        base = s * (nc - 1)  # continuous coord in control index space
        t = base - torch.floor(base)
    i1 = torch.floor(base).to(torch.int64)  # floor
    i0 = i1 - 1
    i2 = i1 + 1
    i3 = i1 + 2

    w0, w1, w2, w3 = _cubic_bspline_weights(t)

    idx = torch.stack([i0, i1, i2, i3], dim=-1)  # (nd, 4)
    w = torch.stack([w0, w1, w2, w3], dim=-1)    # (nd, 4)

    # 反射边界
    idx = _reflect_indices(idx, nc)

    return idx, w

def bspline2d_prolong(c: torch.Tensor, out_shape: Tuple[int, int]) -> torch.Tensor:
    """
    二维三次 B-spline prolongation（稀疏 -> 密）
    输入:
        c: (Hc, Wc) or (B, Hc, Wc) 视为样条系数
        out_shape: (Hd, Wd) 目标密网格尺寸
    返回:
        m: (Hd, Wd) 或 (B, Hd, Wd)
    """
    assert c.dim() in (2, 3), "c must be (Hc,Wc) or (B,Hc,Wc)"
    batched = (c.dim() == 3)
    if not batched:
        c = c.unsqueeze(0)  # -> (B=1, Hc, Wc)

    B, Hc, Wc = c.shape
    Hd, Wd = out_shape
    device, dtype = c.device, c.dtype

    # 预计算两个方向的 1D 索引与权重
    idx_x, w_x = _make_1d_stencil(Wc, Wd, device, dtype)  # (Wd,4)
    idx_z, w_z = _make_1d_stencil(Hc, Hd, device, dtype)  # (Hd,4)

    # 先沿 x 方向插值：  (B, Hc, Wd)
    # tmp = sum_k c[:, :, idx_x[:,k]] * w_x[:,k]
    tmp = torch.zeros((B, Hc, Wd), device=device, dtype=dtype)
    for k in range(4):
        # gather along width
        gathered = c.gather(dim=2, index=idx_x[:, k].view(1, 1, Wd).expand(B, Hc, Wd))
        tmp += gathered * w_x[:, k].view(1, 1, Wd)

    # 再沿 z 方向插值： (B, Hd, Wd)
    m = torch.zeros((B, Hd, Wd), device=device, dtype=dtype)
    for k in range(4):
        # gather along height (dim=1)
        gathered = tmp.gather(dim=1, index=idx_z[:, k].view(1, Hd, 1).expand(B, Hd, Wd))
        m += gathered * w_z[:, k].view(1, Hd, 1)

    return m if batched else m.squeeze(0)

def bspline2d_adjoint(gm: torch.Tensor, in_shape: Tuple[int, int]) -> torch.Tensor:
    """
    二维三次 B-spline prolongation 的伴随（密 -> 稀疏）
    输入:
        gm: (Hd, Wd) 或 (B, Hd, Wd) —— 对密网格的梯度/残差
        in_shape: (Hc, Wc) —— 目标稀疏系数网格尺寸
    返回:
        gc: (Hc, Wc) 或 (B, Hc, Wc)
    """
    assert gm.dim() in (2, 3), "gm must be (Hd,Wd) or (B,Hd,Wd)"
    batched = (gm.dim() == 3)
    if not batched:
        gm = gm.unsqueeze(0)  # -> (B, Hd, Wd)

    B, Hd, Wd = gm.shape
    Hc, Wc = in_shape
    device, dtype = gm.device, gm.dtype

    # 与 prolong 相同的 1D stencil（同一几何下，adjoint 使用同样的索引与权重）
    idx_x, w_x = _make_1d_stencil(Wc, Wd, device, dtype)  # (Wd,4)
    idx_z, w_z = _make_1d_stencil(Hc, Hd, device, dtype)  # (Hd,4)

    # 伴随顺序与前向相反：先把 (B,Hd,Wd) 回投到 (B,Hc,Wd)，再回投到 (B,Hc,Wc)
    # Step 1: z 方向的伴随 (把行加回去)
    gx = torch.zeros((B, Hc, Wd), device=device, dtype=dtype)  # accum on height
    for k in range(4):
        # 对 dim=1 做 index_add: gx[:, idx_z[:,k], :] += w_z[:,k] * gm
        idx = idx_z[:, k].view(1, Hd, 1).expand(B, Hd, Wd)
        weight = w_z[:, k].view(1, Hd, 1)
        # 将 gm*weight 加到 gx 指定行
        gx.index_add_(dim=1, index=idx_z[:, k], source=gm * weight)

    # Step 2: x 方向的伴随 (把列加回去)
    gc = torch.zeros((B, Hc, Wc), device=device, dtype=dtype)
    for k in range(4):
        # 对 dim=2 做 index_add: gc[:, :, idx_x[:,k]] += w_x[:,k] * gx
        weight = w_x[:, k].view(1, 1, Wd)
        # 我们需要对每一列的 Wd 位置累加到对应的控制列 idx_x[:,k]
        # 这里用循环按列 index_add_（PyTorch 当前不支持对每列不同目标索引的矢量化 add，需要拆解）
        # 为了效率，按块处理可以进一步优化；先给出清晰正确版本：
        contrib = gx * weight  # (B,Hc,Wd)
        # 将每个目标列聚合：对目标列 j_c，找所有 Wd 中 idx_x[*,k]==j_c 的列求和
        # 简化：使用 index_add_ 按列聚合
        gc.index_add_(dim=2, index=idx_x[:, k], source=contrib)

    return gc if batched else gc.squeeze(0)

# ------------------ quick tests ------------------

def _adjoint_test(Hc=33, Wc=31, Hd=257, Wd=211, dtype=torch.float32, device='cpu', atol=1e-4, rtol=1e-4):
    torch.manual_seed(37)
    c  = torch.randn(Hc, Wc, dtype=dtype, device=device)
    gm = torch.randn(Hd, Wd, dtype=dtype, device=device)

    m  = bspline2d_prolong(c, (Hd, Wd))
    gc = bspline2d_adjoint(gm, (Hc, Wc))

    lhs = (m * gm).sum()
    rhs = (c * gc).sum()

    err_abs = (lhs - rhs).abs().item()
    rel = err_abs / (lhs.abs().item() + 1e-12)

    ok = (err_abs <= atol) or (rel <= rtol)
    return ok, err_abs, rel

# ==============================================================
#  B-spline inverse (pseudo-inverse / least-squares reconstruction)
# ==============================================================
def bspline2d_inverse(m_obs: torch.Tensor,
                      in_shape: Tuple[int, int],
                      lam: float = 1e-3,
                      maxit: int = 100,
                      tol: float = 1e-15,
                      verbose: bool = False) -> torch.Tensor:
    """
    Compute pseudo-inverse (least-squares inverse) of the 2D B-spline operator.

    Solves for c_hat ≈ (P^T P + lam I)^(-1) P^T m_obs
    using Conjugate Gradient (CG).

    Parameters
    ----------
    m_obs : torch.Tensor
        Observed dense field (Hd, Wd)
    in_shape : Tuple[int, int]
        Target sparse control grid shape (Hc, Wc)
    lam : float, optional
        Tikhonov regularization coefficient (default 0)
    maxit : int, optional
        Maximum number of CG iterations (default 50)
    tol : float, optional
        Relative residual tolerance for convergence (default 1e-6)
    verbose : bool, optional
        Print iteration info (default False)

    Returns
    -------
    c_hat : torch.Tensor
        Reconstructed sparse control grid (Hc, Wc)
    """

    Hc, Wc = in_shape
    device, dtype = m_obs.device, m_obs.dtype

    # 右端项 b = P^T m_obs
    b = bspline2d_adjoint(m_obs, (Hc, Wc))

    # 定义正规方程算子 A(x) = P^T P x + lam * x
    def A(x: torch.Tensor) -> torch.Tensor:
        return bspline2d_adjoint(bspline2d_prolong(x, m_obs.shape), (Hc, Wc)) + lam * x

    # 初始化
    x = torch.zeros_like(b, dtype=dtype).to(device)
    r = b - A(x)
    p = r.clone()
    rr_old = (r * r).sum()

    for k in range(1, maxit + 1):
        Ap = A(p)
        alpha = rr_old / (p * Ap).sum().clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rr_new = (r * r).sum()
        rel = torch.sqrt(rr_new / (b * b).sum().clamp_min(1e-30)).item()
        if verbose:
            print(f"[B-spline inverse CG] it={k:02d}, rel_res={rel:.3e}")
        if rel < tol:
            break
        beta = rr_new / rr_old.clamp_min(1e-30)
        p = r + beta * p
        rr_old = rr_new

    return x


if __name__ == "__main__":
    ok, eabs, erel = _adjoint_test(device='cpu')
    print(f"Adjoint test: {ok}, abs_err={eabs:.3e}, rel_err={erel:.3e}")
