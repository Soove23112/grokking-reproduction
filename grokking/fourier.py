"""Fourier analysis of the model and every measurement from sections 4-5 of Nanda et al. 2023.

Everything runs on the CPU in float64 over all 113*113 pairs (a, b) in a-major order (model.all_pairs).
Logits are taken at the "=" position and only the 113 number logits (not the logit of "=" itself),
as in the authors' code (transformers.py, take_metrics: model(all_data)[:, -1, :-1]).

Fourier basis: 113 vectors of length 113 — the constant, cos(w_k x) and sin(w_k x) for k = 1..56,
w_k = 2*pi*k/113, each normalised to unit length. The basis is orthonormal, so the transform and its
inverse are exact and projections are plain dot products. Indices: 0 is the constant, 2k-1 is cos k,
2k is sin k.
"""
import math
from functools import cached_property

import torch
import torch.nn.functional as F

from grokking import model as gm

P = gm.P
N_FREQ = P // 2  # 56
FREQS = range(1, N_FREQ + 1)


# ---------------------------------------------------------------- basis
def fourier_basis(p=P):
    """(p, p): row i is the i-th basis function evaluated at x = 0..p-1."""
    x = torch.arange(p, dtype=torch.float64)
    rows = [torch.ones(p, dtype=torch.float64)]
    for k in range(1, p // 2 + 1):
        rows.append(torch.cos(2 * math.pi * k * x / p))
        rows.append(torch.sin(2 * math.pi * k * x / p))
    B = torch.stack(rows)
    return B / B.norm(dim=1, keepdim=True)


B = fourier_basis()
NAMES = ["const"] + [f"{t} {k}" for k in FREQS for t in ("cos", "sin")]


def cos_idx(k):
    return 2 * k - 1


def sin_idx(k):
    return 2 * k


def block_idx(k):
    """Basis indices {const, cos k, sin k}: products of two of them span the 2nd-degree polynomials of frequency k."""
    return [0, cos_idx(k), sin_idx(k)]


def per_freq_norm(comp):
    """(113,) component norms -> (56,) norm of each frequency, cos and sin together."""
    return torch.stack([(comp[cos_idx(k)] ** 2 + comp[sin_idx(k)] ** 2).sqrt() for k in FREQS])


def block_share(f2, k):
    """Share of the 2D energy f2 (p, p, ...) that sits in the 3x3 block of frequency k."""
    idx = block_idx(k)
    return f2[idx][:, idx].sum((0, 1))


def to_fourier_2d(t):
    """t: (p*p, ...) over pairs (a, b) -> (p, p, ...) coefficients: [f, g] is basis f along a, basis g along b."""
    t = t.reshape(P, P, -1)
    return torch.einsum("abz,fa,gb->fgz", t, B, B)


def from_fourier_2d(f):
    """Inverse: (p, p, z) -> (p*p, z)."""
    return torch.einsum("fgz,fa,gb->abz", f, B, B).reshape(P * P, -1)


def xpy_directions(k):
    """The two unit directions in 2D Fourier space (p, p) for frequency k:
    cos(w_k(a+b)) = cos a cos b - sin a sin b  and  sin(w_k(a+b)) = sin a cos b + cos a sin b."""
    c, s = cos_idx(k), sin_idx(k)
    dc = torch.zeros(P, P, dtype=torch.float64)
    dc[c, c], dc[s, s] = 1 / math.sqrt(2), -1 / math.sqrt(2)
    ds = torch.zeros(P, P, dtype=torch.float64)
    ds[s, c], ds[c, s] = 1 / math.sqrt(2), 1 / math.sqrt(2)
    return dc, ds


def xpy_part(f, freqs):
    """Projection of 2D coefficients f (p, p, z) onto cos/sin(w_k(a+b)) for every k in freqs."""
    out = torch.zeros_like(f)
    for k in freqs:
        for d in xpy_directions(k):
            coef = torch.einsum("fgz,fg->z", f, d)
            out += d[:, :, None] * coef[None, None, :]
    return out


def remove_xpy(f, freqs):
    """f without its cos/sin(w_k(a+b)) components for k in freqs."""
    return f - xpy_part(f, freqs)


def keep_xpy(f, freqs, keep_const=True):
    """Only the constant (optional) and the cos/sin(w_k(a+b)) components for k in freqs."""
    out = xpy_part(f, freqs)
    if keep_const:
        out[0, 0] += f[0, 0]
    return out


def gini(c):
    """Gini coefficient after Hurley & Rickard 2009: 0 — spread evenly, 1 - 1/n — all in one component."""
    c = c.abs().sort().values
    n = len(c)
    k = torch.arange(1, n + 1, dtype=c.dtype)
    return (1 - 2 * ((c / c.sum()) * (n - k + 0.5) / n).sum()).item()


def key_freqs(freq, fve, thresh=0.85, min_neurons=10):
    """A frequency is key if it is the best frequency, with FVE > thresh, of at least min_neurons neurons.
    Returns (key frequencies, neuron count per frequency)."""
    counts = torch.bincount(freq[fve > thresh], minlength=N_FREQ + 1)
    return [k for k in FREQS if counts[k] >= min_neurons], counts


# ---------------------------------------------------------------- running a model
def load(sd=None, authors=False, path=None):
    """float64 eval-mode model from our state_dict, an authors' state_dict (authors=True), or a file."""
    m = gm.Transformer()
    if path is not None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    if authors:
        gm.load_authors(m, sd)
    else:
        m.load_state_dict(sd)
    return m.double().eval()


class Run:
    """A model run over all 12 769 pairs, with every measurement the paper makes on it."""

    def __init__(self, model):
        self.m = model
        self.x, self.y = gm.all_pairs()
        self.is_train = gm.train_mask()
        with torch.no_grad():
            logits, c = model(self.x, return_cache=True)
        self.logits = logits[:, -1, :P]                     # (p*p, 113)
        self.post = c["post"][:, -1]                         # (p*p, 512) neurons at "="
        self.attn = c["attn"]                                # (p*p, 4, 3, 3)
        mlp_out = self.post @ model.W_out + model.b_out
        self.resid_mid = c["resid_final"][:, -1] - mlp_out   # residual stream before the MLP
        self.W_L = model.W_out.detach() @ model.W_U.detach()[:, :P]   # (512, 113): neuron -> logit
        self.W_E = model.W_E.detach()[:P]                    # (113, 128): number embeddings

    # --- losses
    def loss(self, logits=None, mask=None):
        lg = self.logits if logits is None else logits
        y = self.y
        if mask is not None:
            lg, y = lg[mask], y[mask]
        return F.cross_entropy(lg, y).item()

    def loss_2d(self, f, mask=None):
        """Loss of logits given by their 2D Fourier coefficients f."""
        return self.loss(from_fourier_2d(f), mask)

    def logits_from_post(self, post):
        """Logits when the neuron activations are replaced by `post` (the rest of the network unchanged)."""
        return ((self.resid_mid + post @ self.m.W_out + self.m.b_out) @ self.m.W_U)[:, :P].detach()

    # --- section 4.1: embedding
    def embed_freq_norms(self):
        """Per-component norms of the embedding in the Fourier basis, and per frequency k = 1..56."""
        comp = (B @ self.W_E).norm(dim=1)
        return comp, per_freq_norm(comp)

    # --- section 4.3: neurons
    def neuron_fve(self):
        """For each neuron: its best frequency k and the fraction of variance (FVE) explained by
        the 2nd-degree polynomial of that frequency — the 3x3 block {const, cos k, sin k}^2 in 2D Fourier
        space, (0, 0) excluded by centring. Frequencies 1..56 (the authors' range(1, p//2) skips 56)."""
        f2 = to_fourier_2d(self.post - self.post.mean(0)).pow(2)   # (p, p, 512)
        total = f2.sum((0, 1))
        fve = torch.stack([block_share(f2, k) / total for k in FREQS])
        best, arg = fve.max(0)
        return arg + 1, best, fve

    def key_freqs(self, thresh=0.85, min_neurons=10):
        freq, fve, _ = self.neuron_fve()
        return key_freqs(freq, fve, thresh, min_neurons)

    def head_freq(self, head=0):
        """Dominant frequency of head `head`'s attention from "=" to a, and its share (fig. 4, left)."""
        a = self.attn[:, head, 2, 0]
        f2 = to_fourier_2d((a - a.mean())[:, None])[:, :, 0].pow(2)
        per = torch.stack([block_share(f2, k) for k in FREQS]) / f2.sum()
        return int(per.argmax()) + 1, per.max().item()

    # --- section 4.1: logits in 2D Fourier space
    @cached_property
    def logits_2d(self):
        return to_fourier_2d(self.logits)

    def logit_fourier_norms(self):
        """(p, p): norm over the 113 logits of each 2D Fourier component (fig. 4, right)."""
        return self.logits_2d.norm(dim=-1)

    # --- section 4.2: W_L
    def W_L_fourier(self):
        """(512, 113): W_L in the Fourier basis along the logit axis. Column cos k is u_k, sin k is v_k."""
        return self.W_L @ B.T

    def W_L_freq_norms(self):
        """Norm of the W_L components per frequency (fig. 3, right): norm over neurons, cos and sin together."""
        comp = self.W_L_fourier().norm(dim=0)
        return comp, per_freq_norm(comp)

    def W_L_residual(self, freqs):
        """Share of the Frobenius norm of W_L outside the cos/sin directions of the key frequencies."""
        keep = [i for k in freqs for i in (cos_idx(k), sin_idx(k))]
        approx = self.W_L_fourier()[:, keep] @ B[keep]
        return ((self.W_L - approx).norm() / self.W_L.norm()).item()

    def uv_directions(self, freqs):
        """{(k, 'cos' | 'sin'): direction in neuron space (512,)}."""
        Wf = self.W_L_fourier()
        return {(k, t): Wf[:, i(k)] for k in freqs for t, i in (("cos", cos_idx), ("sin", sin_idx))}

    def table1(self, freqs):
        """Table 1: activations projected on u_k / v_k, and the share of that projection's variance
        explained by the single term cos(w_k(a+b)) or sin(w_k(a+b))."""
        rows = []
        for (k, t), u in self.uv_directions(freqs).items():
            proj = self.post @ u
            f = to_fourier_2d((proj - proj.mean())[:, None])[:, :, 0]
            d = xpy_directions(k)[0 if t == "cos" else 1]
            coef = (f * d).sum()
            rows.append((k, t, coef.item(), (coef ** 2 / f.pow(2).sum()).item()))
        return rows

    def logits_cos_fit(self, freqs):
        """Logits ~ sum_k alpha_k cos(w_k(a+b-c)), alpha by least squares. Returns the FVE, alpha and
        the loss of the fitted logits. Logits are centred over c within each pair: shifting all
        113 logits of a pair by a constant does not change the softmax."""
        a, b = self.x[:, 0].double(), self.x[:, 1].double()
        c = torch.arange(P, dtype=torch.float64)
        X = torch.stack([torch.cos(2 * math.pi * k / P * (a[:, None] + b[:, None] - c[None, :])).flatten()
                         for k in freqs], 1)
        y = (self.logits - self.logits.mean(1, keepdim=True)).flatten()
        alpha = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
        fve = 1 - (y - X @ alpha).pow(2).sum() / y.pow(2).sum()
        return fve.item(), alpha, self.loss((X @ alpha).reshape(P * P, P))

    # --- section 4.4: ablations
    def ablate_neurons_poly(self, thresh=0.85):
        """Replace every neuron with FVE > thresh by the 2nd-degree polynomial of its best frequency
        (mean + its 3x3 block). Returns (loss, number of neurons replaced)."""
        freq, fve, _ = self.neuron_fve()
        sel = (fve > thresh).nonzero()[:, 0].tolist()
        mean = self.post.mean(0)
        f = to_fourier_2d(self.post - mean)
        new = self.post.clone()
        for n in sel:
            idx = block_idx(freq[n].item())
            g = torch.zeros_like(f[:, :, n:n + 1])
            g[[[i] for i in idx], idx] = f[[[i] for i in idx], idx, n:n + 1]
            new[:, n] = mean[n] + from_fourier_2d(g)[:, 0]
        return self.loss(self.logits_from_post(new)), len(sel)

    def ablate_neurons_strict(self, freqs):
        """Every neuron -> its mean + its cos/sin(w_k(a+b)) components for the key frequencies."""
        mean = self.post.mean(0)
        new = mean + from_fourier_2d(keep_xpy(to_fourier_2d(self.post - mean), freqs, keep_const=False))
        return self.loss(self.logits_from_post(new))

    def project_post(self, freqs, null=False):
        """Project the neuron activations onto the 10 directions u_k, v_k (or onto their complement)."""
        U = torch.stack(list(self.uv_directions(freqs).values()), 1)    # (512, 10)
        Q, _ = torch.linalg.qr(U)
        proj = self.post @ Q @ Q.T
        return self.loss(self.logits_from_post(self.post - proj if null else proj))

    def ablate_freq_loss(self, k):
        """Remove cos/sin(w_k(a+b)) of one frequency from the logits; loss over all pairs (fig. 6)."""
        return self.loss_2d(remove_xpy(self.logits_2d, [k]))

    # --- section 5.1: progress measures
    def keep_blocks_loss(self, freqs):
        """Logit ablation from section 4.4: zero every one of the 113*113 components except the
        8*len(freqs) in the 3x3 blocks {const, cos k, sin k}^2 of the key frequencies, (0, 0) excluded.
        The constant is not kept. Loss over all pairs. For 5 frequencies this is the paper's "113*113 - 40"."""
        f = self.logits_2d
        out = torch.zeros_like(f)
        for k in freqs:
            for i in block_idx(k):
                for j in block_idx(k):
                    out[i, j] = f[i, j]
        out[0, 0] = 0
        return self.loss_2d(out)

    def restricted_loss(self, freqs):
        """Keep only the constant and cos/sin(w_k(a+b)) of the key frequencies; loss over all pairs.
        Keeping the constant is the authors' bias_correction."""
        return self.loss_2d(keep_xpy(self.logits_2d, freqs))

    def excluded_loss(self, freqs):
        """Subtract cos/sin(w_k(a+b)) of the key frequencies; loss on the training pairs."""
        return self.loss_2d(remove_xpy(self.logits_2d, freqs), mask=self.is_train)

    def gini_embed(self):
        return gini(self.embed_freq_norms()[0])

    def gini_W_L(self):
        return gini(self.W_L_freq_norms()[0])
