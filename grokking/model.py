"""Data, model and training loop of Nanda et al. 2023 (arXiv 2301.05217).

Checked line by line against the authors' transformers.py and helpers.py
(github.com/mechanistic-interpretability-grokking/progress-measures-paper).
The only difference is how weights are stored: we compute `x @ W`, the authors store W transposed.
load_authors() converts their state_dict into ours.
"""
import math
import random

import torch
import torch.nn.functional as F

P = 113                      # modulus; numbers 0..112
EQ = P                       # token id of "=" (113)
D_VOCAB = P + 1              # 114 tokens
D_MODEL = 128                # residual stream width
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS  # 32
D_MLP = 4 * D_MODEL          # 512 neurons
N_CTX = 3                    # sequence: a, b, =

MATRICES = ["W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"]


# ---------------------------------------------------------------- data
def make_data(p=P, frac=0.3, data_seed=0):
    """All pairs (a, b), shuffled exactly as the authors do; the first `frac` are the training set.

    Returns (train_x, train_y, test_x, test_y): x is a LongTensor (N, 3) of tokens a, b, =;
    y is a LongTensor (N,) with (a + b) mod p.
    """
    pairs = [(a, b, p) for a in range(p) for b in range(p)]
    random.seed(data_seed)   # the stdlib random, as in the authors' code, so the split matches theirs
    random.shuffle(pairs)
    div = int(frac * len(pairs))
    x = torch.tensor(pairs, dtype=torch.long)
    y = (x[:, 0] + x[:, 1]) % p
    return x[:div], y[:div], x[div:], y[div:]


def all_pairs(p=P):
    """All p*p inputs in a-major order (0, 0), (0, 1), ... — the order the Fourier analysis expects."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    x = torch.stack([a, b, torch.full_like(a, p)], dim=1)
    return x, (a + b) % p


def train_mask(frac=0.3, data_seed=0, p=P):
    """Boolean (p*p,) in all_pairs() order: True for pairs in the training set."""
    tx, _, _, _ = make_data(p, frac, data_seed)
    mask = torch.zeros(p * p, dtype=torch.bool)
    mask[tx[:, 0] * p + tx[:, 1]] = True
    return mask


# ---------------------------------------------------------------- model
class Transformer(torch.nn.Module):
    """One-layer transformer without LayerNorm.

    Parameters and shapes (the input multiplies from the left: x @ W):
      W_E   (114, 128)            embedding: one row per token
      W_pos (3, 128)              positional embedding: one row per position
      W_Q, W_K, W_V (4, 128, 32)  one matrix per head
      W_O   (4, 32, 128)          each head's output back into the residual stream
      W_in  (128, 512), b_in (512)   MLP, first layer
      W_out (512, 128), b_out (128)  MLP, second layer
      W_U   (128, 114)            unembedding
    """

    def __init__(self):
        super().__init__()
        e = lambda *shape: torch.nn.Parameter(torch.empty(*shape))
        self.W_E = e(D_VOCAB, D_MODEL)
        self.W_pos = e(N_CTX, D_MODEL)
        self.W_Q = e(N_HEADS, D_MODEL, D_HEAD)
        self.W_K = e(N_HEADS, D_MODEL, D_HEAD)
        self.W_V = e(N_HEADS, D_MODEL, D_HEAD)
        self.W_O = e(N_HEADS, D_HEAD, D_MODEL)
        self.W_in = e(D_MODEL, D_MLP)
        self.b_in = e(D_MLP)
        self.W_out = e(D_MLP, D_MODEL)
        self.b_out = e(D_MODEL)
        self.W_U = e(D_MODEL, D_VOCAB)
        # causal mask: query position q sees key positions k <= q
        self.register_buffer("mask", torch.tril(torch.ones(N_CTX, N_CTX, dtype=torch.bool)))
        init_weights(self, init_seed=0)

    def forward(self, x, return_cache=False):
        """x: LongTensor (batch, 3) -> logits (batch, 3, 114)."""
        # 1. embedding: look up token rows and add the position row
        resid = self.W_E[x] + self.W_pos                                    # (B, 3, 128)

        # 2. attention: each head computes queries, keys and values
        q = torch.einsum("bpd,hde->bhpe", resid, self.W_Q)                  # (B, 4, 3, 32)
        k = torch.einsum("bpd,hde->bhpe", resid, self.W_K)
        v = torch.einsum("bpd,hde->bhpe", resid, self.W_V)
        scores = torch.einsum("bhqe,bhke->bhqk", q, k) / math.sqrt(D_HEAD)  # (B, 4, 3, 3)
        scores = scores.masked_fill(~self.mask, float("-inf"))             # the future is hidden
        attn = scores.softmax(dim=-1)                                        # weights sum to 1 over k
        z = torch.einsum("bhqk,bhke->bhqe", attn, v)                         # (B, 4, 3, 32)
        resid = resid + torch.einsum("bhqe,hed->bqd", z, self.W_O)           # heads summed, residual add

        # 3. MLP: 128 -> 512 -> ReLU -> 128, at every position separately
        pre = resid @ self.W_in + self.b_in                                  # (B, 3, 512)
        post = F.relu(pre)                                                   # neuron activations
        resid = resid + (post @ self.W_out + self.b_out)                     # residual add

        # 4. unembedding: 114 logits at every position
        logits = resid @ self.W_U
        if return_cache:
            return logits, {"attn": attn, "pre": pre, "post": post, "resid_final": resid}
        return logits


def init_weights(model, init_seed=0):
    """The authors' initialisation: N(0,1)/sqrt(128) for every matrix, N(0,1)/sqrt(114) for W_U,
    zero biases. Uses its own generator so the seed is reproducible (the authors did not seed it)."""
    g = torch.Generator().manual_seed(init_seed)
    with torch.no_grad():
        for name in MATRICES:
            w = getattr(model, name)
            scale = math.sqrt(D_VOCAB) if name == "W_U" else math.sqrt(D_MODEL)
            w.copy_(torch.randn(w.shape, generator=g, dtype=torch.float32) / scale)
        model.b_in.zero_()
        model.b_out.zero_()
    return model


def load_authors(model, sd):
    """Load a state_dict in the authors' naming and layout into our model."""
    a = "blocks.0.attn."
    m = "blocks.0.mlp."
    ours = {
        "W_E": sd["embed.W_E"].T,                                   # (128, 114) -> (114, 128)
        "W_pos": sd["pos_embed.W_pos"],
        # theirs: W_Q (head, 32, 128), q = W_Q @ x  ->  ours: x @ W_Q with W_Q (head, 128, 32)
        "W_Q": sd[a + "W_Q"].transpose(1, 2),
        "W_K": sd[a + "W_K"].transpose(1, 2),
        "W_V": sd[a + "W_V"].transpose(1, 2),
        # theirs: W_O (128, 4*32), columns ordered "head, then head dimension"
        "W_O": sd[a + "W_O"].T.reshape(N_HEADS, D_HEAD, D_MODEL),
        "W_in": sd[m + "W_in"].T,
        "b_in": sd[m + "b_in"],
        "W_out": sd[m + "W_out"].T,
        "b_out": sd[m + "b_out"],
        "W_U": sd["unembed.W_U"],
    }
    with torch.no_grad():
        for name, t in ours.items():
            getattr(model, name).copy_(t.to(getattr(model, name).dtype))
    return model


def sum_sq_weights(model):
    return sum(p.detach().pow(2).sum().item() for p in model.parameters())


# ---------------------------------------------------------------- loss and training
def loss_fn(logits, y):
    """Cross-entropy at the "=" position (the last one) over all 114 logits, as the authors do.
    Computed in float64 on the CPU: in float32 losses below ~1e-7 are indistinguishable
    and training shows loss spikes."""
    lg = logits[:, -1].cpu().double()   # move to CPU first: MPS has no float64
    return F.cross_entropy(lg, y.to("cpu"))


def accuracy(logits, y):
    """Share of correct answers; the logit of "=" does not take part."""
    return (logits[:, -1, :P].argmax(-1).cpu() == y.cpu()).double().mean().item()


def make_optimizer(model, lr=1e-3, weight_decay=1.0):
    """AdamW as in the authors' code plus warm-up lr * min(step/10, 1), so the very first step has lr = 0.
    Weight decay applies to every parameter, biases and embeddings included."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(step / 10, 1))
    return opt, sched


def train(model, cfg, log=None, save=None):
    """Full-batch training.

    cfg: dict with keys epochs, lr, weight_decay, frac, data_seed, device, eval_every.
    log(epoch, metrics) is called every eval_every epochs, and so is save(epoch, state_dict).
    Same order as the authors: at epoch e we compute the loss of the current model, take a step,
    then save, so the weights saved at epoch e have taken e + 1 steps.
    Returns the list of training losses, one per epoch.
    """
    dev = cfg.get("device", "cpu")
    tx, ty, vx, vy = make_data(P, cfg.get("frac", 0.3), cfg.get("data_seed", 0))
    tx, vx = tx.to(dev), vx.to(dev)
    model.to(dev)
    opt, sched = make_optimizer(model, cfg.get("lr", 1e-3), cfg.get("weight_decay", 1.0))
    every = cfg.get("eval_every", 100)
    losses = []
    for epoch in range(cfg["epochs"]):
        logits = model(tx)
        loss = loss_fn(logits, ty)
        losses.append(loss.item())
        if epoch % every == 0 and log is not None:
            with torch.no_grad():
                vlogits = model(vx)
            log(epoch, {
                "train_loss": loss.item(), "test_loss": loss_fn(vlogits, vy).item(),
                "train_acc": accuracy(logits, ty), "test_acc": accuracy(vlogits, vy),
                "sum_sq_weights": sum_sq_weights(model),
            })
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
        if epoch % every == 0 and save is not None:
            save(epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    return losses
