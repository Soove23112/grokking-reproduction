import math

import pytest
import torch

from grokking import model as ref
from conftest import needs_authors_run
from grokking import weights


def test_parameter_count_matches_authors_code():
    assert sum(p.numel() for p in ref.Transformer().parameters()) == 226_816


def test_init_scales_like_authors():
    m = ref.init_weights(ref.Transformer(), init_seed=0)
    assert m.W_U.std().item() == pytest.approx(1 / math.sqrt(114), rel=0.02)
    assert m.W_in.std().item() == pytest.approx(1 / math.sqrt(128), rel=0.02)
    assert m.b_in.abs().sum() == 0 and m.b_out.abs().sum() == 0


def test_init_seed_controls_weights():
    a, b, c = (ref.init_weights(ref.Transformer(), s).W_E for s in (0, 0, 1))
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_attention_is_causal():
    m = ref.Transformer().double()
    x = torch.tensor([[5, 7, 113], [5, 90, 113]])
    out = m(x)
    assert torch.allclose(out[0, 0], out[1, 0])
    assert not torch.allclose(out[0, 2], out[1, 2])


def test_first_training_steps_are_pinned():
    m = ref.init_weights(ref.Transformer(), init_seed=0)
    losses = ref.train(m, dict(epochs=3, device="cpu"))
    # step 0 has lr = 0 (warm-up), so the first two losses are equal
    assert losses[0] == pytest.approx(losses[1], rel=1e-12)
    assert losses == pytest.approx(PINNED_LOSSES, rel=1e-6)


@needs_authors_run
def test_authors_final_weights_reproduce_paper_loss():
    m = ref.load_authors(ref.Transformer(), weights.authors_state()).double()
    x, y = ref.all_pairs()
    with torch.no_grad():
        loss = ref.loss_fn(m(x), y).item()
    assert loss == pytest.approx(2.41e-7, rel=0.01)


PINNED_LOSSES = [4.776606311569237, 4.776606311569237, 4.772054352207212]


def test_accuracy_ignores_the_logit_of_equals():
    logits = torch.zeros(2, 3, 114)
    logits[0, -1, 7] = 1.0
    logits[1, -1, 113] = 5.0                           # "=" wins overall but is not a valid answer
    logits[1, -1, 3] = 1.0
    assert ref.accuracy(logits, torch.tensor([7, 3])) == 1.0
    assert ref.accuracy(logits, torch.tensor([7, 4])) == 0.5


def test_relabel_tokens_makes_the_relabeled_model_equivalent():
    """Model with tokens renamed x -> 7x, fed renamed inputs, gives the original logits at the renamed answers."""
    m = ref.init_weights(ref.Transformer(), 3).double()
    r = ref.relabel_tokens(ref.init_weights(ref.Transformer(), 3).double(), 7)
    x, _ = ref.all_pairs()
    rx = torch.stack([x[:, 0] * 7 % 113, x[:, 1] * 7 % 113, x[:, 2]], dim=1)
    out = torch.arange(114)
    out[:113] = out[:113] * 7 % 113
    assert torch.allclose(r(rx)[:, -1, out], m(x)[:, -1], rtol=0, atol=1e-12)
    assert torch.equal(r.W_E[113], m.W_E[113])
