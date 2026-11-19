"""Our model code on the authors' checkpoints reproduces the losses they recorded.

This checks the architecture, the weight conversion and the data split together; a mistake in any
of them changes the losses by orders of magnitude. The tolerance is the pre-registered 1e-6.
"""
import pytest
import torch

from conftest import needs_authors_run
from grokking import model as gm
from grokking import weights

# state_dicts[i] was saved after the step of epoch 100*i, so the authors recorded its training loss
# at the next epoch: train_losses[100*i + 1]
INDICES = [0, 1, 5, 10, 14, 50, 94, 100, 120, 140, 200, 400, 499]


@needs_authors_run
@pytest.mark.parametrize("i", INDICES)
def test_recorded_training_loss_is_reproduced(i):
    run = weights.authors_run()
    m = gm.load_authors(gm.Transformer(), run["state_dicts"][i])
    tx, ty, _, _ = gm.make_data()
    with torch.no_grad():
        ours = gm.loss_fn(m(tx), ty).item()
    assert ours == pytest.approx(run["train_losses"][100 * i + 1], rel=1e-6)
