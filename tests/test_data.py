import torch

from grokking import model as ref


def test_split_sizes_match_paper():
    tx, ty, vx, vy = ref.make_data()
    assert (len(tx), len(vx)) == (3830, 8939)


def test_labels_are_sum_mod_p():
    tx, ty, _, _ = ref.make_data()
    assert torch.equal(ty, (tx[:, 0] + tx[:, 1]) % 113)
    assert (tx[:, 2] == 113).all()


def test_split_is_a_partition_of_all_pairs():
    tx, _, vx, _ = ref.make_data()
    keys = torch.cat([tx, vx])[:, 0] * 113 + torch.cat([tx, vx])[:, 1]
    assert torch.equal(keys.sort().values, torch.arange(113 * 113))


def test_split_matches_authors_shuffle():
    tx, _, _, _ = ref.make_data()
    assert tx[:3, :2].tolist() == [[18, 34], [10, 83], [55, 28]]


def test_relabel_multiplies_every_number_and_keeps_order():
    tx, ty, vx, vy = ref.make_data()
    rx, ry, rvx, rvy = ref.make_data(relabel=7)
    assert torch.equal(rx[:, :2], tx[:, :2] * 7 % 113) and torch.equal(rvx[:, :2], vx[:, :2] * 7 % 113)
    assert torch.equal(ry, ty * 7 % 113) and (rx[:, 2] == 113).all()
