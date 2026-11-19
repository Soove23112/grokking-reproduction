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
