import torch

from src.audit import g6_locality, per_sample_shuffle

# G6 locality classes and the per-sample shuffle used in Table 5.


def test_per_sample_shuffle_preserves_multiset():
    x = torch.arange(8).float().view(1, 8, 1, 1, 1).expand(5, 8, 1, 1, 1).contiguous()
    gen = torch.Generator().manual_seed(0)
    y = per_sample_shuffle(x, gen)
    for i in range(5):
        a = x[i, :, 0, 0, 0].sort().values
        b = y[i, :, 0, 0, 0].sort().values
        torch.testing.assert_close(a, b)
    assert not torch.equal(x, y)


def test_g6_locality_classes():
    assert g6_locality(0.9, 0.5, 0.9, 0.9, False) == "frame-local"
    assert g6_locality(0.5, 0.9, 0.9, 0.9, False) == "order-invariant multi-frame"
    assert g6_locality(0.5, 0.5, 0.9, 0.5, True) == "order-encoded"
    assert g6_locality(0.5, 0.5, 0.9, 0.5, False) == "order-encoded"
    assert g6_locality(0.5, 0.5, 0.5, 0.5, False) == "inconclusive"
