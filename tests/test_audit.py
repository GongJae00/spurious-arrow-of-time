import torch

from src.audit import gate6, per_sample_shuffle

# Gate 6 classes (Algorithm 1 Phase 2) and the per-sample shuffle used in Table 5.


def test_per_sample_shuffle_preserves_multiset():
    x = torch.arange(8).float().view(1, 8, 1, 1, 1).expand(5, 8, 1, 1, 1).contiguous()
    gen = torch.Generator().manual_seed(0)
    y = per_sample_shuffle(x, gen)
    for i in range(5):
        a = x[i, :, 0, 0, 0].sort().values
        b = y[i, :, 0, 0, 0].sort().values
        torch.testing.assert_close(a, b)
    assert not torch.equal(x, y)


def test_gate6_classes():
    assert gate6(0.9, 0.5, 0.9, 0.9, False) == "frame-local"
    assert gate6(0.5, 0.9, 0.9, 0.9, False) == "order-invariant multi-frame"
    assert gate6(0.5, 0.5, 0.9, 0.5, True) == "order-encoded"
    assert gate6(0.5, 0.5, 0.9, 0.5, False) == "order-encoded"
    assert gate6(0.5, 0.5, 0.5, 0.5, False) == "inconclusive"
