import inspect

import torch

from src.audit import Audit, per_sample_shuffle

# Gate 6 cases in Audit.gate6, and the per-sample shuffle used in Table 5.


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
    src = inspect.getsource(Audit.gate6)
    assert "if single >= 0.8:" in src
    assert 'loc = "frame-local"' in src
    assert "elif set_acc >= 0.8:" in src
    assert 'loc = "order-invariant multi-frame"' in src
    assert "<= 0.6" in src
    assert 'loc = "order-encoded"' in src
    assert 'loc = "inconclusive"' in src
