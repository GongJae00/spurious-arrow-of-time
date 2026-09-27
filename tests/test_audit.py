import torch

from src.audit import SetProbe, per_sample_shuffle

# Order invariance of the set probe and multiset preservation under shuffling.


def test_per_sample_shuffle_preserves_multiset():
    x = torch.arange(8).float().view(1, 8, 1, 1, 1).expand(5, 8, 1, 1, 1).contiguous()
    gen = torch.Generator().manual_seed(0)
    y = per_sample_shuffle(x, gen)
    for i in range(5):
        a = x[i, :, 0, 0, 0].sort().values
        b = y[i, :, 0, 0, 0].sort().values
        torch.testing.assert_close(a, b)
    assert not torch.equal(x, y)


def test_set_probe_is_order_invariant():
    torch.manual_seed(0)
    probe = SetProbe(frame_dim=4).eval()
    frames = torch.randn(5, 8, 4)
    shuffled = per_sample_shuffle(frames, torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(probe(frames), probe(shuffled))
        torch.testing.assert_close(probe(frames), probe(frames.flip(1)))
