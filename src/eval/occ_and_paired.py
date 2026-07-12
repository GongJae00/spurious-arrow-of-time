"""(a) Frame-order randomization on the order-encoded-core configuration
(both cues order-dependent): the order-destroying mitigation should fail.
(b) Paired plain-ERM baseline inside the method-bench training stream.
(c) IRMv1 penalty-weight sweep.

Usage: python -m src.eval.occ_and_paired --seeds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import src.eval.robust_methods_bench as rmb
from src.eval.temporal_evidence_audit import (
    to_tensor, train_sequence_model, eval_sequence)
from src.data.irreversible_source_inference import (
    IrreversibleSourceConfig, generate_split)

OCC = {**rmb.BASE, "core_process": "directional_pulse"}


def occ_frame_rand(seed, device):
    cfg = IrreversibleSourceConfig(seed=seed, **rmb.SIZES, **OCC)
    sp = {s: generate_split(cfg, s)
          for s in ["train", "val_iid", "iid_test", "ood_test"]}
    mu = float(np.asarray(sp["train"].mixed).mean())
    sd = float(np.asarray(sp["train"].mixed).std()) or 1.0

    def x(n):
        return to_tensor((np.asarray(sp[n].mixed) - mu) / sd)

    y = {n: torch.from_numpy(sp[n].y) for n in sp}
    m = train_sequence_model(x("train"), y["train"], x("val_iid"), y["val_iid"],
                             seed * 31 + 8, device, shuffle_frames=True,
                             epochs=100, patience=30)
    return (eval_sequence(m, x("iid_test"), y["iid_test"], device, "shuffled", seed),
            eval_sequence(m, x("ood_test"), y["ood_test"], device, "shuffled", seed + 1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default="results/extended/occ_and_paired.json")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}

    rows = [occ_frame_rand(s, device) for s in range(a.seeds)]
    out["occ_frame_rand"] = [{"seed": i, "iid": round(r[0], 4), "ood": round(r[1], 4)}
                             for i, r in enumerate(rows)]
    print("occ_frame_rand:", out["occ_frame_rand"], flush=True)

    rows = [rmb.train_generic("erm", s, device) for s in range(a.seeds)]
    out["paired_erm"] = [{"seed": i, "iid": round(r[0], 4), "ood": round(r[1], 4)}
                         for i, r in enumerate(rows)]
    print("paired_erm:", out["paired_erm"], flush=True)

    for lam in [100.0, 10000.0]:
        import src.eval.robust_methods_bench as m2
        src = open(m2.__file__, encoding="utf-8").read()
        # penalty weight is literal 1000.0 in train_irm; monkeypatch via exec
        ns = {}
        exec(src.replace("1000.0", str(lam)), ns)
        rows = [ns["train_irm"](s, device) for s in range(a.seeds)]
        out[f"irm_lam{int(lam)}"] = [
            {"seed": i, "iid": round(r[0], 4), "ood": round(r[1], 4)}
            for i, r in enumerate(rows)]
        print(f"irm_lam{int(lam)}:", out[f"irm_lam{int(lam)}"], flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
    for k, v in out.items():
        o = np.array([r["ood"] for r in v])
        i = np.array([r["iid"] for r in v])
        print(k, "iid", round(i.mean(), 3), "ood", round(o.mean(), 3),
              "core", int(((i >= 0.8) & (o >= 0.8)).sum()),
              "coll", int(((i >= 0.8) & (o <= 0.2)).sum()))


if __name__ == "__main__":
    main()
