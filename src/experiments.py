import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from src.audit import Audit, per_sample_shuffle, probe, train_mlp_probe
from src.benchmark import CONSTRUCTIONS, SIZES, graph_setup, graph_split, load_forda, load_har, make, overlay_order_pulse, paper_config
from src.data import GeneratorConfig, generate_split, generate_splits
from src.evaluate import accuracy, aggregate, as_tensor, input_gradient_saliency, regime, summarize_results
from src.models import FinalFrameMLP, SegGRU, build_model
from src.train import (
    ARCHS,
    METHODS,
    deep_update,
    eval_sequence,
    set_seed,
    stable_run_seed,
    train_dual,
    train_irm,
    train_one_method,
    train_robust,
    train_sequence,
)

# Table 5 OE-Strict, Table 6/Figure 4 Trail-FL, Table 7 sequence ERM,
# Table 8 MF-Core, Table 9 UCR, then appendix runs.


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def torch_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def seed_list(spec):
    s = spec["seeds"]
    return range(s) if type(s) is int else [int(x) for x in s]


def rounded(r):
    return {k: round(v, 4) if type(v) is float else [round(t, 4) for t in v] for k, v in r.items()}


def run_shortcut_eval(spec: dict, default: dict) -> dict:
    # Table 5. OE-Strict mixed ERM, channel interventions, Gate 6.
    device = torch_device(default["device"])
    outpath = Path(spec["out"])
    out = json.loads(outpath.read_text(encoding="utf-8")) if outpath.exists() else {}
    benchmark = spec["benchmark"]
    nospur = spec["nospurious"]
    corr = 0.5 if nospur else 0.97
    epochs = spec["epochs"]
    patience = spec["patience"]
    L = 8
    for seed in seed_list(spec):
        key = f"seed{seed}"
        if key in out:
            continue
        splits = make(benchmark, seed, corr_train=corr)
        ytr = torch.from_numpy(splits["train"].y)
        yva = torch.from_numpy(splits["val_iid"].y)
        mu = float(splits["train"].mixed.mean())
        sd = float(splits["train"].mixed.std()) or 1.0

        def x(name, field=None):
            a = np.asarray(splits[name].mixed)
            if field is not None:
                a = a[:, :, field : field + 1]
            return as_tensor((a - mu) / sd)

        r = {}
        if nospur:
            m = train_sequence(x("train"), ytr, x("val_iid"), yva, seed, device, epochs=epochs, patience=patience)
            r["nospur"] = (eval_sequence(m, x("iid_test"), torch.from_numpy(splits["iid_test"].y), device), eval_sequence(m, x("ood_test"), torch.from_numpy(splits["ood_test"].y), device))
            out[key] = {k: [round(t, 4) for t in v] for k, v in r.items()}
            write_json(outpath, out)
            continue
        m = train_sequence(x("train"), ytr, x("val_iid"), yva, seed, device, epochs=epochs, patience=patience)
        yo = torch.from_numpy(splits["ood_test"].y)
        xo = x("ood_test")
        r["erm"] = (eval_sequence(m, x("iid_test"), torch.from_numpy(splits["iid_test"].y), device), eval_sequence(m, xo, yo, device))
        perm = torch.randperm(L)
        xs = xo.clone()
        xs[:, :, 1] = xs[:, perm, 1]
        r["shuffle_nuis"] = accuracy(m, xs, yo, device)
        xr = xo.clone()
        xr[:, :, 1] = torch.flip(xr[:, :, 1], dims=[1])
        r["reverse_nuis"] = accuracy(m, xr, yo, device)
        xc = xo.clone()
        xc[:, :, 0] = torch.flip(xc[:, :, 0], dims=[1])
        r["reverse_core"] = accuracy(m, xc, yo, device)
        nu_m = train_sequence(x("train", 1), ytr, x("val_iid", 1), yva, seed + 5000, device, epochs=epochs, patience=patience)
        r["nuisance_only"] = (eval_sequence(nu_m, x("iid_test", 1), torch.from_numpy(splits["iid_test"].y), device), eval_sequence(nu_m, x("ood_test", 1), yo, device))
        if seed < spec["probe_seeds"]:
            nu_tr = torch.from_numpy(np.asarray(splits["train"].mixed)[:, :, 1])
            nu_te = torch.from_numpy(np.asarray(splits["iid_test"].mixed)[:, :, 1])
            dtr = torch.from_numpy((splits["train"].nuisance_direction > 0).astype(np.int64))
            dte = torch.from_numpy((splits["iid_test"].nuisance_direction > 0).astype(np.int64))
            r["firstlast_pair_dir"] = probe(nu_tr[:, [0, L - 1]].reshape(len(nu_tr), -1), dtr, nu_te[:, [0, L - 1]].reshape(len(nu_te), -1), dte, device)
            best = 0.0
            for t in range(L):
                best = max(best, probe(nu_tr[:, t].reshape(len(nu_tr), -1), dtr, nu_te[:, t].reshape(len(nu_te), -1), dte, device, 15))
            r["best_single_frame_dir"] = best
            srt_tr = torch.sort(nu_tr.reshape(len(nu_tr), L, -1), dim=1)[0]
            srt_te = torch.sort(nu_te.reshape(len(nu_te), L, -1), dim=1)[0]
            r["set_dir"] = probe(srt_tr.reshape(len(srt_tr), -1), dtr, srt_te.reshape(len(srt_te), -1), dte, device)
            if benchmark == "oe_strict":
                r["adjacent_pair_dir"] = probe(nu_tr[:, [3, 4]].reshape(len(nu_tr), -1), dtr, nu_te[:, [3, 4]].reshape(len(nu_te), -1), dte, device)
            if benchmark == "set_mf":
                r["temporal_mean_dir"] = probe(nu_tr.mean(1).reshape(len(nu_tr), -1), dtr, nu_te.mean(1).reshape(len(nu_te), -1), dte, device)
        out[key] = rounded(r)
        write_json(outpath, out)
    return out


def run_certify(spec: dict, default: dict) -> dict:
    # Table 5. Nuisance-only ERM under shuffle and reverse.
    device = torch_device(default["device"])
    outpath = Path(spec["out"])
    out = json.loads(outpath.read_text(encoding="utf-8")) if outpath.exists() else {}
    seeds = range(int(spec["seeds"]))
    L = 8
    for seed in seeds:
        key = f"seed{seed}"
        if key in out:
            continue
        splits = make(spec["benchmark"], seed)
        mu = float(np.asarray(splits["train"].mixed).mean())
        sd = float(np.asarray(splits["train"].mixed).std()) or 1.0
        def x(name, field=1):
            a = np.asarray(splits[name].mixed)[:, :, field : field + 1]
            return as_tensor((a - mu) / sd)
        ytr = torch.from_numpy(splits["train"].y)
        m = train_sequence(x("train"), ytr, x("val_iid"), torch.from_numpy(splits["val_iid"].y), seed + 5000, device)
        yo = torch.from_numpy(splits["ood_test"].y)
        xo = x("ood_test")
        r = {
            "ordered_iid": eval_sequence(m, x("iid_test"), torch.from_numpy(splits["iid_test"].y), device),
            "ordered_ood": eval_sequence(m, xo, yo, device),
            "shuffled_ood": accuracy(m, xo[:, torch.randperm(L)], yo, device),
            "reversed_ood": accuracy(m, torch.flip(xo, dims=[1]), yo, device),
        }
        out[key] = {k: round(v, 4) for k, v in r.items()}
        write_json(outpath, out)
    return out


def run_shuffle(spec: dict, default: dict) -> dict:
    # Table 5. Per-sample shuffle of the nuisance channel.
    device = torch_device(default["device"])
    outpath = Path(spec["out"])
    out = json.loads(outpath.read_text(encoding="utf-8")) if outpath.exists() else {}
    for seed in range(int(spec["seeds"])):
        key = f"seed{seed}"
        if key in out:
            continue
        splits = make(spec["benchmark"], seed)
        mu = float(np.asarray(splits["train"].mixed).mean())
        sd = float(np.asarray(splits["train"].mixed).std()) or 1.0
        gen = torch.Generator().manual_seed(seed + 777)
        y = torch.from_numpy(splits["ood_test"].y)
        def xn(name, sl=None):
            a = np.asarray(splits[name].mixed)
            if sl is not None:
                a = a[:, :, sl]
            return as_tensor((a - mu) / sd)
        m = train_sequence(xn("train", slice(1, 2)), torch.from_numpy(splits["train"].y), xn("val_iid", slice(1, 2)), torch.from_numpy(splits["val_iid"].y), seed + 5000, device)
        r = {"reader_persample_shuffle_ood": accuracy(m, per_sample_shuffle(xn("ood_test", slice(1, 2)), gen), y, device)}
        m2 = train_sequence(xn("train"), torch.from_numpy(splits["train"].y), xn("val_iid"), torch.from_numpy(splits["val_iid"].y), seed, device)
        xm = xn("ood_test")
        xs = xm.clone()
        xs[:, :, 1] = per_sample_shuffle(xm[:, :, 1], gen)
        r["mixed_persample_shuffle_nuis_ood"] = accuracy(m2, xs, y, device)
        out[key] = {k: round(v, 4) for k, v in r.items()}
        write_json(outpath, out)
    return out


def run_temporal(spec: dict, default: dict) -> dict:
    # Table 6 / Figure 4a. Gate 6 and Gate 5 order tests.
    device = torch_device(default["device"])
    per_frame_runs, summary_runs, order_runs = [], [], []
    n = int(spec["seeds"])
    for s in range(n):
        out = Audit(make(spec["benchmark"], s), s, device).gate6()
        per_frame_runs.append(out["per_frame"])
        summary_runs.append(out["summary"])
        order_runs.append(out["order"])
    L = len(per_frame_runs[0]["dir_iid"])
    result = {
        "per_frame": {k: [aggregate([r[k][t] for r in per_frame_runs]) for t in range(L)] for k in ["label_iid", "label_ood", "dir_iid", "dir_ood", "core_label_iid", "core_label_ood"]},
        "summary": {name: {k: aggregate([r[name][k] for r in summary_runs]) for k in summary_runs[0][name]} for name in summary_runs[0]},
        "order": {k: aggregate([r[k] for r in order_runs]) for k in order_runs[0]},
    }
    write_json(Path(spec["out"]), result)
    return result


def run_strict_order(spec: dict, default: dict) -> dict:
    # Table 6. Channel probes and Gate 6 set probe.
    device = torch_device(default["device"])
    result = {}
    for name, bench in spec["benchmarks"].items():
        ch_runs, set_runs, erm_runs = [], [], []
        for s in range(int(spec["seeds"])):
            a = Audit(make(bench, s), s, device)
            ch_runs.append(a.channel_probes())
            set_runs.append(a.gate6()["set"])
            if name == "simple_oe":
                erm_runs.append(a.mixed_channel())
        L = len(ch_runs[0]["dir_nuis_only"])
        result[name] = {
            "dir_nuis_only": [aggregate([r["dir_nuis_only"][t] for r in ch_runs]) for t in range(L)],
            "dir_core_only": [aggregate([r["dir_core_only"][t] for r in ch_runs]) for t in range(L)],
            "set_probe_direction": aggregate(set_runs),
        }
        if erm_runs:
            result[name]["mixed_erm_channel"] = {k: aggregate([r[k] for r in erm_runs]) for k in erm_runs[0]}
    write_json(Path(spec["out"]), result)
    return result


def run_nuisance_order(spec: dict, default: dict) -> dict:
    # Figure 4b. Nuisance-only ERM under order interventions.
    device = torch_device(default["device"])
    result = {}
    for name, bench in spec["benchmarks"].items():
        runs = [Audit(make(bench, s), s, device).nuisance_order() for s in range(int(spec["seeds"]))]
        result[name] = {k: aggregate([r[k] for r in runs]) for k in runs[0]}
    write_json(Path(spec["out"]), result)
    return result


def run_endpoint(spec: dict, default: dict) -> dict:
    # Gate 3. Final-frame direction on endpoint-matched vs residue-visible.
    device = torch_device(default["device"])
    result = {}
    for variant in ["endpoint_matched", "residue_visible"]:
        accs = []
        for s in range(int(spec["seeds"])):
            cfg = paper_config(s, n_train=4096, n_val_iid=512, n_iid_test=2048, n_ood_test=512, benchmark_variant=variant)
            splits = {"train": generate_split(cfg, "train"), "iid_test": generate_split(cfg, "iid_test")}
            accs.append(Audit(splits, s, device).gate3())
        result[variant] = aggregate(accs)
    write_json(Path(spec["out"]), result)
    return result


def run_train(spec: dict, default: dict) -> dict:
    # Table 7. Sequence ERM and Table 7 methods on named constructions.
    out_dir = Path(spec["out"])
    seeds = [int(s) for s in spec["seeds"]]
    methods = spec["methods"] if "methods" in spec else list(METHODS)
    scenarios = spec["scenarios"]
    primary = spec["primary_scenario"]
    training = deep_update(default["training"], spec["training"] if "training" in spec else {})
    model_config = deep_update(default["model"], spec["model"] if "model" in spec else {})
    data_base = deep_update(default["data"], spec["data"] if "data" in spec else {})
    device = torch_device(default["device"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()
    all_results = []
    for scenario in scenarios:
        scenario_name = scenario["name"]
        scenario_data = deep_update(data_base, scenario["data"] if "data" in scenario else {})
        scenario_methods = scenario["methods"] if "methods" in scenario else methods
        scenario_training = deep_update(training, scenario["training"] if "training" in scenario else {})
        for seed in seeds:
            set_seed(seed)
            fields = {**scenario_data, "seed": seed}
            dataset_config = GeneratorConfig(**{k: fields[k] for k in GeneratorConfig.__dataclass_fields__ if k in fields})
            bench = spec["benchmark"] if "benchmark" in spec else None
            if bench in CONSTRUCTIONS:
                sizes = {k: getattr(dataset_config, k) for k in ("n_train", "n_val_iid", "n_iid_test", "n_ood_test")}
                extra = {k: v for k, v in scenario_data.items() if k not in sizes}
                splits = make(bench, seed, sizes=sizes, extra=extra)
            else:
                splits = generate_splits(dataset_config)
            for method in scenario_methods:
                run_seed = stable_run_seed(seed, scenario_name, method)
                set_seed(run_seed)
                row = train_one_method(method, splits, dataset_config, scenario_training, model_config, seed, run_seed, device)
                del row["model"]
                row["profile"] = spec["name"]
                row["scenario"] = scenario_name
                all_results.append(row)
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
    summary = summarize_results(all_results, primary_scenario=primary)
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "manifest.json", {"run": spec["name"], "seeds": seeds, "methods": methods, "primary_scenario": primary, "device": str(device)})
    return {"summary": summary}


def run_mf_core_probes(spec: dict, default: dict) -> dict:
    # Table 8. Temporal-mean probe vs core-only GRU.
    device = torch_device(default["device"])
    runs = []
    for s in range(int(spec["seeds"])):
        splits = make("mf_core", s)
        tr, va, it, ot = (splits[k] for k in ["train", "val_iid", "iid_test", "ood_test"])
        ytr, yit = torch.from_numpy(tr.y), torch.from_numpy(it.y)
        mean_feat = lambda x: as_tensor(np.asarray(x.core_only).mean(axis=1).reshape(len(x.y), -1))
        mean_acc = train_mlp_probe(mean_feat(tr), ytr, [(mean_feat(it), yit)], s * 7 + 3, device)[0]
        mu = float(np.asarray(tr.core_only).mean())
        sd = float(np.asarray(tr.core_only).std()) or 1.0
        seq = lambda x: as_tensor(((np.asarray(x.core_only) - mu) / sd)[:, :, None])
        m = train_sequence(seq(tr), ytr, seq(va), torch.from_numpy(va.y), s * 31 + 7, device, epochs=100, patience=30)
        runs.append({"mean_probe": mean_acc, "gru_iid": eval_sequence(m, seq(it), yit, device), "gru_ood": eval_sequence(m, seq(ot), torch.from_numpy(ot.y), device)})
    result = {k: aggregate([r[k] for r in runs]) for k in runs[0]}
    write_json(Path(spec["out"]), result)
    return result


def run_mf_core_perframe(spec: dict, default: dict) -> dict:
    # Table 8. Per-frame core-only label probes.
    device = torch_device(default["device"])
    runs = []
    for s in range(int(spec["seeds"])):
        splits = make("mf_core", s)
        tr, it = splits["train"], splits["iid_test"]
        ytr, yit = torch.from_numpy(tr.y), torch.from_numpy(it.y)
        accs = []
        for t in range(tr.core_only.shape[1]):
            accs.append(train_mlp_probe(as_tensor(np.asarray(tr.core_only)[:, t].reshape(len(ytr), -1)), ytr, [(as_tensor(np.asarray(it.core_only)[:, t].reshape(len(yit), -1)), yit)], s * 100 + t, device)[0])
        runs.append(accs)
    result = {"per_frame": [aggregate([r[t] for r in runs]) for t in range(len(runs[0]))]}
    write_json(Path(spec["out"]), result)
    return result


def run_ucr(spec: dict, default: dict) -> dict:
    # Table 9. FordA / HAR with an order-pulse overlay on the official split.
    device = torch_device(default["device"])
    dataset = spec["dataset"]
    xc_all, y_all, ntr = load_har(dataset) if dataset in ("har", "har2") else load_forda()
    n = len(y_all)
    L, CORR = 10, 0.97
    official = spec["official"]
    cut = [int(ntr * 0.9), ntr, ntr + (n - ntr) // 2, n] if official else [int(n * 0.62), int(n * 0.70), int(n * 0.85), n]
    outpath = Path(spec["out"])
    out = json.loads(outpath.read_text(encoding="utf-8")) if outpath.exists() else {}
    for seed in seed_list(spec):
        key = f"seed{seed}"
        if key in out:
            continue
        rng = np.random.default_rng(seed)
        order = np.concatenate([rng.permutation(ntr), ntr + rng.permutation(n - ntr)]) if official else rng.permutation(n)
        pool = {}
        for name, s, e, corr in [("train", 0, cut[0], CORR), ("val", cut[0], cut[1], CORR), ("iid", cut[1], cut[2], CORR), ("ood", cut[2], cut[3], 1 - CORR)]:
            idx = order[s:e]
            pool[name] = overlay_order_pulse(xc_all[idx], y_all[idx], np.random.default_rng(seed * 7 + s), corr, len(idx))

        def T(name, mode):
            xcs, nu, ys, d = pool[name]
            core = torch.from_numpy(xcs)[:, :, None, :]
            nut = torch.from_numpy(nu)[:, :, None, :]
            x = torch.cat([core, nut], 2) if mode == "mixed" else (core if mode == "core" else nut)
            return x, torch.from_numpy(ys), torch.from_numpy(d)

        def fit(ch, mode):
            torch.manual_seed(seed)
            m = SegGRU(ch).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
            xtr, ytr, _ = T("train", mode)
            xva, yva, _ = T("val", mode)
            best, state, bad = -1, None, 0
            for _ in range(60):
                m.train()
                perm = torch.randperm(len(xtr))
                for i in range(0, len(xtr), 128):
                    j = perm[i:i + 128]
                    opt.zero_grad(set_to_none=True)
                    F.cross_entropy(m(xtr[j].to(device)).logits, ytr[j].to(device)).backward()
                    opt.step()
                m.eval()
                with torch.no_grad():
                    acc = (m(xva.to(device)).logits.argmax(1).cpu() == yva).float().mean().item()
                if acc > best:
                    best, bad, state = acc, 0, {k: v.cpu().clone() for k, v in m.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 15:
                        break
            m.load_state_dict(state)
            m.eval()
            return m

        def acc(m, x, y):
            return float((m(x.to(device)).logits.argmax(1).cpu() == y).float().mean())

        r = {}
        m = fit(1, "core")
        r["core_only"] = [acc(m, *T("iid", "core")[:2]), acc(m, *T("ood", "core")[:2])]
        m = fit(1, "nu")
        r["nuisance_only"] = [acc(m, *T("iid", "nu")[:2]), acc(m, *T("ood", "nu")[:2])]
        erm = fit(2, "mixed")
        xi, yi, _ = T("iid", "mixed")
        xo, yo, _ = T("ood", "mixed")
        r["erm"] = [acc(erm, xi, yi), acc(erm, xo, yo)]
        xs = xo.clone()
        xs[:, :, 1] = xs[:, torch.randperm(L), 1]
        r["erm_shuffle_nuis"] = acc(erm, xs, yo)
        xr = xo.clone()
        xr[:, :, 1] = torch.flip(xr[:, :, 1], dims=[1])
        r["erm_reverse_nuis"] = acc(erm, xr, yo)
        xrc = xo.clone()
        xrc[:, :, 0] = torch.flip(xrc[:, :, 0], dims=[1])
        r["erm_reverse_core"] = acc(erm, xrc, yo)
        ns = {}
        for name in pool:
            xcs, nu, ys, d = pool[name]
            rng2 = np.random.default_rng(seed * 13 + 1009 * ["train", "val", "iid", "ood"].index(name))
            ns[name] = overlay_order_pulse(xcs, ys, rng2, 0.5, len(ys))
        pool_bak = {k: pool[k] for k in pool}
        pool.update(ns)
        m = fit(2, "mixed")
        r["no_spurious"] = [acc(m, *T("iid", "mixed")[:2]), acc(m, *T("ood", "mixed")[:2])]
        pool.update(pool_bak)

        def probe_ucr(x, t, tgt):
            xt = x[:, :, :] if t is None else x[:, t]
            model = nn.Sequential(nn.Linear(xt.reshape(len(xt), -1).shape[1], 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            xtr = xt.reshape(len(xt), -1)
            for _ in range(30):
                pm = torch.randperm(len(xtr))
                for i in range(0, len(xtr), 256):
                    j = pm[i : i + 256]
                    opt.zero_grad(set_to_none=True)
                    F.cross_entropy(model(xtr[j].to(device)), tgt[j].to(device)).backward()
                    opt.step()
            model.eval()
            return model

        xn_tr, ytr_, dtr_ = T("train", "nu")
        xn_iid, yiid_, diid_ = T("iid", "nu")
        best_sf = 0.0
        for t in range(L):
            pm = probe_ucr(xn_tr, t, dtr_)
            with torch.no_grad():
                v = float((pm(xn_iid[:, t].reshape(len(xn_iid), -1).to(device)).argmax(1).cpu() == diid_).float().mean())
            best_sf = max(best_sf, v)
        r["best_single_seg_dir"] = best_sf
        srt = torch.sort(xn_tr.reshape(len(xn_tr), L, -1), dim=1)[0]
        srti = torch.sort(xn_iid.reshape(len(xn_iid), L, -1), dim=1)[0]
        pm = probe_ucr(srt, None, dtr_)
        with torch.no_grad():
            r["unordered_set_dir"] = float((pm(srti.reshape(len(srti), -1).to(device)).argmax(1).cpu() == diid_).float().mean())
        xm_tr, ytr2, dtr2 = T("train", "mixed")
        xm_iid, yiid2, diid2 = T("iid", "mixed")
        pm = probe_ucr(xm_tr, L - 1, ytr2)
        with torch.no_grad():
            r["final_seg_label"] = float((pm(xm_iid[:, L - 1].reshape(len(xm_iid), -1).to(device)).argmax(1).cpu() == yiid2).float().mean())
        pm = probe_ucr(xm_tr, L - 1, dtr2)
        with torch.no_grad():
            r["final_seg_dir"] = float((pm(xm_iid[:, L - 1].reshape(len(xm_iid), -1).to(device)).argmax(1).cpu() == diid2).float().mean())
        out[key] = rounded(r)
        write_json(outpath, out)
    return out


def run_graph(spec: dict, default: dict) -> dict:
    # Table A11. Karate / Les Misérables diffusion core with a directional nuisance.
    device = torch_device(default["device"])
    P, faction, order, N = graph_setup(spec["graph"])
    allres = {}
    for s in range(int(spec["seeds"])):
        torch.manual_seed(s)
        res = {}
        for scen, prefix in [("main", ""), ("no_spurious", "ns_")]:
            tr = graph_split(P, faction, order, N, 4096, s * 100 + 1, prefix + "train")
            va = graph_split(P, faction, order, N, 1024, s * 100 + 2, prefix + "train")
            ii = graph_split(P, faction, order, N, 2048, s * 100 + 3, prefix + "train")
            oo = graph_split(P, faction, order, N, 2048, s * 100 + 4, prefix + "ood")
            methods = {"sequence_erm": ("mixed", "seq"), "core_only": ("core", "seq"), "nuisance_only": ("nuis", "seq"), "final_frame": ("mixed", "mlp")}
            if scen == "no_spurious":
                methods = {"sequence_erm": ("mixed", "seq")}
            for mname, (key, kind) in methods.items():
                mu, sd = tr[key].mean(), max(tr[key].std(), 1e-6)
                norm = lambda a: ((a - mu) / sd).astype(np.float32)
                ch = tr[key].shape[2]
                model = build_model("sequence_cnn_gru", grid_size=N, hidden_dim=64, input_channels=ch).to(device) if kind == "seq" else FinalFrameMLP(grid_size=1, hidden_dim=64, input_dim=ch * N).to(device)
                xtr = torch.from_numpy(norm(tr[key])).to(device)
                ytr = torch.from_numpy(tr["y"]).to(device)
                xval = torch.from_numpy(norm(va[key])).to(device)
                yval = torch.from_numpy(va["y"]).to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                best, best_state, stale = -1, None, 0
                for _ in range(40):
                    model.train()
                    perm = torch.randperm(len(xtr), device=device)
                    for i in range(0, len(xtr), 128):
                        idx = perm[i:i + 128]
                        opt.zero_grad(set_to_none=True)
                        torch.nn.functional.cross_entropy(model(xtr[idx]).logits, ytr[idx]).backward()
                        opt.step()
                    model.eval()
                    with torch.no_grad():
                        acc = float((model(xval).logits.argmax(1) == yval).float().mean())
                    if acc > best:
                        best, best_state, stale = acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
                    else:
                        stale += 1
                        if stale >= 12:
                            break
                model.load_state_dict(best_state)
                with torch.no_grad():
                    res[f"{scen}/{mname}"] = {
                        "iid": float((model(torch.from_numpy(norm(ii[key])).to(device)).logits.argmax(1) == torch.from_numpy(ii["y"]).to(device)).float().mean()),
                        "ood": float((model(torch.from_numpy(norm(oo[key])).to(device)).logits.argmax(1) == torch.from_numpy(oo["y"]).to(device)).float().mean()),
                    }
        allres[s] = res
    agg = {}
    for k in allres[0]:
        for split in ("iid", "ood"):
            vals = [allres[s][k][split] for s in allres]
            agg[f"{k}/{split}"] = aggregate(vals)
    write_json(Path(spec["out"]), agg)
    return agg


def run_multi_init(spec: dict, default: dict) -> dict:
    # Table A18. 15 inits × data seeds, standard vs certification budget.
    device = torch_device(default["device"])
    outpath = Path(spec["out"])
    result = json.loads(outpath.read_text(encoding="utf-8")) if outpath.exists() else {}
    budgets = {"standard": (40, 12), "certification": (100, 30)}
    for vname, bench in [("trail_fl", "trail_fl"), ("simple_oe", "simple_oe")]:
        for ds in range(int(spec["data_seeds"])):
            splits = make(bench, ds)
            mu = float(np.asarray(splits["train"].mixed).mean())
            sd = float(np.asarray(splits["train"].mixed).std()) or 1.0
            x = lambda n: as_tensor((np.asarray(splits[n].mixed) - mu) / sd)
            y = lambda n: torch.from_numpy(splits[n].y)
            for bname, (ep, pat) in budgets.items():
                key = f"{vname}/data{ds}/{bname}"
                if key in result:
                    continue
                rows = []
                for init in range(int(spec["inits"])):
                    m = train_sequence(x("train"), y("train"), x("val_iid"), y("val_iid"), 90001 + 613 * init, device, epochs=ep, patience=pat)
                    iid = eval_sequence(m, x("iid_test"), y("iid_test"), device)
                    ood = eval_sequence(m, x("ood_test"), y("ood_test"), device)
                    rows.append({"init": init, "iid": round(iid, 4), "ood": round(ood, 4), "regime": regime(iid, ood)})
                regs = [r["regime"] for r in rows]
                result[key] = {"core": regs.count("core"), "collapse": regs.count("collapse"), "chance": regs.count("chance"), "n": len(rows), "rows": rows}
                write_json(outpath, result)
    return result


def run_accessibility(spec: dict, default: dict) -> dict:
    # Table A1. Epochs to 95% of the cue ceiling.
    device = torch_device(default["device"])
    cues = {
        "diffusion_core": (dict(), "core_only", 1.0),
        "multiframe_core": (dict(diffusion_start_step=8, diffusion_steps_between_frames=2, core_noise_std=0.045, core_noise_growth_power=0.0, observation_noise_std=0.01), "core_only", 0.961),
        "oe_nuisance": (dict(nuisance_trail_decay=0.0), "nuisance_only", 0.97),
    }
    out = {}
    for cue, (over, field, ceiling) in cues.items():
        for n_train in [1024, 8192]:
            rows = []
            for s in range(int(spec["seeds"])):
                cfg = paper_config(s, n_train=n_train, n_val_iid=2048, n_iid_test=4096, n_ood_test=256, **over)
                sp = {k: generate_split(cfg, k) for k in ["train", "val_iid", "iid_test"]}
                mu = float(np.asarray(getattr(sp["train"], field)).mean())
                sd = float(np.asarray(getattr(sp["train"], field)).std()) or 1.0
                x = lambda name: as_tensor(((np.asarray(getattr(sp[name], field)) - mu) / sd)[:, :, None])
                y = lambda name: torch.from_numpy(sp[name].y)
                torch.manual_seed(s * 31 + 7)
                m = build_model("sequence_cnn_gru", grid_size=16, hidden_dim=64).to(device)
                opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
                crit, hit, best, best_state = 0.95 * ceiling, None, -1.0, None
                xtr, xva, ytr, yva = x("train"), x("val_iid"), y("train"), y("val_iid")
                for ep in range(1, 101):
                    m.train()
                    perm = torch.randperm(len(xtr))
                    for i in range(0, len(xtr), 128):
                        idx = perm[i:i + 128]
                        opt.zero_grad(set_to_none=True)
                        F.cross_entropy(m(xtr[idx].to(device)).logits, ytr[idx].to(device)).backward()
                        opt.step()
                    m.eval()
                    with torch.no_grad():
                        acc = float((m(xva.to(device)).logits.argmax(1) == yva.to(device)).float().mean())
                    if acc > best:
                        best, best_state = acc, {k: v.cpu().clone() for k, v in m.state_dict().items()}
                    if hit is None and acc >= crit:
                        hit = ep
                    if hit is not None and ep >= hit + 5:
                        break
                m.load_state_dict(best_state)
                m.eval()
                with torch.no_grad():
                    iid = float((m(x("iid_test").to(device)).logits.argmax(1) == y("iid_test").to(device)).float().mean())
                rows.append((iid, hit if hit is not None else 101))
            accs = np.array([r[0] for r in rows])
            eps = np.array([r[1] for r in rows])
            out[f"{cue}/n{n_train}"] = {"iid_mean": float(accs.mean()), "iid_std": float(accs.std(ddof=1)), "epochs_to_95pct_ceiling": [int(e) for e in eps], "epochs_median": float(np.median(eps))}
    write_json(Path(spec["out"]), out)
    return out


def run_oe_core_equalized(spec: dict, default: dict) -> dict:
    # OE-Core. Core-direction flip 0.03.
    device = torch_device(default["device"])
    rows = []
    for s in range(int(spec["seeds"])):
        splits = make("oe_core_equalized", s)
        mu = float(np.asarray(splits["train"].mixed).mean())
        sd = float(np.asarray(splits["train"].mixed).std()) or 1.0
        x = lambda n: as_tensor((np.asarray(splits[n].mixed) - mu) / sd)
        y = lambda n: torch.from_numpy(splits[n].y)
        m = train_sequence(x("train"), y("train"), x("val_iid"), y("val_iid"), s * 31 + 8, device, epochs=100, patience=30)
        rows.append({"seed": s, "iid": round(eval_sequence(m, x("iid_test"), y("iid_test"), device), 4), "ood": round(eval_sequence(m, x("ood_test"), y("ood_test"), device), 4)})
    write_json(Path(spec["out"]), {"rows": rows})
    return {"rows": rows}


def run_oe_core_controls(spec: dict, default: dict) -> dict:
    # OE-Core. Frame-rand; paired ERM and IRM λ on Simple OE.
    device = torch_device(default["device"])
    out = {}
    rows = []
    for s in range(int(spec["seeds"])):
        splits = make("oe_core", s)
        mu = float(np.asarray(splits["train"].mixed).mean())
        sd = float(np.asarray(splits["train"].mixed).std()) or 1.0
        x = lambda n: as_tensor((np.asarray(splits[n].mixed) - mu) / sd)
        y = lambda n: torch.from_numpy(splits[n].y)
        m = train_sequence(x("train"), y("train"), x("val_iid"), y("val_iid"), s * 31 + 8, device, shuffle_frames=True, epochs=100, patience=30)
        rows.append({"seed": s, "iid": round(eval_sequence(m, x("iid_test"), y("iid_test"), device, "shuffled", s), 4), "ood": round(eval_sequence(m, x("ood_test"), y("ood_test"), device, "shuffled", s + 1), 4)})
    out["order_rand"] = rows
    rows = []
    for s in range(int(spec["seeds"])):
        splits = make("simple_oe", s)
        iid, ood, _ = train_robust("erm", splits, s, device)
        rows.append({"seed": s, "iid": round(iid, 4), "ood": round(ood, 4)})
    out["paired_erm"] = rows
    for lam in [100.0, 10000.0]:
        rows = []
        for s in range(int(spec["seeds"])):
            splits0 = make("simple_oe", s)
            splits1 = make("simple_oe", s + 7919, extra={"nuisance_correlation": 0.85}, sizes={**SIZES, "n_train": 4096})
            iid, ood, _ = train_irm(splits0, splits1, s, device, irm_lambda=lam)
            rows.append({"seed": s, "iid": round(iid, 4), "ood": round(ood, 4)})
        out[f"irm_lam{int(lam)}"] = rows
    write_json(Path(spec["out"]), out)
    return out


def run_arch_cue(spec: dict, default: dict) -> dict:
    # Table A4–A5. Single-cue LSTM/TCN/Transformer/pool; GroupDRO at ρ=0.70.
    device = torch_device(default["device"])
    out = {}
    for arch, model_type in ARCHS.items():
        if arch == "gru":
            continue
        for field in ["nuisance_only", "core_only"]:
            rows = []
            for s in range(int(spec["seeds"])):
                splits = make("simple_oe", s)
                mu = float(np.asarray(getattr(splits["train"], field)).mean())
                sd = float(np.asarray(getattr(splits["train"], field)).std()) or 1.0
                x = lambda n: as_tensor(((np.asarray(getattr(splits[n], field)) - mu) / sd)[:, :, None])
                y = lambda n: torch.from_numpy(splits[n].y)
                m = train_sequence(x("train"), y("train"), x("val_iid"), y("val_iid"), s * 31 + 7, device, model_type=model_type)
                rows.append((eval_sequence(m, x("iid_test"), y("iid_test"), device), eval_sequence(m, x("ood_test"), y("ood_test"), device)))
            i = np.array([r[0] for r in rows])
            o = np.array([r[1] for r in rows])
            out[f"{arch}/{field}"] = {"iid": float(i.mean()), "iid_std": float(i.std(ddof=1)), "ood": float(o.mean()), "ood_std": float(o.std(ddof=1))}
    rows = []
    for s in range(int(spec["seeds"])):
        splits = make("simple_oe", s, extra={"nuisance_correlation": 0.70})
        iid, ood, _ = train_robust("groupdro", splits, s, device)
        rows.append((iid, ood))
    i = np.array([r[0] for r in rows])
    o = np.array([r[1] for r in rows])
    out["groupdro_corr0.70"] = {"iid": float(i.mean()), "ood": float(o.mean()), "core": int(((i >= 0.8) & (o >= 0.8)).sum())}
    write_json(Path(spec["out"]), out)
    return out


def run_groupdro(spec: dict, default: dict) -> dict:
    # GroupDRO. η and balanced-sampler sweep.
    device = torch_device(default["device"])
    result = {}
    for name, kw in {"eta0.01_balanced": dict(eta=0.01, balanced_sampler=True), "eta0.1_standard": dict(eta=0.1, balanced_sampler=False), "eta0.001_standard": dict(eta=0.001, balanced_sampler=False)}.items():
        rows = []
        for s in range(int(spec["seeds"])):
            splits = make("simple_oe", s)
            iid, ood, _ = train_robust("groupdro", splits, s, device, **kw)
            rows.append({"seed": s, "iid": round(iid, 4), "ood": round(ood, 4)})
        iids = np.array([r["iid"] for r in rows])
        oods = np.array([r["ood"] for r in rows])
        result[name] = {"iid_mean": float(iids.mean()), "ood_mean": float(oods.mean()), "ood_std": float(oods.std(ddof=1)), "core": int(((iids >= 0.8) & (oods >= 0.8)).sum()), "collapse": int(((iids >= 0.8) & (oods <= 0.2)).sum()), "rows": rows}
    write_json(Path(spec["out"]), result)
    return result


def run_gradsal(spec: dict, default: dict) -> dict:
    # Input-gradient. Nuisance-channel share.
    device = torch_device(default["device"])
    out = {}
    for variant, bench in [("trail", "trail_fl"), ("oe", "simple_oe")]:
        shares, profs = [], []
        for s in range(int(spec["seeds"])):
            splits = make(bench, s)
            mu = float(np.asarray(splits["train"].mixed).mean())
            sd = float(np.asarray(splits["train"].mixed).std()) or 1.0
            x = lambda n: as_tensor((np.asarray(splits[n].mixed) - mu) / sd)
            y = lambda n: torch.from_numpy(splits[n].y)
            m = train_sequence(x("train"), y("train"), x("val_iid"), y("val_iid"), s, device)
            share, prof = input_gradient_saliency(m, x("iid_test")[:512], device)
            shares.append(share)
            profs.append(np.asarray(prof).round(4).tolist())
        out[variant] = {"nuis_share_mean": round(float(np.mean(shares)), 4), "nuis_share_std": round(float(np.std(shares)), 4), "frame_profile_mean": np.mean(profs, axis=0).round(4).tolist()}
    write_json(Path(spec["out"]), out)
    return out


def run_video_search(spec: dict, default: dict) -> dict:
    # Table A12. Real-video nuisance: ERM, core-only, nuisance-only, final-frame, no-spurious.
    device = torch_device(default["device"])
    configs = {
        "A_std_c05_n15": {"real_video_standardize": True, "core_scale": 0.5, "nuisance_scale": 1.5},
        "B_std_c035_n15": {"real_video_standardize": True, "core_scale": 0.35, "nuisance_scale": 1.5},
        "C_c05_n15": {"core_scale": 0.5, "nuisance_scale": 1.5},
        "D_std_c05_n15_on08": {"real_video_standardize": True, "core_scale": 0.5, "nuisance_scale": 1.5, "observation_noise_std": 0.08},
        "E_roll_std_c07_n15": {"real_video_standardize": True, "real_video_endpoint_roll": True, "core_scale": 0.7, "nuisance_scale": 1.5},
        "F_roll_std_c085_n15": {"real_video_standardize": True, "real_video_endpoint_roll": True, "core_scale": 0.85, "nuisance_scale": 1.5},
        "G_roll_std_c10_n15": {"real_video_standardize": True, "real_video_endpoint_roll": True, "core_scale": 1.0, "nuisance_scale": 1.5},
        "H_roll_std_c06_n15": {"real_video_standardize": True, "real_video_endpoint_roll": True, "core_scale": 0.6, "nuisance_scale": 1.5},
    }
    outpath = Path(spec["out"])
    out = json.loads(outpath.read_text(encoding="utf-8")) if outpath.exists() else {}
    sizes = dict(n_train=8192, n_val_iid=2048, n_iid_test=2048, n_ood_test=2048)
    for name, over in configs.items():
        for seed in range(int(spec["seeds"])):
            base = f"{name}/seed{seed}"
            if f"{base}/erm" in out:
                continue
            extra = {**over, "nuisance_motion": "real_video", "real_video_cache": "data/real_video/cache_g16_L8_s5.npz"}
            splits = make("trail_fl", seed, sizes=sizes, extra=extra)
            def run_field(field, s):
                a0 = np.asarray(getattr(splits["train"], field))
                if a0.ndim == 4:
                    a0 = a0[:, :, None]
                mu = float(a0.mean())
                sd = float(a0.std()) or 1.0
                def x(n):
                    a = np.asarray(getattr(splits[n], field))
                    if a.ndim == 4:
                        a = a[:, :, None]
                    return as_tensor((a - mu) / sd)
                y = lambda n: torch.from_numpy(splits[n].y)
                m = train_sequence(x("train"), y("train"), x("val_iid"), y("val_iid"), s, device)
                return [round(eval_sequence(m, x("iid_test"), y("iid_test"), device), 4), round(eval_sequence(m, x("ood_test"), y("ood_test"), device), 4)]
            out[f"{base}/erm"] = run_field("mixed", seed * 31 + 11)
            out[f"{base}/core_only"] = run_field("core_only", seed * 31 + 12)
            out[f"{base}/nuis_only"] = run_field("nuisance_only", seed * 31 + 13)
            def last(name):
                a = np.asarray(splits[name].mixed)[:, -1]
                return a.reshape(len(a), -1)
            mu, sd = last("train").mean(), last("train").std() or 1.0
            xlast = {n: torch.from_numpy(((last(n) - mu) / sd).astype(np.float32)) for n in splits}
            y = {n: torch.from_numpy(splits[n].y) for n in splits}
            torch.manual_seed(seed * 31 + 14)
            head = nn.Sequential(nn.Linear(xlast["train"].shape[1], 128), nn.ReLU(), nn.Linear(128, 2)).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(20):
                perm = torch.randperm(len(xlast["train"]))
                for i in range(0, len(perm), 256):
                    idx = perm[i : i + 256]
                    opt.zero_grad(set_to_none=True)
                    F.cross_entropy(head(xlast["train"][idx].to(device)), y["train"][idx].to(device)).backward()
                    opt.step()
            head.eval()
            with torch.no_grad():
                out[f"{base}/final_frame"] = [
                    float((head(xlast["iid_test"].to(device)).argmax(1).cpu() == y["iid_test"]).float().mean()),
                    float((head(xlast["ood_test"].to(device)).argmax(1).cpu() == y["ood_test"]).float().mean()),
                ]
                out[f"{base}/final_frame"] = [round(v, 4) for v in out[f"{base}/final_frame"]]
            spn = make("trail_fl", seed, sizes=sizes, extra={**extra, "train_nuisance_mode": "randomized", "ood_mode": "randomized"})
            splits = spn
            out[f"{base}/no_spurious"] = run_field("mixed", 2, seed * 31 + 11)
            write_json(outpath, out)
    return out


def run_unified(spec: dict, default: dict) -> dict:
    # Table A6. ERM / GroupDRO / IRMv1 / DANN / JTT / frame-rand, sel_iid and sel_shift.
    device = torch_device(default["device"])
    outpath = Path(spec["out"])
    out = json.loads(outpath.read_text(encoding="utf-8")) if outpath.exists() else {}
    jobs = [(m, "gru") for m in ["erm", "groupdro_joint", "irmv1", "dann_dir", "jtt", "frame_rand"]] + [("erm", k) for k in ARCHS if k != "gru"]
    for seed in range(int(spec["seeds"])):
        splits = make("simple_oe", seed)
        for method, arch in jobs:
            key = f"{method}/{arch}/seed{seed}"
            if key in out:
                continue
            r = train_dual(method, arch, seed, splits, device)
            out[key] = {sel: [round(v, 4) for v in r[sel]] for sel in r}
            write_json(outpath, out)
    return out


def run(name: str, config_dir: Path = Path("configs")) -> dict:
    default = load_yaml(config_dir / "default.yaml")
    experiments = load_yaml(config_dir / "experiments.yaml")
    spec = copy.deepcopy(experiments[name])
    spec["name"] = name
    if spec["kind"] == "train":
        benches = load_yaml(config_dir / "benchmarks.yaml")
        if "benchmark" in spec:
            spec["data"] = deep_update(benches[spec["benchmark"]], spec["data"] if "data" in spec else {})
    return {
        "shortcut": run_shortcut_eval,
        "certify": run_certify,
        "shuffle": run_shuffle,
        "temporal": run_temporal,
        "strict_order": run_strict_order,
        "nuisance_order": run_nuisance_order,
        "endpoint": run_endpoint,
        "train": run_train,
        "mf_core_probes": run_mf_core_probes,
        "mf_core_perframe": run_mf_core_perframe,
        "ucr": run_ucr,
        "unified": run_unified,
        "graph": run_graph,
        "video_search": run_video_search,
        "arch_cue": run_arch_cue,
        "groupdro": run_groupdro,
        "gradsal": run_gradsal,
        "multi_init": run_multi_init,
        "accessibility": run_accessibility,
        "oe_core_equalized": run_oe_core_equalized,
        "oe_core_controls": run_oe_core_controls,
    }[spec["kind"]](spec, default)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--configs", default="configs")
    return p.parse_args()


def main():
    args = parse_args()
    result = run(args.run, Path(args.configs))
    print(json.dumps(result["summary"] if "summary" in result else result, indent=2))


if __name__ == "__main__":
    main()
