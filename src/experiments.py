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
    ARCHITECTURES,
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

# Table 5–9, then the appendix kinds. `kind` in configs/experiments.yaml selects the function.


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def torch_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def seed_list(spec):
    seeds = spec["seeds"]
    return range(seeds) if type(seeds) is int else [int(seed) for seed in seeds]


def rounded(row):
    return {k: round(v, 4) if type(v) is float else [round(t, 4) for t in v] for k, v in row.items()}


def run_shortcut_eval(spec: dict, default: dict) -> dict:
    # Table 5. OE-Strict mixed ERM, channel interventions, Gate 6 probes.
    device = torch_device(default["device"])
    output_path = Path(spec["out"])
    results = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    benchmark = spec["benchmark"]
    no_spurious = spec["nospurious"]
    alignment_probability = 0.5 if no_spurious else 0.97
    epochs = spec["epochs"]
    patience = spec["patience"]
    length = 8
    for seed in seed_list(spec):
        key = f"seed{seed}"
        if key in results:
            continue
        splits = make(benchmark, seed, corr_train=alignment_probability)
        y_train = torch.from_numpy(splits["train"].y)
        y_val = torch.from_numpy(splits["val_iid"].y)
        mean = float(splits["train"].mixed.mean())
        std = float(splits["train"].mixed.std()) or 1.0

        def normalized(name, field=None):
            array = np.asarray(splits[name].mixed)
            if field is not None:
                array = array[:, :, field : field + 1]
            return as_tensor((array - mean) / std)

        row = {}

        # no-spurious mixed ERM
        if no_spurious:
            model = train_sequence(normalized("train"), y_train, normalized("val_iid"), y_val, seed, device, epochs=epochs, patience=patience)
            row["nospur"] = (eval_sequence(model, normalized("iid_test"), torch.from_numpy(splits["iid_test"].y), device), eval_sequence(model, normalized("ood_test"), torch.from_numpy(splits["ood_test"].y), device))
            results[key] = {k: [round(t, 4) for t in v] for k, v in row.items()}
            write_json(output_path, results)
            continue

        # mixed ERM
        model = train_sequence(normalized("train"), y_train, normalized("val_iid"), y_val, seed, device, epochs=epochs, patience=patience)
        y_ood = torch.from_numpy(splits["ood_test"].y)
        x_ood = normalized("ood_test")
        row["erm"] = (eval_sequence(model, normalized("iid_test"), torch.from_numpy(splits["iid_test"].y), device), eval_sequence(model, x_ood, y_ood, device))

        # nuisance shuffle
        frame_order = torch.randperm(length)
        shuffled = x_ood.clone()
        shuffled[:, :, 1] = shuffled[:, frame_order, 1]
        row["shuffle_nuis"] = accuracy(model, shuffled, y_ood, device)

        # nuisance reverse
        reversed_nuisance = x_ood.clone()
        reversed_nuisance[:, :, 1] = torch.flip(reversed_nuisance[:, :, 1], dims=[1])
        row["reverse_nuis"] = accuracy(model, reversed_nuisance, y_ood, device)

        # core reverse
        reversed_core = x_ood.clone()
        reversed_core[:, :, 0] = torch.flip(reversed_core[:, :, 0], dims=[1])
        row["reverse_core"] = accuracy(model, reversed_core, y_ood, device)

        # nuisance-only ERM
        nuisance_model = train_sequence(normalized("train", 1), y_train, normalized("val_iid", 1), y_val, seed + 5000, device, epochs=epochs, patience=patience)
        row["nuisance_only"] = (eval_sequence(nuisance_model, normalized("iid_test", 1), torch.from_numpy(splits["iid_test"].y), device), eval_sequence(nuisance_model, normalized("ood_test", 1), y_ood, device))

        # first-last pair, best single frame, unordered set
        if seed < spec["probe_seeds"]:
            nuisance_train = torch.from_numpy(np.asarray(splits["train"].mixed)[:, :, 1])
            nuisance_iid = torch.from_numpy(np.asarray(splits["iid_test"].mixed)[:, :, 1])
            direction_train = torch.from_numpy((splits["train"].nuisance_direction > 0).astype(np.int64))
            direction_iid = torch.from_numpy((splits["iid_test"].nuisance_direction > 0).astype(np.int64))
            row["firstlast_pair_dir"] = probe(nuisance_train[:, [0, length - 1]].reshape(len(nuisance_train), -1), direction_train, nuisance_iid[:, [0, length - 1]].reshape(len(nuisance_iid), -1), direction_iid, device)
            best_frame_accuracy = 0.0
            for t in range(length):
                best_frame_accuracy = max(best_frame_accuracy, probe(nuisance_train[:, t].reshape(len(nuisance_train), -1), direction_train, nuisance_iid[:, t].reshape(len(nuisance_iid), -1), direction_iid, device, 15))
            row["best_single_frame_dir"] = best_frame_accuracy
            sorted_train = torch.sort(nuisance_train.reshape(len(nuisance_train), length, -1), dim=1)[0]
            sorted_iid = torch.sort(nuisance_iid.reshape(len(nuisance_iid), length, -1), dim=1)[0]
            row["set_dir"] = probe(sorted_train.reshape(len(sorted_train), -1), direction_train, sorted_iid.reshape(len(sorted_iid), -1), direction_iid, device)

            # OE-Strict adjacent interior pair (frames 3, 4)
            if benchmark == "oe_strict":
                row["adjacent_pair_dir"] = probe(nuisance_train[:, [3, 4]].reshape(len(nuisance_train), -1), direction_train, nuisance_iid[:, [3, 4]].reshape(len(nuisance_iid), -1), direction_iid, device)

            # MF-Set temporal mean
            if benchmark == "mf_set":
                row["temporal_mean_dir"] = probe(nuisance_train.mean(1).reshape(len(nuisance_train), -1), direction_train, nuisance_iid.mean(1).reshape(len(nuisance_iid), -1), direction_iid, device)
        results[key] = rounded(row)
        write_json(output_path, results)
    return results


def run_certify(spec: dict, default: dict) -> dict:
    # Table 5 cue certification. Nuisance-only ERM under shuffle and reverse.
    device = torch_device(default["device"])
    output_path = Path(spec["out"])
    results = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    seeds = range(int(spec["seeds"]))
    length = 8
    for seed in seeds:
        key = f"seed{seed}"
        if key in results:
            continue
        splits = make(spec["benchmark"], seed)
        mean = float(np.asarray(splits["train"].mixed).mean())
        std = float(np.asarray(splits["train"].mixed).std()) or 1.0
        def normalized(name, field=1):
            array = np.asarray(splits[name].mixed)[:, :, field : field + 1]
            return as_tensor((array - mean) / std)
        y_train = torch.from_numpy(splits["train"].y)
        model = train_sequence(normalized("train"), y_train, normalized("val_iid"), torch.from_numpy(splits["val_iid"].y), seed + 5000, device)
        y_ood = torch.from_numpy(splits["ood_test"].y)
        x_ood = normalized("ood_test")
        row = {
            "ordered_iid": eval_sequence(model, normalized("iid_test"), torch.from_numpy(splits["iid_test"].y), device),
            "ordered_ood": eval_sequence(model, x_ood, y_ood, device),
            "shuffled_ood": accuracy(model, x_ood[:, torch.randperm(length)], y_ood, device),
            "reversed_ood": accuracy(model, torch.flip(x_ood, dims=[1]), y_ood, device),
        }
        results[key] = {k: round(v, 4) for k, v in row.items()}
        write_json(output_path, results)
    return results


def run_shuffle(spec: dict, default: dict) -> dict:
    # Table 5. Per-sample permutation of the nuisance channel.
    device = torch_device(default["device"])
    output_path = Path(spec["out"])
    results = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    for seed in range(int(spec["seeds"])):
        key = f"seed{seed}"
        if key in results:
            continue
        splits = make(spec["benchmark"], seed)
        mean = float(np.asarray(splits["train"].mixed).mean())
        std = float(np.asarray(splits["train"].mixed).std()) or 1.0
        generator = torch.Generator().manual_seed(seed + 777)
        y_ood = torch.from_numpy(splits["ood_test"].y)
        def normalized(name, channel=None):
            array = np.asarray(splits[name].mixed)
            if channel is not None:
                array = array[:, :, channel]
            return as_tensor((array - mean) / std)
        reader = train_sequence(normalized("train", slice(1, 2)), torch.from_numpy(splits["train"].y), normalized("val_iid", slice(1, 2)), torch.from_numpy(splits["val_iid"].y), seed + 5000, device)
        row = {"reader_persample_shuffle_ood": accuracy(reader, per_sample_shuffle(normalized("ood_test", slice(1, 2)), generator), y_ood, device)}
        mixed_model = train_sequence(normalized("train"), torch.from_numpy(splits["train"].y), normalized("val_iid"), torch.from_numpy(splits["val_iid"].y), seed, device)
        x_ood = normalized("ood_test")
        shuffled = x_ood.clone()
        shuffled[:, :, 1] = per_sample_shuffle(x_ood[:, :, 1], generator)
        row["mixed_persample_shuffle_nuis_ood"] = accuracy(mixed_model, shuffled, y_ood, device)
        results[key] = {k: round(v, 4) for k, v in row.items()}
        write_json(output_path, results)
    return results


def run_temporal(spec: dict, default: dict) -> dict:
    # Table 6 / Figure 4a. Gate 6 probes; order_tests are Gate 5 reversal plus Gate 6 shuffle/reverse.
    device = torch_device(default["device"])
    per_frame_runs, summary_runs, order_runs = [], [], []
    seed_count = int(spec["seeds"])
    for seed in range(seed_count):
        audit = Audit(make(spec["benchmark"], seed), seed, device)
        measured = audit.gate6(audit.gate5())
        per_frame_runs.append(measured["per_frame"])
        summary_runs.append(measured["summary"])
        order_runs.append(measured["order"])
    length = len(per_frame_runs[0]["dir_iid"])
    named = {k: v for k, v in CONSTRUCTIONS[spec["benchmark"]].items() if k != "overlay"}
    config = paper_config(0, **named)
    config_keys = [
        "grid_size", "length", "diffusion_alpha", "diffusion_start_step", "diffusion_steps_between_frames",
        "core_noise_std", "observation_noise_std", "core_scale", "nuisance_scale", "nuisance_sigma",
        "nuisance_speed", "nuisance_trail_decay", "nuisance_correlation", "observation_layout",
        "benchmark_variant", "n_train", "n_val_iid", "n_iid_test", "n_ood_test",
    ]
    result = {
        "config": {**{k: getattr(config, k) for k in config_keys}, "seeds": int(spec["seeds"])},
        "per_frame": {k: [aggregate([run[k][t] for run in per_frame_runs]) for t in range(length)] for k in ["label_iid", "label_ood", "dir_iid", "dir_ood", "core_label_iid", "core_label_ood"]},
        "summary_probes": {name: {k: aggregate([run[name][k] for run in summary_runs]) for k in summary_runs[0][name]} for name in summary_runs[0]},
        "order_tests": {k: aggregate([run[k] for run in order_runs]) for k in order_runs[0]},
    }
    write_json(Path(spec["out"]), result)
    return result


def run_strict_order(spec: dict, default: dict) -> dict:
    # Table 6. Channel probes and Gate 6 set probe.
    device = torch_device(default["device"])
    result = {}
    for name, benchmark in spec["benchmarks"].items():
        channel_runs, set_runs, erm_runs = [], [], []
        for seed in range(int(spec["seeds"])):
            audit = Audit(make(benchmark, seed), seed, device)
            channel_runs.append(audit.channel_probes())
            set_runs.append(audit.gate6(audit.gate5())["set"])
            if benchmark == "oe_simple":
                erm_runs.append(audit.mixed_channel())
        length = len(channel_runs[0]["dir_nuis_only"])
        result[name] = {
            "dir_nuis_only": [aggregate([run["dir_nuis_only"][t] for run in channel_runs]) for t in range(length)],
            "dir_core_only": [aggregate([run["dir_core_only"][t] for run in channel_runs]) for t in range(length)],
            "set_probe_direction": aggregate(set_runs),
        }
        if erm_runs:
            result[name]["mixed_erm_channel"] = {k: aggregate([run[k] for run in erm_runs]) for k in erm_runs[0]}
    write_json(Path(spec["out"]), result)
    return result


def run_nuisance_order(spec: dict, default: dict) -> dict:
    # Figure 4b. Nuisance-only ERM under order interventions.
    device = torch_device(default["device"])
    result = {}
    for name, benchmark in spec["benchmarks"].items():
        runs = [Audit(make(benchmark, seed), seed, device).nuisance_order() for seed in range(int(spec["seeds"]))]
        result[name] = {k: aggregate([run[k] for run in runs]) for k in runs[0]}
    write_json(Path(spec["out"]), result)
    return result


def run_endpoint(spec: dict, default: dict) -> dict:
    # Gate 3. Final-frame direction on endpoint-matched vs residue-visible.
    device = torch_device(default["device"])
    result = {}
    for variant in ["endpoint_matched", "residue_visible"]:
        accuracies = []
        for seed in range(int(spec["seeds"])):
            config = paper_config(seed, n_train=4096, n_val_iid=512, n_iid_test=2048, n_ood_test=512, benchmark_variant=variant)
            splits = {"train": generate_split(config, "train"), "iid_test": generate_split(config, "iid_test")}
            accuracies.append(Audit(splits, seed, device).gate3())
        result[variant] = aggregate(accuracies)
    write_json(Path(spec["out"]), result)
    return result


def run_train(spec: dict, default: dict) -> dict:
    # Table 7. Sequence ERM and Table 7 methods on named constructions.
    output_directory = Path(spec["out"])
    seeds = [int(s) for s in spec["seeds"]]
    methods = spec["methods"] if "methods" in spec else list(METHODS)
    scenarios = spec["scenarios"]
    primary = spec["primary_scenario"]
    training = deep_update(default["training"], spec["training"] if "training" in spec else {})
    model_config = deep_update(default["model"], spec["model"] if "model" in spec else {})
    data_base = deep_update(default["data"], spec["data"] if "data" in spec else {})
    device = torch_device(default["device"])
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output_directory / "metrics.jsonl"
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
            benchmark = spec["benchmark"] if "benchmark" in spec else None
            if benchmark in CONSTRUCTIONS:
                sizes = {k: getattr(dataset_config, k) for k in ("n_train", "n_val_iid", "n_iid_test", "n_ood_test")}
                extra = {k: v for k, v in scenario_data.items() if k not in sizes}
                splits = make(benchmark, seed, sizes=sizes, extra=extra)
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
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
    summary = summarize_results(all_results, primary_scenario=primary)
    write_json(output_directory / "summary.json", summary)
    write_json(output_directory / "manifest.json", {"run": spec["name"], "seeds": seeds, "methods": methods, "primary_scenario": primary, "device": str(device)})
    return {"summary": summary}


def run_mf_core_probes(spec: dict, default: dict) -> dict:
    # Table 8. Temporal-mean probe vs core-only GRU.
    device = torch_device(default["device"])
    runs = []
    for seed in range(int(spec["seeds"])):
        splits = make("mf_core", seed)
        train, validation, iid, ood = (splits[name] for name in ("train", "val_iid", "iid_test", "ood_test"))
        y_train, y_iid = torch.from_numpy(train.y), torch.from_numpy(iid.y)
        mean_features = lambda split: as_tensor(np.asarray(split.core_only).mean(axis=1).reshape(len(split.y), -1))
        mean_accuracy = train_mlp_probe(mean_features(train), y_train, [(mean_features(iid), y_iid)], seed * 7 + 3, device)[0]
        mean = float(np.asarray(train.core_only).mean())
        std = float(np.asarray(train.core_only).std()) or 1.0
        core = lambda split: as_tensor(((np.asarray(split.core_only) - mean) / std)[:, :, None])
        model = train_sequence(core(train), y_train, core(validation), torch.from_numpy(validation.y), seed * 31 + 7, device, epochs=100, patience=30)
        runs.append({"mean_probe": mean_accuracy, "gru_iid": eval_sequence(model, core(iid), y_iid, device), "gru_ood": eval_sequence(model, core(ood), torch.from_numpy(ood.y), device)})
    result = {k: aggregate([run[k] for run in runs]) for k in runs[0]}
    write_json(Path(spec["out"]), result)
    return result


def run_mf_core_perframe(spec: dict, default: dict) -> dict:
    # Table 8. Per-frame core-only label probes.
    device = torch_device(default["device"])
    runs = []
    for seed in range(int(spec["seeds"])):
        splits = make("mf_core", seed)
        train, iid = splits["train"], splits["iid_test"]
        y_train, y_iid = torch.from_numpy(train.y), torch.from_numpy(iid.y)
        accuracies = []
        for t in range(train.core_only.shape[1]):
            accuracies.append(train_mlp_probe(as_tensor(np.asarray(train.core_only)[:, t].reshape(len(y_train), -1)), y_train, [(as_tensor(np.asarray(iid.core_only)[:, t].reshape(len(y_iid), -1)), y_iid)], seed * 100 + t, device)[0])
        runs.append(accuracies)
    result = {"per_frame": [aggregate([run[t] for run in runs]) for t in range(len(runs[0]))]}
    write_json(Path(spec["out"]), result)
    return result


def run_ucr(spec: dict, default: dict) -> dict:
    # Table 9. FordA / HAR with an order-pulse overlay on the official split.
    device = torch_device(default["device"])
    dataset = spec["dataset"]
    series, y_all, n_train = load_har(dataset) if dataset in ("har", "har2") else load_forda()
    sample_count = len(y_all)
    length, corr_train = 10, 0.97
    official = spec["official"]
    split_boundaries = [int(n_train * 0.9), n_train, n_train + (sample_count - n_train) // 2, sample_count] if official else [int(sample_count * 0.62), int(sample_count * 0.70), int(sample_count * 0.85), sample_count]
    output_path = Path(spec["out"])
    results = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    for seed in seed_list(spec):
        key = f"seed{seed}"
        if key in results:
            continue
        rng = np.random.default_rng(seed)
        order = np.concatenate([rng.permutation(n_train), n_train + rng.permutation(sample_count - n_train)]) if official else rng.permutation(sample_count)
        pool = {}
        for name, start, end, corr in [("train", 0, split_boundaries[0], corr_train), ("val", split_boundaries[0], split_boundaries[1], corr_train), ("iid", split_boundaries[1], split_boundaries[2], corr_train), ("ood", split_boundaries[2], split_boundaries[3], 1 - corr_train)]:
            split_indices = order[start:end]
            pool[name] = overlay_order_pulse(series[split_indices], y_all[split_indices], np.random.default_rng(seed * 7 + start), corr, len(split_indices))

        def tensors(name, mode):
            core_array, nuisance_array, labels, direction = pool[name]
            core = torch.from_numpy(core_array)[:, :, None, :]
            nuisance = torch.from_numpy(nuisance_array)[:, :, None, :]
            x = torch.cat([core, nuisance], 2) if mode == "mixed" else (core if mode == "core" else nuisance)
            return x, torch.from_numpy(labels), torch.from_numpy(direction)

        def fit(channels, mode):
            torch.manual_seed(seed)
            model = SegGRU(channels).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            x_train, y_train, _ = tensors("train", mode)
            x_val, y_val, _ = tensors("val", mode)
            best_validation_accuracy, state, stale_epochs = -1, None, 0
            for _ in range(60):
                model.train()
                sample_order = torch.randperm(len(x_train))
                for batch_start in range(0, len(x_train), 128):
                    batch_indices = sample_order[batch_start:batch_start + 128]
                    optimizer.zero_grad(set_to_none=True)
                    F.cross_entropy(model(x_train[batch_indices].to(device)).logits, y_train[batch_indices].to(device)).backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    validation_accuracy = (model(x_val.to(device)).logits.argmax(1).cpu() == y_val).float().mean().item()
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy, stale_epochs, state = validation_accuracy, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale_epochs += 1
                    if stale_epochs >= 15:
                        break
            model.load_state_dict(state)
            model.eval()
            return model

        def accuracy_of(model, x, y):
            return float((model(x.to(device)).logits.argmax(1).cpu() == y).float().mean())

        row = {}
        model = fit(1, "core")
        row["core_only"] = [accuracy_of(model, *tensors("iid", "core")[:2]), accuracy_of(model, *tensors("ood", "core")[:2])]
        model = fit(1, "nuisance")
        row["nuisance_only"] = [accuracy_of(model, *tensors("iid", "nuisance")[:2]), accuracy_of(model, *tensors("ood", "nuisance")[:2])]
        erm = fit(2, "mixed")
        x_iid, y_iid, _ = tensors("iid", "mixed")
        x_ood, y_ood, _ = tensors("ood", "mixed")
        row["erm"] = [accuracy_of(erm, x_iid, y_iid), accuracy_of(erm, x_ood, y_ood)]
        shuffled = x_ood.clone()
        shuffled[:, :, 1] = shuffled[:, torch.randperm(length), 1]
        row["erm_shuffle_nuis"] = accuracy_of(erm, shuffled, y_ood)
        reversed_nuisance = x_ood.clone()
        reversed_nuisance[:, :, 1] = torch.flip(reversed_nuisance[:, :, 1], dims=[1])
        row["erm_reverse_nuis"] = accuracy_of(erm, reversed_nuisance, y_ood)
        reversed_core = x_ood.clone()
        reversed_core[:, :, 0] = torch.flip(reversed_core[:, :, 0], dims=[1])
        row["erm_reverse_core"] = accuracy_of(erm, reversed_core, y_ood)
        no_spurious_pool = {}
        for name in pool:
            core_array, _, labels, _ = pool[name]
            no_spurious_generator = np.random.default_rng(seed * 13 + 1009 * ["train", "val", "iid", "ood"].index(name))
            no_spurious_pool[name] = overlay_order_pulse(core_array, labels, no_spurious_generator, 0.5, len(labels))
        pool_saved = {k: pool[k] for k in pool}
        pool.update(no_spurious_pool)
        model = fit(2, "mixed")
        row["no_spurious"] = [accuracy_of(model, *tensors("iid", "mixed")[:2]), accuracy_of(model, *tensors("ood", "mixed")[:2])]
        pool.update(pool_saved)

        def linear_probe(x, t, target):
            frame = x[:, :, :] if t is None else x[:, t]
            probe_model = nn.Sequential(nn.Linear(frame.reshape(len(frame), -1).shape[1], 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
            optimizer = torch.optim.AdamW(probe_model.parameters(), lr=1e-3)
            x_train = frame.reshape(len(frame), -1)
            for _ in range(30):
                sample_order = torch.randperm(len(x_train))
                for batch_start in range(0, len(x_train), 256):
                    batch_indices = sample_order[batch_start : batch_start + 256]
                    optimizer.zero_grad(set_to_none=True)
                    F.cross_entropy(probe_model(x_train[batch_indices].to(device)), target[batch_indices].to(device)).backward()
                    optimizer.step()
            probe_model.eval()
            return probe_model

        nuisance_train, _, direction_train = tensors("train", "nuisance")
        nuisance_iid, _, direction_iid = tensors("iid", "nuisance")
        best_single = 0.0
        for t in range(length):
            probe_model = linear_probe(nuisance_train, t, direction_train)
            with torch.no_grad():
                frame_accuracy = float((probe_model(nuisance_iid[:, t].reshape(len(nuisance_iid), -1).to(device)).argmax(1).cpu() == direction_iid).float().mean())
            best_single = max(best_single, frame_accuracy)
        row["best_single_seg_dir"] = best_single
        sorted_train = torch.sort(nuisance_train.reshape(len(nuisance_train), length, -1), dim=1)[0]
        sorted_iid = torch.sort(nuisance_iid.reshape(len(nuisance_iid), length, -1), dim=1)[0]
        probe_model = linear_probe(sorted_train, None, direction_train)
        with torch.no_grad():
            row["unordered_set_dir"] = float((probe_model(sorted_iid.reshape(len(sorted_iid), -1).to(device)).argmax(1).cpu() == direction_iid).float().mean())
        mixed_train, y_train, direction_train_mixed = tensors("train", "mixed")
        mixed_iid, y_iid_mixed, direction_iid_mixed = tensors("iid", "mixed")
        probe_model = linear_probe(mixed_train, length - 1, y_train)
        with torch.no_grad():
            row["final_seg_label"] = float((probe_model(mixed_iid[:, length - 1].reshape(len(mixed_iid), -1).to(device)).argmax(1).cpu() == y_iid_mixed).float().mean())
        probe_model = linear_probe(mixed_train, length - 1, direction_train_mixed)
        with torch.no_grad():
            row["final_seg_dir"] = float((probe_model(mixed_iid[:, length - 1].reshape(len(mixed_iid), -1).to(device)).argmax(1).cpu() == direction_iid_mixed).float().mean())
        results[key] = rounded(row)
        write_json(output_path, results)
    return results


def run_graph(spec: dict, default: dict) -> dict:
    # Table A11. Karate / Les Misérables diffusion core with a directional nuisance.
    device = torch_device(default["device"])
    transition, faction, order, n_nodes = graph_setup(spec["graph"])
    seed_results = {}
    for seed in range(int(spec["seeds"])):
        torch.manual_seed(seed)
        measurements = {}
        for scenario, prefix in [("main", ""), ("no_spurious", "ns_")]:
            train = graph_split(transition, faction, order, n_nodes, 4096, seed * 100 + 1, prefix + "train")
            validation = graph_split(transition, faction, order, n_nodes, 1024, seed * 100 + 2, prefix + "train")
            iid = graph_split(transition, faction, order, n_nodes, 2048, seed * 100 + 3, prefix + "train")
            ood = graph_split(transition, faction, order, n_nodes, 2048, seed * 100 + 4, prefix + "ood")
            methods = {"sequence_erm": ("mixed", "seq"), "core_only": ("core", "seq"), "nuisance_only": ("nuisance", "seq"), "final_frame": ("mixed", "mlp")}
            if scenario == "no_spurious":
                methods = {"sequence_erm": ("mixed", "seq")}
            for method_name, (key, kind) in methods.items():
                mean, std = train[key].mean(), max(train[key].std(), 1e-6)
                normalize = lambda array: ((array - mean) / std).astype(np.float32)
                channels = train[key].shape[2]
                model = build_model("sequence_cnn_gru", grid_size=n_nodes, hidden_dim=64, input_channels=channels).to(device) if kind == "seq" else FinalFrameMLP(grid_size=1, hidden_dim=64, input_dim=channels * n_nodes).to(device)
                x_train = torch.from_numpy(normalize(train[key])).to(device)
                y_train = torch.from_numpy(train["y"]).to(device)
                x_val = torch.from_numpy(normalize(validation[key])).to(device)
                y_val = torch.from_numpy(validation["y"]).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                best_validation_accuracy, best_state, stale_epochs = -1, None, 0
                for _ in range(40):
                    model.train()
                    sample_order = torch.randperm(len(x_train), device=device)
                    for batch_start in range(0, len(x_train), 128):
                        batch_indices = sample_order[batch_start:batch_start + 128]
                        optimizer.zero_grad(set_to_none=True)
                        torch.nn.functional.cross_entropy(model(x_train[batch_indices]).logits, y_train[batch_indices]).backward()
                        optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        validation_accuracy = float((model(x_val).logits.argmax(1) == y_val).float().mean())
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy, best_state, stale_epochs = validation_accuracy, {k: v.clone() for k, v in model.state_dict().items()}, 0
                    else:
                        stale_epochs += 1
                        if stale_epochs >= 12:
                            break
                model.load_state_dict(best_state)
                with torch.no_grad():
                    measurements[f"{scenario}/{method_name}"] = {
                        "iid": float((model(torch.from_numpy(normalize(iid[key])).to(device)).logits.argmax(1) == torch.from_numpy(iid["y"]).to(device)).float().mean()),
                        "ood": float((model(torch.from_numpy(normalize(ood[key])).to(device)).logits.argmax(1) == torch.from_numpy(ood["y"]).to(device)).float().mean()),
                    }
        seed_results[seed] = measurements
    summary = {}
    for key in seed_results[0]:
        for split in ("iid", "ood"):
            values = [seed_results[seed][key][split] for seed in seed_results]
            summary[f"{key}/{split}"] = aggregate(values)
    write_json(Path(spec["out"]), summary)
    return summary


def run_multi_init(spec: dict, default: dict) -> dict:
    # Table A18. 15 inits × data seeds, standard vs certification budget.
    device = torch_device(default["device"])
    output_path = Path(spec["out"])
    result = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    budgets = {"standard": (40, 12), "certification": (100, 30)}
    for variant, benchmark in [("fl_trail", "fl_trail"), ("oe_simple", "oe_simple")]:
        for data_seed in range(int(spec["data_seeds"])):
            splits = make(benchmark, data_seed)
            mean = float(np.asarray(splits["train"].mixed).mean())
            std = float(np.asarray(splits["train"].mixed).std()) or 1.0
            x = lambda name: as_tensor((np.asarray(splits[name].mixed) - mean) / std)
            y = lambda name: torch.from_numpy(splits[name].y)
            for budget_name, (epoch_budget, patience) in budgets.items():
                key = f"{variant}/data{data_seed}/{budget_name}"
                if key in result:
                    continue
                rows = []
                for initialization in range(int(spec["inits"])):
                    model = train_sequence(x("train"), y("train"), x("val_iid"), y("val_iid"), 90001 + 613 * initialization, device, epochs=epoch_budget, patience=patience)
                    iid = eval_sequence(model, x("iid_test"), y("iid_test"), device)
                    ood = eval_sequence(model, x("ood_test"), y("ood_test"), device)
                    rows.append({"init": initialization, "iid": round(iid, 4), "ood": round(ood, 4), "regime": regime(iid, ood)})
                regimes = [r["regime"] for r in rows]
                result[key] = {"core": regimes.count("core"), "collapse": regimes.count("collapse"), "chance": regimes.count("chance"), "n": len(rows), "rows": rows}
                write_json(output_path, result)
    return result


def run_accessibility(spec: dict, default: dict) -> dict:
    # Table A1. Epochs to 95% of the cue ceiling.
    device = torch_device(default["device"])
    cues = {
        "diffusion_core": (dict(), "core_only", 1.0),
        "multiframe_core": (dict(diffusion_start_step=8, diffusion_steps_between_frames=2, core_noise_std=0.045, core_noise_growth_power=0.0, observation_noise_std=0.01), "core_only", 0.961),
        "oe_nuisance": (dict(nuisance_trail_decay=0.0), "nuisance_only", 0.97),
    }
    results = {}
    for cue, (overrides, field, ceiling) in cues.items():
        for n_train in [1024, 8192]:
            rows = []
            for seed in range(int(spec["seeds"])):
                config = paper_config(seed, n_train=n_train, n_val_iid=2048, n_iid_test=4096, n_ood_test=256, **overrides)
                splits = {k: generate_split(config, k) for k in ["train", "val_iid", "iid_test"]}
                mean = float(np.asarray(getattr(splits["train"], field)).mean())
                std = float(np.asarray(getattr(splits["train"], field)).std()) or 1.0
                x = lambda name: as_tensor(((np.asarray(getattr(splits[name], field)) - mean) / std)[:, :, None])
                y = lambda name: torch.from_numpy(splits[name].y)
                torch.manual_seed(seed * 31 + 7)
                model = build_model("sequence_cnn_gru", grid_size=16, hidden_dim=64).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                accuracy_threshold, threshold_epoch, best_validation_accuracy, best_state = 0.95 * ceiling, None, -1.0, None
                x_train, x_val, y_train, y_val = x("train"), x("val_iid"), y("train"), y("val_iid")
                for epoch in range(1, 101):
                    model.train()
                    sample_order = torch.randperm(len(x_train))
                    for batch_start in range(0, len(x_train), 128):
                        batch_indices = sample_order[batch_start:batch_start + 128]
                        optimizer.zero_grad(set_to_none=True)
                        F.cross_entropy(model(x_train[batch_indices].to(device)).logits, y_train[batch_indices].to(device)).backward()
                        optimizer.step()
                    model.eval()
                    with torch.no_grad():
                        validation_accuracy = float((model(x_val.to(device)).logits.argmax(1) == y_val.to(device)).float().mean())
                    if validation_accuracy > best_validation_accuracy:
                        best_validation_accuracy, best_state = validation_accuracy, {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    if threshold_epoch is None and validation_accuracy >= accuracy_threshold:
                        threshold_epoch = epoch
                    if threshold_epoch is not None and epoch >= threshold_epoch + 5:
                        break
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    iid = float((model(x("iid_test").to(device)).logits.argmax(1) == y("iid_test").to(device)).float().mean())
                rows.append((iid, threshold_epoch if threshold_epoch is not None else 101))
            accuracies = np.array([row[0] for row in rows])
            epochs_hit = np.array([row[1] for row in rows])
            results[f"{cue}/n{n_train}"] = {"iid_mean": float(accuracies.mean()), "iid_std": float(accuracies.std(ddof=1)), "epochs_to_95pct_ceiling": [int(e) for e in epochs_hit], "epochs_median": float(np.median(epochs_hit))}
    write_json(Path(spec["out"]), results)
    return results


def run_oe_core_equalized(spec: dict, default: dict) -> dict:
    # OE-Core equalization. The core direction flips with probability 0.03.
    device = torch_device(default["device"])
    rows = []
    for seed in range(int(spec["seeds"])):
        splits = make("oe_core_equalized", seed)
        mean = float(np.asarray(splits["train"].mixed).mean())
        std = float(np.asarray(splits["train"].mixed).std()) or 1.0
        x = lambda name: as_tensor((np.asarray(splits[name].mixed) - mean) / std)
        y = lambda name: torch.from_numpy(splits[name].y)
        model = train_sequence(x("train"), y("train"), x("val_iid"), y("val_iid"), seed * 31 + 8, device, epochs=100, patience=30)
        rows.append({"seed": seed, "iid": round(eval_sequence(model, x("iid_test"), y("iid_test"), device), 4), "ood": round(eval_sequence(model, x("ood_test"), y("ood_test"), device), 4)})
    write_json(Path(spec["out"]), {"rows": rows})
    return {"rows": rows}


def run_oe_core_controls(spec: dict, default: dict) -> dict:
    # OE-Core controls. Shuffled-frame training, then paired ERM and IRM on OE-Simple.
    device = torch_device(default["device"])
    results = {}
    rows = []
    for seed in range(int(spec["seeds"])):
        splits = make("oe_core", seed)
        mean = float(np.asarray(splits["train"].mixed).mean())
        std = float(np.asarray(splits["train"].mixed).std()) or 1.0
        x = lambda name: as_tensor((np.asarray(splits[name].mixed) - mean) / std)
        y = lambda name: torch.from_numpy(splits[name].y)
        model = train_sequence(x("train"), y("train"), x("val_iid"), y("val_iid"), seed * 31 + 8, device, shuffle_frames=True, epochs=100, patience=30)
        rows.append({"seed": seed, "iid": round(eval_sequence(model, x("iid_test"), y("iid_test"), device, "shuffled", seed), 4), "ood": round(eval_sequence(model, x("ood_test"), y("ood_test"), device, "shuffled", seed + 1), 4)})
    results["order_rand"] = rows
    rows = []
    for seed in range(int(spec["seeds"])):
        splits = make("oe_simple", seed)
        iid, ood, _ = train_robust("erm", splits, seed, device)
        rows.append({"seed": seed, "iid": round(iid, 4), "ood": round(ood, 4)})
    results["paired_erm"] = rows
    for irm_penalty in [100.0, 10000.0]:
        rows = []
        for seed in range(int(spec["seeds"])):
            splits0 = make("oe_simple", seed)
            splits1 = make("oe_simple", seed + 7919, extra={"nuisance_correlation": 0.85}, sizes={**SIZES, "n_train": 4096})
            iid, ood, _ = train_irm(splits0, splits1, seed, device, irm_lambda=irm_penalty)
            rows.append({"seed": seed, "iid": round(iid, 4), "ood": round(ood, 4)})
        results[f"irm_lam{int(irm_penalty)}"] = rows
    write_json(Path(spec["out"]), results)
    return results


def run_arch_cue(spec: dict, default: dict) -> dict:
    # Table A4–A5. Single-cue LSTM/TCN/Transformer/pool; GroupDRO at ρ=0.70.
    device = torch_device(default["device"])
    results = {}
    for architecture, model_type in ARCHITECTURES.items():
        if architecture == "gru":
            continue
        for field in ["nuisance_only", "core_only"]:
            rows = []
            for seed in range(int(spec["seeds"])):
                splits = make("oe_simple", seed)
                mean = float(np.asarray(getattr(splits["train"], field)).mean())
                std = float(np.asarray(getattr(splits["train"], field)).std()) or 1.0
                x = lambda name: as_tensor(((np.asarray(getattr(splits[name], field)) - mean) / std)[:, :, None])
                y = lambda name: torch.from_numpy(splits[name].y)
                model = train_sequence(x("train"), y("train"), x("val_iid"), y("val_iid"), seed * 31 + 7, device, model_type=model_type)
                rows.append((eval_sequence(model, x("iid_test"), y("iid_test"), device), eval_sequence(model, x("ood_test"), y("ood_test"), device)))
            iid_accuracies = np.array([row[0] for row in rows])
            ood_accuracies = np.array([row[1] for row in rows])
            results[f"{architecture}/{field}"] = {"iid": float(iid_accuracies.mean()), "iid_std": float(iid_accuracies.std(ddof=1)), "ood": float(ood_accuracies.mean()), "ood_std": float(ood_accuracies.std(ddof=1))}
    rows = []
    for seed in range(int(spec["seeds"])):
        splits = make("oe_simple", seed, extra={"nuisance_correlation": 0.70})
        iid, ood, _ = train_robust("groupdro", splits, seed, device)
        rows.append((iid, ood))
    iid_accuracies = np.array([row[0] for row in rows])
    ood_accuracies = np.array([row[1] for row in rows])
    results["groupdro_corr0.70"] = {"iid": float(iid_accuracies.mean()), "ood": float(ood_accuracies.mean()), "core": int(((iid_accuracies >= 0.8) & (ood_accuracies >= 0.8)).sum())}
    write_json(Path(spec["out"]), results)
    return results


def run_groupdro(spec: dict, default: dict) -> dict:
    # GroupDRO sweep. Group step 0.01, 0.1, and 0.001, with and without a balanced sampler.
    device = torch_device(default["device"])
    result = {}
    for name, training_settings in {"eta0.01_balanced": dict(group_step=0.01, balanced_sampler=True), "eta0.1_standard": dict(group_step=0.1, balanced_sampler=False), "eta0.001_standard": dict(group_step=0.001, balanced_sampler=False)}.items():
        rows = []
        for seed in range(int(spec["seeds"])):
            splits = make("oe_simple", seed)
            iid, ood, _ = train_robust("groupdro", splits, seed, device, **training_settings)
            rows.append({"seed": seed, "iid": round(iid, 4), "ood": round(ood, 4)})
        iid_accuracies = np.array([r["iid"] for r in rows])
        ood_accuracies = np.array([r["ood"] for r in rows])
        result[name] = {"iid_mean": float(iid_accuracies.mean()), "ood_mean": float(ood_accuracies.mean()), "ood_std": float(ood_accuracies.std(ddof=1)), "core": int(((iid_accuracies >= 0.8) & (ood_accuracies >= 0.8)).sum()), "collapse": int(((iid_accuracies >= 0.8) & (ood_accuracies <= 0.2)).sum()), "rows": rows}
    write_json(Path(spec["out"]), result)
    return result


def run_gradsal(spec: dict, default: dict) -> dict:
    # Input gradient. Nuisance-channel share on trail and OE-Simple.
    device = torch_device(default["device"])
    results = {}
    for variant, benchmark in [("trail", "fl_trail"), ("oe", "oe_simple")]:
        shares, profiles = [], []
        for seed in range(int(spec["seeds"])):
            splits = make(benchmark, seed)
            mean = float(np.asarray(splits["train"].mixed).mean())
            std = float(np.asarray(splits["train"].mixed).std()) or 1.0
            x = lambda name: as_tensor((np.asarray(splits[name].mixed) - mean) / std)
            y = lambda name: torch.from_numpy(splits[name].y)
            model = train_sequence(x("train"), y("train"), x("val_iid"), y("val_iid"), seed, device)
            share, profile = input_gradient_saliency(model, x("iid_test")[:512], device)
            shares.append(share)
            profiles.append(np.asarray(profile).round(4).tolist())
        results[variant] = {"nuis_share_mean": round(float(np.mean(shares)), 4), "nuis_share_std": round(float(np.std(shares)), 4), "frame_profile_mean": np.mean(profiles, axis=0).round(4).tolist()}
    write_json(Path(spec["out"]), results)
    return results


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
    output_path = Path(spec["out"])
    results = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    sizes = dict(n_train=8192, n_val_iid=2048, n_iid_test=2048, n_ood_test=2048)
    for name, overrides in configs.items():
        for seed in range(int(spec["seeds"])):
            base = f"{name}/seed{seed}"
            if f"{base}/erm" in results:
                continue
            extra = {**overrides, "nuisance_motion": "real_video", "real_video_cache": "data/real_video/cache_g16_L8_s5.npz"}
            splits = make("fl_trail", seed, sizes=sizes, extra=extra)
            def run_field(field, run_seed):
                train_field = np.asarray(getattr(splits["train"], field))
                if train_field.ndim == 4:
                    train_field = train_field[:, :, None]
                mean = float(train_field.mean())
                std = float(train_field.std()) or 1.0
                def normalized(name):
                    array = np.asarray(getattr(splits[name], field))
                    if array.ndim == 4:
                        array = array[:, :, None]
                    return as_tensor((array - mean) / std)
                labels = lambda name: torch.from_numpy(splits[name].y)
                model = train_sequence(normalized("train"), labels("train"), normalized("val_iid"), labels("val_iid"), run_seed, device)
                return [round(eval_sequence(model, normalized("iid_test"), labels("iid_test"), device), 4), round(eval_sequence(model, normalized("ood_test"), labels("ood_test"), device), 4)]
            results[f"{base}/erm"] = run_field("mixed", seed * 31 + 11)
            results[f"{base}/core_only"] = run_field("core_only", seed * 31 + 12)
            results[f"{base}/nuis_only"] = run_field("nuisance_only", seed * 31 + 13)
            def last(name):
                array = np.asarray(splits[name].mixed)[:, -1]
                return array.reshape(len(array), -1)
            mean, std = last("train").mean(), last("train").std() or 1.0
            final_frame_inputs = {name: torch.from_numpy(((last(name) - mean) / std).astype(np.float32)) for name in splits}
            y = {name: torch.from_numpy(splits[name].y) for name in splits}
            torch.manual_seed(seed * 31 + 14)
            final_frame = nn.Sequential(nn.Linear(final_frame_inputs["train"].shape[1], 128), nn.ReLU(), nn.Linear(128, 2)).to(device)
            optimizer = torch.optim.AdamW(final_frame.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(20):
                sample_order = torch.randperm(len(final_frame_inputs["train"]))
                for batch_start in range(0, len(sample_order), 256):
                    batch_indices = sample_order[batch_start : batch_start + 256]
                    optimizer.zero_grad(set_to_none=True)
                    F.cross_entropy(final_frame(final_frame_inputs["train"][batch_indices].to(device)), y["train"][batch_indices].to(device)).backward()
                    optimizer.step()
            final_frame.eval()
            with torch.no_grad():
                results[f"{base}/final_frame"] = [
                    float((final_frame(final_frame_inputs["iid_test"].to(device)).argmax(1).cpu() == y["iid_test"]).float().mean()),
                    float((final_frame(final_frame_inputs["ood_test"].to(device)).argmax(1).cpu() == y["ood_test"]).float().mean()),
                ]
                results[f"{base}/final_frame"] = [round(v, 4) for v in results[f"{base}/final_frame"]]
            no_spurious_splits = make("fl_trail", seed, sizes=sizes, extra={**extra, "train_nuisance_mode": "randomized", "ood_mode": "randomized"})
            splits = no_spurious_splits
            results[f"{base}/no_spurious"] = run_field("mixed", seed * 31 + 11)
            write_json(output_path, results)
    return results


def run_unified(spec: dict, default: dict) -> dict:
    # Table A6. ERM / GroupDRO / IRMv1 / DANN / JTT / frame-rand, sel_iid and sel_shift.
    device = torch_device(default["device"])
    output_path = Path(spec["out"])
    results = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    jobs = [(m, "gru") for m in ["erm", "groupdro_joint", "irmv1", "dann_dir", "jtt", "frame_rand"]] + [("erm", k) for k in ARCHITECTURES if k != "gru"]
    for seed in range(int(spec["seeds"])):
        splits = make("oe_simple", seed)
        for method, architecture in jobs:
            key = f"{method}/{architecture}/seed{seed}"
            if key in results:
                continue
            selection_results = train_dual(method, architecture, seed, splits, device)
            results[key] = {selection_rule: [round(v, 4) for v in selection_results[selection_rule]] for selection_rule in selection_results}
            write_json(output_path, results)
    return results


def run(name: str, config_dir: Path = Path("configs")) -> dict:
    default = load_yaml(config_dir / "default.yaml")
    experiments = load_yaml(config_dir / "experiments.yaml")
    spec = copy.deepcopy(experiments[name])
    spec["name"] = name
    if spec["kind"] == "train":
        benchmark_configs = load_yaml(config_dir / "benchmarks.yaml")
        if "benchmark" in spec:
            spec["data"] = deep_update(benchmark_configs[spec["benchmark"]], spec["data"] if "data" in spec else {})
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="")
    parser.add_argument("--configs", default="configs")
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = Path(args.configs)
    if args.run and args.run != "all":
        result = run(args.run, config_dir)
        print(json.dumps(result["summary"] if "summary" in result else result, indent=2))
        return
    experiments = load_yaml(config_dir / "experiments.yaml")
    for name, spec in experiments.items():
        if type(spec) is dict and "kind" in spec:
            output_path = Path(spec["out"])
            if spec["kind"] == "train" and (output_path / "summary.json").exists():
                continue
            if spec["kind"] != "train" and output_path.exists():
                continue
            run(name, config_dir)


if __name__ == "__main__":
    main()
