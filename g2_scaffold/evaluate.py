#!/usr/bin/env python
"""Target-free G2 evaluation, calibration, and prediction artifact writer."""

import argparse
import copy
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from data.datamodule import MaskedFormerTopsWsDataModule
from utils.utils import load_any_config

from g2_scaffold.decoder import decode_batch
from g2_scaffold.metrics import METRIC_KEYS, merge_counts, rates, score_batch
from g2_scaffold.targets import targets_to_g2
from g2_scaffold.train import G2Trainer, build_model


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _take(value, limit):
    if torch.is_tensor(value):
        return value[:limit]
    if isinstance(value, dict):
        return {key: _take(item, limit) for key, item in value.items()}
    return value


def _load_module(config: dict, checkpoint: str, device: torch.device) -> G2Trainer:
    model = build_model(config)
    module = G2Trainer.load_from_checkpoint(
        checkpoint, model=model, config=config, map_location=device
    )
    module.to(device).eval()
    return module


def _loader(config: dict, split: str):
    data = MaskedFormerTopsWsDataModule(config)
    if split == "calibration":
        data.setup("calibrate")
        return data.calibration_dataloader()
    if split == "test":
        data.setup("test")
        return data.test_dataloader()
    raise ValueError("split must be 'calibration' or 'test'")


@contextmanager
def _configured_data(config: dict, input_path: Optional[str], stress_file: Optional[str]):
    """Provide an optional stress file under the datamodule's test filename."""
    cfg = copy.deepcopy(config)
    if input_path:
        cfg["data_modules"]["input_path"] = input_path
    if not stress_file:
        yield cfg
        return

    with tempfile.TemporaryDirectory(prefix="g2-stress-") as temp_dir:
        stress_path = os.path.abspath(stress_file)
        link = os.path.join(temp_dir, "ttbar_contract_processed_test.h5")
        os.symlink(stress_path, link)
        cfg["data_modules"]["input_path"] = temp_dir
        yield cfg


def _prediction_arrays(decoded) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.zeros((len(decoded), 2), dtype=np.int8)
    w_pairs = np.full((len(decoded), 2, 2), -1, dtype=np.int16)
    b_jets = np.full((len(decoded), 2), -1, dtype=np.int16)
    scores = np.zeros(len(decoded), dtype=np.float32)
    for row, event in enumerate(decoded):
        scores[row] = event.score
        for query, chain in enumerate(event.chains):
            states[row, query] = chain.state
            if chain.w_pair is not None:
                w_pairs[row, query] = chain.w_pair
            if chain.b_jet is not None:
                b_jets[row, query] = chain.b_jet
    return states, w_pairs, b_jets, scores


def evaluate_split(
    module: G2Trainer,
    config: dict,
    split: str,
    device: torch.device,
    state_temperature: float = 1.0,
    candidate_temperature: float = 1.0,
    max_events: Optional[int] = None,
    output_prefix: Optional[str] = None,
) -> dict:
    loader = _loader(config, split)
    counts = {key: 0 for key in METRIC_KEYS}
    all_event_ids, all_states, all_w, all_b, all_scores = [], [], [], [], []
    seen = 0

    with torch.inference_mode():
        for inputs, raw_targets in loader:
            remaining = None if max_events is None else max_events - seen
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None:
                batch_size = next(value for value in inputs.values() if torch.is_tensor(value)).shape[0]
                if remaining < batch_size:
                    inputs = _take(inputs, remaining)
                    raw_targets = _take(raw_targets, remaining)

            inputs = _move(inputs, device)
            raw_targets = _move(raw_targets, device)
            targets = targets_to_g2(raw_targets)
            outputs = module(inputs)
            decoded = decode_batch(
                outputs,
                targets["valid_particles"],
                state_temperature=state_temperature,
                candidate_temperature=candidate_temperature,
            )
            merge_counts(counts, score_batch(decoded, targets))
            event_ids = inputs.get("event_id")
            if event_ids is None:
                raise ValueError("G2 evaluation requires stable event_id values")
            states, w_pairs, b_jets, scores = _prediction_arrays(decoded)
            all_event_ids.append(event_ids.cpu().numpy().astype(np.uint64))
            all_states.append(states)
            all_w.append(w_pairs)
            all_b.append(b_jets)
            all_scores.append(scores)
            seen += len(decoded)

    summary = rates(counts)
    summary.update({
        "split": split,
        "state_temperature": state_temperature,
        "candidate_temperature": candidate_temperature,
    })
    if output_prefix:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prefix.with_suffix(".npz"),
            event_id=np.concatenate(all_event_ids),
            predicted_state=np.concatenate(all_states),
            predicted_w_pair=np.concatenate(all_w),
            predicted_b=np.concatenate(all_b),
            score=np.concatenate(all_scores),
        )
        with prefix.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def fit_temperatures(
    module: G2Trainer,
    config: dict,
    device: torch.device,
    max_events: int = 512,
    grid: Iterable[float] = (0.75, 1.0, 1.5),
) -> dict:
    """Choose decoder temperatures on calibration events only."""
    loader = _loader(config, "calibration")
    records = []
    seen = 0
    with torch.inference_mode():
        for inputs, raw_targets in loader:
            remaining = max_events - seen
            if remaining <= 0:
                break
            batch_size = next(value for value in inputs.values() if torch.is_tensor(value)).shape[0]
            if remaining < batch_size:
                inputs = _take(inputs, remaining)
                raw_targets = _take(raw_targets, remaining)
            inputs = _move(inputs, device)
            raw_targets = _move(raw_targets, device)
            targets = targets_to_g2(raw_targets)
            outputs = {key: value.cpu() for key, value in module(inputs).items()}
            records.append(({key: value.cpu() for key, value in targets.items()}, outputs))
            seen += outputs["state_logits"].shape[0]

    best = None
    for state_temperature in grid:
        for candidate_temperature in grid:
            counts = {key: 0 for key in METRIC_KEYS}
            for targets, outputs in records:
                decoded = decode_batch(
                    outputs,
                    targets["valid_particles"],
                    state_temperature=state_temperature,
                    candidate_temperature=candidate_temperature,
                )
                merge_counts(counts, score_batch(decoded, targets))
            summary = rates(counts)
            key = (summary["event_exact"], summary["full_top_exact"], summary["w_exact"])
            if best is None or key > best["score"]:
                best = {
                    "score": key,
                    "state_temperature": state_temperature,
                    "candidate_temperature": candidate_temperature,
                    "events": counts["events"],
                }
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="g2_scaffold/g2_pilot.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("calibration", "test"), default="test")
    parser.add_argument("--input-path")
    parser.add_argument("--stress-file")
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-temperature", type=float, default=1.0)
    parser.add_argument("--candidate-temperature", type=float, default=1.0)
    parser.add_argument("--output-prefix")
    parser.add_argument("--fit-calibration", action="store_true")
    parser.add_argument("--calibration-events", type=int, default=512)
    parser.add_argument("--calibration-output")
    args = parser.parse_args()

    config = load_any_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    with _configured_data(config, args.input_path, args.stress_file) as configured:
        module = _load_module(configured, args.checkpoint, device)
        if args.fit_calibration:
            result = fit_temperatures(module, configured, device, args.calibration_events)
            output = json.dumps(result, indent=2, sort_keys=True)
            print(output)
            if args.calibration_output:
                Path(args.calibration_output).write_text(output + "\n", encoding="utf-8")
            return
        summary = evaluate_split(
            module,
            configured,
            args.split,
            device,
            args.state_temperature,
            args.candidate_temperature,
            args.max_events,
            args.output_prefix,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
