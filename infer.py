"""Batch inference, Haar-state verification, and optional symbolic ZX certification."""

from contextlib import nullcontext
import glob
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import pickle
from queue import Queue
import random
import threading
import time

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import webdataset as wds

from src.data.utils import Vocab
from src.data.zxdiagram_circuit_dataset import CircuitProcessor, make_eval_loader
from src.model.sampler import Sampler
from src.utils.checkpoint import load_checkpoint
from src.utils.compare_circuits import certify_equivalence_zx, string_to_circuit, verify_gen_seqs

OmegaConf.register_new_resolver("eval", eval)
logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _circuit_hash(circ) -> str:
    payload = f"{circ.qubits}|{CircuitProcessor.circuit_to_string(circ)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_vocab_file(cfg: DictConfig) -> str:
    if getattr(cfg, "vocab_path", None) and os.path.exists(cfg.vocab_path):
        return cfg.vocab_path
    parent_vocab = os.path.join(os.path.dirname(os.path.normpath(cfg.data_path)), "vocab.txt")
    if os.path.exists(parent_vocab):
        return parent_vocab
    local_vocab = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "vocab.txt")
    if os.path.exists(local_vocab):
        return local_vocab
    raise FileNotFoundError("vocab.txt not found; pass vocab_path=/path/to/vocab.txt.")


@torch.inference_mode()
def run_inference(cfg: DictConfig, sampler: Sampler, dataset, device: torch.device):
    logger.info("Starting batch inference (K=%d, T=%.3f)...", cfg.num_sample_per_seq, cfg.temperature)
    t0 = time.time()
    os.makedirs(cfg.output_dir, exist_ok=True)
    output_fn = os.path.join(cfg.output_dir, "generate_samples.tar")

    write_queue: Queue = Queue(maxsize=64)

    def writer_thread():
        sink = wds.TarWriter(output_fn)
        while True:
            item = write_queue.get()
            if item is None:
                break
            sink.write(item)
        sink.close()

    wt = threading.Thread(target=writer_thread, daemon=True)
    wt.start()

    sample_id = 0
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    for batch_id, data in enumerate(dataset):
        batch = {k: v.to(device, non_blocking=(device.type == "cuda")) for k, v in data.items() if k != "raw_data"}
        max_len = int(batch["tgt_len"].max().item() * 1.2)
        with amp_ctx:
            _, seqs, _, _, _ = sampler(
                batch,
                temperature=cfg.temperature,
                max_length=max_len,
                num_samples=cfg.num_sample_per_seq,
                top_k=cfg.top_k,
                top_p=cfg.top_p,
                max_chunk_size=getattr(cfg, "sampler_chunk_size", 8),
            )

        for sid, seq_list in enumerate(seqs):
            record = dict(data["raw_data"][sid])
            record["gen_seqs"] = seq_list
            record["src_hash"] = _circuit_hash(record["circ"])
            write_queue.put({"__key__": f"{sample_id:08d}", "pyd": record})
            sample_id += 1
            if sample_id >= cfg.validation_size:
                break

        if (batch_id + 1) % 10 == 0 or sample_id >= cfg.validation_size:
            logger.info("Generated %d / %d circuits (%.1fs)", sample_id, cfg.validation_size, time.time() - t0)
        if sample_id >= cfg.validation_size:
            break

    write_queue.put(None)
    wt.join()
    logger.info("Saved %d generated circuit groups to %s in %.1fs", sample_id, output_fn, time.time() - t0)


def _verify_worker(item):
    q, key, src_hash, src_circ, tgt_circ, gen_seqs, atol, probe_seed, certify_zx = item
    seed = (int(src_hash[:8], 16) ^ int(probe_seed)) if src_hash else None
    fids, best_gates, tgt_gates, src_gates, best_idx = verify_gen_seqs(
        src_circ, tgt_circ, gen_seqs, atol=atol, probe_seed=seed
    )
    zx_certified = None
    if certify_zx and best_idx >= 0:
        gen_circ = string_to_circuit(gen_seqs[best_idx], q)
        zx_certified = certify_equivalence_zx(tgt_circ, gen_circ)
    return q, key, fids, best_gates, tgt_gates, src_gates, zx_certified


def run_verification(cfg: DictConfig):
    logger.info("Starting Haar-state verification (workers=%d, certify_zx=%s)...", cfg.verify_num_workers, cfg.certify_zx)
    t0 = time.time()
    tar_path = os.path.join(cfg.output_dir, "generate_samples.tar")
    dataset = wds.WebDataset(tar_path)
    atol = float(getattr(cfg, "fidelity_tolerance", 1e-4))
    probe_seed = int(getattr(cfg, "probe_seed", 0))
    certify_zx = bool(getattr(cfg, "certify_zx", False))

    def gen_tasks():
        for sample in dataset:
            data = pickle.loads(sample["pyd"])
            src_circ = data["circ"]
            src_hash = data.get("src_hash") or _circuit_hash(src_circ)
            yield (
                src_circ.qubits,
                sample["__key__"],
                src_hash,
                src_circ,
                data["circ_opt"],
                data["gen_seqs"],
                atol,
                probe_seed,
                certify_zx,
            )

    open_files = {}
    summary = {
        "n_circuits": 0,
        "n_solved": 0,
        "pass1_sum": 0.0,
        "n_zx_checked": 0,
        "n_zx_certified": 0,
        "red_all_model": [],
        "red_2q_model": [],
        "red_T_model": [],
        "red_all_ref": [],
        "red_2q_ref": [],
        "red_T_ref": [],
        "better_all": 0,
        "better_2q": 0,
        "better_T": 0,
    }
    per_qubit = {}

    with mp.Pool(cfg.verify_num_workers) as pool:
        for q, key, fids, best_gates, tgt_gates, src_gates, zx_cert in pool.imap_unordered(
            _verify_worker, gen_tasks(), chunksize=1
        ):
            f = open_files.get(q)
            if f is None:
                f = open(os.path.join(cfg.output_dir, f"fidelity_q{q}.txt"), "w", encoding="utf-8")
                open_files[q] = f

            line = " ".join(f"{r:.4f}" for r in fids)
            if best_gates is not None:
                line += (
                    f"\tgen(all,2q,T): {best_gates[0]}, {best_gates[1]}, {best_gates[2]}"
                    f"\tref(all,2q,T): {tgt_gates[0]}, {tgt_gates[1]}, {tgt_gates[2]}"
                    f"\tsrc(all,2q,T): {src_gates[0]}, {src_gates[1]}, {src_gates[2]}"
                )
            f.write(f"{key}, {line}\n")

            is_ok = np.abs(fids - 1.0) <= atol
            solved = bool(is_ok.any())
            summary["n_circuits"] += 1
            summary["pass1_sum"] += float(is_ok.mean())

            qs = per_qubit.setdefault(q, [0, 0, 0, 0])
            qs[0] += 1
            qs[1] += len(fids)
            qs[2] += int(solved)
            qs[3] += int(is_ok.sum())

            if solved and best_gates is not None:
                summary["n_solved"] += 1
                summary["red_all_model"].append(src_gates[0] - best_gates[0])
                summary["red_2q_model"].append(src_gates[1] - best_gates[1])
                summary["red_T_model"].append(src_gates[2] - best_gates[2])
                summary["red_all_ref"].append(src_gates[0] - tgt_gates[0])
                summary["red_2q_ref"].append(src_gates[1] - tgt_gates[1])
                summary["red_T_ref"].append(src_gates[2] - tgt_gates[2])
                summary["better_all"] += int(best_gates[0] < tgt_gates[0])
                summary["better_2q"] += int(best_gates[1] < tgt_gates[1])
                summary["better_T"] += int(best_gates[2] < tgt_gates[2])
                if zx_cert is not None:
                    summary["n_zx_checked"] += 1
                    summary["n_zx_certified"] += int(zx_cert)

    for q, f in open_files.items():
        nc, ns, ncc, ncs = per_qubit[q]
        f.write(f"\nsuccess circuit: {ncc}/{nc} ({100.0 * ncc / max(nc, 1):.4f}%)\n")
        f.write(f"success sequence: {ncs}/{ns} ({100.0 * ncs / max(ns, 1):.4f}%)\n")
        f.close()

    n = max(summary["n_circuits"], 1)
    ns = max(summary["n_solved"], 1)
    report = {
        "n_circuits": summary["n_circuits"],
        "n_solved": summary["n_solved"],
        "pass@1": round(summary["pass1_sum"] / n, 4),
        "pass@K": round(summary["n_solved"] / n, 4),
        "K": int(cfg.num_sample_per_seq),
        "mean_reduced_model": {
            "all": round(float(np.mean(summary["red_all_model"])), 3) if summary["red_all_model"] else None,
            "2q": round(float(np.mean(summary["red_2q_model"])), 3) if summary["red_2q_model"] else None,
            "T": round(float(np.mean(summary["red_T_model"])), 3) if summary["red_T_model"] else None,
        },
        "mean_reduced_pyzx_ref": {
            "all": round(float(np.mean(summary["red_all_ref"])), 3) if summary["red_all_ref"] else None,
            "2q": round(float(np.mean(summary["red_2q_ref"])), 3) if summary["red_2q_ref"] else None,
            "T": round(float(np.mean(summary["red_T_ref"])), 3) if summary["red_T_ref"] else None,
        },
        "better_than_ref_rate": {
            "all": round(summary["better_all"] / ns, 4),
            "2q": round(summary["better_2q"] / ns, 4),
            "T": round(summary["better_T"] / ns, 4),
        },
        "zx_certification": {
            "checked": summary["n_zx_checked"],
            "certified": summary["n_zx_certified"],
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    report_path = Path(cfg.output_dir) / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Verification summary:\n%s", json.dumps(report, indent=2))


@hydra.main(version_base=None, config_path="config", config_name="infer")
def main(cfg: DictConfig):
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    if cfg.inference:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        model = instantiate(cfg.model)
        model = load_checkpoint(model, cfg.checkpoint)
        model.eval().to(device)
        if device.type == "cuda":
            model.bfloat16()

        urls = sorted(glob.glob(os.path.join(cfg.data_path, "*.tar")))
        if not urls:
            raise FileNotFoundError(f"No .tar files found under data_path={cfg.data_path}")
        vocab_file = _resolve_vocab_file(cfg)

        dataset = make_eval_loader(
            urls=urls,
            vocab=Vocab(vocab_file),
            max_num_qubits=cfg.max_num_qubits,
            batch_size=cfg.batch_size,
            choice_graph_mode=cfg.choice_graph_mode,
        )
        sampler = Sampler(model, vocab_path=vocab_file, max_qubits=cfg.max_num_qubits)
        run_inference(cfg, sampler, dataset, device)

    if cfg.verify_circuit:
        run_verification(cfg)


if __name__ == "__main__":
    main()
