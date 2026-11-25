"""Single-circuit quickstart CLI: optimize one Clifford+T circuit with GraftQ."""

import argparse
from pathlib import Path

import numpy as np
import pyzx as zx
import torch
from torch_geometric.data import Batch

from src.data.utils import Vocab
from src.data.zxdiagram_circuit_dataset import CircuitProcessor
from src.model.gnn_transformer import GNNTransformer
from src.model.sampler import Sampler
from src.utils.checkpoint import load_checkpoint
from src.utils.compare_circuits import (
    certify_equivalence_zx,
    string_to_circuit,
    verify_gen_seqs,
)
from src.utils.zx_simplify import flow_opt


def load_source_circuit(args) -> zx.Circuit:
    if args.qasm:
        return zx.Circuit.load(args.qasm).to_basic_gates()
    if args.gates:
        if args.qubits is None:
            raise ValueError("--qubits must be provided when using --gates.")
        return string_to_circuit(args.gates, args.qubits)
    raise ValueError("Provide either --qasm PATH or --gates TOKEN_STRING --qubits N.")


def main():
    parser = argparse.ArgumentParser(description="Optimize a single quantum circuit using GraftQ.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained GraftQ .ckpt file.")
    parser.add_argument("--qasm", type=str, default=None, help="Input OpenQASM 2.0 file.")
    parser.add_argument("--gates", type=str, default=None, help="Input space-separated gate token string.")
    parser.add_argument("--qubits", type=int, default=None, help="Number of qubits (required with --gates).")
    parser.add_argument("--vocab", type=str, default="assets/vocab.txt", help="Path to 984-token vocab.txt.")
    parser.add_argument("--num-samples", "-K", type=int, default=8, help="Sampling budget K (default: 8).")
    parser.add_argument("--temperature", "-T", type=float, default=0.2, help="Decoding temperature (default: 0.2).")
    parser.add_argument("--max-qubits", type=int, default=20, help="Model max_num_qubits (default: 20).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--certify-zx", action="store_true", help="Also certify output with PyZX verify_equality.")
    parser.add_argument("--output-qasm", type=Path, default=None, help="Optional path to write optimized QASM.")
    args = parser.parse_args()

    src_circ = load_source_circuit(args)
    if src_circ.qubits > args.max_qubits:
        raise ValueError(f"Circuit has {src_circ.qubits} qubits, exceeding max_qubits={args.max_qubits}.")

    ref_circ, graph_list = flow_opt(src_circ)
    d2_graph = graph_list[-1]

    vocab = Vocab(args.vocab)
    processor = CircuitProcessor(vocab, max_qubits=args.max_qubits)
    src_tokens = torch.LongTensor(vocab.tokenize(processor.circuit_to_string(src_circ)))
    ref_tokens = torch.LongTensor(vocab.tokenize(processor.circuit_to_string(ref_circ)))
    pyg_graph = processor.pyzx_to_pyg(d2_graph, src_circ.qubits)

    device = torch.device(args.device)
    model = GNNTransformer(
        d_model=1024,
        gnn_in_features=2 * args.max_qubits + 5,
        vocab_size=len(vocab),
        nhead=16,
        dim_feedforward=2048,
        num_encoder_layers=8,
        num_decoder_layers=8,
        num_gnn_layers=4,
        dropout=0.0,
    )
    load_checkpoint(model, args.checkpoint)
    model.eval().to(device)
    if device.type == "cuda":
        model.bfloat16()

    batch = {
        "src": src_tokens.unsqueeze(0).to(device),
        "src_len": torch.LongTensor([len(src_tokens)]).to(device),
        "tgt_len": torch.LongTensor([len(ref_tokens)]).to(device),
        "num_qubits": torch.LongTensor([src_circ.qubits]).to(device),
        "graph": Batch.from_data_list([pyg_graph]).to(device),
    }

    sampler = Sampler(model, vocab_path=args.vocab, max_qubits=args.max_qubits)
    max_length = max(16, int(len(ref_tokens) * 1.2))
    _, seqs, _, _, _ = sampler(
        batch,
        temperature=args.temperature,
        max_length=max_length,
        num_samples=args.num_samples,
        top_k=0,
        top_p=1.0,
        max_chunk_size=8,
    )

    candidates = seqs[0]
    fids, best_gates, ref_gates, src_gates, best_idx = verify_gen_seqs(src_circ, ref_circ, candidates, atol=1e-4)

    print(f"Source circuit      (all, 2Q, T): {src_gates}")
    print(f"PyZX ref circuit    (all, 2Q, T): {ref_gates}")
    print(f"Valid candidates: {int((np.abs(fids - 1.0) <= 1e-4).sum())}/{args.num_samples}")

    if best_idx < 0:
        print("No equivalent candidate found; falling back to PyZX reference.")
        best_circ = ref_circ
    else:
        best_seq = candidates[best_idx]
        best_circ = string_to_circuit(best_seq, src_circ.qubits)
        print(f"GraftQ best circuit (all, 2Q, T): {best_gates}")
        if args.certify_zx:
            certified = certify_equivalence_zx(ref_circ, best_circ)
            print(f"PyZX symbolic certification: {'PROVEN' if certified else 'UNPROVEN'}")
        print(f"Optimized gate tokens:\n{best_seq}")

    if args.output_qasm is not None:
        args.output_qasm.parent.mkdir(parents=True, exist_ok=True)
        args.output_qasm.write_text(best_circ.to_qasm(), encoding="utf-8")
        print(f"Wrote optimized QASM to {args.output_qasm}")


if __name__ == "__main__":
    main()
