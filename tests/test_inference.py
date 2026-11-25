"""Unit tests for GraftQ inference: vocab, masking, model, sampler, checkpoint loading, and verification."""

from pathlib import Path
import tempfile

import pyzx as zx
import torch
from torch_geometric.data import Batch, Data

from src.data.utils import Vocab
from src.data.zxdiagram_circuit_dataset import CircuitProcessor
from src.model.gnn_transformer import GNNTransformer
from src.model.sampler import Sampler, build_qubit_token_mask
from src.utils.checkpoint import load_checkpoint
from src.utils.compare_circuits import certify_equivalence_zx, count_gates, string_to_circuit, verify_gen_seqs

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_small_model() -> GNNTransformer:
    return GNNTransformer(
        d_model=32,
        gnn_in_features=45,
        vocab_size=984,
        nhead=4,
        dim_feedforward=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_gnn_layers=2,
        dropout=0.0,
    ).eval()


def test_vocab_and_qubit_validity_mask():
    vocab_path = REPO_ROOT / "assets" / "vocab.txt"
    assert vocab_path.exists()
    vocab = Vocab(str(vocab_path))
    assert len(vocab) == 984

    ids = vocab.tokenize("HAD_0 CNOT_0_1 T_1 RZ(3/4)_2 CZ_1_2")
    assert ids[0] == vocab.bos_id and ids[-1] == vocab.eos_id
    assert vocab.detokenize(ids[1:-1]) == ["HAD_0", "CNOT_0_1", "T_1", "RZ(3/4)_2", "CZ_1_2"]

    mask = build_qubit_token_mask(vocab, max_qubits=20)
    assert mask.shape == (20, 984)
    assert int(mask[4].sum().item()) == 96  # 95 valid 5-qubit gates + <eos>
    assert mask[4, vocab.token2id["HAD_4"]] == 1.0
    assert mask[4, vocab.token2id["HAD_5"]] == 0.0
    assert mask[4, vocab.token2id["CNOT_0_5"]] == 0.0


def test_paper_model_parameter_count_and_forward():
    full_model = GNNTransformer(
        d_model=1024,
        gnn_in_features=45,
        vocab_size=984,
        nhead=16,
        dim_feedforward=2048,
        num_encoder_layers=8,
        num_decoder_layers=8,
        num_gnn_layers=4,
        dropout=0.0,
    )
    total_params = sum(p.numel() for p in full_model.parameters())
    assert 170_000_000 <= total_params <= 178_000_000

    small = _make_small_model()
    g1 = Data(x=torch.randn(6, 45), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long))
    g2 = Data(x=torch.randn(4, 45), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
    batch = {
        "graph": Batch.from_data_list([g1, g2]),
        "src": torch.tensor([[2, 4, 5, 3, 0], [2, 6, 7, 8, 3]], dtype=torch.long),
        "src_len": torch.tensor([4, 5], dtype=torch.long),
        "tgt": torch.tensor([[2, 4, 3, 0], [2, 6, 7, 3]], dtype=torch.long),
    }
    logits = small(batch)
    assert logits.shape == (2, 3, 984)


def test_checkpoint_loader_formats():
    src_model = _make_small_model()
    sd = src_model.state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Plain weights-only .pt
        pt_plain = tmp / "weights.pt"
        torch.save(sd, pt_plain)
        m1 = _make_small_model()
        load_checkpoint(m1, str(pt_plain))
        for k, v in sd.items():
            assert torch.equal(m1.state_dict()[k], v)

        # 2. Prefixed Lightning-style dict (.ckpt) with extra non-model keys
        ckpt_prefixed = tmp / "lightning.ckpt"
        prefixed = {f"model.{k}": v for k, v in sd.items()}
        prefixed["extra_unused_key"] = torch.zeros(1)
        torch.save({"state_dict": prefixed, "epoch": 10, "global_step": 1000}, ckpt_prefixed)
        m2 = _make_small_model()
        load_checkpoint(m2, str(ckpt_prefixed))
        for k, v in sd.items():
            assert torch.equal(m2.state_dict()[k], v)

        # 3. Optional safetensors roundtrip (if safetensors is installed)
        try:
            from safetensors.torch import save_file

            st_path = tmp / "weights.safetensors"
            save_file(sd, str(st_path))
            m3 = _make_small_model()
            load_checkpoint(m3, str(st_path))
            for k, v in sd.items():
                assert torch.equal(m3.state_dict()[k], v)
        except ImportError:
            pass


def test_sampler_and_verifiers_end_to_end():
    vocab_path = str(REPO_ROOT / "assets" / "vocab.txt")
    vocab = Vocab(vocab_path)
    processor = CircuitProcessor(vocab, max_qubits=20)

    circ_str = "HAD_0 CNOT_0_1 T_1 RZ(3/4)_2 CZ_1_2 HAD_2 CNOT_1_2 T_0"
    circ = string_to_circuit(circ_str, 5)
    assert isinstance(circ, zx.Circuit)

    pyg = processor.pyzx_to_pyg(circ.to_graph(), circ.qubits)
    assert pyg.x.shape[1] == 45

    fids, best_gates, tgt_gates, src_gates, best_idx = verify_gen_seqs(
        circ, circ, [circ_str, "HAD_0"], atol=1e-4, probe_seed=0
    )
    assert abs(fids[0] - 1.0) <= 1e-4
    assert best_idx == 0
    assert best_gates == tgt_gates == src_gates == count_gates(circ)
    assert certify_equivalence_zx(circ, circ)

    small = _make_small_model()
    src_ids = torch.LongTensor(vocab.tokenize(circ_str)).unsqueeze(0)
    batch = {
        "graph": Batch.from_data_list([pyg]),
        "src": src_ids,
        "src_len": torch.LongTensor([src_ids.shape[1]]),
        "num_qubits": torch.LongTensor([5]),
    }

    sampler = Sampler(small, vocab_path, max_qubits=20)
    toks1, seqs1, lg1, lp1, mask1 = sampler(batch, temperature=1.0, max_length=8, num_samples=2)
    assert toks1.shape[:2] == (1, 2) and len(seqs1) == 1 and len(seqs1[0]) == 2
    assert lg1.shape[2] == toks1.shape[2] - 1 and lp1.shape[2] == toks1.shape[2] - 1
    assert mask1.shape == (1, 984)

    toks2, seqs2, lg2, lp2, mask2 = sampler(
        batch, temperature=0.2, max_length=8, num_samples=4, max_chunk_size=2
    )
    assert toks2.shape[:2] == (1, 4) and len(seqs2[0]) == 4
    assert lg2.shape[2] == toks2.shape[2] - 1 and lp2.shape[2] == toks2.shape[2] - 1
    assert torch.equal(mask1, mask2)


def test_hydra_infer_config():
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "config")):
        cfg = compose(config_name="infer")
        assert cfg.model.vocab_size == 984
        assert cfg.model.d_model == 1024
        cfg_small = compose(
            config_name="infer",
            overrides=[
                "model.d_model=32",
                "model.nhead=4",
                "model.dim_feedforward=64",
                "model.num_encoder_layers=1",
                "model.num_decoder_layers=1",
                "model.num_gnn_layers=2",
            ],
        )
        net = instantiate(cfg_small.model)
        assert isinstance(net, GNNTransformer)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS: {name}")
    print("ALL INFERENCE TESTS PASSED")
