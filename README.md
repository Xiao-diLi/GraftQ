# GraftQ: Multimodal Generative Quantum Circuit Optimization (Inference & Checkpoints)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](./LICENSE)
[![Commercial License](https://img.shields.io/badge/License-Commercial%20Dual--License-orange.svg)](./LICENSE-COMMERCIAL.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)

Official **inference code and pretrained model checkpoints** for **GraftQ** (*Multimodal Generative Quantum Circuit Optimization with Verifier-Guided Reinforcement Learning*), a 174M-parameter multimodal GNN-Transformer that jointly encodes a source circuit's gate-token sequence and simplified ZX diagram ($D_2$) to autoregressively synthesize optimized Clifford+$T$ circuits on 5–20 qubits.

---

## Highlights

- **Multimodal GNN-Transformer (174M)**: 4-layer GCN over the simplified ZX diagram ($D_2$) fused with an 8-layer Pre-LN Transformer encoder (with RoPE) and an 8-layer Pre-LN autoregressive decoder.
- **Exact Qubit-Range Validity Masking**: Gate-level vocabulary ($|\mathcal{V}|=984$ for $n \le 20$ qubits) with precomputed validity masks $\mathcal{V}_n$ guaranteeing syntactically valid circuits on the target qubit register.
- **Automatic Equivalence Verification & Symbolic Certification**: Fast Haar-random state-vector equivalence verification via Qulacs (`|F - 1| <= 1e-4`), lexicographic selection minimizing $(N_{\mathrm{all}}, N_{\mathrm{2q}}, N_T)$, and optional deterministic symbolic certification via PyZX (`Circuit.verify_equality`).
- **Flexible Checkpoint Loading**: Supports clean weights-only `.safetensors` / `.pt` files as well as raw `.ckpt` checkpoints with zero training-framework dependencies.

### Benchmark Summary (Held-Out 9,000 Circuits per Bucket, 5–20 Qubits, $K=8, \mathcal{T}=0.2$)

| Source Bucket | Model | pass@1 | pass@8 | Better than PyZX (2Q) |
| :--- | :--- | :---: | :---: | :---: |
| 20–100 | GraftQ (SFT) | 60.8% | 67.7% | 0.1% |
| 20–100 | GraftQ (Stage 1 Hybrid) | 80.4% | 83.5% | 21.8% |
| 20–100 | GraftQ (Stage 2b Hybrid) | **92.0%** | **94.1%** | **23.6%** |
| 101–200 | GraftQ (Stage 2b Hybrid) | **71.4%** | **78.8%** | **28.6%** |
| 201–300 | GraftQ (Stage 2b Hybrid) | **38.8%** | **50.7%** | **20.2%** |
| 301–400 (0-shot) | GraftQ (Stage 2b Hybrid) | **4.1%** | **7.7%** | **16.5%** |

All 159,868 selected benchmark solutions across 34 evaluated model–bucket sets pass deterministic PyZX symbolic certification (`Circuit.verify_equality`, 0% rejection rate).

---

## Pretrained Checkpoints

Download the released weights from GitHub Releases or Hugging Face Hub and place them in a local `checkpoints/` directory:

| Model | Scope | Training Stage | Download |
| :--- | :--- | :--- | :--- |
| `graftq_sft` | 20–100 gates (5–20 qubits) | Stage 0 SFT (step 248k) | *Release link* |
| `graftq_s1_hybrid` | 20–100 gates (5–20 qubits) | Stage 1 Hybrid RL (step 49k) | *Release link* |
| `graftq_s2a_hybrid` | Up to 200 gates (5–20 qubits) | Stage 2a Curriculum RL (step 117k) | *Release link* |
| `graftq_s2b_hybrid` | Up to 300 gates + 301–400 zero-shot | Stage 2b Curriculum RL (step 255k) | *Release link* |

> Checkpoints are governed by the same dual-license terms as this repository; see [Licensing & Patents](#licensing--patents).

---

## Repository Structure

```text
GraftQ_opensource/
├── assets/
│   └── vocab.txt                      # 984-token gate vocabulary (up to 20 qubits)
├── config/
│   ├── model/gnn_transformer.yaml     # 174M GNN-Transformer architecture config
│   └── infer.yaml                     # Batch inference & verification config
├── src/
│   ├── layers/                        # GCN encoder, Pre-LN Transformer, RoPE, KV cache
│   ├── model/                         # GNNTransformer and unified chunked Sampler
│   ├── data/                          # Vocab, CircuitProcessor, eval WebDataset loader
│   └── utils/                         # Checkpoint loader, ZX flow_opt, Qulacs & PyZX verifiers
├── optimize_circuit.py                # Single-circuit optimization CLI (OpenQASM / gate tokens)
├── infer.py                           # Batch WebDataset inference, verification & ZX certification
├── scripts/
│   └── run_infer.sh                   # Portable shell launcher for batch evaluation
├── tests/
│   └── test_inference.py              # Inference, masking, checkpoint & verifier unit tests
├── LICENSE                            # GNU AGPLv3 full text
└── LICENSE-COMMERCIAL.md              # Commercial dual-license & patent notice
```

---

## Installation

Recommended environment: Python 3.11 with CUDA-enabled PyTorch (tested with Python 3.11, PyTorch 2.8.0, PyTorch Geometric 2.6+, Qulacs 0.6.12+).

```bash
conda create -n graftq python=3.11 -y
conda activate graftq

# Install PyTorch matching your CUDA version (example for CUDA 12.8; or CPU-only):
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Install lightweight inference dependencies
pip install -r requirements.txt
```

### Installing PyZX with `flow_2Q_simp` (Required for `optimize_circuit.py`)

Constructing the input simplified ZX diagram $D_2$ on-the-fly (`src/utils/zx_simplify.py`) uses Calum Holker's causal-flow-preserving simplification (`flow_2Q_simp`, arXiv:2312.02793; [zxcalc/pyzx PR #189](https://github.com/zxcalc/pyzx/pull/189)):

```bash
pip install "git+https://github.com/CalumHolker/pyzx.git"
```

Verify installation:

```bash
python -c "import pyzx as zx; assert hasattr(zx, 'flow_2Q_simp'); print('PyZX flow_2Q_simp OK')"
```

---

## Quickstart: Optimize a Single Circuit

Optimize an OpenQASM 2.0 file (5–20 qubits, Clifford+$T$ / $R_Z(k\pi/4)$ gate set):

```bash
python optimize_circuit.py \
  --checkpoint checkpoints/graftq_s2b_hybrid.safetensors \
  --qasm example.qasm \
  --num-samples 8 \
  --temperature 0.2 \
  --certify-zx \
  --output-qasm optimized.qasm
```

Or pass a space-separated gate-token string directly:

```bash
python optimize_circuit.py \
  --checkpoint checkpoints/graftq_s2b_hybrid.safetensors \
  --gates "HAD_0 CNOT_0_1 T_1 RZ(3/4)_2 CZ_1_2 HAD_2 CNOT_1_2 T_0" \
  --qubits 5 \
  --num-samples 8 \
  --temperature 0.2 \
  --certify-zx
```

`optimize_circuit.py` automatically:
1. Simplifies the source circuit with PyZX (`flow_opt`) to build the input ZX diagram $D_2$ and reference circuit $C_{\mathrm{ref}}$.
2. Samples $K$ candidate circuits from GraftQ using exact qubit-range masking $\mathcal{V}_n$.
3. Verifies equivalence against $C_{\mathrm{ref}}$ with Qulacs Haar-state evolution (`|F - 1| <= 1e-4`), selects the valid candidate minimizing $(N_{\mathrm{all}}, N_{\mathrm{2q}}, N_T)$, and optionally certifies it symbolically with PyZX `verify_equality`.

---

## Batch Inference & Verification

To evaluate a directory of WebDataset `.tar` shards (where each `.pyd` record is a dict `{"circ": zx.Circuit, "circ_opt": zx.Circuit, "graphs": [D0, D1, D2], "nqubits": int}`):

```bash
bash scripts/run_infer.sh checkpoints/graftq_s2b_hybrid.safetensors /path/to/test_shards outputs/eval 8
```

Or invoke `infer.py` directly with Hydra overrides (e.g., enabling symbolic ZX certification on all selected candidates):

```bash
python infer.py \
  checkpoint='"checkpoints/graftq_s2b_hybrid.safetensors"' \
  data_path=/path/to/test_shards \
  output_dir=outputs/eval \
  num_sample_per_seq=8 \
  temperature=0.2 \
  validation_size=9000 \
  certify_zx=true
```

Outputs saved to `output_dir`:
- `generate_samples.tar`: generated candidate sequences alongside source/reference circuits and `src_hash`.
- `fidelity_q{n}.txt`: per-circuit candidate fidelities and selected gate counts by qubit register width.
- `summary.json`: aggregate `pass@1`, `pass@K`, mean gate reductions `(all, 2Q, T)`, better-than-PyZX rates, and symbolic certification count.

---

## Testing

Run the self-contained inference unit test suite (vocabulary, qubit mask, 174M parameter count, forward pass, single/chunked sampler parity, safe checkpoint loading across `.pt` / `.safetensors` / `.ckpt`, Qulacs & PyZX verifiers, and Hydra `infer.yaml` composition):

```bash
PYTHONPATH=. python tests/test_inference.py
```

---

## Licensing & Patents

Both the **inference source code** in this repository and the **released pretrained model checkpoints** are dual-licensed:
- **Open-Source**: [GNU Affero General Public License v3.0 (AGPL-3.0)](./LICENSE). Any modified deployment—including network, cloud, or SaaS compilation services—must disclose corresponding source code under AGPLv3.
- **Commercial**: Proprietary, closed-source, or non-AGPL commercial/SaaS deployment requires a separate commercial license. Pending patent applications cover methods implemented in this model and codebase; see [LICENSE-COMMERCIAL.md](./LICENSE-COMMERCIAL.md) for full terms and contact information.

---

## Citation

If you find GraftQ useful in your research, please cite:

```bibtex
@article{li2026graftq,
  title   = {Multimodal Generative Quantum Circuit Optimization with Verifier-Guided Reinforcement Learning},
  author  = {Li, Xiaodi and Chen, Linghui and Zhang, Tao},
  year    = {2026}
}
```
