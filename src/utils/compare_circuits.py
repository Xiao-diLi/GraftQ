"""Circuit gate counting, Qulacs Haar-state verification, and symbolic ZX certification."""

from fractions import Fraction
import re
from typing import List, Optional, Tuple

import numpy as np
import pyzx as zx
import qulacs


def count_2qubit_gates(c) -> int:
    """Count two-qubit gates (CNOT and CZ) in a PyZX Circuit."""
    two_qubit_types = (zx.gates.CNOT, zx.gates.CZ)
    return sum(1 for gate in c.gates if isinstance(gate, two_qubit_types))


def safe_tcount(c) -> int:
    """Count non-Clifford T / ZPhase(k*pi/4, k odd) gates safely for rational or float phases."""
    count = 0
    for g in c.gates:
        if isinstance(g, zx.gates.T):
            count += 1
            continue
        if hasattr(g, "phase"):
            p = g.phase
            if hasattr(p, "denominator"):
                if p.denominator > 2:
                    count += 1
            elif isinstance(p, float):
                if not np.isclose(p % 0.5, 0.0):
                    count += 1
    return count


def count_gates(circ) -> Tuple[int, int, int]:
    """Return (N_all, N_2q, N_T) gate counts for a PyZX Circuit."""
    return len(circ.gates), count_2qubit_gates(circ), safe_tcount(circ)


def pyzx_circuit_to_qulacs(zx_circuit, n_qubits: Optional[int] = None):
    """Convert a PyZX Circuit to a Qulacs QuantumCircuit (up to global phase)."""
    if n_qubits is None:
        n_qubits = zx_circuit.qubits

    try:
        qc = qulacs.QuantumCircuit(n_qubits)
        for gate in zx_circuit.gates:
            if isinstance(gate, zx.gates.CNOT):
                qc.add_CNOT_gate(gate.control, gate.target)
            elif isinstance(gate, zx.gates.CZ):
                qc.add_CZ_gate(gate.control, gate.target)
            elif isinstance(gate, zx.gates.HAD):
                qc.add_H_gate(gate.target)
            elif isinstance(gate, zx.gates.NOT):
                qc.add_X_gate(gate.target)
            elif isinstance(gate, zx.gates.Z):
                qc.add_Z_gate(gate.target)
            elif isinstance(gate, zx.gates.SWAP):
                qc.add_SWAP_gate(gate.control, gate.target)
            elif isinstance(gate, (zx.gates.ZPhase, zx.gates.T, zx.gates.S)):
                qc.add_RZ_gate(gate.target, float(gate.phase) * np.pi)
            elif isinstance(gate, zx.gates.XPhase):
                qc.add_RX_gate(gate.target, float(gate.phase) * np.pi)
            else:
                raise ValueError(f"Unsupported gate type: {gate}")
        return qc
    except Exception:
        return None


def calculate_fidelity(state1, state2) -> float:
    """Compute state fidelity |<state1|state2>|^2 (invariant to global phase)."""
    inner_product = qulacs.state.inner_product(state1, state2)
    return float(abs(inner_product) ** 2)


def circuit_is_equivalent(
    circ1,
    circ2,
    n_qubits: int,
    atol: float = 1e-6,
    seed: Optional[int] = None,
) -> bool:
    """Check equivalence up to global phase via a single Haar-random probe state."""
    state = qulacs.QuantumState(n_qubits)
    if seed is None:
        state.set_Haar_random_state()
    else:
        state.set_Haar_random_state(int(seed))

    state_c1 = state.copy()
    state_c2 = state.copy()

    q1 = pyzx_circuit_to_qulacs(circ1, n_qubits)
    q2 = pyzx_circuit_to_qulacs(circ2, n_qubits)
    if q1 is None or q2 is None:
        return False

    q1.update_quantum_state(state_c1)
    q2.update_quantum_state(state_c2)
    fidelity = calculate_fidelity(state_c1, state_c2)
    return abs(fidelity - 1.0) <= atol


def certify_equivalence_zx(ref_circ, gen_circ) -> bool:
    """Deterministically certify equivalence up to global phase using PyZX symbolic simplification."""
    try:
        return bool(ref_circ.verify_equality(gen_circ, up_to_swaps=False, up_to_global_phase=True))
    except TypeError:
        return bool(ref_circ.verify_equality(gen_circ))
    except Exception:
        return False


def string_to_circuit(line: str, nqubits: int):
    """Parse a space-separated gate-token string into a PyZX Circuit."""
    c = zx.Circuit(nqubits)
    for gate in line.strip().split():
        parts = gate.split("_")
        if len(parts) > 3 or len(parts) < 2:
            raise ValueError(f"Invalid gate token: {gate!r}")

        name = parts[0]
        target = int(parts[-1])
        control = int(parts[1]) if len(parts) == 3 else None

        if name.startswith("R"):
            name_short = name[:2]
            valid_name = "ZPhase" if name_short == "RZ" else "XPhase"
            match = re.search(r"\((\d+/\d+)\)", name)
            if not match:
                raise ValueError(f"Invalid rotation token: {gate!r}")
            num, den = match.group(1).split("/")
            c.add_gate(valid_name, target, phase=Fraction(int(num), int(den)))
        elif name in ("CNOT", "CZ", "SWAP"):
            if control is None:
                raise ValueError(f"Two-qubit gate missing control index: {gate!r}")
            c.add_gate(name, control, target)
        else:
            c.add_gate(name, target)

    return c


def get_num_qubits(circ_str: str) -> int:
    """Infer register width from maximum qubit index in a gate-token string."""
    max_qubit = 0
    for gate in circ_str.strip().split():
        max_qid = max(int(s) for s in gate.split("_")[1:])
        max_qubit = max(max_qubit, max_qid)
    return max_qubit + 1


def verify_gen_seqs(
    src_circ,
    tgt_circ,
    gen_seqs: List[str],
    atol: float = 1e-4,
    probe_seed: Optional[int] = None,
):
    """Verify candidate sequences on a shared Haar-random state and pick the lexicographic best.

    Selection order among valid candidates (|F - 1| <= atol): minimum (N_all, N_2q, N_T).

    Returns:
        fidelities: np.ndarray of shape [K]
        best_gates: Tuple[int, int, int] or None
        tgt_gates: Tuple[int, int, int]
        src_gates: Tuple[int, int, int]
        best_idx: int (-1 if no valid candidate)
    """
    nqubits = src_circ.qubits
    state = qulacs.QuantumState(nqubits)
    if probe_seed is None:
        state.set_Haar_random_state()
    else:
        state.set_Haar_random_state(int(probe_seed))

    src_qulacs = pyzx_circuit_to_qulacs(src_circ, nqubits)
    tgt_qulacs = pyzx_circuit_to_qulacs(tgt_circ, nqubits)

    tgt_state = state.copy()
    tgt_qulacs.update_quantum_state(tgt_state)

    src_state = state.copy()
    src_qulacs.update_quantum_state(src_state)
    fid_src_tgt = calculate_fidelity(src_state, tgt_state)
    assert abs(fid_src_tgt - 1.0) <= atol, f"Reference circuit mismatch: F={fid_src_tgt}"

    tgt_gates = count_gates(tgt_circ)
    src_gates = count_gates(src_circ)

    fidelities = []
    valid_candidates = []
    for idx, seq in enumerate(gen_seqs):
        try:
            circ = string_to_circuit(seq, nqubits)
            circ_qulacs = pyzx_circuit_to_qulacs(circ, nqubits)
            if circ_qulacs is None:
                fidelities.append(0.0)
                continue
            gen_state = state.copy()
            circ_qulacs.update_quantum_state(gen_state)
            fid = calculate_fidelity(tgt_state, gen_state)
            fidelities.append(fid)
            if abs(fid - 1.0) <= atol:
                valid_candidates.append((count_gates(circ), idx))
        except Exception:
            fidelities.append(0.0)

    if valid_candidates:
        best_gates, best_idx = min(valid_candidates, key=lambda item: item[0])
    else:
        best_gates, best_idx = None, -1

    return np.asarray(fidelities, dtype=np.float64), best_gates, tgt_gates, src_gates, best_idx
