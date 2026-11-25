"""PyZX causal-flow-preserving ZX simplification for constructing GraftQ's input diagram D_2."""

import pyzx as zx


def basic_optimise(c: zx.Circuit) -> zx.Circuit:
    """Apply PyZX basic_optimization with and without SWAP rules; keep fewer 2Q gates."""
    c1 = zx.basic_optimization(c.copy(), do_swaps=False).to_basic_gates()
    c2 = zx.basic_optimization(c.copy(), do_swaps=True).to_basic_gates()
    return c2 if c2.twoqubitcount() < c1.twoqubitcount() else c1


def flow_opt(c: zx.Circuit):
    """Construct the simplified ZX diagram D_2 and PyZX reference circuit C_ref.

    Requires PyZX with ``flow_2Q_simp`` (Calum Holker, arXiv:2312.02793; zxcalc/pyzx PR #189).

    Returns:
        circ_ref: Extracted and gate-optimized PyZX reference circuit C_ref.
        g_list:   List [D_0, D_1, D_2] of ZX diagrams (GraftQ conditions on D_2 = g_list[-1]).
    """
    if not hasattr(zx, "flow_2Q_simp"):
        raise ImportError(
            "zx.flow_2Q_simp is not available in the installed PyZX build. "
            "Install Calum Holker's causal-flow PyZX fork (zxcalc/pyzx PR #189): "
            "pip install 'git+https://github.com/CalumHolker/pyzx.git'"
        )

    g_list = []
    g = c.to_graph()
    g_list.append(g.copy())  # D_0: raw ZX diagram

    zx.teleport_reduce(g)
    g_list.append(g.copy())  # D_1: after phase teleportation

    zx.to_graph_like(g)
    zx.flow_2Q_simp(g)
    g_list.append(g.copy())  # D_2: causal-flow-preserving simplified ZX diagram

    c_extracted = zx.extract_simple(g).to_basic_gates()
    return basic_optimise(c_extracted), g_list
