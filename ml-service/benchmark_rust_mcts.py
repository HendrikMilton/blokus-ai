"""
Benchmark comparing Rust MCTS tree operations vs Python implementation.

Run after building rust_mcts:
    cd rust_mcts && maturin develop --release && cd ..
    python benchmark_rust_mcts.py
"""

import time
import sys
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Try to import Rust implementation
try:
    import rust_mcts
    RUST_AVAILABLE = True
    print("✓ Rust MCTS module loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    print(f"✗ Rust MCTS not available: {e}")
    print("  Build with: cd rust_mcts && maturin develop --release")


# ============================================================================
# Python MCTS Implementation (simplified for benchmarking)
# ============================================================================

@dataclass
class PythonMCTSNode:
    """Python MCTS node for comparison."""
    move_id: int
    prior: float
    parent: Optional['PythonMCTSNode'] = None
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, 'PythonMCTSNode'] = None
    child_priors: Dict[int, float] = None
    is_expanded: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.child_priors is None:
            self.child_priors = {}

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        if self.visit_count == 0:
            q = 0.0
        else:
            q = self.q_value
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + exploration

    @staticmethod
    def unexpanded_ucb(prior: float, parent_visits: int, c_puct: float) -> float:
        return c_puct * prior * math.sqrt(parent_visits)


class PythonMCTSTree:
    """Python MCTS tree for comparison."""

    def __init__(self):
        self.root = PythonMCTSNode(move_id=0, prior=1.0)
        self.node_count = 1

    def set_root_policy(self, moves_and_priors: List[Tuple[int, float]]):
        self.root.child_priors = {m: p for m, p in moves_and_priors}

    def select_leaf(self, c_puct: float) -> Tuple[List[PythonMCTSNode], List[float]]:
        """Select a leaf node, returning path and priors."""
        path = []
        node = self.root

        while True:
            path.append(node)

            if not node.is_expanded:
                priors = list(node.child_priors.values())
                return path, priors

            if not node.children and not node.child_priors:
                return path, []

            # Find best child
            best_score = float('-inf')
            best_child = None
            best_unexpanded = None

            parent_visits = node.visit_count

            # Check expanded children
            for move_id, child in node.children.items():
                score = child.ucb_score(parent_visits, c_puct)
                if score > best_score:
                    best_score = score
                    best_child = child
                    best_unexpanded = None

            # Check unexpanded
            expanded_moves = set(node.children.keys())
            for move_id, prior in node.child_priors.items():
                if move_id not in expanded_moves:
                    score = PythonMCTSNode.unexpanded_ucb(prior, parent_visits, c_puct)
                    if score > best_score:
                        best_score = score
                        best_child = None
                        best_unexpanded = (move_id, prior)

            if best_child is not None:
                node = best_child
            elif best_unexpanded is not None:
                priors = [p for m, p in node.child_priors.items() if m not in expanded_moves]
                return path, priors
            else:
                return path, []

    def expand_and_backprop(self, path: List[PythonMCTSNode], children_priors: List[Tuple[int, float]], value: float):
        """Expand leaf and backpropagate."""
        if not path:
            return

        leaf = path[-1]

        # Store priors
        if children_priors:
            leaf.child_priors = {m: p for m, p in children_priors}
            leaf.is_expanded = True

        # Backpropagate
        current_value = value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += current_value
            current_value = -current_value

    def add_child(self, parent: PythonMCTSNode, move_id: int, prior: float) -> PythonMCTSNode:
        """Add a child node."""
        child = PythonMCTSNode(move_id=move_id, prior=prior, parent=parent)
        parent.children[move_id] = child
        self.node_count += 1
        return child


# ============================================================================
# Benchmarks
# ============================================================================

def benchmark_python_tree_ops(num_simulations: int, num_children: int, c_puct: float) -> Tuple[float, float, float]:
    """Benchmark Python tree operations."""
    tree = PythonMCTSTree()

    # Setup priors
    priors = [(i, 1.0 / num_children) for i in range(num_children)]
    tree.set_root_policy(priors)

    # Benchmark select
    start = time.perf_counter()
    for _ in range(num_simulations):
        path, leaf_priors = tree.select_leaf(c_puct)
    select_time = time.perf_counter() - start

    # Benchmark expand
    start = time.perf_counter()
    for i in range(num_simulations):
        tree.expand_and_backprop([tree.root], priors, 0.5)
    expand_time = time.perf_counter() - start

    # Benchmark full cycle
    tree2 = PythonMCTSTree()
    tree2.set_root_policy(priors)

    start = time.perf_counter()
    for i in range(num_simulations):
        path, leaf_priors = tree2.select_leaf(c_puct)
        tree2.expand_and_backprop(path, priors, 1.0 if i % 2 == 0 else -1.0)
    full_time = time.perf_counter() - start

    return select_time, expand_time, full_time


def benchmark_rust_tree_ops(num_simulations: int, num_children: int, c_puct: float) -> Tuple[float, float, float]:
    """Benchmark Rust tree operations."""
    if not RUST_AVAILABLE:
        return 0.0, 0.0, 0.0

    # Use Rust's internal benchmark (no Python overhead)
    return rust_mcts.benchmark_tree_ops(num_simulations, num_children, c_puct)


def benchmark_rust_tree_from_python(num_simulations: int, num_children: int, c_puct: float) -> Tuple[float, float, float]:
    """Benchmark Rust tree operations called from Python (includes call overhead)."""
    if not RUST_AVAILABLE:
        return 0.0, 0.0, 0.0

    tree = rust_mcts.MCTSTree(0)

    # Setup priors
    priors = [(i, 1.0 / num_children) for i in range(num_children)]
    tree.set_root_policy(priors)

    # Benchmark select
    start = time.perf_counter()
    for _ in range(num_simulations):
        result = tree.select_leaf(c_puct)
    select_time = time.perf_counter() - start

    # Benchmark expand
    start = time.perf_counter()
    for i in range(num_simulations):
        tree.expand_and_backprop([0], priors, 0.5)
    expand_time = time.perf_counter() - start

    # Benchmark full cycle
    tree2 = rust_mcts.MCTSTree(0)
    tree2.set_root_policy(priors)

    start = time.perf_counter()
    for i in range(num_simulations):
        result = tree2.select_leaf(c_puct)
        if result:
            path, _ = result
            tree2.expand_and_backprop(path, priors, 1.0 if i % 2 == 0 else -1.0)
    full_time = time.perf_counter() - start

    return select_time, expand_time, full_time


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("\n" + "=" * 70)
    print("MCTS Tree Operations Benchmark: Python vs Rust")
    print("=" * 70)

    configs = [
        (1000, 50, 1.5, "Small (1K sims, 50 children)"),
        (10000, 50, 1.5, "Medium (10K sims, 50 children)"),
        (10000, 200, 1.5, "Large children (10K sims, 200 children)"),
        (100000, 50, 1.5, "Large (100K sims, 50 children)"),
    ]

    for num_sims, num_children, c_puct, label in configs:
        print(f"\n{label}")
        print("-" * 50)

        # Python benchmark
        py_select, py_expand, py_full = benchmark_python_tree_ops(num_sims, num_children, c_puct)
        print(f"Python:           select={py_select*1000:7.2f}ms  expand={py_expand*1000:7.2f}ms  full={py_full*1000:7.2f}ms")

        if RUST_AVAILABLE:
            # Rust internal benchmark (no Python call overhead)
            rs_select, rs_expand, rs_full = benchmark_rust_tree_ops(num_sims, num_children, c_puct)
            print(f"Rust (internal):  select={rs_select*1000:7.2f}ms  expand={rs_expand*1000:7.2f}ms  full={rs_full*1000:7.2f}ms")

            # Rust called from Python
            rs_py_select, rs_py_expand, rs_py_full = benchmark_rust_tree_from_python(num_sims, num_children, c_puct)
            print(f"Rust (from Py):   select={rs_py_select*1000:7.2f}ms  expand={rs_py_expand*1000:7.2f}ms  full={rs_py_full*1000:7.2f}ms")

            # Speedup
            if py_full > 0 and rs_full > 0:
                speedup_internal = py_full / rs_full
                speedup_from_py = py_full / rs_py_full if rs_py_full > 0 else 0
                print(f"Speedup:          internal={speedup_internal:5.1f}x     from_py={speedup_from_py:5.1f}x")

    print("\n" + "=" * 70)
    print("Simulating realistic MCTS usage (400 sims, 100 legal moves)")
    print("=" * 70)

    # Realistic MCTS simulation
    num_sims = 400
    num_moves = 100
    c_puct = 1.5

    print(f"\nPython MCTS (400 simulations)...")
    tree = PythonMCTSTree()
    priors = [(i, 1.0 / num_moves) for i in range(num_moves)]
    tree.set_root_policy(priors)

    start = time.perf_counter()
    for i in range(num_sims):
        path, _ = tree.select_leaf(c_puct)
        tree.expand_and_backprop(path, priors, random.uniform(-1, 1))
    py_time = time.perf_counter() - start
    print(f"  Time: {py_time*1000:.2f}ms ({num_sims/py_time:.0f} sims/sec)")

    if RUST_AVAILABLE:
        print(f"\nRust MCTS (400 simulations)...")
        tree = rust_mcts.MCTSTree(0)
        tree.set_root_policy(priors)

        start = time.perf_counter()
        for i in range(num_sims):
            result = tree.select_leaf(c_puct)
            if result:
                path, _ = result
                tree.expand_and_backprop(path, priors, random.uniform(-1, 1))
        rs_time = time.perf_counter() - start
        print(f"  Time: {rs_time*1000:.2f}ms ({num_sims/rs_time:.0f} sims/sec)")
        print(f"  Speedup: {py_time/rs_time:.1f}x")

    print("\n" + "=" * 70)


def test_correctness():
    """Test that Rust implementation produces correct results."""
    if not RUST_AVAILABLE:
        print("Skipping correctness test - Rust not available")
        return

    print("\n" + "=" * 70)
    print("Correctness Tests")
    print("=" * 70)

    # Test move encoding/decoding
    print("\n1. Move encoding/decoding...")
    for piece in [0, 10, 20]:
        for orient in [0, 4, 7]:
            for row in [0, 10, 19]:
                for col in [0, 10, 19]:
                    encoded = rust_mcts.GameState.encode_move(piece, orient, row, col)
                    decoded = rust_mcts.GameState.decode_move(encoded)
                    assert decoded == (piece, orient, row, col), f"Mismatch: {decoded} != {(piece, orient, row, col)}"
    print("   ✓ Move encoding/decoding correct")

    # Test tree operations
    print("\n2. Tree operations...")
    tree = rust_mcts.MCTSTree(0)
    assert tree.size() == 1, "Initial tree should have 1 node"
    assert tree.root_visits() == 0, "Initial root should have 0 visits"

    # Set policy
    priors = [(i, 0.1) for i in range(10)]
    tree.set_root_policy(priors)

    # Select and expand
    result = tree.select_leaf(1.5)
    assert result is not None, "Should return a leaf"
    path, leaf_priors = result
    assert len(path) > 0, "Path should not be empty"

    # Expand and backprop
    tree.expand_and_backprop(path, priors, 0.5)
    assert tree.root_visits() == 1, "Root should have 1 visit after backprop"

    print("   ✓ Tree operations correct")

    # Test multiple simulations
    print("\n3. Multiple simulations...")
    tree = rust_mcts.MCTSTree(0)
    tree.set_root_policy([(i, 1.0/50) for i in range(50)])

    for i in range(100):
        result = tree.select_leaf(1.5)
        if result:
            path, _ = result
            tree.expand_and_backprop(path, [(i, 1.0/50) for i in range(50)], 0.5 if i % 2 == 0 else -0.5)

    assert tree.root_visits() == 100, f"Root should have 100 visits, got {tree.root_visits()}"
    print(f"   ✓ 100 simulations completed, tree size: {tree.size()} nodes")

    print("\n" + "=" * 70)
    print("All correctness tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_correctness()
    else:
        run_benchmarks()

    if not RUST_AVAILABLE:
        print("\n" + "=" * 70)
        print("To build the Rust module:")
        print("  cd ml-service/rust_mcts")
        print("  pip install maturin")
        print("  maturin develop --release")
        print("=" * 70)
