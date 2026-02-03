"""Benchmark batch preparation to identify bottlenecks."""

import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Tuple
import threading
from queue import Queue

# Simulate training example structure
@dataclass
class FakeTrainingExample:
    board_tensor: np.ndarray      # (8, 20, 20)
    pieces_vector: np.ndarray     # (84,)
    policy: np.ndarray            # variable length
    legal_moves: List[Tuple[int, int, int, int]]  # list of (piece, orient, row, col)
    value: float

def generate_fake_examples(n: int) -> List[FakeTrainingExample]:
    """Generate fake training examples for benchmarking."""
    examples = []
    for _ in range(n):
        # Random number of legal moves (typically 20-100)
        num_moves = np.random.randint(20, 100)
        examples.append(FakeTrainingExample(
            board_tensor=np.random.randn(8, 20, 20).astype(np.float32),
            pieces_vector=np.random.randn(84).astype(np.float32),
            policy=np.random.rand(num_moves).astype(np.float32),
            legal_moves=[(np.random.randint(21), np.random.randint(8),
                         np.random.randint(20), np.random.randint(20))
                        for _ in range(num_moves)],
            value=np.random.uniform(-1, 1),
        ))
    return examples

def benchmark_current_method(batch: List[FakeTrainingExample], device: str = 'cuda'):
    """Current batch preparation method from trainer.py"""
    start = time.perf_counter()

    # Step 1: Stack numpy arrays
    t1 = time.perf_counter()
    board_np = np.stack([ex.board_tensor for ex in batch])
    pieces_np = np.stack([ex.pieces_vector for ex in batch])
    values_np = np.array([ex.value for ex in batch], dtype=np.float32)
    numpy_stack_time = time.perf_counter() - t1

    # Step 2: Convert to tensors and move to GPU
    t2 = time.perf_counter()
    board_tensors = torch.from_numpy(board_np).to(device)
    pieces_vectors = torch.from_numpy(pieces_np).to(device)
    target_values = torch.from_numpy(values_np).unsqueeze(1).to(device)
    to_gpu_time = time.perf_counter() - t2

    # Step 3: Build target policies (padded)
    t3 = time.perf_counter()
    max_policy_len = max(len(ex.policy) for ex in batch)
    target_np = np.zeros((len(batch), max_policy_len), dtype=np.float32)
    mask_np = np.zeros((len(batch), max_policy_len), dtype=np.float32)
    for i, ex in enumerate(batch):
        plen = len(ex.policy)
        if plen > 0:
            target_np[i, :plen] = ex.policy
            mask_np[i, :plen] = 1.0
    target_policies = torch.from_numpy(target_np).to(device)
    policy_mask = torch.from_numpy(mask_np).to(device)
    policy_prep_time = time.perf_counter() - t3

    # Step 4: Extract legal moves list (for network forward)
    t4 = time.perf_counter()
    legal_moves_list = [ex.legal_moves for ex in batch]
    legal_moves_time = time.perf_counter() - t4

    total_time = time.perf_counter() - start

    return {
        'total': total_time,
        'numpy_stack': numpy_stack_time,
        'to_gpu': to_gpu_time,
        'policy_prep': policy_prep_time,
        'legal_moves': legal_moves_time,
    }

def benchmark_optimized_numpy(batch: List[FakeTrainingExample], device: str = 'cuda'):
    """Optimized with pre-allocated arrays."""
    start = time.perf_counter()

    batch_size = len(batch)

    # Step 1: Pre-allocate and fill (avoiding list comprehension overhead)
    t1 = time.perf_counter()
    board_np = np.empty((batch_size, 8, 20, 20), dtype=np.float32)
    pieces_np = np.empty((batch_size, 84), dtype=np.float32)
    values_np = np.empty(batch_size, dtype=np.float32)
    for i, ex in enumerate(batch):
        board_np[i] = ex.board_tensor
        pieces_np[i] = ex.pieces_vector
        values_np[i] = ex.value
    numpy_stack_time = time.perf_counter() - t1

    # Step 2: Convert to tensors and move to GPU
    t2 = time.perf_counter()
    board_tensors = torch.from_numpy(board_np).to(device)
    pieces_vectors = torch.from_numpy(pieces_np).to(device)
    target_values = torch.from_numpy(values_np).unsqueeze(1).to(device)
    to_gpu_time = time.perf_counter() - t2

    # Step 3: Build target policies (same as current)
    t3 = time.perf_counter()
    max_policy_len = max(len(ex.policy) for ex in batch)
    target_np = np.zeros((len(batch), max_policy_len), dtype=np.float32)
    mask_np = np.zeros((len(batch), max_policy_len), dtype=np.float32)
    for i, ex in enumerate(batch):
        plen = len(ex.policy)
        if plen > 0:
            target_np[i, :plen] = ex.policy
            mask_np[i, :plen] = 1.0
    target_policies = torch.from_numpy(target_np).to(device)
    policy_mask = torch.from_numpy(mask_np).to(device)
    policy_prep_time = time.perf_counter() - t3

    # Step 4: Extract legal moves list
    t4 = time.perf_counter()
    legal_moves_list = [ex.legal_moves for ex in batch]
    legal_moves_time = time.perf_counter() - t4

    total_time = time.perf_counter() - start

    return {
        'total': total_time,
        'numpy_stack': numpy_stack_time,
        'to_gpu': to_gpu_time,
        'policy_prep': policy_prep_time,
        'legal_moves': legal_moves_time,
    }

def benchmark_pinned_memory(batch: List[FakeTrainingExample], device: str = 'cuda'):
    """Use pinned memory for faster CPU->GPU transfer."""
    start = time.perf_counter()

    batch_size = len(batch)

    # Step 1: Stack numpy
    t1 = time.perf_counter()
    board_np = np.stack([ex.board_tensor for ex in batch])
    pieces_np = np.stack([ex.pieces_vector for ex in batch])
    values_np = np.array([ex.value for ex in batch], dtype=np.float32)
    numpy_stack_time = time.perf_counter() - t1

    # Step 2: Convert to pinned tensors and async copy to GPU
    t2 = time.perf_counter()
    board_tensors = torch.from_numpy(board_np).pin_memory().to(device, non_blocking=True)
    pieces_vectors = torch.from_numpy(pieces_np).pin_memory().to(device, non_blocking=True)
    target_values = torch.from_numpy(values_np).unsqueeze(1).pin_memory().to(device, non_blocking=True)
    to_gpu_time = time.perf_counter() - t2

    # Step 3: Build target policies
    t3 = time.perf_counter()
    max_policy_len = max(len(ex.policy) for ex in batch)
    target_np = np.zeros((len(batch), max_policy_len), dtype=np.float32)
    mask_np = np.zeros((len(batch), max_policy_len), dtype=np.float32)
    for i, ex in enumerate(batch):
        plen = len(ex.policy)
        if plen > 0:
            target_np[i, :plen] = ex.policy
            mask_np[i, :plen] = 1.0
    target_policies = torch.from_numpy(target_np).pin_memory().to(device, non_blocking=True)
    policy_mask = torch.from_numpy(mask_np).pin_memory().to(device, non_blocking=True)
    policy_prep_time = time.perf_counter() - t3

    # Sync to ensure transfer is complete
    torch.cuda.synchronize()

    # Step 4: Extract legal moves list
    t4 = time.perf_counter()
    legal_moves_list = [ex.legal_moves for ex in batch]
    legal_moves_time = time.perf_counter() - t4

    total_time = time.perf_counter() - start

    return {
        'total': total_time,
        'numpy_stack': numpy_stack_time,
        'to_gpu': to_gpu_time,
        'policy_prep': policy_prep_time,
        'legal_moves': legal_moves_time,
    }

class ReplayBufferDataset(Dataset):
    """Dataset wrapper for replay buffer with prefetching."""

    def __init__(self, examples: List[FakeTrainingExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'board': torch.from_numpy(ex.board_tensor),
            'pieces': torch.from_numpy(ex.pieces_vector),
            'value': ex.value,
            'policy': ex.policy,  # Keep as numpy for variable length
            'legal_moves': ex.legal_moves,
        }


def collate_fn(batch):
    """Custom collate function to handle variable-length policies."""
    boards = torch.stack([b['board'] for b in batch])
    pieces = torch.stack([b['pieces'] for b in batch])
    values = torch.tensor([b['value'] for b in batch], dtype=torch.float32).unsqueeze(1)

    # Handle variable-length policies
    max_policy_len = max(len(b['policy']) for b in batch)
    target_policies = torch.zeros(len(batch), max_policy_len)
    policy_mask = torch.zeros(len(batch), max_policy_len)
    for i, b in enumerate(batch):
        plen = len(b['policy'])
        if plen > 0:
            target_policies[i, :plen] = torch.from_numpy(b['policy'])
            policy_mask[i, :plen] = 1.0

    legal_moves_list = [b['legal_moves'] for b in batch]

    return boards, pieces, values, target_policies, policy_mask, legal_moves_list


class PrefetchingBatchLoader:
    """Simple prefetching loader that prepares next batch while GPU works."""

    def __init__(self, examples: List[FakeTrainingExample], batch_size: int, device: str, num_prefetch: int = 2):
        self.examples = examples
        self.batch_size = batch_size
        self.device = device
        self.queue = Queue(maxsize=num_prefetch)
        self.stop_event = threading.Event()
        self.thread = None

    def _prepare_batch(self):
        """Prepare a single batch (runs in background thread)."""
        indices = np.random.choice(len(self.examples), self.batch_size, replace=False)
        batch = [self.examples[i] for i in indices]

        # Stack numpy
        board_np = np.stack([ex.board_tensor for ex in batch])
        pieces_np = np.stack([ex.pieces_vector for ex in batch])
        values_np = np.array([ex.value for ex in batch], dtype=np.float32)

        # Convert to pinned tensors
        boards = torch.from_numpy(board_np).pin_memory()
        pieces = torch.from_numpy(pieces_np).pin_memory()
        values = torch.from_numpy(values_np).unsqueeze(1).pin_memory()

        # Policies
        max_policy_len = max(len(ex.policy) for ex in batch)
        target_np = np.zeros((len(batch), max_policy_len), dtype=np.float32)
        mask_np = np.zeros((len(batch), max_policy_len), dtype=np.float32)
        for i, ex in enumerate(batch):
            plen = len(ex.policy)
            if plen > 0:
                target_np[i, :plen] = ex.policy
                mask_np[i, :plen] = 1.0
        target_policies = torch.from_numpy(target_np).pin_memory()
        policy_mask = torch.from_numpy(mask_np).pin_memory()

        legal_moves_list = [ex.legal_moves for ex in batch]

        return boards, pieces, values, target_policies, policy_mask, legal_moves_list

    def _prefetch_worker(self, num_batches: int):
        """Worker thread that prefetches batches."""
        for _ in range(num_batches):
            if self.stop_event.is_set():
                break
            batch = self._prepare_batch()
            self.queue.put(batch)

    def start(self, num_batches: int):
        """Start prefetching."""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._prefetch_worker, args=(num_batches,))
        self.thread.start()

    def get_batch(self):
        """Get next batch, moving to GPU."""
        boards, pieces, values, target_policies, policy_mask, legal_moves = self.queue.get()
        return (
            boards.to(self.device, non_blocking=True),
            pieces.to(self.device, non_blocking=True),
            values.to(self.device, non_blocking=True),
            target_policies.to(self.device, non_blocking=True),
            policy_mask.to(self.device, non_blocking=True),
            legal_moves,
        )

    def stop(self):
        """Stop prefetching."""
        self.stop_event.set()
        if self.thread:
            self.thread.join()


def simulate_training_step(boards, pieces, values, device, duration_ms=800):
    """Simulate GPU work (forward + backward pass)."""
    # Simulate specified duration of GPU work
    start = time.perf_counter()
    x = torch.randn(4096, 2048, device=device)
    while (time.perf_counter() - start) * 1000 < duration_ms:
        x = torch.matmul(x, torch.randn(2048, 2048, device=device))
    torch.cuda.synchronize()


def benchmark_with_prefetch(examples: List[FakeTrainingExample], batch_size: int, device: str, num_steps: int = 10):
    """Benchmark with prefetching - simulates actual training loop."""

    # Without prefetch
    print(f"\n--- Without Prefetch (batch_size={batch_size}) ---")
    total_times_no_prefetch = []

    for step in range(num_steps):
        step_start = time.perf_counter()

        # Prepare batch (blocking)
        prep_start = time.perf_counter()
        indices = np.random.choice(len(examples), batch_size, replace=False)
        batch = [examples[i] for i in indices]

        board_np = np.stack([ex.board_tensor for ex in batch])
        pieces_np = np.stack([ex.pieces_vector for ex in batch])
        values_np = np.array([ex.value for ex in batch], dtype=np.float32)

        boards = torch.from_numpy(board_np).to(device)
        pieces = torch.from_numpy(pieces_np).to(device)
        values = torch.from_numpy(values_np).unsqueeze(1).to(device)
        prep_time = time.perf_counter() - prep_start

        # Simulate GPU training (~750ms like real training)
        gpu_start = time.perf_counter()
        simulate_training_step(boards, pieces, values, device, duration_ms=750)
        gpu_time = time.perf_counter() - gpu_start

        total_time = time.perf_counter() - step_start
        total_times_no_prefetch.append(total_time)

        if step == 0:
            print(f"  Step {step}: prep={prep_time*1000:.1f}ms, gpu={gpu_time*1000:.1f}ms, total={total_time*1000:.1f}ms")

    avg_no_prefetch = np.mean(total_times_no_prefetch) * 1000
    print(f"  Average: {avg_no_prefetch:.1f}ms/step ({1000/avg_no_prefetch:.2f} steps/s)")

    # With prefetch
    print(f"\n--- With Prefetch (batch_size={batch_size}) ---")
    loader = PrefetchingBatchLoader(examples, batch_size, device, num_prefetch=2)
    loader.start(num_steps + 2)  # Prefetch a couple extra

    # Wait for first batch to be ready
    time.sleep(0.1)

    total_times_with_prefetch = []

    for step in range(num_steps):
        step_start = time.perf_counter()

        # Get batch (should be instant if prefetched)
        get_start = time.perf_counter()
        boards, pieces, values, target_policies, policy_mask, legal_moves = loader.get_batch()
        torch.cuda.synchronize()  # Ensure transfer complete
        get_time = time.perf_counter() - get_start

        # Simulate GPU training (~750ms like real training)
        gpu_start = time.perf_counter()
        simulate_training_step(boards, pieces, values, device, duration_ms=750)
        gpu_time = time.perf_counter() - gpu_start

        total_time = time.perf_counter() - step_start
        total_times_with_prefetch.append(total_time)

        if step == 0:
            print(f"  Step {step}: get={get_time*1000:.1f}ms, gpu={gpu_time*1000:.1f}ms, total={total_time*1000:.1f}ms")

    loader.stop()

    avg_with_prefetch = np.mean(total_times_with_prefetch) * 1000
    print(f"  Average: {avg_with_prefetch:.1f}ms/step ({1000/avg_with_prefetch:.2f} steps/s)")

    print(f"\n  Speedup with prefetch: {avg_no_prefetch/avg_with_prefetch:.2f}x")

    return avg_no_prefetch, avg_with_prefetch


def main():
    print("Generating fake training data...")
    # Generate 10000 examples (like replay buffer)
    all_examples = generate_fake_examples(10000)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print()

    # Test prefetching with realistic batch sizes
    print("=" * 60)
    print("PREFETCH BENCHMARK (simulates actual training loop)")
    print("=" * 60)

    for batch_size in [4096, 6144]:
        benchmark_with_prefetch(all_examples, batch_size, device, num_steps=10)
        print()

if __name__ == "__main__":
    main()
