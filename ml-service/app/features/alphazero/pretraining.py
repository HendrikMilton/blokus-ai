"""
Heuristic pretraining for AlphaZero cold start problem.

Pretrains the policy head on heuristic-scored moves before self-play begins.
This helps MCTS generate focused targets during early training instead of
uniform visit counts from a random network.

Phase 0 (before self-play):
1. Generate diverse random game positions
2. Score all legal moves with heuristics for each position
3. Train policy head (freeze backbone) for N steps
4. Unfreeze all, continue to normal self-play
"""

import random
from typing import List, Dict, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from app.logging_config import log
from .game import BlokusGame
from .network import BlokusNet
from .heuristics import HeuristicConfig, compute_heuristic_policy


def generate_pretraining_positions(
    num_positions: int = 10000,
    heuristic_config: Optional[HeuristicConfig] = None,
) -> List[Dict]:
    """
    Generate diverse positions by playing random games and sampling states.

    Args:
        num_positions: Target number of positions to generate
        heuristic_config: Configuration for heuristic scoring

    Returns:
        List of position dictionaries with:
        - board_tensor: numpy array for neural network input
        - pieces_vector: numpy array for neural network input
        - legal_moves: List[Move] for this position
        - target_policy: numpy array of heuristic policy (normalized)
    """
    if heuristic_config is None:
        heuristic_config = HeuristicConfig()

    positions = []

    # Estimate games needed (~50 positions per game on average)
    games_needed = (num_positions // 50) + 10  # Add buffer

    log.info(f"Generating {num_positions} pretraining positions from ~{games_needed} random games")

    for game_idx in range(games_needed):
        if len(positions) >= num_positions:
            break

        game = BlokusGame()
        depth = 0

        while not game.is_game_over and depth < 80:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                game.pass_turn()
                depth += 1
                continue

            # Sample position every 4-5 moves (always in early game for more diversity)
            should_sample = (depth % 5 == 0) or (depth < 15 and depth % 3 == 0)

            if should_sample and len(positions) < num_positions:
                # Get state tensors
                board_tensor, pieces_vector = game.get_state_for_neural_net()

                # Compute heuristic policy for this position
                heuristic_policy = compute_heuristic_policy(
                    game, legal_moves, heuristic_config
                )

                positions.append({
                    'board_tensor': board_tensor,
                    'pieces_vector': pieces_vector,
                    'legal_moves': legal_moves,
                    'target_policy': heuristic_policy,
                })

            # Random move to continue the game
            move = random.choice(legal_moves)
            game.make_move(move)
            depth += 1

        if (game_idx + 1) % 100 == 0:
            log.info(f"Generated {len(positions)}/{num_positions} positions from {game_idx + 1} games")

    log.info(f"Generated {len(positions)} pretraining positions")
    return positions[:num_positions]


def compute_policy_cross_entropy(
    policy_logits: torch.Tensor,
    batch: List[Dict],
    device: str,
) -> torch.Tensor:
    """
    Compute cross-entropy loss between predicted logits and heuristic targets.

    Args:
        policy_logits: Tensor of shape (batch_size, max_moves) from network
        batch: List of position dictionaries with target_policy
        device: Device for tensors

    Returns:
        Scalar loss tensor
    """
    max_len = max(len(b['target_policy']) for b in batch)

    # Prepare target and mask arrays
    target_np = np.zeros((len(batch), max_len), dtype=np.float32)
    mask_np = np.zeros((len(batch), max_len), dtype=np.float32)

    for i, b in enumerate(batch):
        plen = len(b['target_policy'])
        target_np[i, :plen] = b['target_policy']
        mask_np[i, :plen] = 1.0

    targets = torch.from_numpy(target_np).to(device)
    mask = torch.from_numpy(mask_np).to(device)

    # Mask invalid positions with -inf before softmax
    masked_logits = policy_logits[:, :max_len].clone()
    masked_logits[mask == 0] = float('-inf')

    # Compute log probabilities
    log_probs = F.log_softmax(masked_logits, dim=1)
    # Replace NaN with 0 (can happen if all logits are -inf)
    log_probs = torch.nan_to_num(log_probs, nan=0.0)

    # Cross-entropy: -sum(target * log_probs) per sample, then mean
    loss = -(targets * log_probs).sum(dim=1).mean()

    return loss


def pretrain_policy_head(
    network: BlokusNet,
    device: str,
    num_positions: int = 10000,
    num_steps: int = 1000,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> Dict[str, float]:
    """
    Pretrain policy head on heuristic-scored positions.

    Freezes backbone (stem, res_tower, value_head), trains only:
    - move_encoder
    - policy_query

    Args:
        network: The BlokusNet model
        device: Device string ('cuda' or 'cpu')
        num_positions: Number of positions to generate for training
        num_steps: Number of training steps
        batch_size: Batch size for training
        learning_rate: Learning rate for policy head
        progress_callback: Optional callback(step, total_steps, loss)

    Returns:
        Dictionary with training statistics
    """
    log.info("=" * 60)
    log.info("PHASE 0: Heuristic Pretraining")
    log.info("=" * 60)
    log.info(f"  Positions: {num_positions}")
    log.info(f"  Steps: {num_steps}")
    log.info(f"  Batch size: {batch_size}")
    log.info(f"  Learning rate: {learning_rate}")

    # Generate training data
    log.info("Generating training positions...")
    training_data = generate_pretraining_positions(num_positions)

    if len(training_data) < batch_size:
        log.warning(f"Not enough training data ({len(training_data)} < {batch_size}), skipping pretraining")
        return {'avg_loss': 0.0, 'skipped': True}

    # Freeze backbone layers, train only policy head
    frozen_count = 0
    trainable_count = 0

    for name, param in network.named_parameters():
        # Keep move_encoder and policy_query trainable
        if 'move_encoder' in name or 'policy_query' in name:
            param.requires_grad_(True)
            trainable_count += 1
        else:
            param.requires_grad_(False)
            frozen_count += 1

    log.info(f"Frozen parameters: {frozen_count}, Trainable: {trainable_count}")

    # Optimizer for policy head only
    policy_params = [p for p in network.parameters() if p.requires_grad]
    optimizer = optim.Adam(policy_params, lr=learning_rate)

    network.train()
    total_loss = 0.0
    loss_history = []

    for step in range(num_steps):
        # Sample batch
        batch = random.sample(training_data, min(batch_size, len(training_data)))

        # Prepare batch tensors
        board_np = np.stack([b['board_tensor'] for b in batch])
        pieces_np = np.stack([b['pieces_vector'] for b in batch])
        legal_moves_list = [b['legal_moves'] for b in batch]

        board_tensors = torch.from_numpy(board_np).to(device)
        pieces_vectors = torch.from_numpy(pieces_np).to(device)

        # Forward pass
        policy_logits, _ = network(board_tensors, pieces_vectors, legal_moves_list)

        # Compute loss
        loss = compute_policy_cross_entropy(policy_logits, batch, device)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_params, 1.0)
        optimizer.step()

        loss_value = loss.item()
        total_loss += loss_value
        loss_history.append(loss_value)

        # Progress callback
        if progress_callback and (step + 1) % 10 == 0:
            progress_callback(step + 1, num_steps, loss_value)

        # Log progress
        if (step + 1) % 100 == 0:
            avg_recent = sum(loss_history[-100:]) / len(loss_history[-100:])
            log.info(f"Pretrain step {step + 1}/{num_steps} - loss: {loss_value:.4f} (avg: {avg_recent:.4f})")

    # Unfreeze all parameters for main training
    for param in network.parameters():
        param.requires_grad_(True)

    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0

    log.info("=" * 60)
    log.info(f"Pretraining complete! Avg loss: {avg_loss:.4f}")
    log.info("=" * 60)

    return {
        'avg_loss': avg_loss,
        'final_loss': loss_history[-1] if loss_history else 0.0,
        'num_positions': len(training_data),
        'num_steps': num_steps,
    }
