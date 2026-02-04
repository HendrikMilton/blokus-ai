"""
Handcrafted heuristics for bootstrapping AlphaZero policy learning.

These heuristics provide an initial policy bias to help MCTS generate
focused targets during early training, addressing the "cold start" problem
where a random network produces uniform MCTS visits.

The heuristics are blended with the neural network output using a decay
schedule: high heuristic weight early in training, fading to pure NN later.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .game import BlokusGame, Move, BOARD_SIZE
from .pieces import PIECE_CATALOG, PIECE_BY_INDEX


# Board center coordinates
CENTER_ROW = (BOARD_SIZE - 1) / 2.0  # 9.5 for 20x20
CENTER_COL = (BOARD_SIZE - 1) / 2.0


@dataclass
class HeuristicConfig:
    """Configuration for heuristic policy bootstrapping."""
    # Blend factor: 0.0 = pure NN, 1.0 = pure heuristic
    # This should decay over training generations
    blend_factor: float = 0.5

    # Feature weights (higher = more important)
    piece_size_weight: float = 1.0      # Prefer larger pieces early
    center_weight: float = 0.5          # Prefer center positions
    corner_creation_weight: float = 0.3  # Prefer moves that create corners
    mobility_weight: float = 0.2        # Prefer moves that maintain flexibility

    # Game phase thresholds
    early_game_turns: int = 20          # Before this, piece size matters most
    mid_game_turns: int = 50            # After this, flexibility matters more


def compute_move_heuristic_score(
    move: Move,
    game: BlokusGame,
    config: HeuristicConfig,
) -> float:
    """
    Compute a heuristic score for a single move.

    Higher score = better move according to handcrafted rules.

    Returns:
        Score >= 0 (not normalized)
    """
    score = 1.0  # Base score for any legal move

    piece = PIECE_BY_INDEX[move.piece_idx]
    turn = game.turn_number

    # 1. PIECE SIZE BONUS (STRONG EFFECT)
    # Early game: STRONGLY prefer larger pieces (5-cells)
    # Rationale: Large pieces are harder to place later when board fills up
    # This is THE most important heuristic for Blokus
    if turn <= config.early_game_turns:
        # Scale: size 1 -> 0.0 bonus, size 5 -> 4.0 bonus
        # Make this exponential to really prefer 5s over 4s over 3s
        size_bonus = (piece.size ** 2) / 5.0  # 1->0.2, 2->0.8, 3->1.8, 4->3.2, 5->5.0
        score += config.piece_size_weight * size_bonus
    elif turn <= config.mid_game_turns:
        # Mid game: moderate piece size preference
        size_bonus = (piece.size ** 1.5) / 5.0
        score += config.piece_size_weight * size_bonus * 0.5

    # 2. CENTER POSITION BONUS
    # Prefer moves that place pieces closer to center
    # Rationale: Central control gives more options for future placements
    piece_cells = piece.get_cells(move.orientation_idx, move.row, move.col)
    avg_row = piece_cells[:, 0].mean()
    avg_col = piece_cells[:, 1].mean()

    # Distance from center (0 = center, ~14 = corner)
    center_dist = np.sqrt((avg_row - CENTER_ROW) ** 2 + (avg_col - CENTER_COL) ** 2)
    max_dist = np.sqrt(2) * CENTER_ROW  # ~13.4

    # Closer to center = higher bonus (0 to 1.5)
    # Use squared distance for stronger center preference
    center_bonus = (1.0 - (center_dist / max_dist)) ** 2 * 1.5
    score += config.center_weight * center_bonus

    # 3. CORNER CREATION BONUS
    # Prefer moves that create more attachment points for future pieces
    # Rationale: More corners = more options = more flexibility
    corners_created = 0
    player_color = move.player + 1
    cell_set = set(map(tuple, piece_cells))

    for r, c in piece_cells:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:  # Diagonals
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                # Check if this could be a valid corner
                if game.board[nr, nc] == 0:  # Empty
                    # Check not edge-adjacent to own color
                    is_valid = True
                    for edr, edc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        er, ec = nr + edr, nc + edc
                        if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE:
                            if game.board[er, ec] == player_color:
                                is_valid = False
                                break
                        # Also check if edge-adjacent to piece being placed
                        if (er, ec) in cell_set:
                            is_valid = False
                            break
                    if is_valid:
                        corners_created += 1

    # Normalize corners (typically 0-8 possible)
    corner_bonus = min(corners_created / 4.0, 2.0)  # Cap at 2.0
    score += config.corner_creation_weight * corner_bonus

    # 4. EDGE AVOIDANCE (more penalty)
    # Penalize moves at board edges - getting trapped is bad
    edge_cells = 0
    for r, c in piece_cells:
        if r <= 1 or r >= BOARD_SIZE - 2 or c <= 1 or c >= BOARD_SIZE - 2:
            edge_cells += 1
    edge_penalty = edge_cells / piece.size * 0.5
    score -= edge_penalty

    # 5. BLOCKING POTENTIAL (late game)
    # In mid/late game, prefer moves that interfere with opponent corners
    if turn > config.early_game_turns:
        blocking_bonus = 0
        for r, c in piece_cells:
            # Check if we're adjacent to opponent pieces (blocking their expansion)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    cell_val = game.board[nr, nc]
                    if cell_val != 0 and cell_val != player_color:
                        blocking_bonus += 0.1
        score += blocking_bonus

    return max(score, 0.01)  # Ensure positive score


def compute_heuristic_policy(
    game: BlokusGame,
    legal_moves: List[Move],
    config: Optional[HeuristicConfig] = None,
) -> np.ndarray:
    """
    Compute a heuristic policy distribution over legal moves.

    Args:
        game: Current game state
        legal_moves: List of legal moves
        config: Heuristic configuration

    Returns:
        Probability distribution over moves (sums to 1)
    """
    if not legal_moves:
        return np.array([])

    if config is None:
        config = HeuristicConfig()

    # Compute raw scores
    scores = np.array([
        compute_move_heuristic_score(move, game, config)
        for move in legal_moves
    ])

    # Apply softmax with temperature to convert to probabilities
    # Lower temperature = more peaked distribution
    temperature = 0.15  # Very low temp makes good moves MUCH more prominent
    scores = scores / temperature
    scores = scores - scores.max()  # Numerical stability
    exp_scores = np.exp(scores)
    policy = exp_scores / exp_scores.sum()

    return policy


def blend_policies(
    nn_policy: np.ndarray,
    heuristic_policy: np.ndarray,
    blend_factor: float,
) -> np.ndarray:
    """
    Blend neural network policy with heuristic policy.

    Args:
        nn_policy: Policy from neural network (normalized)
        heuristic_policy: Policy from heuristics (normalized)
        blend_factor: 0.0 = pure NN, 1.0 = pure heuristic

    Returns:
        Blended policy (normalized)
    """
    if len(nn_policy) != len(heuristic_policy):
        raise ValueError(
            f"Policy lengths don't match: NN={len(nn_policy)}, heur={len(heuristic_policy)}"
        )

    if len(nn_policy) == 0:
        return nn_policy

    # Linear blend
    blended = (1 - blend_factor) * nn_policy + blend_factor * heuristic_policy

    # Renormalize (should already sum to 1, but ensure it)
    total = blended.sum()
    if total > 0:
        blended = blended / total
    else:
        # Fallback to uniform
        blended = np.ones_like(blended) / len(blended)

    return blended


def get_blend_factor_for_generation(
    generation: int,
    initial_blend: float = 0.5,
    decay_generations: int = 30,
    min_blend: float = 0.0,
) -> float:
    """
    Compute blend factor that decays over training generations.

    Starts with initial_blend, decays linearly to min_blend over
    decay_generations, then stays at min_blend.

    Args:
        generation: Current training generation (1-indexed)
        initial_blend: Starting blend factor (high = more heuristic)
        decay_generations: Number of generations to decay over
        min_blend: Minimum blend factor (0 = pure NN after decay)

    Returns:
        Blend factor for current generation
    """
    if generation >= decay_generations:
        return min_blend

    # Linear decay
    progress = generation / decay_generations
    blend = initial_blend * (1 - progress) + min_blend * progress

    return blend


# Quick test if run directly
if __name__ == "__main__":
    # Create a test game
    game = BlokusGame()
    legal_moves = game.get_legal_moves()

    print(f"Testing heuristics on initial game state")
    print(f"Legal moves: {len(legal_moves)}")

    config = HeuristicConfig()
    policy = compute_heuristic_policy(game, legal_moves, config)

    print(f"\nTop 10 moves by heuristic score:")
    sorted_indices = np.argsort(policy)[::-1][:10]
    for i, idx in enumerate(sorted_indices):
        move = legal_moves[idx]
        piece = PIECE_BY_INDEX[move.piece_idx]
        print(f"  {i+1}. {piece.id} at ({move.row}, {move.col}) "
              f"orient={move.orientation_idx}: prob={policy[idx]:.4f}")

    # Test blend factor decay
    print("\nBlend factor decay:")
    for gen in [1, 5, 10, 15, 20, 25, 30, 35, 50]:
        bf = get_blend_factor_for_generation(gen)
        print(f"  Gen {gen}: blend={bf:.3f}")
