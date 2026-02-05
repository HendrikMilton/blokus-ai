"""
Handcrafted heuristics for bootstrapping AlphaZero policy learning.

These heuristics provide an initial policy bias to help MCTS generate
focused targets during early training, addressing the "cold start" problem
where a random network produces uniform MCTS visits.

The heuristics are blended with the neural network output using a decay
schedule: high heuristic weight early in training, fading to pure NN later.

Enhanced with strategic features:
- Mobility lookahead (count moves after placing)
- Corner differential (own corners gained vs opponent corners blocked)
- Opponent corner blocking
- Piece timing strategy
- Territory control
- Phase-adaptive weights
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass

from .game import BlokusGame, Move, BOARD_SIZE, EDGE_OFFSETS, DIAGONAL_OFFSETS
from .pieces import PIECE_CATALOG, PIECE_BY_INDEX, PIECE_INDEX_BY_ID

# Board center coordinates
CENTER_ROW = (BOARD_SIZE - 1) / 2.0  # 9.5 for 20x20
CENTER_COL = (BOARD_SIZE - 1) / 2.0


# ============================================================================
# Piece Timing Strategy
# ============================================================================

# (play_early, play_late) - when each piece is ideally played
PIECE_TIMING: Dict[str, Tuple[bool, bool]] = {
    # 5-cell pieces (pentominoes)
    'F': (True, False),    # Complex F-shape, play early when board is open
    'I5': (False, True),   # Long straight, save for corridors/gaps
    'L5': (True, False),   # L-shape, good early for extending
    'N': (True, False),    # Zigzag, needs space
    'P': (True, False),    # Compact, flexible - good anytime
    'T5': (True, False),   # T-shape, good for central control
    'U': (True, False),    # U-shape, needs space
    'V': (True, False),    # Corner shape, good early
    'W': (True, False),    # Stairs, needs space
    'X': (True, False),    # Plus sign, excellent center control
    'Y': (True, False),    # Y-shape, needs space
    'Z': (True, False),    # Z-shape, needs space

    # 4-cell pieces (tetrominoes)
    'I4': (False, True),   # Long, save for corridors
    'O4': (False, True),   # Square, save for filling gaps
    'T4': (True, True),    # T-shape, flexible
    'L4': (True, True),    # L-shape, flexible
    'S4': (True, True),    # S-shape, flexible

    # 3-cell pieces (triominoes)
    'I3': (False, True),   # Save for tight spots
    'V3': (False, True),   # Save for corner filling

    # Small pieces (always save for endgame)
    '2': (False, True),    # Domino, save for gaps
    '1': (False, True),    # Monomino, key for final placement
}


# ============================================================================
# Phase-Based Weight Profiles
# ============================================================================

@dataclass
class PhaseWeights:
    """Weights for different features in each game phase."""
    piece_size: float
    center: float
    corners: float
    blocking: float
    mobility: float
    corner_diff: float
    territory: float
    piece_timing: float


PHASE_WEIGHTS: Dict[str, PhaseWeights] = {
    'opening': PhaseWeights(
        piece_size=2.0,      # Play big pieces first
        center=1.5,          # Control center early
        corners=0.8,         # Create options
        blocking=0.3,        # Light blocking
        mobility=1.0,        # Maintain flexibility
        corner_diff=0.8,     # Corner balance
        territory=0.5,       # Expand territory
        piece_timing=0.5,    # Piece timing bonus
    ),
    'midgame': PhaseWeights(
        piece_size=1.0,      # Moderate size preference
        center=0.5,          # Less center focus
        corners=1.2,         # Corners matter more
        blocking=1.0,        # Active blocking
        mobility=1.5,        # Maintain flexibility
        corner_diff=1.2,     # Corner balance critical
        territory=0.8,       # Territory control
        piece_timing=0.4,    # Still consider timing
    ),
    'endgame': PhaseWeights(
        piece_size=0.3,      # Use what fits
        center=0.1,          # Center irrelevant
        corners=0.5,         # Corners less critical
        blocking=1.5,        # Block opponents hard
        mobility=2.0,        # Critical to survive
        corner_diff=1.0,     # Still important
        territory=0.3,       # Less relevant
        piece_timing=0.2,    # Less relevant
    ),
}


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

    # SmartPlayer-specific settings
    use_phase_weights: bool = True      # Use phase-adaptive weights
    use_mobility_lookahead: bool = True  # Enable mobility calculation
    use_corner_differential: bool = True  # Enable corner diff calculation
    use_opponent_blocking: bool = True   # Enable blocking calculation
    use_piece_timing: bool = True        # Enable piece timing


# ============================================================================
# Game Phase Detection
# ============================================================================

def get_game_phase(game: BlokusGame) -> str:
    """
    Determine game phase based on board state.

    Returns: 'opening', 'midgame', or 'endgame'
    """
    # Count total pieces played
    total_remaining = sum(len(p.remaining_pieces) for p in game.players)
    pieces_played = 84 - total_remaining  # 21 pieces * 4 players = 84

    fill_ratio = pieces_played / 84.0

    if fill_ratio < 0.25:
        return 'opening'
    elif fill_ratio < 0.60:
        return 'midgame'
    else:
        return 'endgame'


def get_phase_weights(game: BlokusGame) -> PhaseWeights:
    """Get the appropriate weight profile for current game phase."""
    phase = get_game_phase(game)
    return PHASE_WEIGHTS[phase]


# ============================================================================
# Core Scoring Functions
# ============================================================================

def is_edge_adjacent(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
    """Check if two positions are edge-adjacent (not diagonal)."""
    r1, c1 = pos1
    r2, c2 = pos2
    return (abs(r1 - r2) + abs(c1 - c2)) == 1


def count_moves_after(game: BlokusGame, move: Move) -> int:
    """
    Simulate a move and count how many legal moves are available after.
    This is the most important predictor of move quality.

    Note: This only counts moves for the current player, not after
    all opponents have moved (which would be too expensive).
    """
    game_copy = game.clone()
    game_copy.make_move(move)

    # Get moves for the same player in the hypothetical future
    # (after the move, it's the next player's turn, but we want to
    # know how many options THIS player will have)
    return len(game_copy.get_legal_moves(move.player))


def corner_differential(game: BlokusGame, move: Move, player: int) -> float:
    """
    Calculate corner differential:
    (my corners gained) + 0.5 * (opponent corners blocked)

    Higher = better move.
    """
    # Current state
    my_corners_before = len(game._valid_corners[player])

    # Simulate move
    game_copy = game.clone()
    game_copy.make_move(move)

    my_corners_after = len(game_copy._valid_corners[player])

    # Count opponent corners blocked
    blocked = 0
    for opp in range(4):
        if opp != player:
            before = len(game._valid_corners[opp])
            after = len(game_copy._valid_corners[opp])
            blocked += max(0, before - after)  # Only count if reduced

    # Corners gained + half credit for blocking
    return (my_corners_after - my_corners_before) + blocked * 0.5


def blocks_opponent_corners(game: BlokusGame, move: Move) -> float:
    """
    Score how many opponent corners this move blocks.

    A move can block opponent corners by:
    1. Occupying them directly (2.0 points)
    2. Being edge-adjacent to them (1.0 points - blocks diagonal connection)
    """
    piece = PIECE_BY_INDEX[move.piece_idx]
    cells = piece.get_cells(move.orientation_idx, move.row, move.col)
    cell_set = set(map(tuple, cells))
    blocked_score = 0.0

    for opp in range(4):
        if opp != move.player:
            for corner in game._valid_corners[opp]:
                # Check if move directly occupies this corner
                if corner in cell_set:
                    blocked_score += 2.0
                    continue

                # Check if move is edge-adjacent to this corner
                # (blocks opponent from using it due to edge adjacency rule)
                for r, c in cells:
                    if is_edge_adjacent(corner, (r, c)):
                        blocked_score += 1.0
                        break  # Only count once per corner

    return blocked_score


def piece_timing_bonus(piece_id: str, turn: int, total_turns: int = 80) -> float:
    """
    Bonus/penalty for playing a piece at the right time.

    Returns:
        +0.5 if played at ideal time
        -0.3 if played at wrong time
        0.0 otherwise
    """
    timing = PIECE_TIMING.get(piece_id)
    if timing is None:
        return 0.0

    early, late = timing
    phase = turn / total_turns

    if early and phase < 0.3:
        return 0.5  # Good: early piece played early
    if late and phase > 0.6:
        return 0.5  # Good: late piece played late
    if not early and phase < 0.2:
        return -0.3  # Bad: late piece played too early
    if not late and phase > 0.7:
        return -0.3  # Bad: early piece played too late

    return 0.0


def territory_score(game: BlokusGame, move: Move, player: int) -> float:
    """
    Score based on board quadrant control and expansion.

    Rewards:
    - Expanding into new quadrants
    - Strengthening presence in owned quadrants
    """
    piece = PIECE_BY_INDEX[move.piece_idx]
    cells = piece.get_cells(move.orientation_idx, move.row, move.col)

    # Determine which quadrants the move covers
    # Quadrants: (0,0)=top-left, (0,1)=top-right, (1,0)=bottom-left, (1,1)=bottom-right
    quadrants_touched = set()
    for r, c in cells:
        q = (0 if r < 10 else 1, 0 if c < 10 else 1)
        quadrants_touched.add(q)

    # Count current presence in each quadrant
    player_color = player + 1
    quadrant_presence = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if game.board[r, c] == player_color:
                q = (0 if r < 10 else 1, 0 if c < 10 else 1)
                quadrant_presence[q] += 1

    score = 0.0

    # Bonus for expanding into new/weak quadrants
    for q in quadrants_touched:
        presence = quadrant_presence[q]
        if presence == 0:
            score += 0.8  # New quadrant
        elif presence < 10:
            score += 0.4  # Weak presence, strengthen
        else:
            score += 0.1  # Already strong

    # Penalty for concentrating too much in one quadrant
    max_presence = max(quadrant_presence.values())
    if max_presence > 30:
        score -= 0.3

    return score


# ============================================================================
# Main Scoring Function
# ============================================================================

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

    # Get phase-adaptive weights if enabled
    if config.use_phase_weights:
        weights = get_phase_weights(game)
    else:
        weights = None

    # 1. PIECE SIZE BONUS (STRONG EFFECT)
    # Early game: STRONGLY prefer larger pieces (5-cells)
    # Rationale: Large pieces are harder to place later when board fills up
    # This is THE most important heuristic for Blokus
    piece_size_mult = weights.piece_size if weights else config.piece_size_weight

    if turn <= config.early_game_turns:
        # Scale: size 1 -> 0.0 bonus, size 5 -> 4.0 bonus
        # Make this exponential to really prefer 5s over 4s over 3s
        size_bonus = (piece.size ** 2) / 5.0  # 1->0.2, 2->0.8, 3->1.8, 4->3.2, 5->5.0
        score += piece_size_mult * size_bonus
    elif turn <= config.mid_game_turns:
        # Mid game: moderate piece size preference
        size_bonus = (piece.size ** 1.5) / 5.0
        score += piece_size_mult * size_bonus * 0.5
    else:
        # Late game: minimal size preference
        size_bonus = piece.size / 5.0
        score += piece_size_mult * size_bonus * 0.2

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
    center_mult = weights.center if weights else config.center_weight
    center_bonus = (1.0 - (center_dist / max_dist)) ** 2 * 1.5
    score += center_mult * center_bonus

    # 3. CORNER CREATION BONUS
    # Prefer moves that create more attachment points for future pieces
    # Rationale: More corners = more options = more flexibility
    corners_created = 0
    player_color = move.player + 1
    cell_set = set(map(tuple, piece_cells))

    for r, c in piece_cells:
        for dr, dc in DIAGONAL_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                # Check if this could be a valid corner
                if game.board[nr, nc] == 0:  # Empty
                    # Check not edge-adjacent to own color
                    is_valid = True
                    for edr, edc in EDGE_OFFSETS:
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
    corner_mult = weights.corners if weights else config.corner_creation_weight
    corner_bonus = min(corners_created / 4.0, 2.0)  # Cap at 2.0
    score += corner_mult * corner_bonus

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
    blocking_mult = weights.blocking if weights else 0.5
    if turn > config.early_game_turns and config.use_opponent_blocking:
        blocking_score = blocks_opponent_corners(game, move)
        score += blocking_mult * blocking_score * 0.3

    # 6. PIECE TIMING BONUS
    if config.use_piece_timing:
        timing_mult = weights.piece_timing if weights else 0.4
        timing_bonus = piece_timing_bonus(piece.id, turn)
        score += timing_mult * timing_bonus

    # 7. TERRITORY SCORE
    if weights and config.use_phase_weights:
        territory_mult = weights.territory
        terr_score = territory_score(game, move, move.player)
        score += territory_mult * terr_score * 0.3

    return max(score, 0.01)  # Ensure positive score


def compute_move_heuristic_score_deep(
    move: Move,
    game: BlokusGame,
    config: HeuristicConfig,
    include_mobility: bool = True,
    include_corner_diff: bool = True,
) -> float:
    """
    Compute a deeper heuristic score that includes lookahead features.

    This is more expensive than compute_move_heuristic_score but more accurate.
    Used by SmartPlayer for top candidate evaluation.
    """
    # Start with basic score
    score = compute_move_heuristic_score(move, game, config)

    weights = get_phase_weights(game) if config.use_phase_weights else None

    # Add mobility score (expensive - requires game clone)
    if include_mobility and config.use_mobility_lookahead:
        mobility = count_moves_after(game, move)
        mobility_mult = weights.mobility if weights else 1.0
        # Normalize: typical range is 0-200 moves
        score += mobility_mult * (mobility / 50.0)

    # Add corner differential (expensive - requires game clone)
    if include_corner_diff and config.use_corner_differential:
        corner_diff = corner_differential(game, move, move.player)
        corner_diff_mult = weights.corner_diff if weights else 1.0
        score += corner_diff_mult * corner_diff * 0.5

    return score


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
    print(f"Game phase: {get_game_phase(game)}")

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

    # Test phase weights
    print("\nPhase weights:")
    for phase, weights in PHASE_WEIGHTS.items():
        print(f"  {phase}: piece_size={weights.piece_size}, mobility={weights.mobility}")
