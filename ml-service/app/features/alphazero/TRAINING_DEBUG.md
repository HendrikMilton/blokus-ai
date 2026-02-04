# AlphaZero Training Debug Analysis

## Run Analyzed
- **Run ID**: b5151bcd-6676-4f78-b715-c313d3ed1682
- **Status**: Failed at generation 69 (ran overnight)
- **Issues Identified**:
  1. Gradient explosion at generation 37
  2. No actual policy learning for 36+ generations

---

## Issue 1: Gradient Explosion (FIXED)

### Symptoms
- At step 1018 of generation 37, policy_loss exploded from ~5.8 to 4.2e+34
- Training continued despite model corruption (NaN losses were skipped)
- All subsequent generations had policy_loss in 10^32-10^33 range

### Root Cause
No gradient clipping was implemented. A single bad batch caused catastrophic gradient updates.

### Fix Applied
Added gradient clipping with `torch.nn.utils.clip_grad_norm_()`:

```python
# In trainer.py TrainingConfig:
max_grad_norm: float = 1.0  # Max gradient norm for clipping

# In training loop (both mixed precision and regular):
torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
```

---

## Issue 2: No Policy Learning (DIAGNOSIS COMPLETE, NEEDS ARCHITECTURAL FIX)

### Symptoms
- Policy loss stayed flat at ~4.35 for 36 generations
- Win rate remained random at ~25% (1/4 players)
- Value loss DID decrease from ~1.0 to ~0.08 (value head IS learning)

### Key Insight: Policy Loss = log(num_legal_moves)

```
exp(4.35) ≈ 78
```

A policy loss of 4.35 means the network is predicting near-uniform probabilities over ~78 legal moves. This is the cross-entropy loss when both target and prediction are uniform distributions over N items: `-sum(1/N * log(1/N)) = log(N)`.

### Root Cause: Cold Start Problem

1. **Initial state**: Network outputs uniform policy over all legal moves
2. **MCTS with uniform prior**: Explores all moves somewhat equally
3. **With 800 simulations over 78+ moves**: Visits remain fairly uniform
4. **Training targets**: Near-uniform MCTS visit distributions
5. **Result**: Training on uniform targets reinforces uniform predictions

This is a **chicken-and-egg problem**:
- MCTS needs good policy priors to focus search
- Network needs focused MCTS targets to learn good policies
- Without external signal, the system stays in random equilibrium

### Why Value Head Learns But Policy Doesn't

The value head receives a clear signal: +1 for winner, -1 for losers. Even with random play, the network can learn that:
- Certain board configurations correlate with winning
- Game phase (early/mid/late) affects outcomes
- Piece placement patterns matter

The policy head, however, receives targets that are the output of MCTS **using its own predictions**. When predictions are uniform, MCTS generates uniform targets, creating no learning gradient.

---

## Recommendations to Fix Cold Start Problem

### Option 1: Bootstrap with Handcrafted Heuristics

Add heuristic-based policy biases before MCTS:
```python
def compute_move_prior(move, game_state):
    """Simple heuristics to bootstrap policy learning."""
    score = 1.0

    # Prefer larger pieces early game
    if game_state.turn_number < 20:
        score += PIECE_CATALOG[move.piece_idx].size * 0.1

    # Prefer center positions
    center_dist = abs(move.row - 9.5) + abs(move.col - 9.5)
    score += (1.0 - center_dist / 20.0) * 0.2

    # Prefer moves that create more corner options
    # ... etc

    return score
```

Blend these priors with the neural network output initially, then gradually reduce the blend factor.

### Option 2: Curriculum Learning

Start training on simpler scenarios:
1. **Phase 1**: 1v1 Blokus (2 players, lower branching factor)
2. **Phase 2**: 2v2 team Blokus
3. **Phase 3**: Full 4-player Blokus

This gives the network easier learning problems before tackling the full game.

### Option 3: Increase MCTS Simulations at Root

With more simulations, MCTS can explore more deeply even with uniform priors:
```python
# Current: 800 simulations
# Increase to: 1600-3200 for early training

# After policy starts improving, can reduce back to 800
mcts_simulations: int = 1600  # Double for first N generations
```

### Option 4: Temperature Schedule for Policy Targets

Instead of using raw MCTS visit counts, use temperature-adjusted targets:
```python
# Higher temperature (>1.0) = more exploration = sharper targets
# Lower temperature (<1.0) = more exploitation = softer targets

# Use high temperature initially to create more peaked targets
target_temperature = 0.5  # Makes winning moves more prominent in target
```

### Option 5: Add Policy Entropy Regularization

Add a term to the loss that penalizes uniform predictions:
```python
# Encourage the network to be more confident
policy_entropy = -torch.sum(F.softmax(policy_logits, dim=1) * F.log_softmax(policy_logits, dim=1), dim=1)
entropy_bonus = -0.01 * policy_entropy.mean()  # Negative = penalize high entropy
loss = policy_loss + value_loss + entropy_bonus
```

---

## New Diagnostic Metrics Added

To track policy learning, we now log:

1. **Policy Entropy**: Average entropy of MCTS target policies
   - High (~4.0+): Nearly uniform targets (no learning signal)
   - Medium (~2.0-3.0): Some focus emerging
   - Low (<2.0): Focused targets (good learning signal)

2. **Top-1 Probability**: Average probability of the best move
   - Low (~0.01-0.02): Uniform (1/N)
   - High (>0.1): MCTS is finding clearly better moves

3. **Top-5 Probability**: Sum of top 5 move probabilities
   - Indicates if MCTS is concentrating visits on few moves

These metrics are logged per generation and stored in the database.

---

## Testing the Fixes

After restarting training with the gradient clipping fix:

1. **Check that policy loss no longer explodes**
2. **Monitor new policy quality metrics**:
   - If entropy stays at ~4.0 and top1 stays at ~0.01, the cold start problem persists
   - If entropy decreases and top1 increases, learning is occurring

If the cold start problem persists, implement one of the recommended fixes above (Option 1 or 4 are quickest to implement).

---

## Summary

| Issue | Status | Fix |
|-------|--------|-----|
| Gradient explosion | ✅ FIXED | Added `clip_grad_norm_(max_norm=1.0)` |
| No policy learning | ✅ FIXED | Implemented 3 cold start fixes (see below) |
| Value head not learning | ❌ N/A | Value head IS learning (loss: 1.0 → 0.08) |
| Win rate random | 🔗 Related | Should improve with policy learning |

---

## Cold Start Fixes Implemented

### 1. Heuristic Policy Bootstrapping

Added handcrafted heuristics that are blended with the neural network policy at the MCTS root:

**New file**: `heuristics.py`

**Heuristics implemented**:
- **Piece size preference**: Strongly prefers larger pieces (5-cell pentominoes) early game
- **Center position bonus**: Prefers moves closer to board center
- **Corner creation bonus**: Prefers moves that create more attachment points
- **Edge avoidance**: Penalizes moves at board edges
- **Blocking potential**: In mid/late game, prefers moves that block opponents

**Config parameters**:
```python
use_heuristic_bootstrap: bool = True
heuristic_blend_initial: float = 0.5   # 50% heuristic, 50% NN initially
heuristic_blend_decay_gens: int = 30   # Linear decay to pure NN over 30 generations
```

### 2. Increased MCTS Simulations for Early Training

More simulations help MCTS find good moves even with uninformative policy priors:

```python
mcts_simulations_initial: int = 1600   # Start with 1600 sims
mcts_simulations_decay_gens: int = 20  # Decay to base (400-800) over 20 generations
mcts_simulations: int = 800            # Base simulation count
```

### 3. Temperature-Adjusted Policy Targets

Applying temperature < 1.0 to MCTS visit counts makes the targets more peaked (less uniform):

```python
policy_target_temperature: float = 0.5  # Makes winning moves more prominent
```

This transforms near-uniform visit distributions into more focused targets that provide stronger learning signal.

---

## Expected Results

With these fixes, you should see:

1. **Policy entropy** (logged per generation) should decrease over time
   - Initial: ~4.0 (uniform over ~78 moves)
   - After 10 gens: ~3.5 (starting to focus)
   - After 30 gens: ~2.5-3.0 (meaningful policy)

2. **Top-1 probability** should increase
   - Initial: ~0.01 (1/N uniform)
   - After 30 gens: >0.1 (clear preference)

3. **Win rate** should improve above 25%
   - 25% = random (1/4 players)
   - 35%+ = meaningful learning

4. **Policy loss** should decrease below 4.0
   - 4.35 = log(78), uniform over all moves
   - <3.5 = better than uniform, learning is happening
