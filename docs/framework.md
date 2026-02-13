# Framework vs User Control

This document is the canonical strategy contract for StackSats.

## Framework Owns (Non-Negotiable)

- Fixed budget and fixed allocation span (365 or 366 days depending on leap year)
- Uniform initialization of all daily weights
- Day-by-day iterative execution loop
- Locked historical weights (past days are immutable)
- Feasibility projection/clipping at the daily handoff boundary
- Remaining-budget enforcement
- Validation guards (`NaN`/`inf`/range checks) and final invariants

## User Owns (Flexible)

- Feature engineering from lagged/base data
- Signal definitions/formulas
- Signal weights and hyperparameters
- Daily intent output:
  - `propose_weight(state)` for per-day proposals, or
  - `build_target_profile(...)` for a full-window intent series

## Handoff Boundary

The user never writes the framework iteration loop.

User output (`proposed_weight_today` or daily profile intent) is handed to the framework allocation kernel, which computes `final_weight_today` by applying:

1. feasibility clipping
2. remaining-budget rules
3. historical lock rules
4. final invariant checks

## Required Behavior

- Users can strongly influence allocation each day through features/signals/intent.
- Users cannot alter iteration mechanics or rewrite past allocations.
- Local, backtest, and production run the same sealed allocation kernel.

## Production Daily Lifecycle

1. Load locked historical weights for the active allocation span.
2. Build lagged features/signals using information available up to `current_date`.
3. Collect user daily intent (`proposed_weight_today` or profile-derived intent).
4. Project to feasible `final_weight_today` with remaining-budget constraints.
5. Persist today as locked.
6. Advance to next day; past values remain immutable.

## Design Intent

This boundary is deliberate: it maximizes strategy flexibility while preventing forward-looking bias and preserving deterministic allocation semantics.
