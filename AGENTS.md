# Agent System Documentation

Essential agent information for AI assistants. For full architecture details, see `DESIGN.md`.

---

## 1. Overview

All agents implement a common interface and work with `Board` objects (numpy-backed) for performance.

**Key Files:** `agents/base.py`, `agents/expectimax.py`, `agents/mcts.py`, `agents/random_agent.py`

For architecture diagrams, see `DESIGN.md` sections 3.0 and 3.4.

---

## 2. Agent Interface

All agents inherit from `Agent` and implement:

```python
from agents import Agent
from game import Board
from typing import List

class MyAgent(Agent):
    def choose_action(self, state: Board, legal_moves: List[str]) -> str:
        # Return one of legal_moves
        pass

    def reset(self) -> None:  # Optional for stateful agents
        pass
```

**Key Points:** Agents receive `Board` objects (not lists), must return a valid `legal_moves` action. See `agents/base.py` and `DESIGN.md` section 3.4.

---

## 3. Code Standards

### Board Usage
- **Performance ops**: Use `board.array` for numpy operations
- **Copying**: Use `board.copy()` for search trees (~4x faster)
- **Serialization**: Use `board.to_list()` only for JSON/UI

### Integration
- Called by `experiments/run_experiment.py` (not directly by `game_env.py`)
- Receive deterministic `Board` states
- Don't write logs directly (runner handles logging)
- Be deterministic with same state/seed

### Imports
```python
from agents import Agent  # ✅ Package-level
from game import Board
# from agents.base import Agent  # ❌ Avoid
```

For details, see `DESIGN.md` sections 3.2, 3.7, 4, and 6.

---

## 4. Agent Implementations

### ExpectimaxAgent (`agents/expectimax.py`)
- Iterative deepening search with weighted heuristics
- Config: `depth_limit`, `time_limit_ms`, `heuristic_weights`, `use_iterative_deepening`
- Supports fixed (best-known) or GA-optimized weights via `set_heuristic_weights()`
- Uses `heuristics/evaluator.py` and `heuristics/features.py`

### MCTSAgent (`agents/mcts.py`)
- UCT implementation with configurable simulations
- Config: `num_simulations`, `exploration_constant`, `rollout_policy` ("random" or "heuristic"), `time_limit_ms`

### RandomAgent (`agents/random_agent.py`)
- Baseline: uniform random selection from legal moves
- Config: `seed` (optional)

For algorithm details, see `DESIGN.md` section 3.4. For heuristics, see `DESIGN.md` section 3.5.

---

## 5. Heuristics Integration

Agents using heuristics (e.g., Expectimax) follow: `Board → features.py → evaluator.py → Agent`

- `heuristics/features.py`: Stateless feature functions (monotonicity, smoothness, etc.)
- `heuristics/evaluator.py`: Weighted evaluation `H(s) = Σ wi * fi`
- Use `board.array` for vectorized operations (5-10x faster)

See `DESIGN.md` section 3.5 for complete heuristics documentation.

---

## 6. Performance

**Best Practices:** Use `board.array` for numpy ops, `board.copy()` for trees, avoid `board.to_list()` except for serialization.

See `DESIGN.md` section 6 for benchmarks.

---

## References

- Architecture: `DESIGN.md` 3.0 | Agents: `DESIGN.md` 3.4 | Heuristics: `DESIGN.md` 3.5
- Performance: `DESIGN.md` 6 | Imports: `DESIGN.md` 4 | Game Env: `DESIGN.md` 3.2 | Board: `DESIGN.md` 3.1
