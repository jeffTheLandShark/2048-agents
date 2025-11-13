# Board Class Implementation Plan

## Overview

Implement a `Board` class that wraps a numpy array internally, providing:
- Type safety and validation
- Fast numpy operations for agents
- `.array` property for direct performance-critical access
- Fast copying for search trees (Expectimax/MCTS)
- Backward compatibility where needed

## Design Decision

**Hybrid Approach**: Board class with numpy internally, expose `.array` for performance-critical code.

**Benefits**:
- Fast copying (~50ns vs ~200ns) - critical for search trees
- Vectorized heuristics (10-50x faster)
- Type safety maintained
- Agents can opt into performance when needed

---

## Task Breakdown

### Task 1: Create Board Class Foundation
**File**: `game/__init__.py`

**Subtasks**:
1. Add Board class with numpy array internally
   - `_array: np.ndarray` (shape: (n, n), dtype: int32)
   - `size: int` property
   - Validation: ensure square, correct size
2. Add `__init__` accepting `List[List[int]]` or `np.ndarray`
3. Add `@property array` for direct numpy access
4. Add `copy()` method using `np.array.copy()`
5. Add `__getitem__` and `__setitem__` for indexing
6. Add `__eq__` for comparison
7. Add `to_list()` method for backward compatibility
8. Add `from_list()` class method

**Type**: [RELEVANT] - Core foundation

---

### Task 2: Update Game Core Types
**Files**: `game/game_env.py`, `game/utils.py`, `game/board_encoding.py`

**Subtasks**:
1. **game_env.py**:
   - Update `reset()` return type: `Board` instead of `List[List[int]]`
   - Update `step()` return type: `Board` instead of `List[List[int]]`
   - Update `legal_moves()` parameter: `Optional[Board]`
   - Update `is_game_over()` parameter: `Optional[Board]`
   - Update `get_board()` return type: `Board`
   - Internal implementation uses `board.array` for operations

2. **game/utils.py**:
   - Update `spawn_random_tile()` parameter and return: `Board`
   - Update `get_empty_cells()` parameter: `Board`
   - Use `board.array` for numpy operations (e.g., `np.where(board.array == 0)`)

3. **game/board_encoding.py**:
   - Update `encode_board_log2()` parameter: `Board`
   - Update `encode_board_flat()` parameter: `Board`
   - Update `decode_board_log2()` return: `Board`
   - Update `decode_board_flat()` return: `Board`
   - Use `board.array` for encoding operations

**Type**: [RELEVANT] - Core game logic

---

### Task 3: Update Agent Interfaces
**Files**: `agents/base.py`, `agents/expectimax.py`, `agents/mcts.py`, `agents/random_agent.py`

**Subtasks**:
1. **agents/base.py**:
   - Update `choose_action()` parameter: `Board` instead of `List[List[int]]`
   - Update docstring

2. **agents/expectimax.py**:
   - Update `choose_action()` parameter: `Board`
   - Use `board.array` for performance-critical operations
   - Use `board.copy()` for search tree copies
   - Update internal search to work with Board objects

3. **agents/mcts.py**:
   - Update `choose_action()` parameter: `Board`
   - Use `board.array` for performance-critical operations
   - Use `board.copy()` for simulation copies
   - Update rollout logic to work with Board objects

4. **agents/random_agent.py**:
   - Update `choose_action()` parameter: `Board`
   - (No performance changes needed for random agent)

**Type**: [RELEVANT] - Agent performance critical

---

### Task 4: Update Heuristics for Vectorization
**Files**: `heuristics/features.py`, `heuristics/evaluator.py`

**Subtasks**:
1. **heuristics/features.py**:
   - Update all function parameters: `Board` instead of `List[List[int]]`
   - Refactor to use `board.array` for vectorized operations:
     - `compute_empty_tiles()`: `np.sum(board.array == 0)`
     - `compute_max_tile()`: `np.max(board.array)`
     - `compute_sum_tiles()`: `np.sum(board.array)`
     - `compute_monotonicity()`: Use numpy for row/column operations
     - `compute_smoothness()`: Use numpy for adjacent differences
     - `compute_merge_potential()`: Use numpy for adjacent comparisons
   - Update `compute_all_features()` return type (already TypedDict)

2. **heuristics/evaluator.py**:
   - Update `evaluate()` parameter: `Board`
   - Pass `board` to `compute_all_features()`

**Type**: [RELEVANT] - Heuristic performance critical (called millions of times)

---

### Task 5: Update Logging and Experiments
**Files**: `logging/stats_logger.py`, `experiments/run_experiment.py`, `logging/etl.py`

**Subtasks**:
1. **logging/stats_logger.py**:
   - Update `log_step()` parameter: `Board`
   - Convert board to list for encoding: `board.to_list()` or use `board.array.flatten()`
   - Update encoding calls

2. **experiments/run_experiment.py**:
   - Update internal game loop to work with Board objects
   - Convert Board to list for logging if needed

3. **logging/etl.py**:
   - No changes needed (works with dictionaries from JSONL)

**Type**: [RELEVANT] - Data logging

---

### Task 6: Update UI and Other Files
**Files**: `game/pygame_ui.py`

**Subtasks**:
1. **game/pygame_ui.py**:
   - Update `render()` parameter: `Board`
   - Convert to list for rendering if needed: `board.to_list()`
   - Update `run_replay()` to create Board objects from logs

**Type**: [RELEVANT] - UI compatibility

---

### Task 7: Update Type Definitions
**Files**: `game/__init__.py` (ResetInfo, StepInfo), `logging/__init__.py` (StepLog)

**Subtasks**:
1. **game/__init__.py**:
   - Update `ResetInfo` and `StepInfo` TypedDicts if they reference boards
   - May need to keep as lists for JSON serialization, or add conversion

2. **logging/__init__.py**:
   - `StepLog` board field is already `List[int]` (flattened encoding) - no change needed

**Type**: [RELEVANT] - Type consistency

---

## Implementation Strategy

### Phase 1: Foundation (Tasks 1-2)
1. Create Board class with all methods
2. Update game core to use Board
3. Test basic operations

### Phase 2: Agents (Task 3)
1. Update agent interfaces
2. Optimize Expectimax/MCTS with `board.array`
3. Test agent performance

### Phase 3: Heuristics (Task 4)
1. Refactor heuristics for vectorization
2. Test heuristic performance improvements
3. Verify correctness

### Phase 4: Integration (Tasks 5-7)
1. Update logging/experiments
2. Update UI
3. Fix any type issues
4. End-to-end testing

---

## Performance Targets

### Expected Improvements:
- **Board copying**: 4x faster (~50ns vs ~200ns)
- **Heuristic computation**: 5-10x faster (vectorized operations)
- **Expectimax per move**: 3-5x faster (10k evaluations → ~1ms vs ~5ms)
- **MCTS per move**: 2-3x faster (1000 simulations)

### Validation:
- Run benchmarks before/after
- Ensure correctness (same results)
- Measure actual speedup in agent performance

---

## Backward Compatibility

### Conversion Points:
1. **JSONL logging**: Convert Board to list for serialization
2. **UI rendering**: Convert Board to list if pygame needs it
3. **Agent interface**: Accept Board, but can convert internally if needed

### Migration Path:
- Board class provides `to_list()` for compatibility
- Can gradually migrate code to use `board.array` directly

---

## Testing Checklist

- [ ] Board creation from list and numpy array
- [ ] Board copying (fast numpy copy)
- [ ] Board indexing (`board[row, col]`)
- [ ] Board comparison (`board1 == board2`)
- [ ] GameEnv operations with Board
- [ ] Agent operations with Board (all agents)
- [ ] Heuristic computations (all features)
- [ ] Logging with Board (conversion to list)
- [ ] UI rendering with Board
- [ ] Performance benchmarks (before/after)

---

## Risk Assessment

**Low Risk**:
- Board class implementation (straightforward)
- Type updates (mechanical)

**Medium Risk**:
- Heuristic vectorization (need to verify correctness)
- Agent search tree logic (need to ensure Board works correctly)

**Mitigation**:
- Keep old list-based code as fallback initially
- Add comprehensive tests
- Benchmark to verify performance gains

---

## Estimated Effort

- **Task 1**: 1-2 hours (Board class)
- **Task 2**: 2-3 hours (Game core updates)
- **Task 3**: 2-3 hours (Agent updates + optimization)
- **Task 4**: 3-4 hours (Heuristic vectorization)
- **Task 5**: 1 hour (Logging updates)
- **Task 6**: 1 hour (UI updates)
- **Task 7**: 1 hour (Type updates)

**Total**: ~11-16 hours

---

## Notes

- Numpy is already a dependency (for RNG)
- Board size validation ensures square grids
- `.array` property allows agents to opt into performance
- Can add more convenience methods later (e.g., `get_row()`, `get_col()`)

