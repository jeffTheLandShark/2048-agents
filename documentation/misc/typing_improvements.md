# Typing Improvements: Dataclasses and TypedDicts

This document identifies opportunities to replace loose dictionary and tuple types with explicit TypedDicts and dataclasses.

## Summary of Opportunities

### 1. Game Environment Info Dictionaries → TypedDict

**File: `game/game_env.py`**
- `reset()` returns `Dict[str, Any]` → Should be `ResetInfo` TypedDict
- `step()` returns `Dict[str, Any]` → Should be `StepInfo` TypedDict

**Benefits:**
- Explicit fields: score, tile_counts, heuristics, etc.
- Better IDE autocomplete
- Type checking for info dict usage

---

### 2. Game Summary Dictionaries → TypedDict

**Files: `logging/stats_logger.py`, `experiments/run_experiment.py`**
- `end_game(summary: Dict[str, Any])` → Should be `GameSummary` TypedDict
- `run_single_game()` returns `Dict[str, Any]` → Should be `GameSummary` TypedDict
- `run_experiment()` returns `Dict[str, Any]` → Should be `ExperimentSummary` TypedDict

**Fields:**
- `final_score: int`
- `highest_tile: int`
- `game_length: int`
- `final_tile_counts: Dict[str, int]`
- `final_heuristics: Dict[str, float]`
- (for ExperimentSummary) `mean_score`, `std_score`, `games_reached_2048`, etc.

---

### 3. Heuristic Features → TypedDict

**File: `heuristics/features.py`**
- `compute_all_features()` returns `Dict[str, float]` → Should be `HeuristicFeatures` TypedDict

**Fields:**
- `empty: float`
- `monotonicity: float`
- `smoothness: float`
- `merge_potential: float`
- `max_tile: float`
- `sum_tiles: float`

**Benefits:**
- Prevents typos in feature names
- Makes available features explicit

---

### 4. Utility Function Returns → Dataclasses

**File: `game/utils.py`**

#### `merge_line()` → `MergeResult` dataclass
Currently: `Tuple[List[int], int]`
```python
@dataclass
class MergeResult:
    merged_line: List[int]
    score_gained: int
```

#### `spawn_random_tile()` → `SpawnLocation` dataclass
Currently: `Tuple[int, int]` for position
```python
@dataclass
class SpawnLocation:
    row: int
    col: int
```

#### `get_empty_cells()` → `Position` dataclass
Currently: `List[Tuple[int, int]]`
```python
@dataclass
class Position:
    row: int
    col: int
```

**Benefits:**
- Named fields instead of positional tuples
- Better readability: `pos.row` vs `pos[0]`

---

### 5. Logging Step Data → TypedDict

**File: `logging/stats_logger.py`**
- `log_step()` parameters:
  - `tile_counts: Dict[str, int]` → Could be `TileCounts` TypedDict (but flexible dict is fine)
  - `heuristics: Dict[str, float]` → Should use `HeuristicFeatures` TypedDict

---

### 6. ETL Log Structures → TypedDict

**File: `logging/etl.py`**
- `load_jsonl_logs()` returns `List[dict]` → Should be `List[GameLog]` TypedDict
- Other functions take `List[dict]` → Should use `List[GameLog]`

**GameLog TypedDict fields:**
- `game_id: str`
- `agent: str`
- `board_size: int`
- `seed: Optional[int]`
- `config: Dict[str, Any]` (or more specific TypedDict)
- `steps: List[StepLog]` (TypedDict)
- `summary: GameSummary` (TypedDict)

---

### 7. Configuration Dictionary → TypedDict

**File: `logging/stats_logger.py`, `experiments/run_experiment.py`**
- `config: Optional[Dict[str, Any]]` → Could be `AgentConfig` TypedDict

**Fields:**
- `depth_limit: Optional[int]`
- `time_limit_ms: Optional[int]`
- `heuristic_weights: Dict[str, float]`
- etc.

---

## Priority Recommendations

### High Priority (Most Impact)
1. **GameSummary TypedDict** - Used in multiple places, critical for logging
2. **HeuristicFeatures TypedDict** - Used frequently, prevents feature name typos
3. **ExperimentSummary TypedDict** - Important for experiment results

### Medium Priority (Better Code Quality)
4. **Position/SpawnLocation dataclasses** - Improves readability of coordinate handling
5. **ResetInfo/StepInfo TypedDicts** - Better type safety for game env info

### Low Priority (Nice to Have)
6. **GameLog TypedDict** - Complex nested structure, but useful for ETL
7. **AgentConfig TypedDict** - Configuration is often flexible, but could be typed

---

## Implementation Notes

- TypedDicts are good for dictionaries that represent structured data
- Dataclasses are good for small value objects (like Position)
- Keep `Dict[str, int]` for truly dynamic dictionaries (like tile_counts where keys are tile values)
- Consider using `total=False` for TypedDicts with optional fields

