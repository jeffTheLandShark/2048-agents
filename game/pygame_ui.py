"""Pygame UI for human play, replay visualization, and live agent play."""

import pygame
import json
from typing import Optional, List, Dict, Any, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from game import StepInfo, Board, GameEnv, decode_board_log2
from agents import Agent
from stats_logging import StatsLogger, GameLog, StepLog, GameSummary
from stats_logging.etl import load_jsonl_logs, get_game_by_id
from heuristics.features import compute_all_features, compute_tile_counts, create_game_summary
import numpy as np


# Color scheme (2048-style)
COLORS = {
    "background": (250, 248, 239),
    "grid_bg": (187, 173, 160),
    "empty": (205, 193, 180),
    "text_dark": (119, 110, 101),
    "text_light": (249, 246, 242),
    "tiles": {
        0: (205, 193, 180),
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46),
    },
}


@dataclass
class UIConfig:
    """Configuration for pygame UI."""

    window_size: int = 600
    fps: int = 60
    tile_padding: int = 10
    font_size_large: int = 48
    font_size_medium: int = 36
    font_size_small: int = 24
    agent_move_delay_ms: int = 100  # Delay between agent moves in milliseconds
    enable_animations: bool = True  # Enable sliding animations
    animation_duration_ms: int = 150  # Duration of slide animation in milliseconds


class PygameUI:
    """
    Pygame-based UI for 2048 game.

    Supports:
    - Human play mode (keyboard input)
    - Replay mode (from raw JSONL logs)
    - Live agent play mode (watch agent play in real-time)
    """

    def __init__(
        self,
        board_size: int = 4,
        config: Optional[UIConfig] = None,
    ) -> None:
        """
        Initialize pygame UI.

        Args:
            board_size: Size of the game board (nxn).
            config: Optional UI configuration. Uses defaults if None.
        """
        self.board_size = board_size
        self.config = config or UIConfig()

        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("2048")

        # Create window
        self.screen = pygame.display.set_mode(
            (self.config.window_size, self.config.window_size + 100)
        )  # Extra space for score panel
        self.clock = pygame.time.Clock()

        # Load fonts
        try:
            self.font_large = pygame.font.Font(
                None, self.config.font_size_large
            )  # Default font
            self.font_medium = pygame.font.Font(None, self.config.font_size_medium)
            self.font_small = pygame.font.Font(None, self.config.font_size_small)
        except Exception:
            # Fallback to default fonts if custom fonts fail
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), self.config.font_size_large)
            self.font_medium = pygame.font.Font(pygame.font.get_default_font(), self.config.font_size_medium)
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), self.config.font_size_small)

        # Calculate board rendering dimensions
        self.board_padding = 20
        self.tile_size = (
            self.config.window_size - 2 * self.board_padding - (self.board_size + 1) * self.config.tile_padding
        ) // self.board_size

    def _get_tile_color(self, value: int) -> tuple:
        """Get color for a tile value."""
        if value == 0:
            return COLORS["empty"]
        # For values > 2048, use the 2048 color
        max_key = max(k for k in COLORS["tiles"].keys() if k <= value)
        return COLORS["tiles"].get(value, COLORS["tiles"][max_key])

    def _get_text_color(self, value: int) -> tuple:
        """Get text color for a tile (dark for small values, light for large)."""
        if value < 8:
            return COLORS["text_dark"]
        return COLORS["text_light"]

    def _render_tile(self, surface: pygame.Surface, value: int, x: int, y: int, alpha: int = 255) -> None:
        """Render a single tile."""
        color = self._get_tile_color(value)
        rect = pygame.Rect(x, y, self.tile_size, self.tile_size)

        def get_font(value: int) -> pygame.font.Font:
            if value >= 10000:
                return self.font_small
            elif value >= 1000:
                return self.font_medium
            else:
                return self.font_large

        # Create a surface for the tile to support alpha transparency
        if alpha < 255:
            tile_surface = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
            tile_surface.set_alpha(alpha)
            pygame.draw.rect(tile_surface, color, pygame.Rect(0, 0, self.tile_size, self.tile_size), border_radius=5)
            if value > 0:
                text_color = self._get_text_color(value)
                font = get_font(value)
                text = font.render(str(value), True, text_color)
                text_rect = text.get_rect(center=(self.tile_size // 2, self.tile_size // 2))
                tile_surface.blit(text, text_rect)
            surface.blit(tile_surface, (x, y))
        else:
            pygame.draw.rect(surface, color, rect, border_radius=5)
            if value > 0:
                text_color = self._get_text_color(value)
                font = get_font(value)
                text = font.render(str(value), True, text_color)
                text_rect = text.get_rect(center=(x + self.tile_size // 2, y + self.tile_size // 2))
                surface.blit(text, text_rect)

    def _simulate_row_left(self, row: List[Tuple[int, List[Tuple[int, int]]]]) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """
        Simulate sliding a row to the left (index 0).
        """
        # 1. Filter zeros (Compress)
        non_zeros = [x for x in row if x[0] != 0]

        # 2. Merge
        merged = []
        i = 0
        while i < len(non_zeros):
            current = non_zeros[i]
            if i + 1 < len(non_zeros) and current[0] == non_zeros[i+1][0]:
                next_tile = non_zeros[i+1]
                val = current[0] * 2
                src = current[1] + next_tile[1]
                merged.append((val, src))
                i += 2
            else:
                merged.append(current)
                i += 1

        # 3. Pad (Compress result)
        padding_len = len(row) - len(merged)
        padding = [(0, []) for _ in range(padding_len)]

        return merged + padding

    def _calculate_tile_movements(
        self, old_board: Board, new_board: Board, action: str
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Calculate tile movements from old_board to new_board using rotation simulation.
        Matches game/utils.py slide_and_merge logic.
        """
        # Create grid of (value, [source_pos])
        # Using object array to allow rotation
        grid = np.empty((self.board_size, self.board_size), dtype=object)
        for r in range(self.board_size):
            for c in range(self.board_size):
                val = old_board.array[r, c]
                # Source is a list of tuples. If empty/0, source list is empty.
                grid[r, c] = (val, [(r, c)] if val > 0 else [])

        # Determine rotation k to align with LEFT slide
        # UP (Top->Left) = 90 CCW = k=1
        # RIGHT (Right->Left) = 180 = k=2
        # DOWN (Bottom->Left) = 270 (90 CW) = k=3
        # LEFT = k=0
        k = 0
        if action == 'UP': k = 1
        elif action == 'RIGHT': k = 2
        elif action == 'DOWN': k = 3

        if k > 0:
            grid = np.rot90(grid, k)

        # Process each row
        for r in range(self.board_size):
            row_list = grid[r].tolist()
            new_row = self._simulate_row_left(row_list)
            for c in range(self.board_size):
                grid[r, c] = new_row[c]

        # Rotate back
        if k > 0:
            # Rotate back is -k
            grid = np.rot90(grid, -k)

        # Build map
        movements = {}
        for r in range(self.board_size):
            for c in range(self.board_size):
                val, sources = grid[r, c]
                movements[(r, c)] = {
                    'value': val,
                    'sources': sources
                }
        return movements

    def _get_tile_position(self, row: int, col: int) -> Tuple[int, int]:
        """Get pixel position for a tile at (row, col)."""
        board_start_y = 80
        x = (
            self.board_padding
            + self.config.tile_padding
            + col * (self.tile_size + self.config.tile_padding)
        )
        y = (
            board_start_y
            + self.board_padding
            + self.config.tile_padding
            + row * (self.tile_size + self.config.tile_padding)
        )
        return x, y

    def render(
        self,
        board: Board,
        score: int,
        info: Optional[StepInfo] = None,
        mode_text: Optional[str] = None,
        skip_flip: bool = False,
        animation_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Render current board state with optional animations.

        Args:
            board: Current board state as Board instance.
            score: Current game score.
            info: Optional StepInfo with additional metadata to display.
            mode_text: Optional text to display (e.g., "Human Play", "Replay", "Agent Play").
            skip_flip: If True, don't call pygame.display.flip() (useful when drawing overlay after).
            animation_state: Optional dict with 'old_board', 'progress', 'action' for animations.
        """
        # Clear screen
        self.screen.fill(COLORS["background"])

        # Render score panel
        score_y = 10
        score_text = self.font_medium.render(f"Score: {score}", True, COLORS["text_dark"])
        self.screen.blit(score_text, (20, score_y))

        if mode_text:
            mode_text_surface = self.font_small.render(mode_text, True, COLORS["text_dark"])
            self.screen.blit(mode_text_surface, (20, score_y + 30))

        # Render board background
        board_start_y = 80
        board_rect = pygame.Rect(
            self.board_padding,
            board_start_y + self.board_padding,
            self.config.window_size - 2 * self.board_padding,
            self.config.window_size - 2 * self.board_padding,
        )
        pygame.draw.rect(self.screen, COLORS["grid_bg"], board_rect, border_radius=10)

        # Render tiles with or without animation
        if (self.config.enable_animations and animation_state and
            animation_state.get('old_board') is not None and
            animation_state.get('progress', 1.0) < 1.0):
            self._render_animated(board, animation_state)
        else:
            # Normal rendering without animation
            board_list = board.to_list()
            for row in range(self.board_size):
                for col in range(self.board_size):
                    x, y = self._get_tile_position(row, col)
                    value = board_list[row][col]
                    self._render_tile(self.screen, value, x, y)

        if not skip_flip:
            pygame.display.flip()

    def _render_animated(self, new_board: Board, animation_state: Dict[str, Any]) -> None:
        """Render board with sliding animations."""
        old_board = animation_state['old_board']
        progress = animation_state.get('progress', 0.0)
        action = animation_state.get('action', '')

        # Clamp progress to [0, 1]
        progress = max(0.0, min(1.0, progress))

        # Use easing function for smoother animation (ease-out)
        eased_progress = 1 - (1 - progress) ** 3

        # Calculate movements
        movements = self._calculate_tile_movements(old_board, new_board, action)

        # Access arrays for lookups
        old_array = old_board.array
        new_array = new_board.array

        # 1. Draw empty background grid for ALL cells to prevent flashing
        # This ensures that if a tile moves, the spot underneath is "empty" color, not grid bg color.
        for r in range(self.board_size):
            for c in range(self.board_size):
                x, y = self._get_tile_position(r, c)
                self._render_tile(self.screen, 0, x, y)

        # 2. Render moving tiles
        for (new_row, new_col), move_info in movements.items():
            value = move_info['value']
            sources = move_info['sources']

            if not sources:
                # No tile moved here. Check for spawn later.
                continue

            new_x, new_y = self._get_tile_position(new_row, new_col)

            if len(sources) > 1:
                # Merge animation
                if progress < 0.5:
                    # Show sources moving towards dest
                    for src_r, src_c in sources:
                        old_x, old_y = self._get_tile_position(src_r, src_c)
                        cur_x = old_x + (new_x - old_x) * eased_progress
                        cur_y = old_y + (new_y - old_y) * eased_progress
                        old_val = old_array[src_r, src_c]

                        # Fade out
                        alpha = int(255 * (1 - progress * 2))
                        self._render_tile(self.screen, old_val, int(cur_x), int(cur_y), alpha)
                else:
                    # Show merged result tile at dest
                    # It "pops" in
                    self._render_tile(self.screen, value, new_x, new_y)
            else:
                # Slide animation (single source)
                src_r, src_c = sources[0]
                old_x, old_y = self._get_tile_position(src_r, src_c)
                cur_x = old_x + (new_x - old_x) * eased_progress
                cur_y = old_y + (new_y - old_y) * eased_progress
                # Render using old value (should be same as new value for slide, but safe)
                old_val = old_array[src_r, src_c]
                self._render_tile(self.screen, old_val, int(cur_x), int(cur_y))

        # 3. Render spawned tiles (fade in)
        for r in range(self.board_size):
            for c in range(self.board_size):
                # If movement map says empty, but new board has tile, it's a spawn
                if movements[(r, c)]['value'] == 0 and new_array[r, c] > 0:
                    x, y = self._get_tile_position(r, c)
                    alpha = int(255 * progress) if progress < 0.3 else 255
                    self._render_tile(self.screen, new_array[r, c], x, y, alpha)

    def handle_input(self) -> Optional[str]:
        """
        Handle keyboard input for human play.
        Iterates through all events to ensure no inputs are dropped.

        Returns:
            Action string ("UP", "DOWN", "LEFT", "RIGHT", "RESET", "QUIT") or None.
        """
        action = None

        # Process all events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "QUIT"
                elif event.key == pygame.K_r:
                    action = "RESET"
                elif event.key == pygame.K_SPACE:
                    action = "SPACE" # Used in replay mode

                # Direction keys
                key_mapping = {
                    pygame.K_UP: "UP",
                    pygame.K_w: "UP",
                    pygame.K_DOWN: "DOWN",
                    pygame.K_s: "DOWN",
                    pygame.K_LEFT: "LEFT",
                    pygame.K_a: "LEFT",
                    pygame.K_RIGHT: "RIGHT",
                    pygame.K_d: "RIGHT",
                }
                if event.key in key_mapping:
                    # Overwrite previous direction if multiple keys pressed in one frame
                    # (Last key press takes precedence)
                    action = key_mapping[event.key]

        return action

    def run_loop(
        self,
        update_callback: Callable[[Optional[str]], bool],
        render_callback: Callable[[], None]
    ) -> None:
        """
        Standard game loop handling events, updates, and rendering.

        Args:
            update_callback: Function taking (action) and returning True to continue, False to stop.
            render_callback: Function to handle rendering.
        """
        running = True
        while running:
            # 1. Process Input
            action = self.handle_input()
            if action == "QUIT":
                running = False
                break

            # 2. Update Game State
            if not update_callback(action):
                running = False
                break

            # 3. Render
            render_callback()

            # 4. Tick Clock
            self.clock.tick(self.config.fps)

        self.close()

    def run_human_play(
        self,
        env: GameEnv,
        logger: Optional[StatsLogger] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Run interactive human play loop."""
        if seed is not None:
            env.seed(seed)

        # Initialize game counter from existing logs if logger exists
        initial_game_count = logger._get_next_game_number("human", seed) if logger else 1

        # Game state container to be modified by callback
        state = {
            "board": None,
            "score": 0,
            "step_count": 0,
            "game_count": initial_game_count - 1,  # Will be incremented to initial_game_count on first reset
            "done": False,
            "waiting_for_reset": False,
            "animation_state": None,  # Dict with 'old_board', 'progress', 'action', 'start_time'
        }

        def reset_game():
            state["board"], _ = env.reset()
            state["score"] = 0
            state["step_count"] = 0
            state["game_count"] += 1
            state["done"] = False
            state["waiting_for_reset"] = False
            state["animation_state"] = None
            if logger:
                logger.start_game(f"human_play_{seed or 'random'}_{state['game_count']}", seed)

        # Initial reset
        reset_game()

        def update(action: Optional[str]) -> bool:
            if state["waiting_for_reset"]:
                if action == "RESET":
                    reset_game()
                return True

            if action == "RESET":
                reset_game()
                return True

            # Update animation progress if animating
            if state["animation_state"] is not None:
                current_time = pygame.time.get_ticks()
                elapsed = current_time - state["animation_state"]["start_time"]
                duration = self.config.animation_duration_ms
                progress = min(1.0, elapsed / duration)
                state["animation_state"]["progress"] = progress

                # If animation complete, clear animation state
                if progress >= 1.0:
                    state["animation_state"] = None
                else:
                    # Still animating, block input
                    return True

            if action and action in ["UP", "DOWN", "LEFT", "RIGHT"] and not state["done"]:
                legal_moves = env.legal_moves(state["board"])
                if action in legal_moves:
                    # Save old board for animation
                    old_board = state["board"].copy() if self.config.enable_animations else None

                    state["board"], reward, state["done"], info = env.step(action)
                    state["score"] += int(reward)
                    state["step_count"] += 1

                    # Start animation if enabled
                    if self.config.enable_animations and old_board is not None:
                        state["animation_state"] = {
                            "old_board": old_board,
                            "progress": 0.0,
                            "action": action,
                            "start_time": pygame.time.get_ticks(),
                        }

                    if logger:
                        tile_counts = compute_tile_counts(state["board"])
                        heuristics = compute_all_features(state["board"])

                        logger.log_step(
                            t=state["step_count"],
                            board=state["board"],
                            action=action,
                            reward=reward,
                            score=state["score"],
                            tile_counts=tile_counts,
                            heuristics=heuristics,
                            done=state["done"],
                        )

                    if state["done"]:
                        state["waiting_for_reset"] = True
                        if logger:
                            tile_counts = compute_tile_counts(state["board"])
                            heuristics = compute_all_features(state["board"])
                            summary = create_game_summary(
                                state["board"], state["score"], state["step_count"], heuristics, tile_counts
                            )
                            logger.end_game(summary)
            return True

        def render():
            if state["waiting_for_reset"]:
                # Draw board first, then overlay, then flip once
                self.render(state["board"], state["score"], mode_text="Game Over!", skip_flip=True)
                self._show_game_over(state["score"], state["step_count"])
            else:
                self.render(
                    state["board"],
                    state["score"],
                    mode_text="Human Play - Arrow Keys/WASD to move, R to reset, ESC to quit",
                    animation_state=state["animation_state"]
                )

        self.run_loop(update, render)

    def run_replay(self, log_file: Path, game_id: Optional[str] = None, speed: float = 1.0) -> None:
        """
        Run replay visualization from JSONL log file.

        Args:
            log_file: Path to JSONL log file.
            game_id: Optional ID of the game to replay. If None, replays the first game found.
            speed: Playback speed multiplier.
        """
        # Select game
        game_log: Optional[GameLog] = None

        if game_id:
            # Optimized lookup: streams until ID is found
            game_log = get_game_by_id(log_file, game_id)
            if not game_log:
                print(f"Error: Game ID '{game_id}' not found in log file.")
                return
        else:
            # Optimized fallback: reads only the first line
            try:
                game_log = next(load_jsonl_logs(log_file))
                print(f"No game ID provided. Replaying first game: {game_log.get('game_id', 'unknown')}")
            except StopIteration:
                print(f"Error: No valid games found in log file {log_file}")
                return

        steps = game_log.get("steps", [])
        if not steps:
            print(f"Error: No steps found in game log {game_log.get('game_id', 'unknown')}")
            return

        board_size = game_log.get("board_size", 4)
        if board_size != self.board_size:
            print(f"Warning: Log board size ({board_size}) != UI board size ({self.board_size})")

        # Speed steps: 50ms to 400ms in 50ms increments, then 100ms increments up to 1000ms
        speed_steps = [50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000]

        # Find closest initial delay
        initial_delay = int(250 / speed)
        initial_idx = min(range(len(speed_steps)), key=lambda i: abs(speed_steps[i] - initial_delay))

        # Replay state
        state = {
            "current_step_idx": 0,
            "playing": True,
            "last_step_time": pygame.time.get_ticks(),
            "step_delay_ms": speed_steps[initial_idx],
            "speed_idx": initial_idx,
            "animation_state": None,  # Dict with 'old_board', 'progress', 'action', 'start_time'
            "previous_board": None,  # Track previous board for animations
        }

        def update(action: Optional[str]) -> bool:
            # Update animation progress if animating
            if state["animation_state"] is not None:
                current_time = pygame.time.get_ticks()
                elapsed = current_time - state["animation_state"]["start_time"]
                duration = self.config.animation_duration_ms
                progress = min(1.0, elapsed / duration)
                state["animation_state"]["progress"] = progress

                # If animation complete, clear animation state
                if progress >= 1.0:
                    state["animation_state"] = None
                else:
                    # Still animating, block auto-advance but allow manual controls
                    if action not in ["LEFT", "RIGHT", "SPACE", "RESET", "UP", "DOWN"]:
                        return True

            # Determine if animations should be enabled based on play state
            # Always enable in step-through mode (paused), conditionally in auto-play
            should_animate = False
            if not state["playing"]:
                # Step-through mode: always enable animations
                should_animate = True
            else:
                # Auto-play mode: enable if step delay >= animation duration
                should_animate = state["step_delay_ms"] >= self.config.animation_duration_ms

            # Temporarily override config for this frame
            original_animations = self.config.enable_animations
            self.config.enable_animations = should_animate

            if action == "UP":
                # Increase speed (decrease delay) -> decrease index
                new_idx = max(0, state["speed_idx"] - 1)
                state["speed_idx"] = new_idx
                state["step_delay_ms"] = speed_steps[new_idx]
            elif action == "DOWN":
                # Decrease speed (increase delay) -> increase index
                new_idx = min(len(speed_steps) - 1, state["speed_idx"] + 1)
                state["speed_idx"] = new_idx
                state["step_delay_ms"] = speed_steps[new_idx]
            elif action == "RESET":
                state["current_step_idx"] = 0
                state["last_step_time"] = pygame.time.get_ticks()
                state["animation_state"] = None
                state["previous_board"] = None
            elif action == "RIGHT":
                if state["current_step_idx"] < len(steps) - 1:
                    # Save old board for animation (current board before moving)
                    if should_animate:
                        current_step = steps[state["current_step_idx"]]
                        next_step = steps[state["current_step_idx"] + 1]
                        old_board = decode_board_log2(current_step["board"], board_size)
                        action_taken = next_step.get("action", "")

                        state["current_step_idx"] = min(state["current_step_idx"] + 1, len(steps) - 1)
                        state["animation_state"] = {
                            "old_board": old_board,
                            "progress": 0.0,
                            "action": action_taken,
                            "start_time": pygame.time.get_ticks(),
                        }
                    else:
                        state["current_step_idx"] = min(state["current_step_idx"] + 1, len(steps) - 1)
                    state["last_step_time"] = pygame.time.get_ticks()
            elif action == "LEFT":
                # Disable animation for backward steps to avoid visual artifacts
                state["animation_state"] = None
                if state["current_step_idx"] > 0:
                    state["current_step_idx"] = max(state["current_step_idx"] - 1, 0)
                    state["last_step_time"] = pygame.time.get_ticks()
            elif action == "SPACE":
                state["playing"] = not state["playing"]
                state["last_step_time"] = pygame.time.get_ticks()

            # Auto-advance (only if not animating)
            if state["playing"] and state["current_step_idx"] < len(steps) - 1 and state["animation_state"] is None:
                current_time = pygame.time.get_ticks()
                if current_time - state["last_step_time"] >= state["step_delay_ms"]:
                    # Save old board for animation
                    if should_animate:
                        current_step = steps[state["current_step_idx"]]
                        next_step = steps[state["current_step_idx"] + 1]
                        old_board = decode_board_log2(current_step["board"], board_size)
                        action_taken = next_step.get("action", "")

                        state["current_step_idx"] += 1
                        state["animation_state"] = {
                            "old_board": old_board,
                            "progress": 0.0,
                            "action": action_taken,
                            "start_time": current_time,
                        }
                    else:
                        state["current_step_idx"] += 1
                    state["last_step_time"] = current_time

            # Restore original animation setting
            self.config.enable_animations = original_animations

            return True

        def render():
            step = steps[state["current_step_idx"]]
            board = decode_board_log2(step["board"], board_size)
            score = step.get("score", 0)
            game_id_disp = game_log.get('game_id', 'unknown')

            # Update previous board for next animation
            state["previous_board"] = board.copy()

            # Determine if animations should be enabled
            should_animate = False
            if not state["playing"]:
                should_animate = True  # Always enable in step-through mode
            else:
                should_animate = state["step_delay_ms"] >= self.config.animation_duration_ms

            # Temporarily override config for rendering
            original_animations = self.config.enable_animations
            self.config.enable_animations = should_animate

            mode_text = f"Replay {game_id_disp} ({state['current_step_idx'] + 1}/{len(steps)}) - Space: pause, Left/Right: step, R: restart"
            self.render(board, score, mode_text=mode_text, animation_state=state["animation_state"])

            # Restore original animation setting
            self.config.enable_animations = original_animations

        self.run_loop(update, render)

    def run_agent(
        self,
        agent: Agent,
        env: GameEnv,
        logger: Optional[StatsLogger] = None,
        seed: Optional[int] = None,
        move_delay_ms: Optional[int] = None,
    ) -> None:
        """Run live agent play mode."""
        if seed is not None:
            env.seed(seed)

        delay_ms = move_delay_ms or self.config.agent_move_delay_ms

        # Initialize game counter from existing logs if logger exists
        agent_name = getattr(agent, "__class__", type(agent)).__name__
        initial_game_count = logger._get_next_game_number(agent_name, seed) if logger else 1

        state = {
            "board": None,
            "score": 0,
            "step_count": 0,
            "game_count": initial_game_count - 1,  # Will be incremented to initial_game_count on first reset
            "done": False,
            "waiting_for_reset": False,
            "last_move_time": pygame.time.get_ticks(),
            "animation_state": None,  # Dict with 'old_board', 'progress', 'action', 'start_time'
        }

        def reset_game():
            agent.reset()
            state["board"], _ = env.reset()
            state["score"] = 0
            state["step_count"] = 0
            state["game_count"] += 1
            state["done"] = False
            state["waiting_for_reset"] = False
            state["last_move_time"] = pygame.time.get_ticks()
            state["animation_state"] = None
            if logger:
                agent_name = getattr(agent, "__class__", type(agent)).__name__
                logger.start_game(f"{agent_name}_{seed or 'random'}_{state['game_count']}", seed)

        reset_game()

        def update(action: Optional[str]) -> bool:
            if action == "RESET":
                reset_game()
                return True

            if state["waiting_for_reset"]:
                return True

            # Update animation progress if animating
            if state["animation_state"] is not None:
                current_time = pygame.time.get_ticks()
                elapsed = current_time - state["animation_state"]["start_time"]
                duration = self.config.animation_duration_ms
                progress = min(1.0, elapsed / duration)
                state["animation_state"]["progress"] = progress

                # If animation complete, clear animation state
                if progress >= 1.0:
                    state["animation_state"] = None
                else:
                    # Still animating, don't process moves yet
                    return True

            current_time = pygame.time.get_ticks()
            if not state["done"] and (current_time - state["last_move_time"] >= delay_ms):
                legal_moves = env.legal_moves(state["board"])
                if legal_moves:
                    # Save old board for animation
                    old_board = state["board"].copy() if self.config.enable_animations else None

                    agent_action = agent.choose_action(state["board"], legal_moves)
                    state["board"], reward, state["done"], info = env.step(agent_action)
                    state["score"] += int(reward)
                    state["step_count"] += 1

                    # Start animation if enabled
                    if self.config.enable_animations and old_board is not None:
                        state["animation_state"] = {
                            "old_board": old_board,
                            "progress": 0.0,
                            "action": agent_action,
                            "start_time": pygame.time.get_ticks(),
                        }

                    if logger:
                        tile_counts = compute_tile_counts(state["board"])
                        heuristics = compute_all_features(state["board"])
                        logger.log_step(
                            t=state["step_count"],
                            board=state["board"],
                            action=agent_action,
                            reward=reward,
                            score=state["score"],
                            tile_counts=tile_counts,
                            heuristics=heuristics,
                            done=state["done"],
                        )

                    state["last_move_time"] = current_time
                else:
                    state["done"] = True

            if state["done"] and not state["waiting_for_reset"]:
                state["waiting_for_reset"] = True
                if logger:
                    tile_counts = compute_tile_counts(state["board"])
                    heuristics = compute_all_features(state["board"])
                    summary = create_game_summary(
                        state["board"], state["score"], state["step_count"], heuristics, tile_counts
                    )
                    logger.end_game(summary)

            return True

        def render():
            if state["waiting_for_reset"]:
                # Draw board first, then overlay, then flip once
                self.render(state["board"], state["score"], mode_text="Game Over!", skip_flip=True)
                self._show_game_over(state["score"], state["step_count"])
            else:
                agent_name = getattr(agent, "__class__", type(agent)).__name__
                mode_text = f"Agent Play: {agent_name} - R: reset, ESC: quit"
                self.render(state["board"], state["score"], mode_text=mode_text, animation_state=state["animation_state"])

        self.run_loop(update, render)

    def _show_game_over(self, score: int, steps: int) -> None:
        """Display game over message."""
        # Overlay semi-transparent surface
        overlay = pygame.Surface((self.config.window_size, self.config.window_size + 100))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        # Game over text
        game_over_text = self.font_large.render("Game Over!", True, COLORS["text_light"])
        score_text = self.font_medium.render(f"Final Score: {score}", True, COLORS["text_light"])
        steps_text = self.font_small.render(f"Steps: {steps}", True, COLORS["text_light"])
        restart_text = self.font_small.render("Press R to restart or ESC to quit", True, COLORS["text_light"])

        center_x = self.config.window_size // 2
        y_offset = (self.config.window_size + 100) // 2 - 60

        self.screen.blit(game_over_text, game_over_text.get_rect(center=(center_x, y_offset)))
        self.screen.blit(score_text, score_text.get_rect(center=(center_x, y_offset + 50)))
        self.screen.blit(steps_text, steps_text.get_rect(center=(center_x, y_offset + 80)))
        self.screen.blit(restart_text, restart_text.get_rect(center=(center_x, y_offset + 110)))

        # Flip after drawing overlay
        pygame.display.flip()

    def close(self) -> None:
        """Close pygame window and cleanup resources."""
        pygame.quit()


# Public API functions
def run_human_ui(
    env: Optional[GameEnv] = None,
    board_size: int = 4,
    seed: Optional[int] = None,
    logger: Optional[StatsLogger] = None,
    config: Optional[UIConfig] = None,
) -> None:
    """
    Run human play UI.

    Args:
        env: Optional GameEnv instance. Creates new one if None.
        board_size: Board size if creating new env.
        seed: Optional random seed.
        logger: Optional StatsLogger for logging.
        config: Optional UI configuration.
    """
    if env is None:
        from game import GameEnv

        env = GameEnv(board_size=board_size, seed=seed)

    ui = PygameUI(board_size=board_size, config=config)
    ui.run_human_play(env, logger=logger, seed=seed)


def run_replay_from_log(
    log_file: Path,
    game_id: Optional[str] = None,
    board_size: int = 4,
    speed: float = 1.0,
    config: Optional[UIConfig] = None,
) -> None:
    """
    Run replay visualization from JSONL log file.

    Args:
        log_file: Path to JSONL log file.
        game_id: Optional ID of the game to replay.
        board_size: Board size (should match log file).
        speed: Playback speed multiplier.
        config: Optional UI configuration.
    """
    # Auto-configure animations for replay
    if config is None:
        config = UIConfig()

    # Calculate step delay from speed (speed=1.0 means 1 step per second = 1000ms delay)
    step_delay_ms = int(1000 / speed)

    # Enable animations if step delay >= animation duration (for auto-play)
    # Animations will be dynamically enabled/disabled based on play/pause state
    # When paused (step-through mode), animations are always enabled
    config.enable_animations = step_delay_ms >= config.animation_duration_ms

    ui = PygameUI(board_size=board_size, config=config)
    ui.run_replay(log_file, game_id=game_id, speed=speed)


def run_agent(
    agent: Agent,
    env: Optional[GameEnv] = None,
    board_size: int = 4,
    seed: Optional[int] = None,
    logger: Optional[StatsLogger] = None,
    move_delay_ms: Optional[int] = None,
    config: Optional[UIConfig] = None,
) -> None:
    """
    Run live agent play UI - watch an agent play in real-time.

    Args:
        agent: Agent instance to play with.
        env: Optional GameEnv instance. Creates new one if None.
        board_size: Board size if creating new env.
        seed: Optional random seed.
        logger: Optional StatsLogger for logging.
        move_delay_ms: Optional delay between moves in milliseconds.
        config: Optional UI configuration.
    """
    if env is None:
        from game import GameEnv

        env = GameEnv(board_size=board_size, seed=seed)

    # Auto-enable animations for agent games if move delay > animation duration
    if config is None:
        config = UIConfig()

    # Use the provided move_delay_ms or the config default
    actual_delay = move_delay_ms or config.agent_move_delay_ms

    # Enable animations only if move delay is greater than animation duration
    # This prevents animations from causing lag when agent moves too fast
    config.enable_animations = actual_delay >= config.animation_duration_ms

    ui = PygameUI(board_size=board_size, config=config)
    ui.run_agent(agent, env, logger=logger, seed=seed, move_delay_ms=move_delay_ms)
