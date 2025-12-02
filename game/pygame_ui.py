"""Pygame UI for human play, replay visualization, and live agent play."""

import pygame
import json
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from game import StepInfo, Board, GameEnv, decode_board_log2
from agents import Agent
from stats_logging import StatsLogger, GameLog, StepLog, GameSummary
from stats_logging.etl import load_jsonl_logs, get_game_by_id
from heuristics.features import compute_all_features, compute_tile_counts, create_game_summary


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
    font_size_medium: int = 24
    font_size_small: int = 18
    agent_move_delay_ms: int = 100  # Delay between agent moves in milliseconds


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

    def _render_tile(self, surface: pygame.Surface, value: int, x: int, y: int) -> None:
        """Render a single tile."""
        color = self._get_tile_color(value)
        rect = pygame.Rect(x, y, self.tile_size, self.tile_size)
        pygame.draw.rect(surface, color, rect, border_radius=5)

        if value > 0:
            text_color = self._get_text_color(value)
            # Choose appropriate font size based on number length
            if value >= 1000:
                font = self.font_small
            elif value >= 100:
                font = self.font_medium
            else:
                font = self.font_large

            text = font.render(str(value), True, text_color)
            text_rect = text.get_rect(center=(x + self.tile_size // 2, y + self.tile_size // 2))
            surface.blit(text, text_rect)

    def render(
        self,
        board: Board,
        score: int,
        info: Optional[StepInfo] = None,
        mode_text: Optional[str] = None,
        skip_flip: bool = False,
    ) -> None:
        """
        Render current board state.

        Args:
            board: Current board state as Board instance.
            score: Current game score.
            info: Optional StepInfo with additional metadata to display.
            mode_text: Optional text to display (e.g., "Human Play", "Replay", "Agent Play").
            skip_flip: If True, don't call pygame.display.flip() (useful when drawing overlay after).
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

        # Render tiles
        board_list = board.to_list()
        for row in range(self.board_size):
            for col in range(self.board_size):
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
                value = board_list[row][col]
                self._render_tile(self.screen, value, x, y)

        if not skip_flip:
            pygame.display.flip()

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
            "waiting_for_reset": False
        }

        def reset_game():
            state["board"], _ = env.reset()
            state["score"] = 0
            state["step_count"] = 0
            state["game_count"] += 1
            state["done"] = False
            state["waiting_for_reset"] = False
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

            if action and action in ["UP", "DOWN", "LEFT", "RIGHT"] and not state["done"]:
                legal_moves = env.legal_moves(state["board"])
                if action in legal_moves:
                    state["board"], reward, state["done"], info = env.step(action)
                    state["score"] += int(reward)
                    state["step_count"] += 1

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
                    mode_text="Human Play - Arrow Keys/WASD to move, R to reset, ESC to quit"
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

        # Replay state
        state = {
            "current_step_idx": 0,
            "playing": True,
            "last_step_time": pygame.time.get_ticks(),
            "step_delay_ms": int(1000 / speed)
        }

        def update(action: Optional[str]) -> bool:
            if action == "RESET":
                state["current_step_idx"] = 0
                state["last_step_time"] = pygame.time.get_ticks()
            elif action == "RIGHT":
                state["current_step_idx"] = min(state["current_step_idx"] + 1, len(steps) - 1)
                state["last_step_time"] = pygame.time.get_ticks()
            elif action == "LEFT":
                state["current_step_idx"] = max(state["current_step_idx"] - 1, 0)
                state["last_step_time"] = pygame.time.get_ticks()
            elif action == "SPACE":
                state["playing"] = not state["playing"]
                state["last_step_time"] = pygame.time.get_ticks()

            # Auto-advance
            if state["playing"] and state["current_step_idx"] < len(steps) - 1:
                current_time = pygame.time.get_ticks()
                if current_time - state["last_step_time"] >= state["step_delay_ms"]:
                    state["current_step_idx"] += 1
                    state["last_step_time"] = current_time
            return True

        def render():
            step = steps[state["current_step_idx"]]
            board = decode_board_log2(step["board"], board_size)
            score = step.get("score", 0)
            game_id_disp = game_log.get('game_id', 'unknown')
            mode_text = f"Replay {game_id_disp} ({state['current_step_idx'] + 1}/{len(steps)}) - Space: pause, Left/Right: step, R: restart"
            self.render(board, score, mode_text=mode_text)

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
            "last_move_time": pygame.time.get_ticks()
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

            current_time = pygame.time.get_ticks()
            if not state["done"] and (current_time - state["last_move_time"] >= delay_ms):
                legal_moves = env.legal_moves(state["board"])
                if legal_moves:
                    agent_action = agent.choose_action(state["board"], legal_moves)
                    state["board"], reward, state["done"], info = env.step(agent_action)
                    state["score"] += int(reward)
                    state["step_count"] += 1

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
                self.render(state["board"], state["score"], mode_text=mode_text)

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

    ui = PygameUI(board_size=board_size, config=config)
    ui.run_agent(agent, env, logger=logger, seed=seed, move_delay_ms=move_delay_ms)
