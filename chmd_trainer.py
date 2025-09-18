import os
import shutil
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn

from chmd_core import (
    APP_PATHS,
    BOOKMARKS,
    CLOCK,
    COMMENTARY,
    DIFFICULTY_MODEL,
    DIAGNOSTICS,
    ENGINE_QUEUE,
    ENGINE_SETTINGS,
    EVENT_BUS,
    EVALUATIONS,
    HINT_CACHE,
    HOTKEYS,
    LIFECYCLE,
    LOGGER,
    NOTES,
    PGN_SERIALIZER,
    PROFILE_MANAGER,
    SCOREBOARD,
    SETTINGS,
    STREAK_TRACKER,
    TASK_SCHEDULER,
    TELEMETRY,
    TRAINER_STATE,
    TRAINING_PLANNER,
    VARIATION_TREE,
    EngineCommand,
    EngineResult,
    HintCacheEntry,
    MoveRecord,
    validate_fen,
)
from chmd_core import dump_diagnostics, initialize_core, shutdown_core
from chmd_vision import start_async_capture, stop_async_capture
from chmd_ui import ApplicationController


# --------------------------------------------------------------------------------------
# Engine process handling
# --------------------------------------------------------------------------------------


class EngineFailure(RuntimeError):
    pass


@dataclass
class EngineInfo:
    name: str
    author: str
    options: Dict[str, str] = field(default_factory=dict)


class StockfishController(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="StockfishController", daemon=True)
        self.process: Optional[subprocess.Popen[str]] = None
        self.command_queue: "queue.Queue[EngineCommand]" = queue.Queue()
        self.running = True
        self.info = EngineInfo("", "")
        self.lock = threading.Lock()
        self.current_fen = "startpos"
        self._connect()

    def _find_engine(self) -> str:
        path = SETTINGS.get("engine_path", "stockfish")
        if os.path.isfile(path):
            return path
        candidates = [path, "stockfish", "/usr/bin/stockfish", "/usr/local/bin/stockfish"]
        for candidate in candidates:
            if shutil.which(candidate):
                return candidate
        raise EngineFailure("Stockfish binary not found")

    def _connect(self) -> None:
        engine_path = self._find_engine()
        try:
            self.process = subprocess.Popen(
                [engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise EngineFailure(f"Failed to launch engine: {exc}")
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise EngineFailure("Process missing streams")
        self._send("uci")
        start_time = time.time()
        while True:
            line = self.process.stdout.readline().strip()
            if not line:
                if time.time() - start_time > 5:
                    raise EngineFailure("Engine did not respond to uci")
                continue
            if line.startswith("id name"):
                self.info.name = line.split("id name", 1)[1].strip()
            elif line.startswith("id author"):
                self.info.author = line.split("id author", 1)[1].strip()
            elif line.startswith("option "):
                parts = line.split("option ", 1)[1]
                option_name = parts.split("name ", 1)[1].split(" type", 1)[0].strip()
                self.info.options[option_name] = line
            elif line == "uciok":
                break
        self._configure()
        LOGGER.info("Engine initialized: %s by %s", self.info.name, self.info.author)

    def _configure(self) -> None:
        self.set_option("Threads", str(SETTINGS.get("engine_threads", 4)))
        self.set_option("Hash", str(SETTINGS.get("engine_hash", 512)))
        self.set_option("MultiPV", str(SETTINGS.get("multi_pv", 3)))
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, command: str) -> None:
        if not self.process or not self.process.stdin:
            return
        LOGGER.debug("Engine <= %s", command)
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _wait_for(self, marker: str, timeout: float = 5.0) -> None:
        if not self.process or not self.process.stdout:
            return
        start = time.time()
        while True:
            line = self.process.stdout.readline().strip()
            LOGGER.debug("Engine => %s", line)
            if marker in line:
                return
            if time.time() - start > timeout:
                raise EngineFailure(f"Timeout waiting for {marker}")

    def run(self) -> None:
        while self.running:
            try:
                command = self.command_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if command.command == "quit":
                self._send("quit")
                break
            if command.command == "analyze":
                self._analyze(command.payload)
            elif command.command == "setoption":
                self.set_option(command.payload["name"], command.payload["value"])

    def set_option(self, name: str, value: str) -> None:
        self._send(f"setoption name {name} value {value}")

    def position(self, fen: str) -> None:
        if not validate_fen(fen):
            LOGGER.warning("Invalid FEN passed to engine: %s", fen)
            return
        self.current_fen = fen
        self._send(f"position fen {fen}")

    def go(self, depth: int) -> None:
        self._send(f"go depth {depth}")

    def _parse_info(self, line: str) -> Optional[EngineResult]:
        if " pv " not in line:
            return None
        parts = line.split()
        depth = int(parts[1]) if "depth" in parts else 0
        score = 0.0
        nodes = 0
        nps = 0
        best_move = ""
        pv_moves: List[str] = []
        for index, token in enumerate(parts):
            if token == "score" and index + 2 < len(parts):
                if parts[index + 1] == "cp":
                    score = float(parts[index + 2]) / 100.0
                elif parts[index + 1] == "mate":
                    mate_value = int(parts[index + 2])
                    score = 100.0 if mate_value > 0 else -100.0
            elif token == "nodes" and index + 1 < len(parts):
                nodes = int(parts[index + 1])
            elif token == "nps" and index + 1 < len(parts):
                nps = int(parts[index + 1])
        pv_index = line.index(" pv ") + 4
        pv_text = line[pv_index:]
        pv_moves = pv_text.strip().split()
        if pv_moves:
            best_move = pv_moves[0]
        return EngineResult(self.current_fen, best_move, [" ".join(pv_moves)], score, depth, nodes, nps)

    def _analyze(self, payload: Dict[str, object]) -> None:
        fen = payload.get("fen", self.current_fen)
        depth = int(payload.get("depth", SETTINGS.get("analysis_depth", 18)))
        multi_pv = int(payload.get("multi_pv", SETTINGS.get("multi_pv", 3)))
        self.position(str(fen))
        self._send(f"setoption name MultiPV value {multi_pv}")
        self._send(f"go depth {depth}")
        if not self.process or not self.process.stdout:
            return
        pv_lines: Dict[int, Tuple[str, float]] = {}
        start_time = time.time()
        while True:
            line = self.process.stdout.readline().strip()
            if not line:
                if time.time() - start_time > 10:
                    break
                continue
            LOGGER.debug("Engine => %s", line)
            if line.startswith("info") and " pv " in line:
                result = self._parse_info(line)
                if result:
                    pv_index = int(line.split("multipv", 1)[1].split()[0]) if "multipv" in line else 1
                    pv_lines[pv_index] = (" ".join(result.pv_lines[0].split()), result.evaluation)
                    ENGINE_QUEUE.push_result(result)
                    HINT_CACHE.update(
                        HintCacheEntry(
                            result.fen,
                            result.best_move,
                            result.pv_lines,
                            result.evaluation,
                            result.depth,
                        )
                    )
                    EVALUATIONS.add(result.evaluation)
                    VARIATION_TREE.update([pv for pv, _ in pv_lines.values()], [score for _, score in pv_lines.values()])
                    TELEMETRY.log("engine_eval", result.evaluation)
            elif line.startswith("bestmove"):
                parts = line.split()
                best_move = parts[1]
                ENGINE_QUEUE.push_result(
                    EngineResult(
                        fen=str(fen),
                        best_move=best_move,
                        pv_lines=[" ".join(result[0] for result in pv_lines.values())],
                        evaluation=pv_lines.get(1, ("", 0.0))[1],
                        depth=depth,
                        nodes=0,
                        nps=0,
                    )
                )
                break

    def stop_engine(self) -> None:
        self.running = False
        self.command_queue.put(EngineCommand("quit"))
        if self.process:
            self.process.terminate()


# --------------------------------------------------------------------------------------
# Game management
# --------------------------------------------------------------------------------------


@dataclass
class GameSession:
    board: chess.Board = field(default_factory=chess.Board)
    history: List[MoveRecord] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    mode: str = "player_vs_ai"
    active: bool = True

    def add_move(self, move: chess.Move, evaluation: float, best_line: str) -> None:
        san = self.board.san(move)
        self.board.push(move)
        record = MoveRecord(
            move_number=len(self.history) + 1,
            san=san,
            fen=self.board.fen(),
            evaluation=evaluation,
            best_line=best_line,
            comment=COMMENTARY.annotate(evaluation),
        )
        self.history.append(record)
        TRAINER_STATE.add_move(record)

    def undo(self) -> None:
        if self.board.move_stack:
            self.board.pop()
            self.history.pop()
            TRAINER_STATE.set_fen(self.board.fen())


class GameManager:
    def __init__(self) -> None:
        self.session = GameSession()
        self.lock = threading.Lock()

    def reset(self, fen: str = chess.STARTING_FEN) -> None:
        with self.lock:
            self.session = GameSession(board=chess.Board(fen))
            TRAINER_STATE.reset()
            TRAINER_STATE.set_fen(fen)
            LOGGER.info("Game reset to %s", fen)

    def apply_move(self, move: chess.Move, evaluation: float = 0.0, best_line: str = "") -> None:
        with self.lock:
            self.session.add_move(move, evaluation, best_line)
            if self.session.board.is_game_over():
                result = self.session.board.result()
                PROFILE_MANAGER.record(result)
                SCOREBOARD.update("Player", DIFFICULTY_MODEL.rating)
                TRAINER_STATE.history.result = result

    def legal_moves(self) -> List[chess.Move]:
        with self.lock:
            return list(self.session.board.legal_moves)

    def current_board(self) -> chess.Board:
        with self.lock:
            return self.session.board.copy()


GAME_MANAGER = GameManager()


# --------------------------------------------------------------------------------------
# Training mode controller
# --------------------------------------------------------------------------------------


class TrainingController:
    def __init__(self) -> None:
        self.current_task = TRAINING_PLANNER.random_task()
        self.active = False

    def start(self) -> None:
        self.current_task = TRAINING_PLANNER.random_task()
        GAME_MANAGER.reset(self.current_task.fen)
        self.active = True
        EVENT_BUS.publish("training.started", {"name": self.current_task.name})

    def verify_solution(self, move: str) -> bool:
        if not self.active:
            return False
        correct = move in self.current_task.solution
        EVENT_BUS.publish("tactic.result", {"correct": correct})
        return correct


TRAINING_CONTROLLER = TrainingController()


# --------------------------------------------------------------------------------------
# Engine integration helpers
# --------------------------------------------------------------------------------------


def submit_analysis(fen: str) -> None:
    depth = DIFFICULTY_MODEL.target_depth()
    ENGINE_QUEUE.submit(EngineCommand("analyze", {"fen": fen, "depth": depth, "multi_pv": SETTINGS.get("multi_pv", 3)}))


def on_engine_result(topic: str, payload: Dict[str, object]) -> None:
    result = payload.get("result")
    if not isinstance(result, EngineResult):
        return
    hints = HINT_CACHE.get(result.fen)
    if hints is None or hints.best_move != result.best_move:
        HINT_CACHE.update(HintCacheEntry(result.fen, result.best_move, result.pv_lines, result.evaluation, result.depth))
    TELEMETRY.log("engine_depth", result.depth)
    DIAGNOSTICS.set("engine_eval", result.evaluation, "Engine Evaluation")


EVENT_BUS.subscribe("engine.result", on_engine_result)


# --------------------------------------------------------------------------------------
# Move suggestion logic
# --------------------------------------------------------------------------------------


class HintController:
    def __init__(self) -> None:
        self.visible = SETTINGS.get("hints_enabled", True)

    def toggle(self) -> None:
        self.visible = not self.visible
        SETTINGS.set("hints_enabled", self.visible)

    def best_move(self, fen: str) -> Optional[str]:
        if not self.visible:
            return None
        entry = HINT_CACHE.get(fen)
        return entry.best_move if entry else None


HINT_CONTROLLER = HintController()


# --------------------------------------------------------------------------------------
# Command handlers for hotkeys
# --------------------------------------------------------------------------------------


def register_hotkeys() -> None:
    HOTKEYS.register("Ctrl+N", "New game", lambda: GAME_MANAGER.reset())
    HOTKEYS.register("Ctrl+R", "Restart training", TRAINING_CONTROLLER.start)
    HOTKEYS.register("Ctrl+H", "Toggle hints", HINT_CONTROLLER.toggle)


register_hotkeys()


# --------------------------------------------------------------------------------------
# Engine worker thread binding queue to process
# --------------------------------------------------------------------------------------


class EngineWorker(threading.Thread):
    def __init__(self, controller: StockfishController) -> None:
        super().__init__(name="EngineWorker", daemon=True)
        self.controller = controller
        self.running = True

    def run(self) -> None:
        while self.running:
            command = ENGINE_QUEUE.get_command(timeout=0.1)
            if command is None:
                continue
            if command.command == "analyze":
                self.controller.command_queue.put(command)
            elif command.command == "setoption":
                self.controller.command_queue.put(command)

    def stop(self) -> None:
        self.running = False


# --------------------------------------------------------------------------------------
# PGN export utilities
# --------------------------------------------------------------------------------------


def export_current_game() -> None:
    history = TRAINER_STATE.history
    PGN_SERIALIZER.save(history, APP_PATHS.default_pgn)


# --------------------------------------------------------------------------------------
# Event handlers for trainer state
# --------------------------------------------------------------------------------------


def on_fen_changed(topic: str, payload: Dict[str, object]) -> None:
    fen = payload.get("fen")
    if isinstance(fen, str) and validate_fen(fen):
        submit_analysis(fen)
        TELEMETRY.log("fen_update", time.time())


EVENT_BUS.subscribe("trainer.fen.update", on_fen_changed)


# --------------------------------------------------------------------------------------
# Main trainer orchestration
# --------------------------------------------------------------------------------------


class TrainerOrchestrator:
    def __init__(self) -> None:
        self.engine_controller = StockfishController()
        self.engine_worker = EngineWorker(self.engine_controller)
        self.ui_controller = ApplicationController()
        self.threads: List[threading.Thread] = []

    def start(self) -> None:
        initialize_core()
        self.engine_controller.start()
        self.engine_worker.start()
        start_async_capture()
        TRAINING_CONTROLLER.start()
        self.ui_controller.start()

    def stop(self) -> None:
        stop_async_capture()
        self.engine_controller.stop_engine()
        self.engine_worker.stop()
        shutdown_core()


# --------------------------------------------------------------------------------------
# CLI integration
# --------------------------------------------------------------------------------------


def main() -> None:
    orchestrator = TrainerOrchestrator()
    try:
        orchestrator.start()
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        orchestrator.stop()


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------
# Move planning and automation
# --------------------------------------------------------------------------------------


@dataclass
class MovePlan:
    source: str
    target: str
    promotion: Optional[str] = None
    score: float = 0.0


class MovePlanner:
    def __init__(self) -> None:
        self.plan: Optional[MovePlan] = None
        self.lock = threading.Lock()

    def update_plan(self, fen: str) -> None:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            self.plan = None
            return
        entry = HINT_CACHE.get(fen)
        if entry:
            move = chess.Move.from_uci(entry.best_move)
        else:
            move = legal_moves[0]
        self.plan = MovePlan(move.uci()[:2], move.uci()[2:4], move.uci()[4:] or None, entry.evaluation if entry else 0.0)

    def consume(self) -> Optional[MovePlan]:
        with self.lock:
            plan = self.plan
            self.plan = None
            return plan


MOVE_PLANNER = MovePlanner()


# --------------------------------------------------------------------------------------
# Player controllers
# --------------------------------------------------------------------------------------


class PlayerController:
    def __init__(self, color: chess.Color) -> None:
        self.color = color

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        raise NotImplementedError


class HumanPlayer(PlayerController):
    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        return None


class EnginePlayer(PlayerController):
    def __init__(self, color: chess.Color, controller: StockfishController) -> None:
        super().__init__(color)
        self.controller = controller

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        fen = board.fen()
        entry = HINT_CACHE.get(fen)
        if entry:
            return chess.Move.from_uci(entry.best_move)
        submit_analysis(fen)
        time.sleep(0.5)
        entry = HINT_CACHE.get(fen)
        return chess.Move.from_uci(entry.best_move) if entry else None


# --------------------------------------------------------------------------------------
# Auto play logic
# --------------------------------------------------------------------------------------


class AutoPlayThread(threading.Thread):
    def __init__(self, game_manager: GameManager, white: PlayerController, black: PlayerController) -> None:
        super().__init__(name="AutoPlay", daemon=True)
        self.game_manager = game_manager
        self.white = white
        self.black = black
        self.running = True

    def run(self) -> None:
        while self.running:
            board = self.game_manager.current_board()
            if board.is_game_over():
                break
            player = self.white if board.turn == chess.WHITE else self.black
            move = player.choose_move(board)
            if move is None:
                time.sleep(0.2)
                continue
            self.game_manager.apply_move(move, 0.0, "")
            TRAINER_STATE.set_fen(self.game_manager.current_board().fen())
            submit_analysis(self.game_manager.current_board().fen())
            time.sleep(0.1)

    def stop(self) -> None:
        self.running = False


# --------------------------------------------------------------------------------------
# PGN utilities
# --------------------------------------------------------------------------------------


class PGNAnalyzer:
    def __init__(self) -> None:
        self.cache: Dict[str, int] = {}

    def analyze(self, path: Path) -> Dict[str, int]:
        if not path.exists():
            return {}
        results: Dict[str, int] = {"white": 0, "black": 0, "draw": 0}
        with path.open("r", encoding="utf-8") as handle:
            game = chess.pgn.read_game(handle)
            while game:
                result = game.headers.get("Result", "*")
                if result == "1-0":
                    results["white"] += 1
                elif result == "0-1":
                    results["black"] += 1
                elif result == "1/2-1/2":
                    results["draw"] += 1
                game = chess.pgn.read_game(handle)
        self.cache[str(path)] = sum(results.values())
        return results


PGN_ANALYZER = PGNAnalyzer()


# --------------------------------------------------------------------------------------
# Training evaluator to score user performance
# --------------------------------------------------------------------------------------


class TrainingEvaluator:
    def __init__(self) -> None:
        self.successes = 0
        self.failures = 0
        self.lock = threading.Lock()

    def record(self, correct: bool) -> None:
        with self.lock:
            if correct:
                self.successes += 1
            else:
                self.failures += 1
            SCOREBOARD.update("Training", self.successes - self.failures)

    def accuracy(self) -> float:
        with self.lock:
            total = self.successes + self.failures
            return self.successes / total if total else 0.0


TRAINING_EVALUATOR = TrainingEvaluator()


EVENT_BUS.subscribe(
    "tactic.result",
    lambda topic, payload: TRAINING_EVALUATOR.record(bool(payload.get("correct", False))),
)


# --------------------------------------------------------------------------------------
# Replay controller for saved games
# --------------------------------------------------------------------------------------


class ReplayController(threading.Thread):
    def __init__(self, path: Path) -> None:
        super().__init__(name="ReplayController", daemon=True)
        self.path = path
        self.running = True

    def run(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            game = chess.pgn.read_game(handle)
            if not game:
                return
            board = game.board()
            for move in game.mainline_moves():
                if not self.running:
                    break
                board.push(move)
                TRAINER_STATE.set_fen(board.fen())
                submit_analysis(board.fen())
                time.sleep(1.0)

    def stop(self) -> None:
        self.running = False


# --------------------------------------------------------------------------------------
# Engine benchmarking
# --------------------------------------------------------------------------------------


class EngineBenchmark(threading.Thread):
    def __init__(self, controller: StockfishController) -> None:
        super().__init__(name="EngineBenchmark", daemon=True)
        self.controller = controller
        self.results: List[Tuple[int, float]] = []
        self.running = True

    def run(self) -> None:
        positions = [chess.STARTING_FEN, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
        for index, fen in enumerate(positions, start=1):
            if not self.running:
                break
            start = time.time()
            self.controller.command_queue.put(EngineCommand("analyze", {"fen": fen, "depth": 14, "multi_pv": 1}))
            time.sleep(2.0)
            entry = HINT_CACHE.get(fen)
            if entry:
                duration = time.time() - start
                self.results.append((index, duration))
                DIAGNOSTICS.set(f"benchmark_{index}", duration, "Benchmark seconds")

    def stop(self) -> None:
        self.running = False


# --------------------------------------------------------------------------------------
# Commentary log
# --------------------------------------------------------------------------------------


class CommentaryLog:
    def __init__(self) -> None:
        self.entries: List[str] = []

    def add(self, text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}"
        self.entries.append(line)
        if len(self.entries) > 200:
            self.entries = self.entries[-200:]
        LOGGER.info("Commentary: %s", line)

    def export(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(self.entries))


COMMENTARY_LOG = CommentaryLog()


# --------------------------------------------------------------------------------------
# Integration with events for commentary
# --------------------------------------------------------------------------------------


def on_overlay_update(topic: str, payload: Dict[str, object]) -> None:
    fen = payload.get("fen")
    best_move = payload.get("best_move")
    if isinstance(fen, str) and isinstance(best_move, str):
        COMMENTARY_LOG.add(f"Best move {best_move} for position {fen[:32]}…")


EVENT_BUS.subscribe("overlay.update", on_overlay_update)


# --------------------------------------------------------------------------------------
# Diagnostics refresh loop
# --------------------------------------------------------------------------------------


class DiagnosticsLoop(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="DiagnosticsLoop", daemon=True)
        self.running = True

    def run(self) -> None:
        while self.running:
            dump_diagnostics(APP_PATHS.root / "live_diagnostics.json")
            time.sleep(15.0)

    def stop(self) -> None:
        self.running = False


DIAGNOSTICS_LOOP = DiagnosticsLoop()


# --------------------------------------------------------------------------------------
# Study planner bridging bookmarks and notes
# --------------------------------------------------------------------------------------


class StudyPlanner:
    def __init__(self) -> None:
        self.plan: List[Tuple[str, str]] = []

    def refresh(self) -> None:
        self.plan = BOOKMARKS.list()
        if not self.plan:
            NOTES.add(chess.STARTING_FEN, "Study the basics of development")
            self.plan = BOOKMARKS.list()

    def schedule(self) -> List[Tuple[str, str]]:
        self.refresh()
        return self.plan


STUDY_PLANNER = StudyPlanner()


# --------------------------------------------------------------------------------------
# Trainer CLI for headless usage
# --------------------------------------------------------------------------------------


class TrainerCLI:
    def __init__(self) -> None:
        self.orchestrator = TrainerOrchestrator()
        self.replay: Optional[ReplayController] = None

    def run(self) -> None:
        self.orchestrator.start()

    def replay_game(self, path: Path) -> None:
        self.replay = ReplayController(path)
        self.replay.start()

    def stop(self) -> None:
        if self.replay:
            self.replay.stop()
        self.orchestrator.stop()


# --------------------------------------------------------------------------------------
# Advanced trainer orchestrator with diagnostics
# --------------------------------------------------------------------------------------


class AdvancedTrainerOrchestrator(TrainerOrchestrator):
    def __init__(self) -> None:
        super().__init__()
        self.auto_play: Optional[AutoPlayThread] = None
        self.diagnostics_loop = DIAGNOSTICS_LOOP
        self.benchmark = EngineBenchmark(self.engine_controller)

    def start(self) -> None:  # type: ignore[override]
        DIAGNOSTICS_LOOP.start()
        super().start()
        white = EnginePlayer(chess.WHITE, self.engine_controller)
        black = EnginePlayer(chess.BLACK, self.engine_controller)
        self.auto_play = AutoPlayThread(GAME_MANAGER, white, black)
        self.auto_play.start()
        self.benchmark.start()

    def stop(self) -> None:  # type: ignore[override]
        if self.auto_play:
            self.auto_play.stop()
        self.benchmark.stop()
        self.diagnostics_loop.stop()
        super().stop()


# --------------------------------------------------------------------------------------
# Public factory for orchestrator selection
# --------------------------------------------------------------------------------------


def build_orchestrator(advanced: bool = False) -> TrainerOrchestrator:
    if advanced:
        return AdvancedTrainerOrchestrator()
    return TrainerOrchestrator()


__all__ = [
    "TrainerOrchestrator",
    "AdvancedTrainerOrchestrator",
    "build_orchestrator",
    "main",
]

# --------------------------------------------------------------------------------------
# Session recorder for analytics
# --------------------------------------------------------------------------------------


class SessionRecorder:
    def __init__(self) -> None:
        self.path = APP_PATHS.root / "session_log.txt"
        self.lock = threading.Lock()

    def record(self, text: str) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(f"{time.time():.3f}: {text}\n")


SESSION_RECORDER = SessionRecorder()


def log_move(move: chess.Move, evaluation: float) -> None:
    SESSION_RECORDER.record(f"Move {move.uci()} eval {evaluation:+.2f}")


EVENT_BUS.subscribe(
    "overlay.update",
    lambda topic, payload: SESSION_RECORDER.record(f"Overlay {payload.get('best_move', '')}"),
)


# --------------------------------------------------------------------------------------
# Move annotation service
# --------------------------------------------------------------------------------------


class MoveAnnotationService:
    def __init__(self) -> None:
        self.cache: Dict[str, str] = {}

    def annotate(self, move: chess.Move, board: chess.Board) -> str:
        key = f"{board.fen()}|{move.uci()}"
        if key in self.cache:
            return self.cache[key]
        board.push(move)
        fen = board.fen()
        entry = HINT_CACHE.get(fen)
        annotation = COMMENTARY.annotate(entry.evaluation if entry else 0.0)
        board.pop()
        self.cache[key] = annotation
        return annotation


MOVE_ANNOTATION = MoveAnnotationService()


# --------------------------------------------------------------------------------------
# Adaptive difficulty scheduler
# --------------------------------------------------------------------------------------


class DifficultyScheduler:
    def __init__(self) -> None:
        self.history: List[float] = []

    def record(self, evaluation: float) -> None:
        self.history.append(evaluation)
        if len(self.history) > 50:
            self.history = self.history[-50:]
        avg = sum(self.history) / len(self.history)
        if avg > 1.0:
            SETTINGS.set("analysis_depth", SETTINGS.get("analysis_depth", 18) + 1)
        elif avg < -1.0:
            SETTINGS.set("analysis_depth", max(10, SETTINGS.get("analysis_depth", 18) - 1))


DIFFICULTY_SCHEDULER = DifficultyScheduler()


EVENT_BUS.subscribe(
    "overlay.update",
    lambda topic, payload: DIFFICULTY_SCHEDULER.record(float(payload.get("evaluation", 0.0))),
)


# --------------------------------------------------------------------------------------
# Move ingestion from UI events
# --------------------------------------------------------------------------------------


def on_user_move(topic: str, payload: Dict[str, object]) -> None:
    move_uci = payload.get("move")
    fen = payload.get("fen")
    if not isinstance(move_uci, str) or not isinstance(fen, str):
        return
    board = GAME_MANAGER.current_board()
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        LOGGER.warning("Invalid move from UI: %s", move_uci)
        return
    if move not in board.legal_moves:
        LOGGER.warning("Illegal move attempted: %s", move_uci)
        return
    board.push(move)
    evaluation = EVALUATIONS.last()
    best_line = " ".join(VARIATION_TREE.best_line())
    GAME_MANAGER.apply_move(move, evaluation, best_line)
    log_move(move, evaluation)
    TRAINER_STATE.set_fen(board.fen())


EVENT_BUS.subscribe("ui.move", on_user_move)


# --------------------------------------------------------------------------------------
# Auto-save hooks
# --------------------------------------------------------------------------------------


def autosave_loop() -> None:
    export_current_game()
    COMMENTARY_LOG.export(APP_PATHS.root / "commentary.log")


TASK_SCHEDULER.post("autosave", autosave_loop)
