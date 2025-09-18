import json
import logging
import os
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


# --------------------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------------------

def configure_logging(log_path: Optional[Path] = None) -> logging.Logger:
    """Configure a process wide logger with rotating file support."""
    logger = logging.getLogger("chmd.core")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(threadName)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_path:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            logger.warning("Could not create log file at %s", log_path)
    logger.debug("Logger configured")
    return logger


LOGGER = configure_logging(Path.home() / ".chmd" / "trainer.log")


# --------------------------------------------------------------------------------------
# Utility values and helper primitives
# --------------------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def rolling_average(buffer: List[float]) -> float:
    if not buffer:
        return 0.0
    return sum(buffer) / float(len(buffer))


def ensure_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


class AtomicFlag:
    def __init__(self) -> None:
        self._flag = False
        self._lock = threading.Lock()

    def set(self, value: bool) -> None:
        with self._lock:
            self._flag = value
            LOGGER.debug("AtomicFlag set to %s", value)

    def is_set(self) -> bool:
        with self._lock:
            return self._flag


@dataclass
class AppPaths:
    root: Path = field(default_factory=lambda: Path.home() / ".chmd")

    def __post_init__(self) -> None:
        ensure_directory(self.root)
        ensure_directory(self.root / "captures")
        ensure_directory(self.root / "logs")
        ensure_directory(self.root / "pgn")

    @property
    def settings_file(self) -> Path:
        return self.root / "settings.json"

    @property
    def profile_file(self) -> Path:
        return self.root / "profile.json"

    @property
    def default_pgn(self) -> Path:
        return self.root / "pgn" / "saved_games.pgn"

    @property
    def capture_dir(self) -> Path:
        return self.root / "captures"


APP_PATHS = AppPaths()


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------


DEFAULT_SETTINGS: Dict[str, Any] = {
    "engine_path": "stockfish",
    "engine_threads": 4,
    "engine_hash": 512,
    "multi_pv": 3,
    "analysis_depth": 20,
    "capture_source": "screen",
    "board_theme": "default",
    "overlay_opacity": 0.85,
    "overlay_font": "Roboto",
    "overlay_size": 1.0,
    "training_mode": "tactics",
    "difficulty": 5,
    "move_time": 2.5,
    "auto_save": True,
    "max_saved_games": 50,
}


class SettingsManager:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._settings: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self.load()

    def _write_default(self) -> None:
        self._settings = dict(DEFAULT_SETTINGS)
        try:
            with self._path.open("w", encoding="utf-8") as handle:
                json.dump(self._settings, handle, indent=2)
            LOGGER.info("Default settings written to %s", self._path)
        except OSError as exc:
            LOGGER.error("Failed to write default settings: %s", exc)

    def load(self) -> None:
        if not self._path.exists():
            self._write_default()
            return
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("Settings file corrupt")
            self._settings = {**DEFAULT_SETTINGS, **data}
            LOGGER.debug("Settings loaded from %s", self._path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.error("Failed to load settings: %s, rewriting defaults", exc)
            self._write_default()

    def save(self) -> None:
        with self._lock:
            try:
                with self._path.open("w", encoding="utf-8") as handle:
                    json.dump(self._settings, handle, indent=2)
                LOGGER.debug("Settings saved to %s", self._path)
            except OSError as exc:
                LOGGER.error("Could not save settings: %s", exc)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._settings[key] = value
            LOGGER.debug("Setting %s updated to %s", key, value)
            self.save()

    def all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._settings)


SETTINGS = SettingsManager(APP_PATHS.settings_file)


# --------------------------------------------------------------------------------------
# Profilers and instrumentation utilities
# --------------------------------------------------------------------------------------


@dataclass
class Sample:
    timestamp: float
    value: float


class RollingMetric:
    def __init__(self, maxlen: int = 120) -> None:
        self.maxlen = maxlen
        self.samples: List[Sample] = []
        self._lock = threading.Lock()

    def add(self, value: float) -> None:
        with self._lock:
            now = time.time()
            self.samples.append(Sample(now, value))
            if len(self.samples) > self.maxlen:
                self.samples = self.samples[-self.maxlen :]
            LOGGER.debug("Metric sample appended: %s", value)

    def average(self) -> float:
        with self._lock:
            return rolling_average([sample.value for sample in self.samples])

    def min(self) -> float:
        with self._lock:
            if not self.samples:
                return 0.0
            return min(sample.value for sample in self.samples)

    def max(self) -> float:
        with self._lock:
            if not self.samples:
                return 0.0
            return max(sample.value for sample in self.samples)

    def recent(self, seconds: float) -> List[Sample]:
        threshold = time.time() - seconds
        with self._lock:
            return [sample for sample in self.samples if sample.timestamp >= threshold]


class Profiler:
    def __init__(self) -> None:
        self.metrics: Dict[str, RollingMetric] = {}
        self._stack: List[Tuple[str, float]] = []
        self._lock = threading.Lock()

    def start(self, name: str) -> None:
        with self._lock:
            self._stack.append((name, time.perf_counter()))

    def stop(self, name: str) -> float:
        with self._lock:
            if not self._stack:
                return 0.0
            stack_name, start_time = self._stack.pop()
            if stack_name != name:
                LOGGER.warning("Profiler mismatch: expected %s but got %s", stack_name, name)
            duration = time.perf_counter() - start_time
            metric = self.metrics.setdefault(name, RollingMetric())
            metric.add(duration)
            LOGGER.debug("Profiler %s recorded duration %.6f", name, duration)
            return duration

    def summary(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            result: Dict[str, Dict[str, float]] = {}
            for name, metric in self.metrics.items():
                result[name] = {
                    "avg": metric.average(),
                    "min": metric.min(),
                    "max": metric.max(),
                    "samples": len(metric.samples),
                }
            return result


PROFILER = Profiler()


# --------------------------------------------------------------------------------------
# Event bus
# --------------------------------------------------------------------------------------


EventCallback = Callable[[str, Dict[str, Any]], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[EventCallback]] = {}
        self._lock = threading.RLock()

    def subscribe(self, topic: str, callback: EventCallback) -> None:
        with self._lock:
            callbacks = self._subscribers.setdefault(topic, [])
            callbacks.append(callback)
            LOGGER.debug("Subscribed %s to topic %s", callback, topic)

    def unsubscribe(self, topic: str, callback: EventCallback) -> None:
        with self._lock:
            if topic not in self._subscribers:
                return
            try:
                self._subscribers[topic].remove(callback)
                LOGGER.debug("Unsubscribed %s from %s", callback, topic)
            except ValueError:
                LOGGER.debug("Callback not found for topic %s", topic)

    def publish(self, topic: str, payload: Optional[Dict[str, Any]] = None) -> None:
        payload = payload or {}
        callbacks: List[EventCallback]
        with self._lock:
            callbacks = list(self._subscribers.get(topic, []))
        for callback in callbacks:
            try:
                callback(topic, payload)
            except Exception:
                LOGGER.exception("Event callback failure on %s", topic)


EVENT_BUS = EventBus()


# --------------------------------------------------------------------------------------
# Task scheduler
# --------------------------------------------------------------------------------------


class ThreadedTaskScheduler:
    def __init__(self) -> None:
        self._queue: "queue.Queue[Tuple[str, Callable[[], None]]]" = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="TaskScheduler", daemon=True)
        self._running = AtomicFlag()

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set(True)
        self._thread.start()
        LOGGER.info("ThreadedTaskScheduler started")

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.set(False)
        self._queue.put(("__stop__", lambda: None))
        self._thread.join(timeout=2.0)
        LOGGER.info("ThreadedTaskScheduler stopped")

    def post(self, name: str, callback: Callable[[], None]) -> None:
        self._queue.put((name, callback))

    def _worker(self) -> None:
        while True:
            name, callback = self._queue.get()
            if name == "__stop__":
                break
            PROFILER.start(name)
            try:
                callback()
            except Exception:
                LOGGER.exception("Task %s raised an exception", name)
            finally:
                PROFILER.stop(name)


TASK_SCHEDULER = ThreadedTaskScheduler()


# --------------------------------------------------------------------------------------
# Chess data structures
# --------------------------------------------------------------------------------------


FILES = "abcdefgh"
RANKS = "12345678"
PIECE_SYMBOLS = {
    "P": "white_pawn",
    "N": "white_knight",
    "B": "white_bishop",
    "R": "white_rook",
    "Q": "white_queen",
    "K": "white_king",
    "p": "black_pawn",
    "n": "black_knight",
    "b": "black_bishop",
    "r": "black_rook",
    "q": "black_queen",
    "k": "black_king",
}


@dataclass
class MoveRecord:
    move_number: int
    san: str
    fen: str
    comment: str = ""
    evaluation: Optional[float] = None
    best_line: Optional[str] = None


@dataclass
class GameHistory:
    records: List[MoveRecord] = field(default_factory=list)
    result: str = "*"

    def append(self, record: MoveRecord) -> None:
        LOGGER.debug("Appending move record %s", record)
        self.records.append(record)

    def clear(self) -> None:
        LOGGER.info("Clearing game history")
        self.records.clear()
        self.result = "*"

    def to_pgn(self) -> str:
        lines: List[str] = []
        white_to_move = True
        move_counter = 1
        for record in self.records:
            prefix = f"{move_counter}. " if white_to_move else ""
            lines.append(prefix + record.san)
            white_to_move = not white_to_move
            if white_to_move:
                move_counter += 1
        lines.append(self.result)
        return " ".join(lines)

    def last_fen(self) -> str:
        if not self.records:
            return ""
        return self.records[-1].fen


class PGNSerializer:
    def __init__(self) -> None:
        self.headers: Dict[str, str] = {
            "Event": "CHMD Training Game",
            "Site": "Local",
            "Round": "-",
            "White": "Player",
            "Black": "Coach",
            "Result": "*",
        }

    def set_header(self, key: str, value: str) -> None:
        LOGGER.debug("PGN header %s set to %s", key, value)
        self.headers[key] = value

    def serialize(self, history: GameHistory) -> str:
        self.headers["Result"] = history.result
        header_lines = [f"[{key} \"{value}\"]" for key, value in self.headers.items()]
        moves = history.to_pgn()
        return "\n".join(header_lines + ["", moves, ""])

    def save(self, history: GameHistory, path: Path) -> None:
        ensure_directory(path.parent)
        content = self.serialize(history)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(content)
        LOGGER.info("Game saved to %s", path)


PGN_SERIALIZER = PGNSerializer()


# --------------------------------------------------------------------------------------
# Evaluation smoothing and annotations
# --------------------------------------------------------------------------------------


class EvaluationTracker:
    def __init__(self) -> None:
        self.values: List[Tuple[float, float]] = []
        self.window = 120
        self._lock = threading.Lock()

    def add(self, evaluation: float) -> None:
        with self._lock:
            timestamp = time.time()
            self.values.append((timestamp, evaluation))
            if len(self.values) > self.window:
                self.values = self.values[-self.window :]
            LOGGER.debug("Evaluation sample %.2f captured", evaluation)

    def trend(self, last_seconds: float = 30.0) -> float:
        cutoff = time.time() - last_seconds
        with self._lock:
            recent = [value for (ts, value) in self.values if ts >= cutoff]
        if len(recent) < 2:
            return 0.0
        return recent[-1] - recent[0]

    def last(self) -> float:
        with self._lock:
            if not self.values:
                return 0.0
            return self.values[-1][1]


EVALUATIONS = EvaluationTracker()


class CommentaryGenerator:
    def __init__(self) -> None:
        self.templates = {
            "brilliant": "A brilliant tactical shot!",
            "blunder": "That move drops the evaluation dramatically.",
            "solid": "Keeping the position under control.",
            "advantage": "Press the advantage with active play.",
            "defense": "Hold tight and neutralize threats.",
        }

    def annotate(self, delta: float) -> str:
        LOGGER.debug("Generating annotation for delta %.2f", delta)
        if delta >= 1.5:
            return self.templates["brilliant"]
        if delta <= -2.0:
            return self.templates["blunder"]
        if abs(delta) < 0.3:
            return self.templates["solid"]
        if delta > 0:
            return self.templates["advantage"]
        return self.templates["defense"]


COMMENTARY = CommentaryGenerator()


# --------------------------------------------------------------------------------------
# Engine command queue and analysis results
# --------------------------------------------------------------------------------------


@dataclass
class EngineCommand:
    command: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineResult:
    fen: str
    best_move: str
    pv_lines: List[str]
    evaluation: float
    depth: int
    nodes: int
    nps: int


class EngineCommandQueue:
    def __init__(self) -> None:
        self._queue: "queue.Queue[EngineCommand]" = queue.Queue()
        self._results: "queue.Queue[EngineResult]" = queue.Queue()
        self._lock = threading.Lock()

    def submit(self, command: EngineCommand) -> None:
        LOGGER.debug("Engine command queued: %s", command)
        self._queue.put(command)

    def get_command(self, timeout: float = 0.5) -> Optional[EngineCommand]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def push_result(self, result: EngineResult) -> None:
        LOGGER.debug("Engine result received: %s", result)
        self._results.put(result)

    def poll_result(self, timeout: float = 0.0) -> Optional[EngineResult]:
        try:
            return self._results.get(timeout=timeout)
        except queue.Empty:
            return None


ENGINE_QUEUE = EngineCommandQueue()


# --------------------------------------------------------------------------------------
# Opening book and training data
# --------------------------------------------------------------------------------------


class OpeningBook:
    def __init__(self) -> None:
        self.lines: Dict[str, List[str]] = {}
        self._build_default_book()

    def _build_default_book(self) -> None:
        LOGGER.debug("Building default opening book")
        self.lines = {
            "startpos": [
                "e2e4 e7e5 g1f3 b8c6 f1b5",
                "d2d4 d7d5 c1f4 g8f6",
                "c2c4 e7e5 g1c3 g8f6",
            ],
            "e2e4 e7e5": ["g1f3 b8c6", "f1c4 b8c6", "d2d4 e5d4"],
            "d2d4 d7d5": ["c2c4 e7e6", "g1f3 g8f6"],
        }

    def suggest(self, key: str) -> Optional[str]:
        lines = self.lines.get(key)
        if not lines:
            return None
        choice = random.choice(lines)
        LOGGER.debug("Opening suggestion for %s: %s", key, choice)
        return choice


OPENING_BOOK = OpeningBook()


# --------------------------------------------------------------------------------------
# Training plans
# --------------------------------------------------------------------------------------


@dataclass
class TrainingTask:
    name: str
    description: str
    fen: str
    solution: List[str]


class TrainingPlanner:
    def __init__(self) -> None:
        self.tasks: List[TrainingTask] = []
        self._populate_default_tasks()

    def _populate_default_tasks(self) -> None:
        LOGGER.info("Loading built-in training tasks")
        self.tasks = [
            TrainingTask(
                name="MateInTwo",
                description="Classic smothered mate pattern",
                fen="6k1/5ppp/8/8/8/5Q2/6PP/6K1 w - - 0 1",
                solution=["Qa8#"],
            ),
            TrainingTask(
                name="ForkTactic",
                description="Knight fork against the queen and king",
                fen="r1bqk2r/pppp1ppp/2n2n2/2b1p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 0 1",
                solution=["Nxe5"],
            ),
            TrainingTask(
                name="PinnedPiece",
                description="Exploit the pin on the e-file",
                fen="rnbq1rk1/ppp2ppp/3bpn2/3p4/3P4/2N1PN2/PPPB1PPP/R2QKB1R w KQ - 4 7",
                solution=["e4"],
            ),
        ]

    def random_task(self) -> TrainingTask:
        choice = random.choice(self.tasks)
        LOGGER.debug("Random training task selected: %s", choice.name)
        return choice


TRAINING_PLANNER = TrainingPlanner()


# --------------------------------------------------------------------------------------
# Matchmaking and difficulty scaling
# --------------------------------------------------------------------------------------


class DifficultyModel:
    def __init__(self) -> None:
        self.rating = 1200
        self.history: List[Tuple[float, float]] = []

    def update(self, evaluation_delta: float, outcome: str) -> None:
        LOGGER.debug("Updating difficulty model: delta=%.2f outcome=%s", evaluation_delta, outcome)
        adjustment = clamp(evaluation_delta / 2.0, -200, 200)
        if outcome == "1-0":
            adjustment += 40
        elif outcome == "0-1":
            adjustment -= 40
        else:
            adjustment += 5
        self.rating = clamp(self.rating + adjustment, 800, 2800)
        self.history.append((time.time(), self.rating))

    def target_depth(self) -> int:
        if self.rating < 1200:
            return 12
        if self.rating < 1600:
            return 16
        if self.rating < 2000:
            return 18
        return 22

    def target_nodes(self) -> int:
        base = max(300000, int(self.rating * 500))
        LOGGER.debug("Target nodes computed: %d", base)
        return base


DIFFICULTY_MODEL = DifficultyModel()


# --------------------------------------------------------------------------------------
# Telemetry & diagnostics
# --------------------------------------------------------------------------------------


class TelemetryCollector:
    def __init__(self) -> None:
        self.data: Dict[str, List[Tuple[float, float]]] = {}
        self._lock = threading.Lock()

    def log(self, name: str, value: float) -> None:
        with self._lock:
            entries = self.data.setdefault(name, [])
            entries.append((time.time(), value))
            if len(entries) > 240:
                entries[:] = entries[-240:]
            LOGGER.debug("Telemetry %s logged value %.3f", name, value)

    def snapshot(self) -> Dict[str, List[Tuple[float, float]]]:
        with self._lock:
            return {name: list(values) for name, values in self.data.items()}


TELEMETRY = TelemetryCollector()


# --------------------------------------------------------------------------------------
# Game clock management
# --------------------------------------------------------------------------------------


class ChessClock:
    def __init__(self, initial_time: float = 300.0) -> None:
        self.initial_time = initial_time
        self.white_time = initial_time
        self.black_time = initial_time
        self.increment = 2.0
        self.active_color = "white"
        self.last_tick = time.time()
        self.paused = True

    def reset(self) -> None:
        LOGGER.info("Clock reset")
        self.white_time = self.initial_time
        self.black_time = self.initial_time
        self.last_tick = time.time()
        self.paused = True

    def start(self, color: str) -> None:
        LOGGER.info("Clock started for %s", color)
        self.active_color = color
        self.last_tick = time.time()
        self.paused = False

    def stop(self) -> None:
        LOGGER.info("Clock stopped")
        self.paused = True

    def toggle(self) -> None:
        LOGGER.debug("Clock toggled")
        self.active_color = "black" if self.active_color == "white" else "white"
        self.last_tick = time.time()

    def update(self) -> None:
        if self.paused:
            return
        now = time.time()
        delta = now - self.last_tick
        self.last_tick = now
        if self.active_color == "white":
            self.white_time = max(0.0, self.white_time - delta)
        else:
            self.black_time = max(0.0, self.black_time - delta)
        TELEMETRY.log("clock_delta", delta)

    def apply_increment(self, color: str) -> None:
        if color == "white":
            self.white_time += self.increment
        else:
            self.black_time += self.increment
        LOGGER.debug("Increment applied to %s", color)


CLOCK = ChessClock()


# --------------------------------------------------------------------------------------
# Data persistence helpers
# --------------------------------------------------------------------------------------


class JSONStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_directory(path.parent)
        self._lock = threading.Lock()

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        with self._lock:
            with self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

    def save(self, data: Dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)


PROFILE_STORE = JSONStore(APP_PATHS.profile_file)


# --------------------------------------------------------------------------------------
# Player profiles and analytics
# --------------------------------------------------------------------------------------


@dataclass
class PlayerProfile:
    name: str = "Player"
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    highest_rating: int = 1200
    lowest_rating: int = 1200
    last_login: float = field(default_factory=time.time)
    achievements: List[str] = field(default_factory=list)

    def record_game(self, result: str, rating: int) -> None:
        LOGGER.debug("Recording game result %s for rating %d", result, rating)
        self.games_played += 1
        if result == "1-0":
            self.wins += 1
        elif result == "0-1":
            self.losses += 1
        else:
            self.draws += 1
        self.highest_rating = max(self.highest_rating, rating)
        self.lowest_rating = min(self.lowest_rating, rating)
        self.last_login = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "highest_rating": self.highest_rating,
            "lowest_rating": self.lowest_rating,
            "last_login": self.last_login,
            "achievements": list(self.achievements),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlayerProfile":
        profile = cls()
        profile.name = data.get("name", profile.name)
        profile.games_played = data.get("games_played", profile.games_played)
        profile.wins = data.get("wins", profile.wins)
        profile.losses = data.get("losses", profile.losses)
        profile.draws = data.get("draws", profile.draws)
        profile.highest_rating = data.get("highest_rating", profile.highest_rating)
        profile.lowest_rating = data.get("lowest_rating", profile.lowest_rating)
        profile.last_login = data.get("last_login", profile.last_login)
        profile.achievements = data.get("achievements", profile.achievements)
        return profile


class ProfileManager:
    def __init__(self) -> None:
        self.profile = PlayerProfile()
        self.load()

    def load(self) -> None:
        data = PROFILE_STORE.load()
        if data:
            self.profile = PlayerProfile.from_dict(data)
            LOGGER.info("Player profile loaded for %s", self.profile.name)

    def save(self) -> None:
        PROFILE_STORE.save(self.profile.to_dict())
        LOGGER.debug("Player profile saved")

    def record(self, result: str) -> None:
        rating = DIFFICULTY_MODEL.rating
        self.profile.record_game(result, rating)
        self.save()


PROFILE_MANAGER = ProfileManager()


# --------------------------------------------------------------------------------------
# Engine hint cache for overlays
# --------------------------------------------------------------------------------------


class HintCacheEntry:
    def __init__(self, fen: str, best_move: str, pv_lines: List[str], evaluation: float, depth: int) -> None:
        self.fen = fen
        self.best_move = best_move
        self.pv_lines = pv_lines
        self.evaluation = evaluation
        self.depth = depth
        self.timestamp = time.time()

    def is_valid(self, fen: str, freshness: float = 5.0) -> bool:
        if self.fen != fen:
            return False
        return (time.time() - self.timestamp) <= freshness


class HintCache:
    def __init__(self) -> None:
        self._entry: Optional[HintCacheEntry] = None
        self._lock = threading.Lock()

    def get(self, fen: str) -> Optional[HintCacheEntry]:
        with self._lock:
            if self._entry and self._entry.is_valid(fen):
                return self._entry
        return None

    def update(self, entry: HintCacheEntry) -> None:
        with self._lock:
            self._entry = entry
            LOGGER.debug("Hint cache updated for fen %s", entry.fen)


HINT_CACHE = HintCache()


# --------------------------------------------------------------------------------------
# Undo/redo stack for move analysis
# --------------------------------------------------------------------------------------


class UndoRedoStack:
    def __init__(self) -> None:
        self.past: List[str] = []
        self.future: List[str] = []
        self._lock = threading.Lock()

    def record(self, fen: str) -> None:
        with self._lock:
            self.past.append(fen)
            self.future.clear()
            LOGGER.debug("Undo stack recorded new state")

    def undo(self) -> Optional[str]:
        with self._lock:
            if len(self.past) < 2:
                return None
            current = self.past.pop()
            self.future.append(current)
            LOGGER.debug("Undo executed")
            return self.past[-1]

    def redo(self) -> Optional[str]:
        with self._lock:
            if not self.future:
                return None
            state = self.future.pop()
            self.past.append(state)
            LOGGER.debug("Redo executed")
            return state


UNDO_STACK = UndoRedoStack()


# --------------------------------------------------------------------------------------
# Spectator mode analytics
# --------------------------------------------------------------------------------------


class SpectatorAnalytics:
    def __init__(self) -> None:
        self.moves_observed = 0
        self.decisions: List[Tuple[str, float]] = []

    def observe(self, move: str, evaluation: float) -> None:
        self.moves_observed += 1
        self.decisions.append((move, evaluation))
        LOGGER.debug("Spectator observed move %s with eval %.2f", move, evaluation)

    def favorite_move(self) -> Optional[str]:
        if not self.decisions:
            return None
        best = max(self.decisions, key=lambda item: item[1])
        return best[0]


SPECTATOR_ANALYTICS = SpectatorAnalytics()


# --------------------------------------------------------------------------------------
# Hotkey bindings
# --------------------------------------------------------------------------------------


class HotkeyBinding:
    def __init__(self, description: str, callback: Callable[[], None]) -> None:
        self.description = description
        self.callback = callback


class HotkeyRegistry:
    def __init__(self) -> None:
        self.bindings: Dict[str, HotkeyBinding] = {}

    def register(self, key: str, description: str, callback: Callable[[], None]) -> None:
        LOGGER.debug("Hotkey registered: %s -> %s", key, description)
        self.bindings[key] = HotkeyBinding(description, callback)

    def execute(self, key: str) -> None:
        binding = self.bindings.get(key)
        if not binding:
            LOGGER.warning("Hotkey %s not registered", key)
            return
        try:
            binding.callback()
        except Exception:
            LOGGER.exception("Hotkey %s callback failed", key)

    def list_bindings(self) -> List[Tuple[str, str]]:
        return [(key, binding.description) for key, binding in self.bindings.items()]


HOTKEYS = HotkeyRegistry()


# --------------------------------------------------------------------------------------
# FEN utilities
# --------------------------------------------------------------------------------------


def board_to_fen(board: List[List[str]], active: str = "w", castling: str = "KQkq", en_passant: str = "-", halfmove: int = 0, fullmove: int = 1) -> str:
    parts: List[str] = []
    for rank in board:
        empty = 0
        row_parts: List[str] = []
        for piece in rank:
            if piece == "":
                empty += 1
            else:
                if empty:
                    row_parts.append(str(empty))
                    empty = 0
                row_parts.append(piece)
        if empty:
            row_parts.append(str(empty))
        parts.append("".join(row_parts))
    fen_board = "/".join(parts)
    fen = f"{fen_board} {active} {castling} {en_passant} {halfmove} {fullmove}"
    LOGGER.debug("Board converted to FEN: %s", fen)
    return fen


def parse_fen(fen: str) -> Tuple[List[List[str]], str, str, str, int, int]:
    board_part, active, castling, en_passant, halfmove, fullmove = fen.split()
    board_rows = board_part.split("/")
    board: List[List[str]] = []
    for row in board_rows:
        squares: List[str] = []
        for char in row:
            if char.isdigit():
                squares.extend([""] * int(char))
            else:
                squares.append(char)
        board.append(squares)
    return board, active, castling, en_passant, int(halfmove), int(fullmove)


# --------------------------------------------------------------------------------------
# Aggregated status model consumed by UI
# --------------------------------------------------------------------------------------


@dataclass
class OverlayStatus:
    fen: str
    best_move: str
    evaluation: float
    pv_lines: List[str]
    depth: int
    nodes: int
    nps: int


@dataclass
class TrainerStatus:
    current_fen: str
    overlay: Optional[OverlayStatus]
    commentary: str
    clock_white: float
    clock_black: float
    difficulty: int


# --------------------------------------------------------------------------------------
# High level controller for the trainer state
# --------------------------------------------------------------------------------------


class TrainerState:
    def __init__(self) -> None:
        self.history = GameHistory()
        self.current_fen = ""
        self.status_listeners: List[Callable[[TrainerStatus], None]] = []
        self._lock = threading.RLock()

    def set_fen(self, fen: str) -> None:
        with self._lock:
            LOGGER.debug("TrainerState set_fen %s", fen)
            self.current_fen = fen
            UNDO_STACK.record(fen)
            EVENT_BUS.publish("trainer.fen.update", {"fen": fen})
            self._notify()

    def add_move(self, record: MoveRecord) -> None:
        with self._lock:
            self.history.append(record)
            self.current_fen = record.fen
            self._notify()

    def reset(self) -> None:
        with self._lock:
            LOGGER.info("TrainerState reset")
            self.history.clear()
            self.current_fen = ""
            UNDO_STACK.past.clear()
            UNDO_STACK.future.clear()
            self._notify()

    def subscribe(self, callback: Callable[[TrainerStatus], None]) -> None:
        with self._lock:
            self.status_listeners.append(callback)

    def _notify(self) -> None:
        overlay_entry = HINT_CACHE.get(self.current_fen)
        overlay = None
        if overlay_entry:
            overlay = OverlayStatus(
                fen=overlay_entry.fen,
                best_move=overlay_entry.best_move,
                evaluation=overlay_entry.evaluation,
                pv_lines=overlay_entry.pv_lines,
                depth=overlay_entry.depth,
                nodes=0,
                nps=0,
            )
        status = TrainerStatus(
            current_fen=self.current_fen,
            overlay=overlay,
            commentary=COMMENTARY.annotate(EVALUATIONS.trend()),
            clock_white=CLOCK.white_time,
            clock_black=CLOCK.black_time,
            difficulty=DIFFICULTY_MODEL.rating,
        )
        for listener in list(self.status_listeners):
            try:
                listener(status)
            except Exception:
                LOGGER.exception("Trainer status listener failed")


TRAINER_STATE = TrainerState()


# --------------------------------------------------------------------------------------
# Capture scheduling helpers
# --------------------------------------------------------------------------------------


class CaptureSchedule:
    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.last_capture = 0.0

    def should_capture(self) -> bool:
        now = time.time()
        if now - self.last_capture >= self.interval:
            self.last_capture = now
            return True
        return False


CAPTURE_SCHEDULE = CaptureSchedule()


# --------------------------------------------------------------------------------------
# Application lifecycle helpers
# --------------------------------------------------------------------------------------


class Lifecycle:
    def __init__(self) -> None:
        self.running = AtomicFlag()
        self.shutdown_callbacks: List[Callable[[], None]] = []

    def register(self, callback: Callable[[], None]) -> None:
        self.shutdown_callbacks.append(callback)

    def start(self) -> None:
        LOGGER.info("Lifecycle starting")
        self.running.set(True)
        TASK_SCHEDULER.start()

    def stop(self) -> None:
        LOGGER.info("Lifecycle stopping")
        self.running.set(False)
        for callback in self.shutdown_callbacks:
            try:
                callback()
            except Exception:
                LOGGER.exception("Lifecycle shutdown callback failed")
        TASK_SCHEDULER.stop()


LIFECYCLE = Lifecycle()


# --------------------------------------------------------------------------------------
# Simple search tree for variations
# --------------------------------------------------------------------------------------


@dataclass
class VariationNode:
    move: str
    evaluation: float
    children: List["VariationNode"] = field(default_factory=list)

    def add_child(self, node: "VariationNode") -> None:
        LOGGER.debug("VariationNode add_child %s -> %s", self.move, node.move)
        self.children.append(node)

    def best_child(self) -> Optional["VariationNode"]:
        if not self.children:
            return None
        return max(self.children, key=lambda node: node.evaluation)

    def to_lines(self, prefix: Optional[List[str]] = None) -> List[str]:
        prefix = prefix or []
        lines: List[str] = []
        current = prefix + [self.move]
        if not self.children:
            lines.append(" ".join(current))
        else:
            for child in self.children:
                lines.extend(child.to_lines(current))
        return lines


class VariationTree:
    def __init__(self) -> None:
        self.root = VariationNode("start", 0.0)

    def update(self, pv_lines: List[str], evaluations: List[float]) -> None:
        LOGGER.debug("VariationTree update with %d lines", len(pv_lines))
        self.root.children.clear()
        for pv, evaluation in zip(pv_lines, evaluations):
            moves = pv.split()
            current = self.root
            for move in moves:
                child = next((node for node in current.children if node.move == move), None)
                if not child:
                    child = VariationNode(move, evaluation)
                    current.add_child(child)
                current = child
                current.evaluation = evaluation

    def best_line(self) -> List[str]:
        node = self.root.best_child()
        if not node:
            return []
        result: List[str] = []
        while node:
            result.append(node.move)
            node = node.best_child()
        return result


VARIATION_TREE = VariationTree()


# --------------------------------------------------------------------------------------
# Analytics for tactic solving streaks
# --------------------------------------------------------------------------------------


class StreakTracker:
    def __init__(self) -> None:
        self.current = 0
        self.best = 0

    def record(self, correct: bool) -> None:
        if correct:
            self.current += 1
            self.best = max(self.best, self.current)
        else:
            self.current = 0
        LOGGER.debug("Streak updated current=%d best=%d", self.current, self.best)


STREAK_TRACKER = StreakTracker()


# --------------------------------------------------------------------------------------
# Adaptive polling logic for capture/engine
# --------------------------------------------------------------------------------------


class AdaptiveInterval:
    def __init__(self, base_interval: float = 0.5) -> None:
        self.base_interval = base_interval
        self.current_interval = base_interval
        self.min_interval = 0.1
        self.max_interval = 2.0

    def record_load(self, load: float) -> None:
        target = clamp(self.base_interval * load, self.min_interval, self.max_interval)
        self.current_interval = lerp(self.current_interval, target, 0.3)
        LOGGER.debug("Adaptive interval adjusted to %.3f", self.current_interval)


CAPTURE_INTERVAL = AdaptiveInterval(0.4)
ENGINE_INTERVAL = AdaptiveInterval(0.2)


# --------------------------------------------------------------------------------------
# Utility for smoothing overlay updates
# --------------------------------------------------------------------------------------


class SmoothedValue:
    def __init__(self, alpha: float = 0.3) -> None:
        self.alpha = alpha
        self.current = 0.0
        self.initialized = False

    def update(self, value: float) -> float:
        if not self.initialized:
            self.current = value
            self.initialized = True
        else:
            self.current = (1 - self.alpha) * self.current + self.alpha * value
        LOGGER.debug("Smoothed value updated to %.3f", self.current)
        return self.current


OVERLAY_EVAL_SMOOTH = SmoothedValue()


# --------------------------------------------------------------------------------------
# Persistent capture index storage
# --------------------------------------------------------------------------------------


class CaptureIndexStore:
    def __init__(self) -> None:
        self.path = APP_PATHS.root / "capture_index.json"
        self.store = JSONStore(self.path)
        self.index = 0
        self.load()

    def load(self) -> None:
        data = self.store.load()
        if data:
            self.index = int(data.get("index", 0))
        LOGGER.debug("Capture index loaded: %d", self.index)

    def save(self) -> None:
        self.store.save({"index": self.index})
        LOGGER.debug("Capture index saved: %d", self.index)

    def set_index(self, value: int) -> None:
        self.index = value
        self.save()


CAPTURE_INDEX = CaptureIndexStore()


# --------------------------------------------------------------------------------------
# Theme manager for UI
# --------------------------------------------------------------------------------------


@dataclass
class Theme:
    name: str
    background: str
    foreground: str
    accent: str
    board_light: str
    board_dark: str


class ThemeManager:
    def __init__(self) -> None:
        self.themes: Dict[str, Theme] = {}
        self._load_default_themes()
        self.active_theme = self.themes["default"]

    def _load_default_themes(self) -> None:
        LOGGER.info("Loading themes")
        self.themes = {
            "default": Theme("default", "#121212", "#f5f5f5", "#4caf50", "#d0d0d0", "#5d5d5d"),
            "aurora": Theme("aurora", "#0b1a2a", "#d6f2ff", "#3fc1c9", "#93deff", "#126d9b"),
            "sunset": Theme("sunset", "#2e1a47", "#ffead0", "#ff9a8d", "#ffe3d8", "#8c4b87"),
        }

    def set_theme(self, name: str) -> Theme:
        theme = self.themes.get(name)
        if not theme:
            LOGGER.warning("Theme %s not found, using default", name)
            theme = self.themes["default"]
        self.active_theme = theme
        SETTINGS.set("board_theme", theme.name)
        LOGGER.debug("Theme set to %s", theme.name)
        return theme


THEME_MANAGER = ThemeManager()


# --------------------------------------------------------------------------------------
# Diagnostics aggregator for UI display
# --------------------------------------------------------------------------------------


class DiagnosticsRegistry:
    def __init__(self) -> None:
        self.values: Dict[str, Tuple[float, str]] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: float, label: str) -> None:
        with self._lock:
            self.values[key] = (value, label)
            LOGGER.debug("Diagnostics updated %s=%.2f", key, value)

    def get_snapshot(self) -> Dict[str, Tuple[float, str]]:
        with self._lock:
            return dict(self.values)


DIAGNOSTICS = DiagnosticsRegistry()


# --------------------------------------------------------------------------------------
# Export utilities
# --------------------------------------------------------------------------------------


def export_history_as_json(history: GameHistory, path: Path) -> None:
    ensure_directory(path.parent)
    data = [record.__dict__ for record in history.records]
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"records": data, "result": history.result}, handle, indent=2)
    LOGGER.info("History exported to %s", path)


def import_history_from_json(path: Path) -> GameHistory:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = [MoveRecord(**item) for item in payload.get("records", [])]
    history = GameHistory(records=records, result=payload.get("result", "*"))
    LOGGER.info("History imported from %s", path)
    return history


# --------------------------------------------------------------------------------------
# Manual save slot handling
# --------------------------------------------------------------------------------------


class SaveSlotManager:
    def __init__(self) -> None:
        self.slots_dir = APP_PATHS.root / "saves"
        ensure_directory(self.slots_dir)

    def _slot_path(self, name: str) -> Path:
        return self.slots_dir / f"{name}.json"

    def save(self, name: str, history: GameHistory) -> None:
        export_history_as_json(history, self._slot_path(name))

    def load(self, name: str) -> Optional[GameHistory]:
        path = self._slot_path(name)
        if not path.exists():
            LOGGER.warning("Save slot %s missing", name)
            return None
        return import_history_from_json(path)

    def list_slots(self) -> List[str]:
        return [path.stem for path in self.slots_dir.glob("*.json")]


SAVE_SLOTS = SaveSlotManager()


# --------------------------------------------------------------------------------------
# Bookmarks for study positions
# --------------------------------------------------------------------------------------


class PositionBookmarks:
    def __init__(self) -> None:
        self.bookmarks: Dict[str, str] = {}
        self.path = APP_PATHS.root / "bookmarks.json"
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            self.bookmarks = json.load(handle)
        LOGGER.debug("Bookmarks loaded: %d entries", len(self.bookmarks))

    def save(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.bookmarks, handle, indent=2)
        LOGGER.debug("Bookmarks saved: %d entries", len(self.bookmarks))

    def add(self, name: str, fen: str) -> None:
        self.bookmarks[name] = fen
        self.save()

    def remove(self, name: str) -> None:
        if name in self.bookmarks:
            del self.bookmarks[name]
            self.save()

    def get(self, name: str) -> Optional[str]:
        return self.bookmarks.get(name)

    def list(self) -> List[Tuple[str, str]]:
        return sorted(self.bookmarks.items())


BOOKMARKS = PositionBookmarks()


# --------------------------------------------------------------------------------------
# Notes storage
# --------------------------------------------------------------------------------------


class NotesStore:
    def __init__(self) -> None:
        self.path = APP_PATHS.root / "notes.json"
        ensure_directory(self.path.parent)
        self.notes: Dict[str, str] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            self.notes = json.load(handle)
        LOGGER.debug("Notes loaded: %d entries", len(self.notes))

    def save(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.notes, handle, indent=2)
        LOGGER.debug("Notes saved: %d entries", len(self.notes))

    def add(self, fen: str, note: str) -> None:
        self.notes[fen] = note
        self.save()

    def get(self, fen: str) -> str:
        return self.notes.get(fen, "")


NOTES = NotesStore()


# --------------------------------------------------------------------------------------
# FEN validators
# --------------------------------------------------------------------------------------


def validate_fen(fen: str) -> bool:
    parts = fen.split()
    if len(parts) != 6:
        return False
    board, active, castling, en_passant, halfmove, fullmove = parts
    if active not in ("w", "b"):
        return False
    for row in board.split("/"):
        count = 0
        for char in row:
            if char.isdigit():
                count += int(char)
            elif char in PIECE_SYMBOLS:
                count += 1
            else:
                return False
        if count != 8:
            return False
    try:
        int(halfmove)
        int(fullmove)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------------------
# Stockfish management placeholder (integrated with engine module)
# --------------------------------------------------------------------------------------


class EngineSettings:
    def __init__(self) -> None:
        self.path = SETTINGS.get("engine_path", "stockfish")
        self.threads = SETTINGS.get("engine_threads", 4)
        self.hash_size = SETTINGS.get("engine_hash", 512)
        self.multi_pv = SETTINGS.get("multi_pv", 3)
        self.depth = SETTINGS.get("analysis_depth", 18)

    def refresh(self) -> None:
        LOGGER.debug("Refreshing engine settings")
        self.path = SETTINGS.get("engine_path", self.path)
        self.threads = SETTINGS.get("engine_threads", self.threads)
        self.hash_size = SETTINGS.get("engine_hash", self.hash_size)
        self.multi_pv = SETTINGS.get("multi_pv", self.multi_pv)
        self.depth = SETTINGS.get("analysis_depth", self.depth)


ENGINE_SETTINGS = EngineSettings()


# --------------------------------------------------------------------------------------
# Real-time scoreboard aggregator
# --------------------------------------------------------------------------------------


class Scoreboard:
    def __init__(self) -> None:
        self.entries: List[Tuple[str, float]] = []

    def update(self, player: str, score: float) -> None:
        for index, (name, _) in enumerate(self.entries):
            if name == player:
                self.entries[index] = (player, score)
                break
        else:
            self.entries.append((player, score))
        self.entries.sort(key=lambda item: item[1], reverse=True)
        LOGGER.debug("Scoreboard updated: %s", self.entries)

    def top(self, count: int = 3) -> List[Tuple[str, float]]:
        return self.entries[:count]


SCOREBOARD = Scoreboard()


# --------------------------------------------------------------------------------------
# Movement heuristics for engine hints
# --------------------------------------------------------------------------------------


def describe_move(move: str) -> str:
    if len(move) < 4:
        return ""
    start = move[:2]
    end = move[2:4]
    desc = f"Move from {start} to {end}"
    if len(move) > 4:
        desc += f" promoting to {move[4]}"
    return desc


def explain_hint(fen: str, best_move: str, evaluation: float) -> str:
    commentary = describe_move(best_move)
    if evaluation > 1.5:
        commentary += ", winning advantage"
    elif evaluation > 0.5:
        commentary += ", slight edge"
    elif evaluation > -0.5:
        commentary += ", balanced"
    else:
        commentary += ", defensive task"
    LOGGER.debug("Hint explanation: %s", commentary)
    return commentary


# --------------------------------------------------------------------------------------
# Adaptive sampling of evaluation metrics
# --------------------------------------------------------------------------------------


class EvaluationSampler:
    def __init__(self) -> None:
        self.values: List[float] = []
        self.maxlen = 60

    def add(self, value: float) -> None:
        self.values.append(value)
        if len(self.values) > self.maxlen:
            self.values = self.values[-self.maxlen :]
        LOGGER.debug("Evaluation sampler appended value %.2f", value)

    def average(self) -> float:
        return rolling_average(self.values)

    def volatility(self) -> float:
        avg = self.average()
        return rolling_average([(value - avg) ** 2 for value in self.values]) ** 0.5


EVAL_SAMPLER = EvaluationSampler()


# --------------------------------------------------------------------------------------
# Tactical motif recognizer (simple pattern matching)
# --------------------------------------------------------------------------------------


class TacticalMotifRecognizer:
    def __init__(self) -> None:
        self.patterns: Dict[str, str] = {
            "back_rank": "Look for back-rank mate threats",
            "fork": "Knights love forks: search for double attacks",
            "pin": "Exploit pins along files and diagonals",
        }

    def classify(self, fen: str) -> str:
        board, _, _, _, _, _ = parse_fen(fen)
        white_back_rank = board[-1]
        black_back_rank = board[0]
        if white_back_rank.count("k") == 1 and white_back_rank.count("r") >= 1:
            return self.patterns["back_rank"]
        if black_back_rank.count("K") == 1 and black_back_rank.count("R") >= 1:
            return self.patterns["back_rank"]
        knights = sum(row.count("N") + row.count("n") for row in board)
        if knights >= 2:
            return self.patterns["fork"]
        return self.patterns["pin"]


TACTICS_RECOGNIZER = TacticalMotifRecognizer()


# --------------------------------------------------------------------------------------
# Export diagnostics for debugging
# --------------------------------------------------------------------------------------


def dump_diagnostics(path: Path) -> None:
    ensure_directory(path.parent)
    payload = {
        "settings": SETTINGS.all(),
        "profile": PROFILE_MANAGER.profile.to_dict(),
        "telemetry": TELEMETRY.snapshot(),
        "diagnostics": DIAGNOSTICS.get_snapshot(),
        "evaluations": EVALUATIONS.values,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOGGER.info("Diagnostics dumped to %s", path)


# --------------------------------------------------------------------------------------
# Scheduled tasks hooking into event bus
# --------------------------------------------------------------------------------------


def schedule_periodic_tasks() -> None:
    def capture_task() -> None:
        EVENT_BUS.publish("capture.request")
        TASK_SCHEDULER.post("capture_loop", capture_task)

    def engine_task() -> None:
        EVENT_BUS.publish("engine.request")
        TASK_SCHEDULER.post("engine_loop", engine_task)

    def clock_task() -> None:
        CLOCK.update()
        TASK_SCHEDULER.post("clock_loop", clock_task)

    TASK_SCHEDULER.post("capture_loop", capture_task)
    TASK_SCHEDULER.post("engine_loop", engine_task)
    TASK_SCHEDULER.post("clock_loop", clock_task)


# --------------------------------------------------------------------------------------
# Initialization sequence for the module
# --------------------------------------------------------------------------------------


def initialize_core() -> None:
    LOGGER.info("Initializing CHMD core")
    LIFECYCLE.start()
    schedule_periodic_tasks()


# --------------------------------------------------------------------------------------
# Shutdown sequence
# --------------------------------------------------------------------------------------


def shutdown_core() -> None:
    LOGGER.info("Shutting down CHMD core")
    LIFECYCLE.stop()


__all__ = [
    "LOGGER",
    "SETTINGS",
    "PROFILER",
    "EVENT_BUS",
    "TASK_SCHEDULER",
    "ENGINE_QUEUE",
    "TRAINER_STATE",
    "initialize_core",
    "shutdown_core",
    "ENGINE_SETTINGS",
    "PGN_SERIALIZER",
    "TRAINING_PLANNER",
    "DIFFICULTY_MODEL",
    "THEME_MANAGER",
    "BOOKMARKS",
    "NOTES",
    "SCOREBOARD",
    "CLOCK",
]
