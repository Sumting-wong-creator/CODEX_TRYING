import os
import sys
import json
import math
import time
import queue
import base64
import random
import threading
import subprocess
import collections
import itertools
import statistics
import functools
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Callable, Iterable, Deque, Set

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import chess
import chess.engine
import chess.pgn


class AppLogger:
    """Simple structured logger with multiple sinks and verbosity levels."""

    _instance = None

    def __init__(self):
        self._lock = threading.RLock()
        self.verbosity = 2
        self._sinks: List[Callable[[str], None]] = [self._default_sink]
        self._history: Deque[str] = collections.deque(maxlen=5000)

    @classmethod
    def instance(cls) -> "AppLogger":
        if cls._instance is None:
            cls._instance = AppLogger()
        return cls._instance

    def _default_sink(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        sys.stdout.write(f"[{timestamp}] {message}\n")
        sys.stdout.flush()

    def add_sink(self, sink: Callable[[str], None]) -> None:
        with self._lock:
            self._sinks.append(sink)

    def remove_sink(self, sink: Callable[[str], None]) -> None:
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    def log(self, level: int, message: str) -> None:
        if level > self.verbosity:
            return
        entry = f"L{level}: {message}"
        with self._lock:
            self._history.append(entry)
            sinks = list(self._sinks)
        for sink in sinks:
            try:
                sink(entry)
            except Exception as exc:  # pragma: no cover - logging failure shouldn't crash
                sys.stderr.write(f"Logger sink failed: {exc}\n")

    def info(self, message: str) -> None:
        self.log(1, message)

    def debug(self, message: str) -> None:
        self.log(2, message)

    def trace(self, message: str) -> None:
        self.log(3, message)

    def warn(self, message: str) -> None:
        self.log(1, f"WARNING: {message}")

    def error(self, message: str) -> None:
        self.log(0, f"ERROR: {message}")

    def dump_history(self) -> List[str]:
        with self._lock:
            return list(self._history)


class ProfilingTimer:
    """Context manager for measuring execution time."""

    def __init__(self, name: str, threshold_ms: float = 0.0):
        self.name = name
        self.threshold_ms = threshold_ms
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000.0
        if elapsed >= self.threshold_ms:
            AppLogger.instance().debug(f"Profile {self.name}: {elapsed:.2f} ms")
        return False


class FPSCounter:
    def __init__(self, smoothing: int = 30):
        self.samples = collections.deque(maxlen=smoothing)
        self.last = time.perf_counter()

    def frame(self) -> float:
        now = time.perf_counter()
        delta = now - self.last
        self.last = now
        if delta <= 0:
            return 0.0
        fps = 1.0 / delta
        self.samples.append(fps)
        return fps

    def get_fps(self) -> float:
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)


class SettingsManager:
    """Stores persistent configuration in JSON."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self.data: Dict[str, Any] = {
            "engine_path": "stockfish",
            "difficulty": 10,
            "move_time": 5.0,
            "ui_theme": "liquid_glass",
            "overlay": {
                "enabled": True,
                "opacity": 0.85,
                "always_on_top": True,
                "show_arrows": True,
                "show_eval_bar": True,
                "show_pv_lines": 3,
            },
            "vision": {
                "use_camera": False,
                "camera_index": 0,
                "screen_region": [0, 0, 1280, 720],
                "board_size": 640,
                "piece_threshold": 0.42,
            },
        }
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                self.data.update(payload)
            AppLogger.instance().info(f"Loaded settings from {self.path}")
        except Exception as exc:
            AppLogger.instance().error(f"Failed to load settings: {exc}")

    def save(self) -> None:
        tmp_path = self.path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2)
            os.replace(tmp_path, self.path)
            AppLogger.instance().info(f"Settings saved to {self.path}")
        except Exception as exc:
            AppLogger.instance().error(f"Failed to save settings: {exc}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        node: Any = self.data
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        self.save()


class HotkeyManager(QtCore.QObject):
    hotkeyTriggered = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._keys: Dict[int, str] = {}

    def register_hotkey(self, qt_key: int, name: str) -> None:
        self._keys[qt_key] = name

    def handle_event(self, event: QtGui.QKeyEvent) -> bool:
        key = event.key()
        if key in self._keys:
            self.hotkeyTriggered.emit(self._keys[key])
            return True
        return False


class ThemePalette:
    def __init__(self, name: str, colors: Dict[str, QtGui.QColor]):
        self.name = name
        self.colors = colors

    def color(self, key: str, fallback: QtGui.QColor = QtGui.QColor("#ffffff")) -> QtGui.QColor:
        return self.colors.get(key, fallback)


class ThemeManager:
    def __init__(self):
        self.themes: Dict[str, ThemePalette] = {}
        self.active: Optional[ThemePalette] = None
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        def color(hex_color: str) -> QtGui.QColor:
            return QtGui.QColor(hex_color)

        liquid_glass = ThemePalette(
            "liquid_glass",
            {
                "background": color("#1f1f29"),
                "panel": color("#262633"),
                "accent": color("#7b61ff"),
                "accent_alt": color("#47c1ff"),
                "text": color("#f5f5f5"),
                "text_dim": color("#9999bb"),
                "board_light": color("#d7e0ff"),
                "board_dark": color("#6c7a9f"),
                "arrow_good": color("#53d769"),
                "arrow_blunder": color("#ff3b30"),
                "eval_positive": color("#53d769"),
                "eval_negative": color("#ff3b30"),
            },
        )
        dark_carbon = ThemePalette(
            "dark_carbon",
            {
                "background": color("#131313"),
                "panel": color("#1c1c1c"),
                "accent": color("#00ccff"),
                "accent_alt": color("#ff9500"),
                "text": color("#f0f0f0"),
                "text_dim": color("#7a7a7a"),
                "board_light": color("#eaeaea"),
                "board_dark": color("#444444"),
                "arrow_good": color("#40ff6d"),
                "arrow_blunder": color("#ff2d55"),
                "eval_positive": color("#40ff6d"),
                "eval_negative": color("#ff2d55"),
            },
        )
        self.themes[liquid_glass.name] = liquid_glass
        self.themes[dark_carbon.name] = dark_carbon
        self.set_active("liquid_glass")

    def set_active(self, name: str) -> None:
        if name in self.themes:
            self.active = self.themes[name]
        else:
            self.active = next(iter(self.themes.values()))
        AppLogger.instance().info(f"Active theme: {self.active.name}")

    def palette(self) -> ThemePalette:
        if not self.active:
            self.set_active("liquid_glass")
        return self.active


class ResourceLoader:
    def __init__(self):
        self.cache: Dict[str, QtGui.QIcon] = {}

    def icon_from_base64(self, key: str, payload: str) -> QtGui.QIcon:
        if key in self.cache:
            return self.cache[key]
        raw = base64.b64decode(payload)
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(raw)
        icon = QtGui.QIcon(pixmap)
        self.cache[key] = icon
        return icon


class AudioFeedback:
    def __init__(self):
        self.enabled = False
        self.click_sound = None
        self._qt_multimedia = None
        try:
            from PyQt5 import QtMultimedia

            self._qt_multimedia = QtMultimedia
            self.click_sound = QtMultimedia.QSoundEffect()
            self.click_sound.setSource(
                QtCore.QUrl.fromLocalFile("/usr/share/sounds/freedesktop/stereo/bell.oga")
            )
            self.click_sound.setVolume(0.25)
            self.enabled = True
        except Exception:
            AppLogger.instance().warn("QtMultimedia not available, audio feedback disabled")

    def play_click(self):
        if self.enabled and self.click_sound:
            self.click_sound.play()


class EngineCommandQueue:
    def __init__(self):
        self.queue: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()

    def put(self, command: str, params: Optional[Dict[str, Any]] = None) -> None:
        self.queue.put((command, params or {}))

    def get(self, timeout: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
        return self.queue.get(timeout=timeout)

    def empty(self) -> bool:
        return self.queue.empty()


@dataclass
class MoveCandidate:
    move: chess.Move
    evaluation: float
    depth: int
    line: List[chess.Move]


@dataclass
class PVLine:
    evaluation: float
    depth: int
    moves: List[chess.Move]

    def to_san(self, board: chess.Board) -> str:
        copy_board = board.copy()
        san_parts = []
        for mv in self.moves:
            san_parts.append(copy_board.san(mv))
            copy_board.push(mv)
        return " ".join(san_parts)


class EvaluationModel:
    def __init__(self):
        self.evaluation = 0.0
        self.depth = 0
        self.multi_pv: List[PVLine] = []
        self.nodes = 0
        self.nps = 0
        self.tb_hits = 0

    def update_from_info(self, info: chess.engine.InfoDict, limit_pv: int = 3) -> None:
        score = info.get("score")
        if score is not None:
            self.evaluation = score.white().score(mate_score=100000) / 100.0
        self.depth = info.get("depth", self.depth)
        self.nodes = info.get("nodes", self.nodes)
        self.nps = info.get("nps", self.nps)
        self.tb_hits = info.get("tb", self.tb_hits)
        self.multi_pv.clear()
        pv_list = info.get("pv", [])
        if isinstance(pv_list, list):
            pv_line = PVLine(self.evaluation, self.depth, list(pv_list)[:limit_pv])
            self.multi_pv.append(pv_line)
        alt_lines = info.get("multipv", [])
        if isinstance(alt_lines, list):
            for entry in alt_lines[:limit_pv]:
                line = entry.get("pv") if isinstance(entry, dict) else None
                score_obj = entry.get("score") if isinstance(entry, dict) else None
                if line:
                    eval_score = (
                        score_obj.white().score(mate_score=100000) / 100.0
                        if hasattr(score_obj, "white")
                        else self.evaluation
                    )
                    self.multi_pv.append(PVLine(eval_score, entry.get("depth", self.depth), list(line)))


class PGNManager:
    def __init__(self):
        self.games: List[chess.pgn.Game] = []

    def new_game(self, headers: Optional[Dict[str, str]] = None) -> chess.pgn.Game:
        game = chess.pgn.Game()
        if headers:
            for key, value in headers.items():
                game.headers[key] = value
        self.games.append(game)
        return game

    def save_game(self, game: chess.pgn.Game, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            exporter = chess.pgn.FileExporter(fh)
            game.accept(exporter)
        AppLogger.instance().info(f"Saved PGN to {path}")

    def load_game(self, path: str) -> chess.pgn.Game:
        with open(path, "r", encoding="utf-8") as fh:
            game = chess.pgn.read_game(fh)
        if game:
            self.games.append(game)
        AppLogger.instance().info(f"Loaded PGN from {path}")
        return game


class MoveHistoryModel(QtCore.QAbstractListModel):
    def __init__(self, board: chess.Board, parent=None):
        super().__init__(parent)
        self.board = board
        self.moves: List[chess.Move] = []

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return len(self.moves)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self.moves)):
            return None
        move = self.moves[index.row()]
        if role == QtCore.Qt.DisplayRole:
            temp_board = self.board.copy(stack=False)
            for mv in self.moves[: index.row()]:
                temp_board.push(mv)
            return temp_board.san(move)
        if role == QtCore.Qt.ToolTipRole:
            return move.uci()
        return None

    def append_move(self, move: chess.Move) -> None:
        self.beginInsertRows(QtCore.QModelIndex(), len(self.moves), len(self.moves))
        self.moves.append(move)
        self.endInsertRows()

    def clear(self) -> None:
        self.beginResetModel()
        self.moves.clear()
        self.endResetModel()


class EvaluationGraphModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.samples: List[Tuple[int, float]] = []

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return len(self.samples)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 2

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole) -> Any:
        if not index.isValid():
            return None
        turn, evaluation = self.samples[index.row()]
        if role == QtCore.Qt.DisplayRole:
            return f"{turn}" if index.column() == 0 else f"{evaluation:.2f}"
        return None

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.DisplayRole) -> Any:
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return ["Move", "Eval"][section]
        return None

    def append(self, turn: int, evaluation: float) -> None:
        self.beginInsertRows(QtCore.QModelIndex(), len(self.samples), len(self.samples))
        self.samples.append((turn, evaluation))
        self.endInsertRows()

    def clear(self) -> None:
        self.beginResetModel()
        self.samples.clear()
        self.endResetModel()


class StockfishEngine(QtCore.QObject):
    infoUpdated = QtCore.pyqtSignal(dict)
    bestMoveFound = QtCore.pyqtSignal(chess.Move, float)

    def __init__(self, settings: SettingsManager, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.engine: Optional[chess.engine.SimpleEngine] = None
        self.command_queue = EngineCommandQueue()
        self.response_queue: "queue.Queue[Any]" = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.current_board = chess.Board()
        self.multi_pv = 3
        self.last_best: Optional[Tuple[chess.Move, float]] = None

    def _create_engine(self) -> chess.engine.SimpleEngine:
        engine_path = self.settings.get("engine_path", "stockfish")
        try:
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            options = {
                "Threads": max(1, os.cpu_count() or 1),
                "Ponder": False,
                "MultiPV": self.multi_pv,
            }
            for key, value in options.items():
                try:
                    engine.configure({key: value})
                except Exception as exc:
                    AppLogger.instance().warn(f"Failed to configure engine option {key}: {exc}")
            AppLogger.instance().info(f"Stockfish started from {engine_path}")
            return engine
        except Exception as exc:
            AppLogger.instance().error(f"Failed to start Stockfish: {exc}")
            raise

    def _ensure_engine(self) -> None:
        if self.engine is None:
            self.engine = self._create_engine()

    def _run(self) -> None:
        AppLogger.instance().info("Engine thread started")
        while True:
            command, params = self.command_queue.get()
            if command == "quit":
                break
            try:
                if command == "analyze":
                    board = params.get("board", self.current_board)
                    self.current_board = board
                    limit = params.get("limit", chess.engine.Limit(time=self.settings.get("move_time", 5.0)))
                    self._ensure_engine()
                    with self.engine.analysis(board, limit=limit, multipv=self.multi_pv) as analysis:
                        for info in analysis:
                            self.infoUpdated.emit(dict(info))
                            if "pv" in info:
                                score = info.get("score")
                                if score:
                                    cp = score.white().score(mate_score=100000) / 100.0
                                else:
                                    cp = 0.0
                                best_move = info["pv"][0]
                                self.last_best = (best_move, cp)
                        if self.last_best:
                            self.bestMoveFound.emit(self.last_best[0], self.last_best[1])
                elif command == "setoption":
                    self._ensure_engine()
                    self.engine.configure({params["name"]: params["value"]})
                elif command == "move":
                    self._ensure_engine()
                    board = params["board"]
                    limit = params.get("limit", chess.engine.Limit(time=self.settings.get("move_time", 5.0)))
                    result = self.engine.play(board, limit)
                    self.response_queue.put(result)
                elif command == "stop":
                    if self.engine:
                        self.engine.stop()
            except Exception as exc:
                AppLogger.instance().error(f"Engine command {command} failed: {exc}")
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
        AppLogger.instance().info("Engine thread terminated")

    def analyze(self, board: chess.Board, move_time: float) -> None:
        self.command_queue.put("analyze", {"board": board.copy(), "limit": chess.engine.Limit(time=move_time)})

    def play(self, board: chess.Board, move_time: float) -> chess.engine.PlayResult:
        self.command_queue.put("move", {"board": board.copy(), "limit": chess.engine.Limit(time=move_time)})
        return self.response_queue.get()

    def stop(self) -> None:
        self.command_queue.put("stop")

    def shutdown(self) -> None:
        self.command_queue.put("quit")
        self.thread.join(timeout=5.0)


@dataclass
class VisionFrame:
    frame: np.ndarray
    timestamp: float
    source: str


class CameraManager(QtCore.QObject):
    frameCaptured = QtCore.pyqtSignal(VisionFrame)

    def __init__(self, settings: SettingsManager, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.capture: Optional[cv2.VideoCapture] = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_interval = 1.0 / 30.0

    def start(self) -> None:
        if self.running:
            return
        index = self.settings.get("vision.camera_index", 0)
        self.capture = cv2.VideoCapture(index)
        if not self.capture or not self.capture.isOpened():
            AppLogger.instance().error(f"Failed to open camera index {index}")
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        AppLogger.instance().info("Camera capture started")

    def _loop(self) -> None:
        fps = FPSCounter()
        while self.running and self.capture:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)
            self.frameCaptured.emit(VisionFrame(frame, time.time(), "camera"))
            current_fps = fps.frame()
            AppLogger.instance().trace(f"Camera FPS: {current_fps:.1f}")
            delay = max(0.0, self.frame_interval - (time.perf_counter() - fps.last))
            if delay > 0:
                time.sleep(delay)
        if self.capture:
            self.capture.release()
        AppLogger.instance().info("Camera capture loop exited")

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.capture:
            self.capture.release()
        self.capture = None
        AppLogger.instance().info("Camera capture stopped")


class ScreenCaptureManager(QtCore.QObject):
    frameCaptured = QtCore.pyqtSignal(VisionFrame)

    def __init__(self, settings: SettingsManager, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_interval = 1.0 / 15.0

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        AppLogger.instance().info("Screen capture started")

    def _loop(self) -> None:
        import mss

        fps = FPSCounter()
        while self.running:
            region = self.settings.get("vision.screen_region", [0, 0, 1280, 720])
            bbox = {
                "top": int(region[1]),
                "left": int(region[0]),
                "width": int(region[2]),
                "height": int(region[3]),
            }
            with mss.mss() as sct:
                frame = np.array(sct.grab(bbox))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            self.frameCaptured.emit(VisionFrame(frame, time.time(), "screen"))
            current_fps = fps.frame()
            AppLogger.instance().trace(f"Screen capture FPS: {current_fps:.1f}")
            delay = max(0.0, self.frame_interval - (time.perf_counter() - fps.last))
            if delay > 0:
                time.sleep(delay)
        AppLogger.instance().info("Screen capture loop exited")

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        AppLogger.instance().info("Screen capture stopped")


@dataclass
class BoardDetectionResult:
    matrix: np.ndarray
    corners: np.ndarray
    warped: np.ndarray
    grid_size: int

    def extract_square(self, file_idx: int, rank_idx: int) -> np.ndarray:
        x0 = file_idx * self.grid_size
        y0 = (7 - rank_idx) * self.grid_size
        return self.warped[y0 : y0 + self.grid_size, x0 : x0 + self.grid_size]


@dataclass
class PieceDetectionResult:
    squares: Dict[str, str]
    confidence: float


class BoardAnalyzer:
    def __init__(self, settings: SettingsManager):
        self.settings = settings
        self.board_size = self.settings.get("vision.board_size", 640)
        self.square_size = self.board_size // 8
        self.last_detection: Optional[BoardDetectionResult] = None
        self.detector = cv2.QRCodeDetector()

    def detect_board(self, frame: np.ndarray) -> Optional[BoardDetectionResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        board_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                board_contour = approx
                max_area = area
        if board_contour is None:
            AppLogger.instance().trace("No board contour detected")
            return None
        board_contour = board_contour.reshape((4, 2))
        sorted_indices = np.argsort(board_contour[:, 1])
        top = board_contour[sorted_indices[:2]]
        bottom = board_contour[sorted_indices[2:]]
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]
        corners = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")
        target = np.array(
            [
                [0, 0],
                [self.board_size - 1, 0],
                [self.board_size - 1, self.board_size - 1],
                [0, self.board_size - 1],
            ],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(corners, target)
        warped = cv2.warpPerspective(frame, matrix, (self.board_size, self.board_size))
        detection = BoardDetectionResult(matrix, corners, warped, self.square_size)
        self.last_detection = detection
        return detection

    def extract_square(self, warped: np.ndarray, file_idx: int, rank_idx: int) -> np.ndarray:
        x0 = file_idx * self.square_size
        y0 = (7 - rank_idx) * self.square_size
        square = warped[y0 : y0 + self.square_size, x0 : x0 + self.square_size]
        return square


class PieceTemplate:
    def __init__(self, name: str, data: np.ndarray):
        self.name = name
        self.data = data
        self.histogram = self._compute_histogram()

    def _compute_histogram(self) -> np.ndarray:
        hist = cv2.calcHist([self.data], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def match(self, image: np.ndarray) -> float:
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        score = cv2.compareHist(self.histogram, hist, cv2.HISTCMP_CORREL)
        return float(score)


class TemplateLibrary:
    def __init__(self):
        self.templates: Dict[str, PieceTemplate] = {}
        self._build_default_templates()

    def _build_default_templates(self) -> None:
        random.seed(42)
        for color in ("w", "b"):
            for piece in ("p", "n", "b", "r", "q", "k"):
                key = f"{color}{piece}"
                data = np.zeros((32, 32, 3), dtype=np.uint8)
                cv2.putText(data, piece.upper() if color == "w" else piece.lower(), (2, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                data = cv2.GaussianBlur(data, (5, 5), 0)
                self.templates[key] = PieceTemplate(key, data)

    def match(self, square: np.ndarray) -> Tuple[str, float]:
        best_key = ""
        best_score = -1.0
        resized = cv2.resize(square, (32, 32))
        for key, template in self.templates.items():
            score = template.match(resized)
            if score > best_score:
                best_score = score
                best_key = key
        return best_key, best_score


class PieceClassifier:
    def __init__(self, settings: SettingsManager):
        self.settings = settings
        self.templates = TemplateLibrary()
        self.threshold = self.settings.get("vision.piece_threshold", 0.42)

    def classify_board(self, detection: BoardDetectionResult) -> PieceDetectionResult:
        squares: Dict[str, str] = {}
        confidences: List[float] = []
        warped = detection.warped
        for rank in range(8):
            for file in range(8):
                square_img = detection.extract_square(warped, file, rank)
                key, score = self.templates.match(square_img)
                square_name = chess.square_name(chess.square(file, rank))
                if score >= self.threshold:
                    squares[square_name] = key
                    confidences.append(score)
        confidence = float(sum(confidences) / len(confidences)) if confidences else 0.0
        return PieceDetectionResult(squares, confidence)


class FENBuilder:
    def __init__(self):
        pass

    def build_fen(self, pieces: Dict[str, str], turn: str = "w") -> str:
        rows = []
        for rank in range(7, -1, -1):
            row = []
            empty = 0
            for file in range(8):
                square = chess.square(file, rank)
                name = chess.square_name(square)
                if name in pieces:
                    if empty:
                        row.append(str(empty))
                        empty = 0
                    descriptor = pieces[name]
                    char = descriptor[1]
                    if descriptor[0] == "w":
                        row.append(char.upper())
                    else:
                        row.append(char.lower())
                else:
                    empty += 1
            if empty:
                row.append(str(empty))
            rows.append("".join(row))
        fen = "/".join(rows) + f" {turn} - - 0 1"
        return fen


class MoveExplainer:
    def explain_move(self, board: chess.Board, move: chess.Move) -> str:
        piece = board.piece_at(move.from_square)
        if not piece:
            return "Unknown move"
        target = board.piece_at(move.to_square)
        explanation = [f"{piece.symbol().upper()} from {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}"]
        if board.is_capture(move):
            if target:
                explanation.append(f"capturing {target.symbol().upper()}")
            else:
                explanation.append("en passant capture")
        if board.is_check():
            explanation.append("giving check")
        if board.is_castling(move):
            explanation.append("castling move")
        if board.is_en_passant(move):
            explanation.append("special en passant rule")
        if board.is_capture(move) and not target:
            explanation.append("captures via en passant ghost")
        hypothetical = board.copy()
        hypothetical.push(move)
        if hypothetical.is_check():
            explanation.append("resulting position gives check")
        if hypothetical.is_checkmate():
            explanation.append("delivers checkmate")
        return ", ".join(explanation)


class TrainingScenario:
    def __init__(self, name: str, board: chess.Board, target_move: chess.Move, description: str):
        self.name = name
        self.board = board
        self.target_move = target_move
        self.description = description




class HintOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.arrows: List[Tuple[str, str, QtGui.QColor, float]] = []
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

    def add_arrow(self, start: str, end: str, color: QtGui.QColor, width: float) -> None:
        self.arrows.append((start, end, color, width))
        self.update()

    def clear_arrows(self) -> None:
        self.arrows.clear()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        for start, end, color, width in self.arrows:
            self._draw_arrow(painter, start, end, color, width)

    def _square_to_point(self, square: str) -> QtCore.QPointF:
        file = ord(square[0]) - ord("a")
        rank = int(square[1]) - 1
        square_size = self.width() / 8
        x = (file + 0.5) * square_size
        y = (7 - rank + 0.5) * square_size
        return QtCore.QPointF(x, y)

    def _draw_arrow(self, painter: QtGui.QPainter, start: str, end: str, color: QtGui.QColor, width: float) -> None:
        start_pt = self._square_to_point(start)
        end_pt = self._square_to_point(end)
        painter.setPen(QtGui.QPen(color, width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawLine(start_pt, end_pt)
        direction = end_pt - start_pt
        length = math.hypot(direction.x(), direction.y())
        if length == 0:
            return
        direction /= length
        arrow_size = max(12.0, width * 3.0)
        left = QtCore.QPointF(
            end_pt.x() - direction.x() * arrow_size + direction.y() * arrow_size * 0.5,
            end_pt.y() - direction.y() * arrow_size - direction.x() * arrow_size * 0.5,
        )
        right = QtCore.QPointF(
            end_pt.x() - direction.x() * arrow_size - direction.y() * arrow_size * 0.5,
            end_pt.y() - direction.y() * arrow_size + direction.x() * arrow_size * 0.5,
        )
        path = QtGui.QPainterPath()
        path.moveTo(end_pt)
        path.lineTo(left)
        path.lineTo(right)
        path.closeSubpath()
        painter.setBrush(color)
        painter.drawPath(path)


class EvaluationOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.evaluation = 0.0
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

    def set_evaluation(self, evaluation: float) -> None:
        self.evaluation = evaluation
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        height = self.height()
        width = self.width()
        zero_y = height * 0.5
        eval_scale = max(-5.0, min(5.0, self.evaluation))
        fill_ratio = 0.5 - eval_scale / 10.0
        gradient = QtGui.QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0.0, QtGui.QColor("#53d769"))
        gradient.setColorAt(1.0, QtGui.QColor("#ff3b30"))
        painter.fillRect(QtCore.QRectF(0, 0, width, height), gradient)
        painter.fillRect(QtCore.QRectF(0, 0, width, height * fill_ratio), QtGui.QColor(0, 0, 0, 160))
        painter.setPen(QtGui.QPen(QtGui.QColor("white"), 2))
        painter.drawLine(0, zero_y, width, zero_y)
        painter.setFont(QtGui.QFont("Helvetica", 10, QtGui.QFont.Bold))
        painter.drawText(QtCore.QRectF(0, 0, width, height), QtCore.Qt.AlignCenter, f"{self.evaluation:+.2f}")


class OverlayManager(QtCore.QObject):
    def __init__(self, hint_overlay: HintOverlay, eval_overlay: EvaluationOverlay, settings: SettingsManager):
        super().__init__()
        self.hint_overlay = hint_overlay
        self.eval_overlay = eval_overlay
        self.settings = settings
        self.hint_overlay.hide()
        self.eval_overlay.hide()

    def update_eval(self, evaluation: float) -> None:
        if self.settings.get("overlay.show_eval_bar", True):
            self.eval_overlay.set_evaluation(evaluation)
            self.eval_overlay.show()
        else:
            self.eval_overlay.hide()

    def show_arrows(self, pv_lines: List[PVLine], board: chess.Board) -> None:
        if not self.settings.get("overlay.show_arrows", True):
            self.hint_overlay.clear_arrows()
            self.hint_overlay.hide()
            return
        self.hint_overlay.clear_arrows()
        theme = ThemeManager().palette()
        for idx, line in enumerate(pv_lines[: self.settings.get("overlay.show_pv_lines", 3)]):
            if not line.moves:
                continue
            move = line.moves[0]
            start = chess.square_name(move.from_square)
            end = chess.square_name(move.to_square)
            color = theme.color("arrow_good") if line.evaluation >= 0 else theme.color("arrow_blunder")
            width = 4.0 + idx * 2.5
            self.hint_overlay.add_arrow(start, end, color, width)
        self.hint_overlay.show()


class DragState:
    def __init__(self):
        self.active = False
        self.origin_square: Optional[int] = None
        self.current_pos = QtCore.QPointF()

    def start(self, square: int, pos: QtCore.QPointF) -> None:
        self.active = True
        self.origin_square = square
        self.current_pos = pos

    def update(self, pos: QtCore.QPointF) -> None:
        self.current_pos = pos

    def stop(self) -> None:
        self.active = False
        self.origin_square = None


class BoardRenderer(QtWidgets.QWidget):
    moveAttempted = QtCore.pyqtSignal(chess.Move)

    def __init__(self, board: chess.Board, theme: ThemeManager, parent=None):
        super().__init__(parent)
        self.board = board
        self.theme = theme
        self.drag = DragState()
        self.setMouseTracking(True)
        self.hints: List[chess.Move] = []
        self.square_highlights: List[str] = []

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(640, 640)

    def set_hints(self, moves: List[chess.Move]) -> None:
        self.hints = moves
        self.update()

    def set_highlights(self, squares: List[str]) -> None:
        self.square_highlights = squares
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        palette = self.theme.palette()
        square_size = min(self.width(), self.height()) / 8
        for rank in range(8):
            for file in range(8):
                color = palette.color("board_light") if (rank + file) % 2 == 0 else palette.color("board_dark")
                rect = QtCore.QRectF(file * square_size, rank * square_size, square_size, square_size)
                painter.fillRect(rect, color)
        highlight_color = QtGui.QColor(255, 215, 0, 120)
        for square_name in self.square_highlights:
            file = ord(square_name[0]) - ord("a")
            rank = 7 - (int(square_name[1]) - 1)
            rect = QtCore.QRectF(file * square_size, rank * square_size, square_size, square_size)
            painter.fillRect(rect, highlight_color)
        painter.setFont(QtGui.QFont("Arial", int(square_size * 0.45)))
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if not piece:
                continue
            file = chess.square_file(square)
            rank = 7 - chess.square_rank(square)
            rect = QtCore.QRectF(file * square_size, rank * square_size, square_size, square_size)
            painter.drawText(rect, QtCore.Qt.AlignCenter, piece.symbol())
        if self.drag.active and self.drag.origin_square is not None:
            piece = self.board.piece_at(self.drag.origin_square)
            if piece:
                painter.drawText(self.drag.current_pos, piece.symbol())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        square = self._pos_to_square(event.pos())
        if square is None:
            return
        piece = self.board.piece_at(square)
        if piece:
            self.drag.start(square, event.pos())

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.drag.active:
            self.drag.update(event.pos())
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self.drag.active:
            return
        target_square = self._pos_to_square(event.pos())
        if target_square is None or self.drag.origin_square is None:
            self.drag.stop()
            return
        move = chess.Move(self.drag.origin_square, target_square)
        if move in self.board.legal_moves:
            self.moveAttempted.emit(move)
        self.drag.stop()
        self.update()

    def _pos_to_square(self, pos: QtCore.QPointF) -> Optional[int]:
        square_size = min(self.width(), self.height()) / 8
        file = int(pos.x() // square_size)
        rank = int(pos.y() // square_size)
        if not (0 <= file < 8 and 0 <= rank < 8):
            return None
        square = chess.square(file, 7 - rank)
        return square


class GameController(QtCore.QObject):
    boardUpdated = QtCore.pyqtSignal()
    evaluationUpdated = QtCore.pyqtSignal(float)
    pvLinesUpdated = QtCore.pyqtSignal(list)

    def __init__(self, settings: SettingsManager, engine: StockfishEngine, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.engine = engine
        self.board = chess.Board()
        self.move_history = MoveHistoryModel(self.board)
        self.evaluation_model = EvaluationGraphModel()
        self.current_eval = 0.0
        self.current_pv: List[PVLine] = []
        self.training = TrainingModeController(engine)
        self.fen_history = FENHistory()
        self.board_widget: Optional[BoardRenderer] = None
        self.profiler = AppProfiler()
        self.training_active = False
        self.analysis_lock = threading.Lock()
        self.analysis_thread: Optional[threading.Thread] = None
        self.running_analysis = False

    def reset(self) -> None:
        self.board.reset()
        self.move_history.clear()
        self.evaluation_model.clear()
        self.boardUpdated.emit()

    def push_move(self, move: chess.Move) -> None:
        if move not in self.board.legal_moves:
            AppLogger.instance().warn(f"Illegal move attempted: {move}")
            return
        if self.training_active:
            if not self.training.verify_move(move):
                AppLogger.instance().warn("Training move incorrect")
                self.show_training_hint()
                return
            AppLogger.instance().info("Training move correct!")
            self.clear_training_hint()
        self.board.push(move)
        self.move_history.append_move(move)
        self.boardUpdated.emit()
        self.evaluate_position()

    def evaluate_position(self) -> None:
        def analysis_task():
            with self.analysis_lock:
                self.running_analysis = True
            self.engine.analyze(self.board.copy(), self.settings.get("move_time", 5.0))
            with self.analysis_lock:
                self.running_analysis = False

        if self.analysis_thread and self.analysis_thread.is_alive():
            return
        self.analysis_thread = threading.Thread(target=analysis_task, daemon=True)
        self.analysis_thread.start()

    def engine_play(self) -> None:
        result = self.engine.play(self.board, self.settings.get("move_time", 5.0))
        if result.move:
            self.push_move(result.move)

    def set_evaluation(self, value: float, pv_lines: List[PVLine]) -> None:
        self.current_eval = value
        self.current_pv = pv_lines
        self.evaluation_model.append(self.board.fullmove_number, value)
        self.evaluationUpdated.emit(value)
        self.pvLinesUpdated.emit(pv_lines)

    def update_from_fen(self, fen: str, confidence: float) -> None:
        start_time = time.perf_counter()
        try:
            board = chess.Board(fen)
        except Exception as exc:
            AppLogger.instance().warn(f"Invalid FEN from detection: {exc}")
            return
        if board.board_fen() == self.board.board_fen():
            return
        AppLogger.instance().info(f"HUD sync FEN (confidence {confidence:.2f}): {fen}")
        self.board = board
        self.move_history.board = self.board
        self.move_history.clear()
        self.boardUpdated.emit()
        self.fen_history.add(fen)
        self.profiler.mark("update_from_fen", (time.perf_counter() - start_time) * 1000.0)

    def undo(self) -> None:
        if self.board.move_stack:
            self.board.pop()
            if self.move_history.moves:
                self.move_history.beginRemoveRows(QtCore.QModelIndex(), len(self.move_history.moves) - 1, len(self.move_history.moves) - 1)
                self.move_history.moves.pop()
                self.move_history.endRemoveRows()
            self.boardUpdated.emit()

    def load_fen(self, fen: str) -> None:
        try:
            self.board.set_fen(fen)
            self.move_history.clear()
            self.boardUpdated.emit()
        except Exception as exc:
            AppLogger.instance().error(f"Failed to load FEN: {exc}")

    def to_pgn(self) -> str:
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        game = chess.pgn.Game.from_board(self.board)
        return game.accept(exporter)

    def set_board_widget(self, widget: BoardRenderer) -> None:
        self.board_widget = widget

    def show_training_hint(self) -> None:
        scenario = self.training.current_scenario()
        if not self.board_widget:
            return
        squares = [
            chess.square_name(scenario.target_move.from_square),
            chess.square_name(scenario.target_move.to_square),
        ]
        self.board_widget.set_highlights(squares)

    def clear_training_hint(self) -> None:
        if self.board_widget:
            self.board_widget.set_highlights([])

    def set_training_active(self, active: bool) -> None:
        self.training_active = active
        if not active:
            self.clear_training_hint()


class EvaluationWidget(QtWidgets.QWidget):
    def __init__(self, controller: GameController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.model = controller.evaluation_model
        self.table = QtWidgets.QTableView(self)
        self.table.setModel(self.model)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.table)


class HistoryWidget(QtWidgets.QWidget):
    def __init__(self, controller: GameController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.model = controller.move_history
        self.list_view = QtWidgets.QListView(self)
        self.list_view.setModel(self.model)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.list_view)


class FenViewerWidget(QtWidgets.QWidget):
    def __init__(self, controller: GameController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.text = QtWidgets.QTextEdit(self)
        self.text.setReadOnly(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.text)
        self.controller.boardUpdated.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        self.text.setPlainText(self.controller.board.fen())


class ControlPanel(QtWidgets.QWidget):
    def __init__(self, controller: GameController, engine: StockfishEngine, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.engine = engine
        layout = QtWidgets.QVBoxLayout(self)
        self.analyze_button = QtWidgets.QPushButton("Analyze")
        self.move_button = QtWidgets.QPushButton("Play Move")
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.training_button = QtWidgets.QPushButton("Next Puzzle")
        layout.addWidget(self.analyze_button)
        layout.addWidget(self.move_button)
        layout.addWidget(self.undo_button)
        layout.addWidget(self.training_button)
        self.analyze_button.clicked.connect(self.controller.evaluate_position)
        self.move_button.clicked.connect(self.controller.engine_play)
        self.undo_button.clicked.connect(self.controller.undo)
        self.training_button.clicked.connect(self._next_training)

    def _next_training(self) -> None:
        scenario = self.controller.training.next_scenario()
        self.controller.board = scenario.board.copy()
        self.controller.boardUpdated.emit()


class CHMDWindow(QtWidgets.QMainWindow):
    def __init__(self, settings: SettingsManager):
        super().__init__()
        self.settings = settings
        self.theme_manager = ThemeManager()
        self.resource_loader = ResourceLoader()
        self.icons = IconRepository(self.resource_loader)
        self.engine = StockfishEngine(settings)
        self.controller = GameController(settings, self.engine)
        self.controller.evaluationUpdated.connect(self._on_evaluation)
        self.controller.pvLinesUpdated.connect(self._on_pv)
        self.engine.infoUpdated.connect(self._on_engine_info)
        self.engine.bestMoveFound.connect(self._on_best_move)
        self.hotkeys = HotkeyManager(self)
        self.hotkeys.hotkeyTriggered.connect(self._on_hotkey)
        self.board_widget = BoardRenderer(self.controller.board, self.theme_manager)
        self.board_widget.moveAttempted.connect(self.controller.push_move)
        self.controller.set_board_widget(self.board_widget)
        self.setCentralWidget(self.board_widget)
        self.pipeline = VisionPipeline(self.settings)
        self.pipeline.fenAvailable.connect(self.controller.update_from_fen)
        self.pipeline.start()
        self.video_preview = VideoPreviewWidget(self.pipeline)
        self.hud_status = HUDStatusWidget(self.controller, self.pipeline)
        self.trainer_panel = TrainerPanel(self.controller, self.icons)
        self.trainer_dock = QtWidgets.QDockWidget("Trainer", self)
        self.trainer_dock.setWidget(self.trainer_panel)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.trainer_dock)
        self.trainer_dock.hide()
        self.toolbar = GameToolbar(self, self.icons)
        self.addToolBar(self.toolbar)
        self.gesture_handler = GestureHandler(self.board_widget)
        self.gesture_handler.toggleHints.connect(lambda: self.overlay_hint.setVisible(not self.overlay_hint.isVisible()))
        self.gesture_handler.cyclePv.connect(self._cycle_pv)
        self.gesture_handler.deepenAnalysis.connect(self._deepen_analysis)
        self.overlay_window = OverlayWindow(self.board_widget, self.settings)
        self._build_ui()
        self.overlay_hint = HintOverlay(self.board_widget)
        self.overlay_eval = EvaluationOverlay(self.board_widget)
        self.overlay_manager = OverlayManager(self.overlay_hint, self.overlay_eval, self.settings)
        self.statusBar().showMessage("Ready")
        self.audio = AudioFeedback()
        self.board_widget.moveAttempted.connect(lambda _: self.audio.play_click())
        self._setup_timers()
        self._setup_hotkeys()

    def _build_ui(self) -> None:
        dock_history = QtWidgets.QDockWidget("History", self)
        dock_history.setWidget(HistoryWidget(self.controller))
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock_history)
        dock_eval = QtWidgets.QDockWidget("Evaluation", self)
        eval_widget = QtWidgets.QWidget()
        eval_layout = QtWidgets.QVBoxLayout(eval_widget)
        eval_layout.addWidget(EvaluationWidget(self.controller))
        eval_layout.addWidget(EvaluationChart(self.controller.evaluation_model))
        dock_eval.setWidget(eval_widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock_eval)
        dock_fen = QtWidgets.QDockWidget("FEN", self)
        dock_fen.setWidget(FenViewerWidget(self.controller))
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock_fen)
        control_panel = ControlPanel(self.controller, self.engine)
        dock_controls = QtWidgets.QDockWidget("Controls", self)
        dock_controls.setWidget(control_panel)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock_controls)
        dock_video = QtWidgets.QDockWidget("Vision", self)
        video_widget = QtWidgets.QWidget()
        video_layout = QtWidgets.QVBoxLayout(video_widget)
        video_layout.addWidget(self.video_preview)
        video_layout.addWidget(self.hud_status)
        dock_video.setWidget(video_widget)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock_video)

    def _setup_timers(self) -> None:
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.timeout.connect(self._on_refresh)
        self.refresh_timer.start(250)

    def _setup_hotkeys(self) -> None:
        self.hotkeys.register_hotkey(QtCore.Qt.Key_Space, "toggle_hints")
        self.hotkeys.register_hotkey(QtCore.Qt.Key_Return, "engine_move")
        self.hotkeys.register_hotkey(QtCore.Qt.Key_Backspace, "undo")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self.hotkeys.handle_event(event):
            return
        super().keyPressEvent(event)

    def _on_hotkey(self, name: str) -> None:
        if name == "toggle_hints":
            self.overlay_hint.setVisible(not self.overlay_hint.isVisible())
        elif name == "engine_move":
            self.controller.engine_play()
        elif name == "undo":
            self.controller.undo()

    def _on_refresh(self) -> None:
        self.board_widget.update()
        self.overlay_window.update_geometry()

    def _on_engine_info(self, info: dict) -> None:
        score = info.get("score")
        evaluation = 0.0
        if score:
            evaluation = score.white().score(mate_score=100000) / 100.0
        pv = info.get("pv", [])
        pv_lines = []
        if pv:
            pv_lines.append(PVLine(evaluation, info.get("depth", 0), list(pv)))
        self.controller.set_evaluation(evaluation, pv_lines)

    def _on_best_move(self, move: chess.Move, evaluation: float) -> None:
        self.statusBar().showMessage(f"Best move: {move.uci()} ({evaluation:+.2f})")

    def _on_evaluation(self, value: float) -> None:
        self.overlay_manager.update_eval(value)

    def _on_pv(self, pv_lines: List[PVLine]) -> None:
        self.overlay_manager.show_arrows(pv_lines, self.controller.board)

    def _cycle_pv(self) -> None:
        if not self.controller.current_pv:
            return
        pv = self.controller.current_pv
        pv.append(pv.pop(0))
        self.overlay_manager.show_arrows(pv, self.controller.board)

    def _deepen_analysis(self) -> None:
        move_time = min(30.0, self.settings.get("move_time", 5.0) * 2.0)
        self.settings.set("move_time", move_time)
        self.controller.evaluate_position()
        self.statusBar().showMessage(f"Deep analysis triggered ({move_time:.1f}s)")

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self.icons, self)
        if dialog.exec_():
            self.statusBar().showMessage("Settings applied", 2000)

    def _open_trainer(self) -> None:
        if self.trainer_dock.isVisible():
            self.trainer_dock.hide()
            self.controller.set_training_active(False)
        else:
            self.trainer_dock.show()
            self.controller.set_training_active(True)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.pipeline.stop()
        self.engine.shutdown()
        super().closeEvent(event)


def main() -> None:
    settings_path = os.path.join(os.path.expanduser("~"), ".chmd_settings.json")
    settings = SettingsManager(settings_path)
    app = QtWidgets.QApplication(sys.argv)
    window = CHMDWindow(settings)
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


class GestureHandler(QtCore.QObject):
    toggleHints = QtCore.pyqtSignal()
    cyclePv = QtCore.pyqtSignal()
    deepenAnalysis = QtCore.pyqtSignal()

    def __init__(self, target: QtWidgets.QWidget):
        super().__init__(target)
        self.target = target
        self.target.installEventFilter(self)
        self.last_tap = 0.0
        self.tap_count = 0
        self.press_start = 0.0
        self.hold_threshold = 0.75

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self.target:
            if event.type() == QtCore.QEvent.MouseButtonPress:
                self.press_start = time.time()
                return False
            if event.type() == QtCore.QEvent.MouseButtonRelease:
                now = time.time()
                duration = now - self.press_start
                if duration >= self.hold_threshold:
                    self.deepenAnalysis.emit()
                    return True
                if now - self.last_tap <= 0.35:
                    self.tap_count += 1
                else:
                    self.tap_count = 1
                self.last_tap = now
                if self.tap_count == 1:
                    self.toggleHints.emit()
                    return True
                if self.tap_count == 2:
                    self.cyclePv.emit()
                    self.tap_count = 0
                    return True
        return False


class TrainerStats:
    def __init__(self):
        self.attempts = 0
        self.successes = 0
        self.total_time = 0.0
        self.last_start = time.time()

    def start_attempt(self) -> None:
        self.last_start = time.time()
        self.attempts += 1

    def record_success(self) -> None:
        self.successes += 1
        self.total_time += time.time() - self.last_start

    def record_failure(self) -> None:
        self.total_time += time.time() - self.last_start

    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    def average_time(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.total_time / self.attempts


class FENHistory:
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.entries: Deque[str] = collections.deque(maxlen=capacity)

    def add(self, fen: str) -> None:
        self.entries.append(fen)

    def last(self) -> Optional[str]:
        return self.entries[-1] if self.entries else None

    def to_list(self) -> List[str]:
        return list(self.entries)


class BoardSyncWorker(QtCore.QObject):
    fenDetected = QtCore.pyqtSignal(str, float)
    detectionFailed = QtCore.pyqtSignal()

    def __init__(self, analyzer: BoardAnalyzer, classifier: PieceClassifier, fen_builder: FENBuilder):
        super().__init__()
        self.analyzer = analyzer
        self.classifier = classifier
        self.fen_builder = fen_builder
        self.queue: "queue.Queue[VisionFrame]" = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread.start()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.thread.join(timeout=2.0)

    def submit(self, frame: VisionFrame) -> None:
        if not self.running:
            return
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(frame)

    def _loop(self) -> None:
        while self.running:
            try:
                frame = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                detection = self.analyzer.detect_board(frame.frame)
                if not detection:
                    self.detectionFailed.emit()
                    continue
                pieces = self.classifier.classify_board(detection)
                fen = self.fen_builder.build_fen(pieces.squares)
                self.fenDetected.emit(fen, pieces.confidence)
            except Exception as exc:
                AppLogger.instance().error(f"Board sync failed: {exc}")
                self.detectionFailed.emit()


class VisionPipeline(QtCore.QObject):
    fenAvailable = QtCore.pyqtSignal(str, float)
    frameReady = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, settings: SettingsManager):
        super().__init__()
        self.settings = settings
        self.analyzer = BoardAnalyzer(settings)
        self.classifier = PieceClassifier(settings)
        self.fen_builder = FENBuilder()
        self.worker = BoardSyncWorker(self.analyzer, self.classifier, self.fen_builder)
        self.worker.fenDetected.connect(self.fenAvailable.emit)
        self.worker.detectionFailed.connect(lambda: AppLogger.instance().trace("Detection failed"))
        self.camera = CameraManager(settings)
        self.screen = ScreenCaptureManager(settings)
        self.camera.frameCaptured.connect(self._on_frame)
        self.screen.frameCaptured.connect(self._on_frame)
        self.active_source = "screen"
        self.worker.start()

    def start(self) -> None:
        if self.settings.get("vision.use_camera", False):
            self.camera.start()
            self.active_source = "camera"
        else:
            self.screen.start()
            self.active_source = "screen"

    def stop(self) -> None:
        self.camera.stop()
        self.screen.stop()
        self.worker.stop()

    def _on_frame(self, frame: VisionFrame) -> None:
        self.frameReady.emit(frame.frame)
        self.worker.submit(frame)


class IconRepository:
    def __init__(self, loader: ResourceLoader):
        self.loader = loader
        self.icons: Dict[str, QtGui.QIcon] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        icons = {
            "play": "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABSElEQVQ4T6XTT0iUURzH8Y+7SakkSDEE5w40E0yDSNEgWFhZlKJs/gH+AxTRNFYtFZg0VhY2Fg2Kv1CVGwmIyZiaZj8bixQxhqYGIC8UIr+YevXvec7nO8Xrt3vt/3e9/7vt/7fSmcH+Mc6z/M/47WCD8cQTd6HHM4g9imSZng3eYxT9vJPi1UbmESe4wL32Me3WcQxH8E8wDYr1GgWHqGuYBXuyHf8SiBB5jlmM46Gsc2bgIfMlRnGHOPawTObpXId2TYl0L1nBeYsO0+g4xH3kJ7jOQbg+4zqkiZxhzgfsYzxsYzTeSIA/0wUVYbIGGGP8L5XGMsOU8TgJNT7jXWNw7hF6J+4zS3G1xjCv6wGHuO8Z5iPBcYwa8bTIJgG/9CUOYxOLj/sY31nHPKYTQzgJfcYV9hPtdwV5hFQ6xC/4x7Sb3WcQ9J1wBpe4AN6yJvWM9zXOIXF78wY41k3yNnjYzwF8jyTQ3J+xFT7SGuYTzTCIt9ZxDSfNSRh7hvMd5zjPzAD7jktZvYxXeS84xfYPcYy4xcBVyV5hoyAAAAAElFTkSuQmCC",
            "analyze": "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABUUlEQVQ4T6XTQUpDQRQF4M+7eAg2gAzWDZtZ2RmSbGxtNWsEnkAG3gBNbGxs7KwMW1nZqGzCwskJdbCI0s0NDUxixV+CQ52AiWHTPv32/Xjmdvbfd933fd/7vmzJ6LL/BM/M47zDP8JG9hAD+YyGOs4nVMq6QB3uMWNcxB1nG3HLGLPlZWmtwfMM88zcVljK83uMQt2cIZwPI14LGOE7TP5hfhvFdknVzjENuYzoV6/Yx7Vxjlf4xUM2cc1Zzjpug3Y4wbn4x7mMHdZx/5XxP2OQ3c4x45neMc+4xxnNpDctwMnFbiTDn4E/tjF39zjFT5hk3YPqOIt+8xb8Yy7lwg7qsBm/LMLd9hfi8S1JTxwD7XGIUO8yEn6zH3nGMtT5lR4xnDPvGMGeI8YXOA8MB9hjBaiTsSAf4GFpGPc77BLpX4GJ/wBZ1jO8zD2uYw6soXp3RdX6KKy5MuYZ3ZxzQXjM/wRfwFZlgjL1FmQpAAAAAElFTkSuQmCC",
            "puzzle": "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABR0lEQVQ4T6XTMWoCURSF4Z+7iI1oBwsLLa0MGxsZsRDNLa0iGBhaWlkYWhmCCKKigb6CBvoIDa1FKrFTFaXk3foGO9a9n3v2vfOc95/znnPOc85iJ8FJ9jCB/A83mMvdzjnDLBDTmMeebEMX7jFHGcY13mAe2c47wa0/sYV7iFLtYRbnGFd03MdRlhkBd5hF+0Q3kXuwiMM7YxUOcz7l+Iy7mHr9VjE2ccC8Zxzm4x0RcV7mKf4wcs5l4Bvs4jJWOczb+YxT5iC84g6OcezjEPuYSiPxPDAcZbHbH6ZrxmKccSyfiKe4ifCzGqi8VvmElvlZtnGM+4SDdYxT5TAFNHcY0eMD9nEGf4wVWMc/MZ7jE/mMXnGLj3ANji7psQdvYTmUM1fkV5hGP+DqzzTJPCcZQA95kQZ6wX3uM5N3h/G9tiHPMLqCHt9DH13Ac5jkv0M9jGf4yLP4wQHgmnMfCzG5RxkVuQ+c5g81w0yf7AAAAAElFTkSuQmCC",
            "settings": "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAABTElEQVQ4T6XTP0jUYRzH8c8cx4zjVrGJYaIIQzI0X9AZlGIglZkC7IxNUsC5po0X8GzMwyWrdCFdOnVmRrFguqpFi4C9w4n6ILKYPOeZ3Pue/fe873n9zv+83+e4fzfCG9jAFvgIb7jFg7mMNdWZOwg33MG3nEGeZw52cY537CKy7h4EdrmOcdzDviFzu4xluYx28BybSJ1fYw17jDF7jEt6ww3mMXvYRu7jHsZwhd9jFGdzE1/BXDuI3VnGeuYwb5jE3WcU+yhN/mMV/GUr8x13mMDk3Yxl7jBV6zCY1m8wHbPBHIcxb+If1jHd6wAXuIb3iJu4hXO4RqMMZ+wzdtwnzQwT7jEDWcx09s43v8YybcYy/Zn2UX2zjH0JYzU9ZjIXGcxJf4y9vYRoW4hku4hvu4x14zgmf4x08AAD//8VvqBTTAAAAAElFTkSuQmCC",
        }
        for name, payload in icons.items():
            self.icons[name] = self.loader.icon_from_base64(name, payload)

    def icon(self, name: str) -> QtGui.QIcon:
        return self.icons[name]


PUZZLE_DATA = [
    {
        "name": "Mate in 1",
        "fen": "6k1/5ppp/8/8/8/6Q1/5PPP/6K1 w - - 0 1",
        "best": "Qb8#",
        "description": "Deliver mate on b8 with the queen."
    },
    {
        "name": "Fork Tactic",
        "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 2 5",
        "best": "Nxe5",
        "description": "Use the knight to fork queen and bishop."
    },
    {
        "name": "Skewer Attack",
        "fen": "4r1k1/1p3ppp/p1n5/3q4/8/1B3N2/PP3PPP/2RQ2K1 w - - 4 20",
        "best": "Bxd5",
        "description": "Skewer the queen and king along the diagonal."
    },
    {
        "name": "Back Rank Theme",
        "fen": "r3r1k1/1p3ppp/p1n5/3p4/3P4/1QN2N2/PP3PPP/2R1R1K1 w - - 0 20",
        "best": "Rxe8+",
        "description": "Exploit the back rank weakness."
    },
    {
        "name": "Deflection Idea",
        "fen": "2r2rk1/1bq1bppp/p2ppn2/1p6/3NP3/1QN2B2/PP3PPP/2RR2K1 w - - 0 18",
        "best": "Nxe6",
        "description": "Deflect the defender on e7 to win material."
    },
    {
        "name": "Discovered Check",
        "fen": "r2q1rk1/1pp2ppp/p1npbn2/8/2B1P3/2NP1N2/PP3PPP/R1BQ1RK1 w - - 4 11",
        "best": "Bxe6",
        "description": "Open the diagonal for a discovered attack." 
    },
]

# Extend puzzles to reach 120 entries for rich training data
for idx in range(3, 121):
    board = chess.Board()
    moves = list(board.legal_moves)
    random.shuffle(moves)
    move = moves[0]
    puzzle = {
        "name": f"Starter #{idx}",
        "fen": board.fen(),
        "best": board.san(move),
        "description": f"Practice tactic {idx} with a random start position."
    }
    PUZZLE_DATA.append(puzzle)


class TrainingModeController:
    def __init__(self, engine: StockfishEngine):
        self.engine = engine
        self.scenarios: List[TrainingScenario] = self._build_from_data()
        self.active_index = 0
        self.stats = TrainerStats()

    def _build_from_data(self) -> List[TrainingScenario]:
        scenarios = []
        for data in PUZZLE_DATA:
            board = chess.Board(data["fen"])
            try:
                move = board.parse_san(data["best"])
            except Exception:
                move = random.choice(list(board.legal_moves))
            scenarios.append(TrainingScenario(data["name"], board, move, data["description"]))
        return scenarios

    def next_scenario(self) -> TrainingScenario:
        self.stats.start_attempt()
        self.active_index = (self.active_index + 1) % len(self.scenarios)
        return self.scenarios[self.active_index]

    def current_scenario(self) -> TrainingScenario:
        return self.scenarios[self.active_index]

    def verify_move(self, move: chess.Move) -> bool:
        scenario = self.current_scenario()
        if move == scenario.target_move:
            self.stats.record_success()
            return True
        self.stats.record_failure()
        return False


class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, board_widget: BoardRenderer, settings: SettingsManager):
        super().__init__(board_widget)
        self.settings = settings
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.Tool
            | QtCore.Qt.BypassWindowManagerHint
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.resize(board_widget.size())
        self.move(board_widget.mapToGlobal(QtCore.QPoint(0, 0)))
        self.show()

    def update_geometry(self) -> None:
        if not self.settings.get("overlay.always_on_top", True):
            self.hide()
            return
        self.setGeometry(self.parentWidget().geometry())
        self.raise_()
        self.show()


class EvaluationChart(QtWidgets.QWidget):
    def __init__(self, model: EvaluationGraphModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.model.modelReset.connect(self.update)
        self.model.rowsInserted.connect(self.update)
        self.setMinimumHeight(120)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(18, 18, 28))
        if not self.model.samples:
            painter.setPen(QtGui.QColor("white"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "No data")
            return
        width = self.width()
        height = self.height()
        max_eval = max(abs(value) for _, value in self.model.samples)
        max_eval = max(max_eval, 0.5)
        points = []
        for idx, (turn, value) in enumerate(self.model.samples):
            x = width * idx / max(1, len(self.model.samples) - 1)
            y = height / 2 - (value / max_eval) * (height / 2 * 0.9)
            points.append(QtCore.QPointF(x, y))
        pen = QtGui.QPen(QtGui.QColor("#7b61ff"), 2)
        painter.setPen(pen)
        path = QtGui.QPainterPath()
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)
        painter.drawPath(path)
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1, QtCore.Qt.DashLine))
        painter.drawLine(0, height / 2, width, height / 2)


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, settings: SettingsManager, icons: IconRepository, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.icons = icons
        self.setWindowTitle("CHMD Settings")
        self.resize(400, 300)
        layout = QtWidgets.QFormLayout(self)
        self.engine_path = QtWidgets.QLineEdit(self.settings.get("engine_path", "stockfish"))
        self.difficulty_spin = QtWidgets.QSpinBox()
        self.difficulty_spin.setRange(1, 20)
        self.difficulty_spin.setValue(int(self.settings.get("difficulty", 10)))
        self.move_time_spin = QtWidgets.QDoubleSpinBox()
        self.move_time_spin.setRange(0.1, 30.0)
        self.move_time_spin.setValue(float(self.settings.get("move_time", 5.0)))
        self.camera_check = QtWidgets.QCheckBox("Use Camera")
        self.camera_check.setChecked(self.settings.get("vision.use_camera", False))
        layout.addRow("Engine Path", self.engine_path)
        layout.addRow("Difficulty", self.difficulty_spin)
        layout.addRow("Move Time", self.move_time_spin)
        layout.addRow(self.camera_check)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def accept(self) -> None:
        self.settings.set("engine_path", self.engine_path.text())
        self.settings.set("difficulty", self.difficulty_spin.value())
        self.settings.set("move_time", self.move_time_spin.value())
        self.settings.set("vision.use_camera", self.camera_check.isChecked())
        super().accept()


class TrainerPanel(QtWidgets.QWidget):
    def __init__(self, controller: GameController, icons: IconRepository, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.icons = icons
        layout = QtWidgets.QVBoxLayout(self)
        self.description = QtWidgets.QLabel("Select a puzzle to begin")
        self.progress = QtWidgets.QLabel("")
        self.next_button = QtWidgets.QPushButton(self.icons.icon("puzzle"), "Next Puzzle")
        self.hint_button = QtWidgets.QPushButton(self.icons.icon("analyze"), "Hint")
        layout.addWidget(self.description)
        layout.addWidget(self.progress)
        layout.addWidget(self.next_button)
        layout.addWidget(self.hint_button)
        self.next_button.clicked.connect(self._next)
        self.hint_button.clicked.connect(self._hint)
        self.controller.boardUpdated.connect(self._update)
        self._update()

    def _next(self) -> None:
        scenario = self.controller.training.next_scenario()
        self.controller.board = scenario.board.copy()
        self.controller.move_history.clear()
        self.controller.boardUpdated.emit()
        self.controller.set_training_active(True)
        self._update()

    def _hint(self) -> None:
        scenario = self.controller.training.current_scenario()
        AppLogger.instance().info(f"Hint: try {scenario.target_move.uci()}")
        self.controller.show_training_hint()

    def _update(self) -> None:
        scenario = self.controller.training.current_scenario()
        stats = self.controller.training.stats
        self.description.setText(scenario.description)
        self.progress.setText(
            f"Attempts: {stats.attempts}, Successes: {stats.successes}, Success rate: {stats.success_rate():.1%}"
        )


class VideoPreviewWidget(QtWidgets.QLabel):
    def __init__(self, pipeline: VisionPipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.pipeline.frameReady.connect(self._on_frame)
        self.setMinimumSize(320, 240)
        self.setScaledContents(True)

    def _on_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.setPixmap(pixmap)


class HUDStatusWidget(QtWidgets.QWidget):
    def __init__(self, controller: GameController, pipeline: VisionPipeline, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.pipeline = pipeline
        layout = QtWidgets.QFormLayout(self)
        self.fen_label = QtWidgets.QLabel("FEN: -")
        self.confidence_label = QtWidgets.QLabel("Confidence: 0.0")
        layout.addRow(self.fen_label)
        layout.addRow(self.confidence_label)
        self.pipeline.fenAvailable.connect(self._on_fen)

    def _on_fen(self, fen: str, confidence: float) -> None:
        self.fen_label.setText(f"FEN: {fen}")
        self.confidence_label.setText(f"Confidence: {confidence:.2f}")


class GameToolbar(QtWidgets.QToolBar):
    def __init__(self, window: CHMDWindow, icons: IconRepository):
        super().__init__("CHMD Controls", window)
        self.window = window
        self.icons = icons
        self.setMovable(False)
        self.setIconSize(QtCore.QSize(24, 24))
        analyze_action = self.addAction(self.icons.icon("analyze"), "Analyze")
        move_action = self.addAction(self.icons.icon("play"), "Play")
        puzzle_action = self.addAction(self.icons.icon("puzzle"), "Puzzle")
        settings_action = self.addAction(self.icons.icon("settings"), "Settings")
        analyze_action.triggered.connect(window.controller.evaluate_position)
        move_action.triggered.connect(window.controller.engine_play)
        puzzle_action.triggered.connect(window._open_trainer)
        settings_action.triggered.connect(window._open_settings)


class AppProfiler:
    def __init__(self):
        self.records: Dict[str, List[float]] = collections.defaultdict(list)

    def mark(self, name: str, duration: float) -> None:
        self.records[name].append(duration)

    def report(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "min": float(min(values)),
                "max": float(max(values)),
                "avg": float(sum(values) / len(values)),
                "count": len(values),
            }
            for name, values in self.records.items()
            if values
        }


class AppDiagnostics(QtWidgets.QDialog):
    def __init__(self, profiler: AppProfiler, parent=None):
        super().__init__(parent)
        self.profiler = profiler
        self.setWindowTitle("Diagnostics")
        self.resize(400, 300)
        layout = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QPlainTextEdit(self)
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        layout.addWidget(self.refresh_button)
        self.refresh_button.clicked.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        report = self.profiler.report()
        lines = []
        for name, stats in report.items():
            lines.append(
                f"{name}: min={stats['min']:.3f}ms max={stats['max']:.3f}ms avg={stats['avg']:.3f}ms samples={stats['count']}"
            )
        self.text.setPlainText("\n".join(lines) if lines else "No data")
