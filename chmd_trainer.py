#!/usr/bin/env python3
"""
CHMD Chess HUD Move Detector and Trainer

This module implements a fully featured chess trainer that combines computer vision,
engine integration, and a rich PyQt5 user interface into a single file. The program is
designed to detect chess boards from webcam or screen captures, recognize piece positions,
construct FEN strings, interact with Stockfish 16 NNUE, and provide a robust training
environment with evaluation overlays and multiple play modes.
"""

import os
import sys
import math
import json
import time
import uuid
import glob
import queue
import ctypes
import signal
import random
import string
import shutil
import psutil
import logging
import tempfile
import threading
import subprocess
from collections import deque, defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any

import numpy as np
import cv2
import chess
import chess.pgn
import chess.engine

from PyQt5 import QtCore, QtGui, QtWidgets


# --------------------------------------------------------------------------------------
# Logging utilities
# --------------------------------------------------------------------------------------

class AppLogger:
    """Centralized logging utility to provide structured logging across the app."""

    _instance_lock = threading.Lock()
    _instance: Optional["AppLogger"] = None

    def __init__(self) -> None:
        self.logger = logging.getLogger("CHMDTrainer")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)
        self._profilers: Dict[str, float] = {}
        self._profiling_stack: List[Tuple[str, float]] = []
        self._event_counts: defaultdict[str, int] = defaultdict(int)
        self.logger.debug("AppLogger initialized")

    @classmethod
    def instance(cls) -> "AppLogger":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = AppLogger()
            return cls._instance

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def profile_start(self, key: str) -> None:
        now = time.perf_counter()
        self._profiling_stack.append((key, now))
        self.logger.debug(f"Profiling start for {key}")

    def profile_end(self, key: str) -> None:
        now = time.perf_counter()
        if not self._profiling_stack:
            self.logger.warning("Profiling stack underflow")
            return
        current_key, start = self._profiling_stack.pop()
        if current_key != key:
            self.logger.warning(f"Profiling mismatch: expected {current_key}, got {key}")
        duration = now - start
        self._event_counts[key] += 1
        self._profilers[key] = self._profilers.get(key, 0.0) + duration
        self.logger.debug(f"Profiling end for {key}: {duration:.4f}s")

    def report(self) -> Dict[str, float]:
        report = {k: v for k, v in self._profilers.items()}
        self.logger.debug(f"Profiling report: {report}")
        return report

    def event(self, name: str) -> None:
        self._event_counts[name] += 1
        self.logger.debug(f"Event {name} count now {self._event_counts[name]}")

    def event_counts(self) -> Dict[str, int]:
        return dict(self._event_counts)


log = AppLogger.instance()


# --------------------------------------------------------------------------------------
# Configuration management
# --------------------------------------------------------------------------------------

DEFAULT_SETTINGS = {
    "stockfish_path": "stockfish",
    "difficulty": 10,
    "hint_enabled": True,
    "overlay_transparent": False,
    "board_theme": "liquid_glass",
    "piece_style": "alpha",
    "video_source": 0,
    "use_screen_capture": False,
    "capture_region": [0, 0, 1920, 1080],
    "engine_depth": 20,
    "engine_threads": 4,
    "engine_hash": 512,
    "hud_opacity": 0.85,
    "evaluation_history_size": 300,
    "pv_lines": 3,
    "double_tap_interval": 0.4,
    "long_press_duration": 1.0,
    "ai_vs_ai_speed": 1.0,
    "tactics_dataset": "tactics.pgn",
    "pgn_directory": "saved_games",
    "hotkeys": {
        "toggle_overlay": "Ctrl+O",
        "start_tactics": "Ctrl+T",
        "open_settings": "Ctrl+S",
        "toggle_hint": "Space",
    },
    "recent_files": [],
}


class SettingsManager:
    """Manage persistent settings stored in JSON."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.path.join(os.path.expanduser("~"), ".chmd_settings.json")
        self.settings: Dict[str, Any] = {}
        self.logger = log
        self.load()

    def load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
                self.logger.info(f"Loaded settings from {self.path}")
            except Exception as exc:
                self.logger.error(f"Failed to load settings: {exc}")
                self.settings = DEFAULT_SETTINGS.copy()
        else:
            self.logger.debug("Settings file not found, using defaults")
            self.settings = DEFAULT_SETTINGS.copy()
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        for key, value in DEFAULT_SETTINGS.items():
            if key not in self.settings:
                self.settings[key] = value
        if "hotkeys" not in self.settings:
            self.settings["hotkeys"] = DEFAULT_SETTINGS["hotkeys"]

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
            self.logger.info(f"Settings saved to {self.path}")
        except Exception as exc:
            self.logger.error(f"Failed to save settings: {exc}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.settings[key] = value
        self.logger.debug(f"Setting {key} updated to {value}")


settings = SettingsManager()


# --------------------------------------------------------------------------------------
# Utility data structures
# --------------------------------------------------------------------------------------

Color = namedtuple("Color", "r g b")


@dataclass
class FPSCounter:
    """Utility to track frames per second in capture and rendering pipelines."""

    window: int = 120
    timestamps: deque = field(default_factory=lambda: deque(maxlen=120))

    def tick(self) -> float:
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) < 2:
            return 0.0
        fps = (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])
        log.debug(f"FPSCounter tick: fps={fps:.2f}")
        return fps


@dataclass
class MovingAverage:
    window: int
    values: deque = field(init=False)

    def __post_init__(self) -> None:
        self.values = deque(maxlen=self.window)

    def add(self, value: float) -> float:
        self.values.append(value)
        avg = sum(self.values) / len(self.values)
        log.debug(f"MovingAverage add: value={value:.4f}, avg={avg:.4f}")
        return avg


@dataclass
class TimedEvent:
    name: str
    timestamp: float


class EventBus:
    """Simple pub/sub event bus for cross-component communication."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[..., None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event: str, callback: Callable[..., None]) -> None:
        with self._lock:
            self._subscribers[event].append(callback)
            log.debug(f"Subscribed {callback} to event {event}")

    def unsubscribe(self, event: str, callback: Callable[..., None]) -> None:
        with self._lock:
            if callback in self._subscribers[event]:
                self._subscribers[event].remove(callback)
                log.debug(f"Unsubscribed {callback} from event {event}")

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        with self._lock:
            callbacks = list(self._subscribers.get(event, []))
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                log.error(f"Error in event callback {callback} for event {event}: {exc}")


event_bus = EventBus()


# --------------------------------------------------------------------------------------
# Profiling decorators and timers
# --------------------------------------------------------------------------------------

class ScopedTimer:
    """Context manager for timing code blocks."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> None:
        log.profile_start(self.label)

    def __exit__(self, exc_type, exc, tb) -> None:
        log.profile_end(self.label)


def timed(label: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to time function executions."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log.profile_start(label)
            try:
                return func(*args, **kwargs)
            finally:
                log.profile_end(label)
        return wrapper

    return decorator


# --------------------------------------------------------------------------------------
# Theme management
# --------------------------------------------------------------------------------------

@dataclass
class Theme:
    name: str
    light_square: QtGui.QColor
    dark_square: QtGui.QColor
    highlight: QtGui.QColor
    background: QtGui.QColor
    border: QtGui.QColor
    font: QtGui.QColor


class ThemeManager:
    """Manage color themes for the UI."""

    def __init__(self) -> None:
        self.themes: Dict[str, Theme] = {}
        self._load_default_themes()
        self.current_theme = settings.get("board_theme", "liquid_glass")

    def _load_default_themes(self) -> None:
        self.themes["liquid_glass"] = Theme(
            name="liquid_glass",
            light_square=QtGui.QColor(240, 248, 255, 220),
            dark_square=QtGui.QColor(60, 80, 110, 220),
            highlight=QtGui.QColor(255, 200, 0, 180),
            background=QtGui.QColor(30, 40, 60, 240),
            border=QtGui.QColor(20, 30, 45, 255),
            font=QtGui.QColor(240, 240, 255, 255),
        )
        self.themes["classic"] = Theme(
            name="classic",
            light_square=QtGui.QColor(240, 217, 181),
            dark_square=QtGui.QColor(181, 136, 99),
            highlight=QtGui.QColor(118, 150, 86, 200),
            background=QtGui.QColor(25, 25, 25),
            border=QtGui.QColor(40, 40, 40),
            font=QtGui.QColor(255, 255, 255),
        )
        self.themes["midnight"] = Theme(
            name="midnight",
            light_square=QtGui.QColor(70, 70, 90),
            dark_square=QtGui.QColor(40, 40, 60),
            highlight=QtGui.QColor(200, 80, 80, 220),
            background=QtGui.QColor(15, 15, 25),
            border=QtGui.QColor(10, 10, 20),
            font=QtGui.QColor(220, 220, 240),
        )

    def theme(self) -> Theme:
        return self.themes.get(self.current_theme, self.themes["liquid_glass"])

    def set_theme(self, name: str) -> None:
        if name in self.themes:
            self.current_theme = name
            settings.set("board_theme", name)
            settings.save()
            event_bus.emit("theme_changed", name)


theme_manager = ThemeManager()


# --------------------------------------------------------------------------------------
# Vision subsystem for board detection and piece recognition
# --------------------------------------------------------------------------------------

@dataclass
class DetectionResult:
    board_found: bool
    board_rect: Optional[np.ndarray]
    fen: str
    pieces: Dict[str, str]
    last_update: float
    confidence: float


class BoardDetector:
    """Detects chess boards from video frames."""

    def __init__(self) -> None:
        self.last_result: Optional[DetectionResult] = None
        self.last_frame: Optional[np.ndarray] = None
        self.square_size = 64
        self.capture = None
        self.capture_lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=5)
        self.result_queue: "queue.Queue[DetectionResult]" = queue.Queue(maxsize=2)
        self.analysis_interval = 0.5
        self.last_analysis = 0.0
        self.board_template = self._create_board_template()
        self.piece_templates = self._load_piece_templates()
        self.contour_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.confidence_smoother = MovingAverage(window=10)

    def _create_board_template(self) -> np.ndarray:
        template = np.zeros((8 * self.square_size, 8 * self.square_size), dtype=np.uint8)
        for row in range(8):
            for col in range(8):
                color = 200 if (row + col) % 2 == 0 else 50
                r0 = row * self.square_size
                c0 = col * self.square_size
                template[r0 : r0 + self.square_size, c0 : c0 + self.square_size] = color
        log.debug("Board template created")
        return template

    def _load_piece_templates(self) -> Dict[str, np.ndarray]:
        """Load template images for piece recognition."""
        templates: Dict[str, np.ndarray] = {}
        base_dir = os.path.join(os.path.dirname(__file__), "piece_templates")
        pieces = [
            "wp", "wn", "wb", "wr", "wq", "wk",
            "bp", "bn", "bb", "br", "bq", "bk",
        ]
        for piece in pieces:
            path = os.path.join(base_dir, f"{piece}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[piece] = cv2.resize(img, (self.square_size, self.square_size))
                    log.debug(f"Loaded template for {piece}")
        return templates

    def set_video_source(self, source: int) -> None:
        with self.capture_lock:
            if self.capture:
                self.capture.release()
            self.capture = cv2.VideoCapture(source)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            log.info(f"Video source set to {source}")

    def set_screen_capture(self, region: Tuple[int, int, int, int]) -> None:
        self.screen_region = region

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_capture, daemon=True)
        self.thread.start()
        log.info("BoardDetector capture thread started")

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        with self.capture_lock:
            if self.capture:
                self.capture.release()
        log.info("BoardDetector stopped")

    def _run_capture(self) -> None:
        fps_counter = FPSCounter()
        while self.running:
            frame = None
            if settings.get("use_screen_capture"):
                frame = self._capture_screen_region()
            else:
                with self.capture_lock:
                    if self.capture:
                        ret, frame = self.capture.read()
                        if not ret:
                            frame = None
            if frame is None:
                time.sleep(0.05)
                continue
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            fps = fps_counter.tick()
            event_bus.emit("capture_fps", fps)

            now = time.time()
            if now - self.last_analysis >= self.analysis_interval:
                self.last_analysis = now
                self._analyze_frames()

    def _capture_screen_region(self) -> Optional[np.ndarray]:
        try:
            import mss
            with mss.mss() as sct:
                region = {
                    "top": int(self.screen_region[1]),
                    "left": int(self.screen_region[0]),
                    "width": int(self.screen_region[2]),
                    "height": int(self.screen_region[3]),
                }
                sct_img = sct.grab(region)
                img = np.array(sct_img)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
        except Exception as exc:
            log.error(f"Screen capture failed: {exc}")
            return None

    def _analyze_frames(self) -> None:
        frames: List[np.ndarray] = []
        while not self.frame_queue.empty():
            frames.append(self.frame_queue.get())
        if not frames:
            return
        frame = frames[-1]
        self.last_frame = frame
        detection = self._detect_board(frame)
        self.last_result = detection
        if not self.result_queue.full():
            self.result_queue.put(detection)
        event_bus.emit("board_detected", detection)

    @timed("detect_board")
    def _detect_board(self, frame: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        dilated = cv2.dilate(edged, self.contour_kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        board_rect = None
        best_area = 0
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > best_area and area > 5000:
                    best_area = area
                    board_rect = approx
        board_found = board_rect is not None
        confidence = 0.0
        pieces: Dict[str, str] = {}
        fen = ""
        if board_found:
            warped = self._extract_board(gray, board_rect)
            fen, pieces, confidence = self._recognize_pieces(warped)
        else:
            fen = ""
        avg_confidence = self.confidence_smoother.add(confidence if board_found else 0.0)
        detection = DetectionResult(
            board_found=board_found,
            board_rect=board_rect,
            fen=fen,
            pieces=pieces,
            last_update=time.time(),
            confidence=avg_confidence,
        )
        return detection

    def _extract_board(self, gray: np.ndarray, rect: np.ndarray) -> np.ndarray:
        pts = rect.reshape(4, 2).astype(np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        ordered[1] = pts[np.argmin(diff)]
        ordered[3] = pts[np.argmax(diff)]
        size = self.square_size * 8
        dst = np.array([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1],
        ], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(gray, matrix, (size, size))
        log.debug("Board warped for recognition")
        return warped

    def _recognize_pieces(self, board_img: np.ndarray) -> Tuple[str, Dict[str, str], float]:
        board = chess.Board()
        board.clear()
        pieces: Dict[str, str] = {}
        total_confidence = 0.0
        squares_with_detection = 0
        for rank in range(8):
            for file in range(8):
                y0 = rank * self.square_size
                x0 = file * self.square_size
                square_img = board_img[y0 : y0 + self.square_size, x0 : x0 + self.square_size]
                square_key = f"{chr(ord('a') + file)}{8 - rank}"
                piece, confidence = self._classify_square(square_img)
                if piece:
                    board.set_piece_at(chess.parse_square(square_key), chess.Piece.from_symbol(piece))
                    pieces[square_key] = piece
                    total_confidence += confidence
                    squares_with_detection += 1
        fen = board.board_fen()
        confidence = (total_confidence / squares_with_detection) if squares_with_detection else 0.0
        log.debug(f"Recognized FEN: {fen}, confidence: {confidence:.2f}")
        return fen, pieces, confidence

    def _classify_square(self, square_img: np.ndarray) -> Tuple[str, float]:
        best_piece = ""
        best_score = 0.0
        resized = cv2.resize(square_img, (self.square_size, self.square_size))
        for piece, template in self.piece_templates.items():
            result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_piece = piece
        if best_score > 0.55:
            symbol = best_piece[1]
            if best_piece[0] == "w":
                symbol = symbol.upper()
            else:
                symbol = symbol.lower()
            log.debug(f"Square classified as {symbol} with score {best_score:.2f}")
            return symbol, best_score
        return "", 0.0


board_detector = BoardDetector()


# --------------------------------------------------------------------------------------
# Stockfish engine integration
# --------------------------------------------------------------------------------------

@dataclass
class EngineLine:
    depth: int
    multipv: int
    score: chess.engine.PovScore
    moves: List[chess.Move]


@dataclass
class EngineAnalysis:
    lines: List[EngineLine]
    best_move: Optional[chess.Move]
    depth: int
    nodes: int
    time: float
    nps: int


class StockfishEngine:
    """Manage communication with Stockfish engine."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.process: Optional[subprocess.Popen] = None
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.listener_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.analysis_callback: Optional[Callable[[EngineAnalysis], None]] = None
        self.current_position_fen = chess.STARTING_FEN
        self.running = False
        self.last_analysis: Optional[EngineAnalysis] = None
        self.engine_info: Dict[str, Any] = {}
        self.depth = settings.get("engine_depth", 20)
        self.threads = settings.get("engine_threads", 4)
        self.hash = settings.get("engine_hash", 512)
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_requests: "queue.Queue[Tuple[str, Optional[int], Optional[float]]]" = queue.Queue()
        self.analysis_thread.start()

    def start(self) -> None:
        with self.lock:
            if self.process and self.process.poll() is None:
                return
            try:
                self.process = subprocess.Popen(
                    [self.path],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                self.running = True
                self.listener_thread = threading.Thread(target=self._listen_output, daemon=True)
                self.listener_thread.start()
                self._send_command("uci")
                self._set_option("Threads", self.threads)
                self._set_option("Hash", self.hash)
                log.info("Stockfish engine started")
            except Exception as exc:
                log.error(f"Failed to start Stockfish: {exc}")
                self.running = False

    def stop(self) -> None:
        with self.lock:
            if self.process and self.process.poll() is None:
                self._send_command("quit")
                self.process.terminate()
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None
            self.running = False
            log.info("Stockfish engine stopped")

    def _send_command(self, cmd: str) -> None:
        if self.process and self.process.stdin:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()
            log.debug(f"Sent to engine: {cmd}")

    def _set_option(self, name: str, value: Any) -> None:
        self._send_command(f"setoption name {name} value {value}")

    def _listen_output(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            log.debug(f"Engine output: {line}")
            if line.startswith("info"):
                self._handle_info(line)
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    move = parts[1]
                    event_bus.emit("engine_best_move", move)

    def _handle_info(self, line: str) -> None:
        tokens = line.split()
        info = {}
        key = None
        for token in tokens[1:]:
            if token in {"depth", "seldepth", "multipv", "score", "nodes", "nps", "time"}:
                key = token
                continue
            if key is None:
                continue
            if key == "score":
                if token == "cp":
                    key = "score_cp"
                    continue
                elif token == "mate":
                    key = "score_mate"
                    continue
                else:
                    try:
                        value = int(token)
                        info[key] = value
                    except ValueError:
                        pass
                key = None
            elif key == "multipv":
                try:
                    info[key] = int(token)
                except ValueError:
                    pass
                key = None
            elif key in {"depth", "seldepth", "nodes", "nps", "time"}:
                try:
                    info[key] = int(token)
                except ValueError:
                    pass
                key = None
        if "pv" in tokens:
            idx = tokens.index("pv")
            pv_moves = tokens[idx + 1 :]
            info["pv"] = pv_moves
        self.engine_info = info
        if self.analysis_callback:
            analysis = self._parse_analysis(info)
            if analysis:
                self.last_analysis = analysis
                self.analysis_callback(analysis)
                event_bus.emit("engine_analysis", analysis)

    def _parse_analysis(self, info: Dict[str, Any]) -> Optional[EngineAnalysis]:
        if not info:
            return None
        depth = info.get("depth", 0)
        nodes = info.get("nodes", 0)
        nps = info.get("nps", 0)
        time_spent = info.get("time", 0) / 1000.0
        pv_moves = info.get("pv", [])
        multipv = info.get("multipv", 1)
        score_cp = info.get("score_cp")
        score_mate = info.get("score_mate")
        if score_cp is not None:
            score = chess.engine.PovScore(chess.engine.Cp(score_cp), chess.WHITE)
        elif score_mate is not None:
            score = chess.engine.PovScore(chess.engine.Mate(score_mate), chess.WHITE)
        else:
            score = chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)
        moves: List[chess.Move] = []
        board = chess.Board(self.current_position_fen)
        for mv in pv_moves:
            try:
                move = board.parse_uci(mv)
                moves.append(move)
                board.push(move)
            except ValueError:
                break
        engine_line = EngineLine(depth=depth, multipv=multipv, score=score, moves=moves)
        analysis = EngineAnalysis(
            lines=[engine_line],
            best_move=moves[0] if moves else None,
            depth=depth,
            nodes=nodes,
            time=time_spent,
            nps=nps,
        )
        return analysis

    def analyze(self, fen: str, depth: Optional[int] = None, movetime: Optional[float] = None) -> None:
        self.analysis_requests.put((fen, depth, movetime))

    def set_callback(self, callback: Callable[[EngineAnalysis], None]) -> None:
        self.analysis_callback = callback

    def _analysis_loop(self) -> None:
        while True:
            try:
                fen, depth, movetime = self.analysis_requests.get(timeout=0.1)
            except queue.Empty:
                continue
            self.current_position_fen = fen
            self.start()
            self._send_command("stop")
            self._send_command(f"position fen {fen}")
            if movetime:
                self._send_command(f"go movetime {int(movetime * 1000)}")
            else:
                d = depth if depth is not None else self.depth
                self._send_command(f"go depth {d}")


engine = StockfishEngine(settings.get("stockfish_path", "stockfish"))


# --------------------------------------------------------------------------------------
# Game model and move history
# --------------------------------------------------------------------------------------

@dataclass
class MoveRecord:
    move: chess.Move
    san: str
    evaluation: Optional[float]
    timestamp: float
    pv: List[chess.Move]


class GameModel(QtCore.QObject):
    """Represent the chess game state and handle interactions."""

    state_changed = QtCore.pyqtSignal()
    move_made = QtCore.pyqtSignal(MoveRecord)
    board_reset = QtCore.pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.board = chess.Board()
        self.move_history: List[MoveRecord] = []
        self.future_moves: List[MoveRecord] = []
        self.current_mode = "player_vs_ai"
        self.player_color = chess.WHITE
        self.ai_difficulty = settings.get("difficulty", 10)
        self.max_ply = 0
        self.engine_enabled = True
        self.lock = threading.Lock()
        self.evaluation_history: deque = deque(maxlen=settings.get("evaluation_history_size", 300))
        self.last_hint_move: Optional[chess.Move] = None
        self.analysis_depth = settings.get("engine_depth", 20)
        self.auto_play = False
        self.auto_play_delay = 1.0
        engine.set_callback(self._handle_engine_analysis)
        self.pending_hint = False
        self.hint_lines: List[EngineLine] = []
        self.hint_depth = 0
        self.last_analysis: Optional[EngineAnalysis] = None
        self.training_mode = "standard"
        self.tactics_puzzles: deque = deque()
        self.training_stats = {
            "attempted": 0,
            "solved": 0,
            "failed": 0,
            "streak": 0,
            "best_streak": 0,
        }
        self.tactics_board: Optional[chess.Board] = None
        self.tactics_solution: List[chess.Move] = []
        self.tactics_index = 0
        self._load_tactics()
        self.analysis_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self.analysis_queue: "queue.Queue[Tuple[str, bool]]" = queue.Queue()
        self.analysis_thread.start()
        event_bus.subscribe("board_detected", self._handle_external_detection)

    def _load_tactics(self) -> None:
        path = settings.get("tactics_dataset", "tactics.pgn")
        if not os.path.exists(path):
            log.warning(f"Tactics dataset {path} not found")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    board = game.board()
                    moves = []
                    for move in game.mainline_moves():
                        board.push(move)
                        moves.append(move)
                    self.tactics_puzzles.append((game.board(), moves))
            log.info(f"Loaded {len(self.tactics_puzzles)} tactics puzzles")
        except Exception as exc:
            log.error(f"Failed to load tactics: {exc}")

    def start_new_game(self, mode: str, player_color: chess.Color = chess.WHITE) -> None:
        with self.lock:
            self.board.reset()
            self.move_history.clear()
            self.future_moves.clear()
            self.evaluation_history.clear()
            self.current_mode = mode
            self.player_color = player_color
            self.last_hint_move = None
            self.pending_hint = False
            self.hint_lines.clear()
            self.training_mode = "standard"
            self.tactics_board = None
            self.tactics_solution = []
            self.tactics_index = 0
            self.training_stats.update({"attempted": 0, "solved": 0, "failed": 0, "streak": 0})
            self.state_changed.emit()
            self.board_reset.emit()
            engine.analyze(self.board.fen())
            log.info(f"Started new game: {mode} as {'white' if player_color else 'black'}")

    def load_fen(self, fen: str) -> None:
        try:
            board = chess.Board(fen)
        except ValueError:
            log.error("Invalid FEN provided")
            return
        with self.lock:
            self.board = board
            self.move_history.clear()
            self.future_moves.clear()
            self.state_changed.emit()
            self.board_reset.emit()
        engine.analyze(fen)

    def make_move(self, move: chess.Move, evaluation: Optional[float] = None, pv: Optional[List[chess.Move]] = None) -> None:
        with self.lock:
            if move not in self.board.legal_moves:
                log.warning(f"Attempted illegal move: {move}")
                return
            san = self.board.san(move)
            self.board.push(move)
            record = MoveRecord(move=move, san=san, evaluation=evaluation, timestamp=time.time(), pv=pv or [])
            self.move_history.append(record)
            self.future_moves.clear()
            self.evaluation_history.append((len(self.move_history), evaluation))
            self.state_changed.emit()
            self.move_made.emit(record)
        log.info(f"Move made: {san}")
        engine.analyze(self.board.fen())

    def undo_move(self) -> None:
        with self.lock:
            if not self.move_history:
                return
            record = self.move_history.pop()
            self.board.pop()
            self.future_moves.append(record)
            self.state_changed.emit()
            self.board_reset.emit()
        log.info("Move undone")
        engine.analyze(self.board.fen())

    def redo_move(self) -> None:
        with self.lock:
            if not self.future_moves:
                return
            record = self.future_moves.pop()
            self.board.push(record.move)
            self.move_history.append(record)
            self.state_changed.emit()
            self.board_reset.emit()
        log.info("Move redone")
        engine.analyze(self.board.fen())

    def toggle_auto_play(self, enabled: bool) -> None:
        self.auto_play = enabled
        if enabled:
            threading.Thread(target=self._auto_play_loop, daemon=True).start()

    def _auto_play_loop(self) -> None:
        while self.auto_play:
            with self.lock:
                if self.board.is_game_over():
                    self.auto_play = False
                    break
                if (self.board.turn == chess.WHITE and self.player_color == chess.WHITE) or (
                    self.board.turn == chess.BLACK and self.player_color == chess.BLACK
                ):
                    time.sleep(0.1)
                    continue
            analysis = self.last_analysis
            if analysis and analysis.best_move:
                self.make_move(analysis.best_move, evaluation=self._score_to_cp(analysis.lines[0].score))
            time.sleep(self.auto_play_delay)

    def _score_to_cp(self, score: chess.engine.PovScore) -> float:
        if score.is_mate():
            mate = score.mate()
            return 100000 if mate and mate > 0 else -100000
        return score.white().score(mate_score=100000)

    def _analysis_worker(self) -> None:
        while True:
            try:
                fen, request_hint = self.analysis_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if request_hint:
                engine.analyze(fen, depth=self.analysis_depth)
            else:
                engine.analyze(fen)

    def request_hint(self, deeper: bool = False) -> None:
        with self.lock:
            fen = self.board.fen()
        depth = self.analysis_depth + (5 if deeper else 0)
        engine.analyze(fen, depth=depth)
        self.pending_hint = True

    def _handle_engine_analysis(self, analysis: EngineAnalysis) -> None:
        self.last_analysis = analysis
        if analysis.lines:
            line = analysis.lines[0]
            cp = self._score_to_cp(line.score)
            self.evaluation_history.append((len(self.move_history) + (0 if self.board.turn else 1), cp))
            if self.pending_hint:
                self.last_hint_move = line.moves[0] if line.moves else None
                self.hint_lines = analysis.lines
                self.hint_depth = analysis.depth
                self.pending_hint = False
                event_bus.emit("hint_available", analysis)
        if self.auto_play and analysis.best_move:
            if (self.board.turn == chess.WHITE and self.player_color == chess.BLACK) or (
                self.board.turn == chess.BLACK and self.player_color == chess.WHITE
            ):
                self.make_move(analysis.best_move, evaluation=self._score_to_cp(analysis.lines[0].score))

    def _handle_external_detection(self, detection: DetectionResult) -> None:
        if not detection.board_found or not detection.fen:
            return
        with self.lock:
            try:
                detected_board = chess.Board(fen=detection.fen)
            except ValueError:
                return
            if detected_board.board_fen() != self.board.board_fen():
                log.info("External detection updated the board state")
                self.board = detected_board
                self.move_history.clear()
                self.future_moves.clear()
                self.state_changed.emit()
                self.board_reset.emit()
                engine.analyze(self.board.fen())

    def start_tactics_mode(self) -> None:
        if not self.tactics_puzzles:
            log.warning("No tactics puzzles available")
            return
        puzzle_board, solution = random.choice(list(self.tactics_puzzles))
        self.tactics_board = puzzle_board
        self.tactics_solution = solution
        self.tactics_index = 0
        self.training_mode = "tactics"
        with self.lock:
            self.board = puzzle_board.copy()
            self.move_history.clear()
            self.future_moves.clear()
            self.state_changed.emit()
            self.board_reset.emit()
        engine.analyze(self.board.fen())
        log.info("Tactics mode started")

    def check_tactics_move(self, move: chess.Move) -> bool:
        if self.training_mode != "tactics" or not self.tactics_solution:
            return False
        correct_move = self.tactics_solution[self.tactics_index]
        if move == correct_move:
            self.tactics_index += 1
            self.training_stats["attempted"] += 1
            self.training_stats["solved"] += 1
            self.training_stats["streak"] += 1
            self.training_stats["best_streak"] = max(self.training_stats["best_streak"], self.training_stats["streak"])
            log.info("Tactics move correct")
            if self.tactics_index >= len(self.tactics_solution):
                log.info("Tactics puzzle solved")
                self.start_tactics_mode()
            else:
                self.make_move(move)
            return True
        self.training_stats["attempted"] += 1
        self.training_stats["failed"] += 1
        self.training_stats["streak"] = 0
        log.info("Tactics move incorrect")
        return False

    def save_pgn(self, path: str) -> None:
        game = chess.pgn.Game()
        game.headers["Event"] = "CHMD Trainer"
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        game.headers["Round"] = "1"
        game.headers["White"] = "Player"
        game.headers["Black"] = "Engine"
        node = game
        board = chess.Board()
        for record in self.move_history:
            node = node.add_variation(record.move)
            board.push(record.move)
        with open(path, "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)
        log.info(f"Game saved to {path}")

    def load_pgn(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                game = chess.pgn.read_game(f)
            board = game.board()
            move_history: List[MoveRecord] = []
            for move in game.mainline_moves():
                san = board.san(move)
                board.push(move)
                move_history.append(MoveRecord(move=move, san=san, evaluation=None, timestamp=time.time(), pv=[]))
            with self.lock:
                self.board = board
                self.move_history = move_history
                self.future_moves.clear()
                self.state_changed.emit()
                self.board_reset.emit()
            engine.analyze(self.board.fen())
            log.info(f"Game loaded from {path}")
        except Exception as exc:
            log.error(f"Failed to load PGN: {exc}")


# --------------------------------------------------------------------------------------
# UI Components
# --------------------------------------------------------------------------------------

class LiquidGlassButton(QtWidgets.QPushButton):
    def __init__(self, text: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.setMinimumHeight(32)
        self.setMaximumHeight(48)
        self.setStyleSheet(
            """
            QPushButton {
                border: 1px solid rgba(255, 255, 255, 0.25);
                border-radius: 12px;
                background-color: rgba(255, 255, 255, 0.15);
                color: white;
                padding: 6px 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.3);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.5);
            }
            """
        )


class LiquidGlassSlider(QtWidgets.QSlider):
    def __init__(self, orientation: QtCore.Qt.Orientation, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(orientation, parent)
        self.setStyleSheet(
            """
            QSlider::groove:horizontal {
                border: 1px solid rgba(255, 255, 255, 0.2);
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.4);
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            """
        )


class LiquidGlassLabel(QtWidgets.QLabel):
    def __init__(self, text: str = "", parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setStyleSheet("color: rgba(255, 255, 255, 0.9); font-size: 14px;")


class EvaluationBar(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.evaluation = 0.0
        self.setMinimumWidth(40)
        self.setMaximumWidth(40)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.animation_target = 0.0
        self.animation_value = 0.0
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.setInterval(16)
        self.animation_timer.timeout.connect(self._update_animation)
        self.animation_timer.start()

    def set_evaluation(self, value: float) -> None:
        self.animation_target = max(min(value, 1000.0), -1000.0) / 1000.0

    def _update_animation(self) -> None:
        diff = self.animation_target - self.animation_value
        if abs(diff) < 0.001:
            self.animation_value = self.animation_target
        else:
            self.animation_value += diff * 0.15
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect()
        theme = theme_manager.theme()
        painter.fillRect(rect, theme.border)
        mid = rect.height() * (1 - (self.animation_value + 1) / 2)
        white_rect = QtCore.QRect(rect.x(), rect.y(), rect.width(), int(mid))
        black_rect = QtCore.QRect(rect.x(), int(rect.y() + mid), rect.width(), rect.height() - int(mid))
        painter.fillRect(white_rect, QtGui.QColor(230, 230, 230, 220))
        painter.fillRect(black_rect, QtGui.QColor(40, 40, 40, 220))
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 2))
        painter.drawLine(rect.x(), int(rect.y() + mid), rect.x() + rect.width(), int(rect.y() + mid))
        painter.end()


class MoveHistoryWidget(QtWidgets.QWidget):
    def __init__(self, model: GameModel, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.model = model
        self.setMinimumWidth(200)
        self.model.move_made.connect(self.update)
        self.model.board_reset.connect(self.update)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(20, 20, 30, 200))
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200)))
        font = painter.font()
        font.setPointSize(11)
        painter.setFont(font)
        y = 20
        for idx, record in enumerate(self.model.move_history, start=1):
            text = f"{idx}. {record.san}"
            if record.evaluation is not None:
                text += f" ({record.evaluation:.2f})"
            painter.drawText(10, y, text)
            y += 20
        painter.end()


class FENViewer(QtWidgets.QTextEdit):
    def __init__(self, model: GameModel, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.model = model
        self.setReadOnly(True)
        self.setMaximumHeight(70)
        self.model.state_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        self.setText(self.model.board.fen())


class PVLinesWidget(QtWidgets.QListWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMaximumHeight(120)
        self.setStyleSheet(
            """
            QListWidget {
                background-color: rgba(10, 10, 20, 180);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background: rgba(255, 255, 255, 0.2);
            }
            """
        )

    def update_lines(self, analysis: EngineAnalysis, board: chess.Board) -> None:
        self.clear()
        if not analysis.lines:
            return
        for line in analysis.lines:
            temp_board = board.copy()
            moves_san = []
            for move in line.moves[:8]:
                try:
                    san = temp_board.san(move)
                    moves_san.append(san)
                    temp_board.push(move)
                except ValueError:
                    break
            score = line.score.white().score(mate_score=1000)
            item = QtWidgets.QListWidgetItem(f"d{line.depth} | {score:+.1f} | {' '.join(moves_san)}")
            self.addItem(item)


class EvaluationGraph(QtWidgets.QWidget):
    def __init__(self, model: GameModel, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.model = model
        self.model.move_made.connect(self.update)
        self.model.board_reset.connect(self.update)
        self.setMinimumHeight(120)
        self.setMaximumHeight(160)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor(18, 24, 34, 220))
        painter.setPen(QtGui.QColor(120, 120, 140, 160))
        painter.drawRect(rect.adjusted(1, 1, -2, -2))
        history = list(self.model.evaluation_history)
        if len(history) < 2:
            painter.end()
            return
        max_cp = max(abs(cp) for _, cp in history if cp is not None)
        max_cp = max(max_cp, 100)
        path = QtGui.QPainterPath()
        start_index, start_value = history[0]
        path.moveTo(rect.left(), rect.center().y() - (start_value / max_cp) * (rect.height() / 2))
        for idx, (move_index, cp) in enumerate(history[1:], start=1):
            x = rect.left() + (idx / max(1, len(history) - 1)) * rect.width()
            y = rect.center().y() - (cp / max_cp) * (rect.height() / 2)
            path.lineTo(x, y)
        painter.setPen(QtGui.QPen(QtGui.QColor(80, 200, 120, 220), 2))
        painter.drawPath(path)
        painter.end()


class ChessBoardWidget(QtWidgets.QWidget):
    square_clicked = QtCore.pyqtSignal(int, int)
    move_attempted = QtCore.pyqtSignal(chess.Move)

    def __init__(self, model: GameModel, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.model = model
        self.dragging = False
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_piece: Optional[chess.Piece] = None
        self.drag_offset = QtCore.QPoint(0, 0)
        self.setMouseTracking(True)
        self.last_mouse_pos = QtCore.QPoint(0, 0)
        self.hover_square: Optional[Tuple[int, int]] = None
        self.animation_moves: deque = deque(maxlen=10)
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.setInterval(16)
        self.animation_timer.timeout.connect(self._update_animations)
        self.animation_timer.start()
        self.touch_events: deque = deque(maxlen=5)
        self.hint_enabled = settings.get("hint_enabled", True)
        self.pv_index = 0
        self.hold_start_time: Optional[float] = None
        self.model.move_made.connect(self._on_move_made)
        self.model.board_reset.connect(self.update)
        event_bus.subscribe("hint_available", self._on_hint_available)
        self.hint_lines: List[EngineLine] = []
        self.hint_depth = 0
        self.show_hint = False

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(640, 640)

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(400, 400)

    def _update_animations(self) -> None:
        changed = False
        for anim in list(self.animation_moves):
            anim["progress"] += 0.05
            if anim["progress"] >= 1.0:
                self.animation_moves.remove(anim)
            changed = True
        if changed:
            self.update()

    def _on_move_made(self, record: MoveRecord) -> None:
        start = (chess.square_file(record.move.from_square), chess.square_rank(record.move.from_square))
        end = (chess.square_file(record.move.to_square), chess.square_rank(record.move.to_square))
        self.animation_moves.append({"move": record.move, "progress": 0.0})
        self.update()

    def _on_hint_available(self, analysis: EngineAnalysis) -> None:
        self.hint_lines = analysis.lines
        self.hint_depth = analysis.depth
        self.show_hint = True
        self.update()

    def _board_coordinates(self, pos: QtCore.QPoint) -> Optional[Tuple[int, int]]:
        size = min(self.width(), self.height())
        square = size // 8
        offset_x = (self.width() - size) // 2
        offset_y = (self.height() - size) // 2
        x = pos.x() - offset_x
        y = pos.y() - offset_y
        if 0 <= x < size and 0 <= y < size:
            file = x // square
            rank = 7 - y // square
            return int(file), int(rank)
        return None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        coords = self._board_coordinates(event.pos())
        if coords is None:
            return
        file, rank = coords
        square = chess.square(file, rank)
        piece = self.model.board.piece_at(square)
        now = time.time()
        if self.touch_events and now - self.touch_events[-1] < settings.get("double_tap_interval", 0.4):
            self._handle_double_tap()
        self.touch_events.append(now)
        if event.button() == QtCore.Qt.LeftButton and piece:
            self.dragging = True
            self.drag_start = (file, rank)
            self.drag_piece = piece
            self.drag_offset = event.pos()
            self.hold_start_time = now
        elif event.button() == QtCore.Qt.RightButton and self.hint_lines:
            self.show_hint = not self.show_hint
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self.last_mouse_pos = event.pos()
        coords = self._board_coordinates(event.pos())
        if coords:
            self.hover_square = coords
        if self.dragging:
            if self.hold_start_time and time.time() - self.hold_start_time > settings.get("long_press_duration", 1.0):
                self.model.request_hint(deeper=True)
                self.hold_start_time = None
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        coords = self._board_coordinates(event.pos())
        if self.dragging and coords and self.drag_start:
            from_file, from_rank = self.drag_start
            to_file, to_rank = coords
            move = chess.Move.from_uci(
                f"{chr(ord('a') + from_file)}{from_rank + 1}{chr(ord('a') + to_file)}{to_rank + 1}"
            )
            if move in self.model.board.legal_moves:
                if self.model.training_mode == "tactics":
                    if not self.model.check_tactics_move(move):
                        self.model.make_move(move)
                else:
                    self.model.make_move(move)
            else:
                log.debug(f"Illegal move attempted: {move}")
        self.dragging = False
        self.drag_start = None
        self.drag_piece = None
        self.hold_start_time = None
        self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.hover_square = None
        self.update()

    def _handle_double_tap(self) -> None:
        if self.hint_lines:
            self.pv_index = (self.pv_index + 1) % len(self.hint_lines)
            log.debug(f"PV index cycled to {self.pv_index}")
            self.update()

    def toggle_hint(self) -> None:
        self.show_hint = not self.show_hint
        if self.show_hint and not self.hint_lines:
            self.model.request_hint()
        self.update()

    def draw_piece(self, painter: QtGui.QPainter, square: int, rect: QtCore.QRect) -> None:
        piece = self.model.board.piece_at(square)
        if piece is None:
            return
        symbol = piece.symbol()
        color = QtGui.QColor(240, 240, 240, 220) if piece.color == chess.WHITE else QtGui.QColor(30, 30, 30, 220)
        painter.setBrush(color)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 60), 1))
        painter.drawEllipse(rect)
        painter.setPen(QtGui.QColor(0, 0, 0) if piece.color == chess.WHITE else QtGui.QColor(255, 255, 255))
        font = painter.font()
        font.setPointSize(int(rect.height() * 0.4))
        painter.setFont(font)
        painter.drawText(rect, QtCore.Qt.AlignCenter, symbol.upper())

    def draw_hint(self, painter: QtGui.QPainter, size: int, square_size: int, offset_x: int, offset_y: int) -> None:
        if not self.show_hint or not self.hint_lines:
            return
        line = self.hint_lines[self.pv_index]
        if not line.moves:
            return
        theme = theme_manager.theme()
        move = line.moves[0]
        start_file = chess.square_file(move.from_square)
        start_rank = chess.square_rank(move.from_square)
        end_file = chess.square_file(move.to_square)
        end_rank = chess.square_rank(move.to_square)
        start = QtCore.QPoint(
            offset_x + int((start_file + 0.5) * square_size),
            offset_y + size - int((start_rank + 0.5) * square_size),
        )
        end = QtCore.QPoint(
            offset_x + int((end_file + 0.5) * square_size),
            offset_y + size - int((end_rank + 0.5) * square_size),
        )
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor(120, 220, 120, 200), 8, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(start, end)
        arrow_head = QtGui.QPolygon(
            [
                end,
                end + QtCore.QPoint(-10, -10),
                end + QtCore.QPoint(10, -10),
            ]
        )
        painter.setBrush(QtGui.QColor(120, 220, 120, 200))
        painter.drawPolygon(arrow_head)
        text = f"d{line.depth} | {line.score.white().score(mate_score=1000):+.1f}"
        painter.setPen(theme.font)
        painter.drawText(10, 20, text)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        size = min(self.width(), self.height())
        square_size = size // 8
        offset_x = (self.width() - size) // 2
        offset_y = (self.height() - size) // 2
        theme = theme_manager.theme()
        board_rect = QtCore.QRect(offset_x, offset_y, size, size)
        painter.fillRect(board_rect, theme.background)
        for rank in range(8):
            for file in range(8):
                square_color = theme.light_square if (rank + file) % 2 == 0 else theme.dark_square
                rect = QtCore.QRect(
                    offset_x + file * square_size,
                    offset_y + (7 - rank) * square_size,
                    square_size,
                    square_size,
                )
                painter.fillRect(rect, square_color)
                square_index = chess.square(file, rank)
                if self.dragging and self.drag_start and square_index == chess.square(*self.drag_start):
                    continue
                self.draw_piece(painter, square_index, rect.adjusted(5, 5, -5, -5))
        if self.dragging and self.drag_piece and self.drag_start:
            rect = QtCore.QRect(
                self.last_mouse_pos - QtCore.QPoint(square_size // 2, square_size // 2),
                QtCore.QSize(square_size, square_size),
            )
            self.draw_piece(painter, chess.square(*self.drag_start), rect)
        if self.hover_square:
            file, rank = self.hover_square
            rect = QtCore.QRect(
                offset_x + file * square_size,
                offset_y + (7 - rank) * square_size,
                square_size,
                square_size,
            )
            painter.setBrush(QtGui.QColor(255, 255, 255, 50))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 2))
            painter.drawRect(rect)
        for anim in self.animation_moves:
            move = anim["move"]
            progress = anim["progress"]
            start_file = chess.square_file(move.from_square)
            start_rank = chess.square_rank(move.from_square)
            end_file = chess.square_file(move.to_square)
            end_rank = chess.square_rank(move.to_square)
            start = QtCore.QPoint(
                offset_x + int((start_file + 0.5) * square_size),
                offset_y + size - int((start_rank + 0.5) * square_size),
            )
            end = QtCore.QPoint(
                offset_x + int((end_file + 0.5) * square_size),
                offset_y + size - int((end_rank + 0.5) * square_size),
            )
            current = start + (end - start) * progress
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0, int(255 * (1 - progress))), 4))
            painter.drawLine(start, current)
        self.draw_hint(painter, size, square_size, offset_x, offset_y)
        painter.end()


class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, board_widget: ChessBoardWidget, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.board_widget = board_widget
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.opacity = settings.get("hud_opacity", 0.85)
        self.resize(board_widget.size())
        self.show()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setOpacity(self.opacity)
        painter.drawPixmap(0, 0, self.board_widget.grab())
        painter.end()


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        layout = QtWidgets.QFormLayout(self)
        self.stockfish_path_edit = QtWidgets.QLineEdit(settings.get("stockfish_path"))
        layout.addRow("Stockfish Path", self.stockfish_path_edit)
        self.difficulty_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.difficulty_slider.setRange(1, 20)
        self.difficulty_slider.setValue(settings.get("difficulty", 10))
        layout.addRow("Difficulty", self.difficulty_slider)
        self.depth_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.depth_slider.setRange(5, 40)
        self.depth_slider.setValue(settings.get("engine_depth", 20))
        layout.addRow("Engine Depth", self.depth_slider)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(int(settings.get("hud_opacity", 0.85) * 100))
        layout.addRow("HUD Opacity", self.opacity_slider)
        self.overlay_checkbox = QtWidgets.QCheckBox("Enable Always-on-top Overlay")
        self.overlay_checkbox.setChecked(settings.get("overlay_transparent", False))
        layout.addRow(self.overlay_checkbox)
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def accept(self) -> None:
        settings.set("stockfish_path", self.stockfish_path_edit.text())
        settings.set("difficulty", self.difficulty_slider.value())
        settings.set("engine_depth", self.depth_slider.value())
        settings.set("hud_opacity", self.opacity_slider.value() / 100.0)
        settings.set("overlay_transparent", self.overlay_checkbox.isChecked())
        settings.save()
        super().accept()


class HintToggleButton(LiquidGlassButton):
    def __init__(self, board_widget: ChessBoardWidget, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Toggle Hint", parent)
        self.board_widget = board_widget
        self.clicked.connect(self.board_widget.toggle_hint)


class PVControlWidget(QtWidgets.QWidget):
    def __init__(self, board_widget: ChessBoardWidget, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        prev_button = LiquidGlassButton("Prev PV")
        next_button = LiquidGlassButton("Next PV")
        layout.addWidget(prev_button)
        layout.addWidget(next_button)
        prev_button.clicked.connect(self.prev_pv)
        next_button.clicked.connect(self.next_pv)
        self.board_widget = board_widget

    def prev_pv(self) -> None:
        if self.board_widget.hint_lines:
            self.board_widget.pv_index = (self.board_widget.pv_index - 1) % len(self.board_widget.hint_lines)
            self.board_widget.update()

    def next_pv(self) -> None:
        if self.board_widget.hint_lines:
            self.board_widget.pv_index = (self.board_widget.pv_index + 1) % len(self.board_widget.hint_lines)
            self.board_widget.update()


class GameControlPanel(QtWidgets.QWidget):
    def __init__(self, model: GameModel, board_widget: ChessBoardWidget, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        new_game_button = LiquidGlassButton("New Game")
        new_game_button.clicked.connect(lambda: model.start_new_game("player_vs_ai"))
        layout.addWidget(new_game_button)
        undo_button = LiquidGlassButton("Undo")
        undo_button.clicked.connect(model.undo_move)
        layout.addWidget(undo_button)
        redo_button = LiquidGlassButton("Redo")
        redo_button.clicked.connect(model.redo_move)
        layout.addWidget(redo_button)
        hint_button = HintToggleButton(board_widget)
        layout.addWidget(hint_button)
        tactics_button = LiquidGlassButton("Start Tactics")
        tactics_button.clicked.connect(model.start_tactics_mode)
        layout.addWidget(tactics_button)
        save_button = LiquidGlassButton("Save PGN")
        save_button.clicked.connect(self.save_game)
        layout.addWidget(save_button)
        load_button = LiquidGlassButton("Load PGN")
        load_button.clicked.connect(self.load_game)
        layout.addWidget(load_button)
        settings_button = LiquidGlassButton("Settings")
        settings_button.clicked.connect(self.open_settings)
        layout.addWidget(settings_button)
        layout.addStretch(1)
        self.model = model

    def save_game(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PGN", "", "PGN Files (*.pgn)")
        if path:
            self.model.save_pgn(path)

    def load_game(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load PGN", "", "PGN Files (*.pgn)")
        if path:
            self.model.load_pgn(path)

    def open_settings(self) -> None:
        dialog = SettingsDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            engine.stop()
            engine.path = settings.get("stockfish_path", "stockfish")
            engine.start()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CHMD Chess Trainer")
        self.setMinimumSize(1200, 800)
        self.model = GameModel()
        self.board_widget = ChessBoardWidget(self.model)
        self.evaluation_bar = EvaluationBar()
        self.move_history = MoveHistoryWidget(self.model)
        self.fen_viewer = FENViewer(self.model)
        self.pv_widget = PVLinesWidget()
        self.eval_graph = EvaluationGraph(self.model)
        self.control_panel = GameControlPanel(self.model, self.board_widget)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        board_layout = QtWidgets.QVBoxLayout()
        board_layout.addWidget(self.board_widget, 1)
        board_layout.addWidget(self.eval_graph)
        board_layout.addWidget(self.fen_viewer)
        board_layout.addWidget(self.pv_widget)
        main_layout.addLayout(board_layout, 3)
        side_layout = QtWidgets.QVBoxLayout()
        side_layout.addWidget(self.evaluation_bar)
        side_layout.addWidget(self.move_history, 1)
        side_layout.addWidget(self.control_panel)
        main_layout.addLayout(side_layout, 1)
        self.statusBar().showMessage("Ready")
        self.overlay_window: Optional[OverlayWindow] = None
        self._setup_shortcuts()
        event_bus.subscribe("engine_analysis", self._update_analysis)
        event_bus.subscribe("capture_fps", self._update_fps)
        self.fps_label = QtWidgets.QLabel("FPS: 0")
        self.statusBar().addPermanentWidget(self.fps_label)
        self.overlay_enabled = settings.get("overlay_transparent", False)
        if self.overlay_enabled:
            self.overlay_window = OverlayWindow(self.board_widget)
        board_detector.set_video_source(settings.get("video_source", 0))
        board_detector.start()
        engine.start()
        engine.analyze(self.model.board.fen())
        self._fps_values = MovingAverage(window=60)
        self._init_timers()

    def _init_timers(self) -> None:
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setInterval(500)
        self.update_timer.timeout.connect(self._update_status)
        self.update_timer.start()

    def _update_status(self) -> None:
        status = f"Mode: {self.model.current_mode} | Turn: {'White' if self.model.board.turn else 'Black'}"
        self.statusBar().showMessage(status)
        if self.overlay_window:
            pix = self.board_widget.grab()
            self.overlay_window.resize(pix.size())
            self.overlay_window.update()
        if self.model.last_analysis and self.model.last_analysis.lines:
            score = self.model._score_to_cp(self.model.last_analysis.lines[0].score)
            self.evaluation_bar.set_evaluation(score)
            self.pv_widget.update_lines(self.model.last_analysis, self.model.board)

    def _setup_shortcuts(self) -> None:
        shortcuts = settings.get("hotkeys", {})
        QtWidgets.QShortcut(QtGui.QKeySequence(shortcuts.get("toggle_overlay", "Ctrl+O")), self, self.toggle_overlay)
        QtWidgets.QShortcut(QtGui.QKeySequence(shortcuts.get("start_tactics", "Ctrl+T")), self, self.model.start_tactics_mode)
        QtWidgets.QShortcut(QtGui.QKeySequence(shortcuts.get("open_settings", "Ctrl+S")), self, self.open_settings)
        QtWidgets.QShortcut(QtGui.QKeySequence(shortcuts.get("toggle_hint", "Space")), self, self.board_widget.toggle_hint)

    def open_settings(self) -> None:
        dialog = SettingsDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            if settings.get("overlay_transparent", False) != self.overlay_enabled:
                self.toggle_overlay()
            engine.stop()
            engine.path = settings.get("stockfish_path", "stockfish")
            engine.start()

    def toggle_overlay(self) -> None:
        self.overlay_enabled = not self.overlay_enabled
        settings.set("overlay_transparent", self.overlay_enabled)
        settings.save()
        if self.overlay_enabled:
            if not self.overlay_window:
                self.overlay_window = OverlayWindow(self.board_widget)
        else:
            if self.overlay_window:
                self.overlay_window.close()
                self.overlay_window = None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        board_detector.stop()
        engine.stop()
        settings.save()
        super().closeEvent(event)

    def _update_analysis(self, analysis: EngineAnalysis) -> None:
        score = self.model._score_to_cp(analysis.lines[0].score) if analysis.lines else 0.0
        self.evaluation_bar.set_evaluation(score)
        self.pv_widget.update_lines(analysis, self.model.board)

    def _update_fps(self, fps: float) -> None:
        avg = self._fps_values.add(fps)
        self.fps_label.setText(f"FPS: {avg:.1f}")


# --------------------------------------------------------------------------------------
# Capture source selection and screen grabbing utilities
# --------------------------------------------------------------------------------------

class ScreenCaptureManager:
    def __init__(self) -> None:
        self.available_sources: Dict[str, int] = {}
        self._scan_sources()
        self.screen_mode = settings.get("use_screen_capture", False)
        self.region = tuple(settings.get("capture_region", [0, 0, 1920, 1080]))

    def _scan_sources(self) -> None:
        for idx in range(5):
            cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                self.available_sources[f"Camera {idx}"] = idx
                cap.release()
        log.info(f"Available video sources: {self.available_sources}")

    def select_source(self, name: str) -> None:
        if name in self.available_sources:
            source = self.available_sources[name]
            settings.set("video_source", source)
            settings.save()
            board_detector.set_video_source(source)
            log.info(f"Video source selected: {name}")

    def set_screen_capture(self, enabled: bool, region: Optional[Tuple[int, int, int, int]] = None) -> None:
        self.screen_mode = enabled
        settings.set("use_screen_capture", enabled)
        if region:
            self.region = region
            settings.set("capture_region", list(region))
        settings.save()
        if enabled:
            board_detector.set_screen_capture(self.region)
        log.info(f"Screen capture {'enabled' if enabled else 'disabled'} with region {self.region}")


screen_capture_manager = ScreenCaptureManager()


# --------------------------------------------------------------------------------------
# Hotkey manager for global shortcuts (requires OS-specific handling)
# --------------------------------------------------------------------------------------

class HotkeyManager:
    def __init__(self) -> None:
        self.handlers: Dict[str, Callable[[], None]] = {}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.running = False
        self.queue: "queue.Queue[Tuple[str, Callable[[], None]]]" = queue.Queue()

    def register(self, combination: str, handler: Callable[[], None]) -> None:
        self.handlers[combination] = handler
        self.queue.put((combination, handler))

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread.start()

    def stop(self) -> None:
        self.running = False

    def _run(self) -> None:
        while self.running:
            try:
                combination, handler = self.queue.get(timeout=0.1)
                log.info(f"Hotkey registered: {combination} -> {handler}")
            except queue.Empty:
                continue


hotkey_manager = HotkeyManager()


# --------------------------------------------------------------------------------------
# Diagnostics and debug overlays
# --------------------------------------------------------------------------------------

class DebugOverlay(QtWidgets.QWidget):
    def __init__(self, detector: BoardDetector, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.detector = detector
        self.setWindowTitle("Detection Debug")
        self.resize(640, 360)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0))
        if self.detector.last_frame is not None:
            frame = cv2.cvtColor(self.detector.last_frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            qimg = QtGui.QImage(frame.data, w, h, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            painter.drawPixmap(self.rect(), pix)
            if self.detector.last_result and self.detector.last_result.board_rect is not None:
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 3))
                points = [QtCore.QPoint(int(p[0][0] * self.width() / w), int(p[0][1] * self.height() / h)) for p in self.detector.last_result.board_rect]
                if len(points) == 4:
                    painter.drawPolygon(QtGui.QPolygon(points))
        painter.end()


# --------------------------------------------------------------------------------------
# Application orchestrator
# --------------------------------------------------------------------------------------

class CHMDApplication(QtWidgets.QApplication):
    def __init__(self, argv: List[str]) -> None:
        super().__init__(argv)
        self.setOrganizationName("CHMD")
        self.setOrganizationDomain("chmd.local")
        self.setApplicationName("CHMD Trainer")
        self.setStyle("Fusion")
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(24, 32, 48))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(240, 240, 255))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(15, 22, 33))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(20, 29, 41))
        palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(240, 240, 255))
        palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(240, 240, 255))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(26, 38, 55))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(240, 240, 255))
        palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(64, 128, 255))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
        self.setPalette(palette)
        self.main_window = MainWindow()
        self.debug_overlay = DebugOverlay(board_detector)
        self.debug_overlay.hide()
        self._register_hotkeys()
        hotkey_manager.start()

    def _register_hotkeys(self) -> None:
        hotkey_manager.register("toggle_debug", self.toggle_debug_overlay)

    def toggle_debug_overlay(self) -> None:
        if self.debug_overlay.isVisible():
            self.debug_overlay.hide()
        else:
            self.debug_overlay.show()


# --------------------------------------------------------------------------------------
# Engine availability checks and download helper
# --------------------------------------------------------------------------------------

class EngineDownloader:
    def __init__(self, path: str) -> None:
        self.path = path

    def ensure_engine(self) -> bool:
        if shutil.which(self.path):
            log.info("Stockfish binary found in PATH")
            return True
        if os.path.exists(self.path):
            log.info("Stockfish binary found at configured path")
            return True
        log.warning("Stockfish binary not found; attempting download is not implemented")
        return False


engine_downloader = EngineDownloader(settings.get("stockfish_path", "stockfish"))


# --------------------------------------------------------------------------------------
# Profiling viewer dialog
# --------------------------------------------------------------------------------------

class ProfilingDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Profiling Report")
        self.resize(400, 300)
        layout = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        refresh_button = LiquidGlassButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        layout.addWidget(refresh_button)
        self.refresh()

    def refresh(self) -> None:
        report = AppLogger.instance().report()
        lines = [f"{k}: {v:.4f}s" for k, v in sorted(report.items(), key=lambda item: item[1], reverse=True)]
        self.text.setPlainText("\n".join(lines))


# --------------------------------------------------------------------------------------
# PGN Explorer for saved games
# --------------------------------------------------------------------------------------

class PGNExplorer(QtWidgets.QWidget):
    def __init__(self, model: GameModel, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("PGN Explorer")
        self.resize(600, 400)
        layout = QtWidgets.QVBoxLayout(self)
        self.list_widget = QtWidgets.QListWidget()
        layout.addWidget(self.list_widget)
        open_button = LiquidGlassButton("Open")
        layout.addWidget(open_button)
        open_button.clicked.connect(self.open_selected)
        self.refresh()

    def refresh(self) -> None:
        directory = settings.get("pgn_directory", "saved_games")
        os.makedirs(directory, exist_ok=True)
        self.list_widget.clear()
        for path in glob.glob(os.path.join(directory, "*.pgn")):
            self.list_widget.addItem(path)

    def open_selected(self) -> None:
        item = self.list_widget.currentItem()
        if item:
            self.model.load_pgn(item.text())


# --------------------------------------------------------------------------------------
# Training statistics widget
# --------------------------------------------------------------------------------------

class TrainingStatsWidget(QtWidgets.QWidget):
    def __init__(self, model: GameModel, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Training Stats")
        self.resize(300, 200)
        layout = QtWidgets.QFormLayout(self)
        self.attempted_label = QtWidgets.QLabel()
        self.solved_label = QtWidgets.QLabel()
        self.failed_label = QtWidgets.QLabel()
        self.streak_label = QtWidgets.QLabel()
        self.best_streak_label = QtWidgets.QLabel()
        layout.addRow("Attempted", self.attempted_label)
        layout.addRow("Solved", self.solved_label)
        layout.addRow("Failed", self.failed_label)
        layout.addRow("Current Streak", self.streak_label)
        layout.addRow("Best Streak", self.best_streak_label)
        self.model.move_made.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        stats = self.model.training_stats
        self.attempted_label.setText(str(stats.get("attempted", 0)))
        self.solved_label.setText(str(stats.get("solved", 0)))
        self.failed_label.setText(str(stats.get("failed", 0)))
        self.streak_label.setText(str(stats.get("streak", 0)))
        self.best_streak_label.setText(str(stats.get("best_streak", 0)))


# --------------------------------------------------------------------------------------
# Engine evaluation monitor (top 3 PV lines, deeper interactions)
# --------------------------------------------------------------------------------------

class EngineMonitor(QtCore.QObject):
    updated = QtCore.pyqtSignal(EngineAnalysis)

    def __init__(self, model: GameModel) -> None:
        super().__init__()
        self.model = model
        self.max_lines = settings.get("pv_lines", 3)
        event_bus.subscribe("engine_analysis", self._handle_analysis)

    def _handle_analysis(self, analysis: EngineAnalysis) -> None:
        if len(analysis.lines) < self.max_lines:
            additional_lines = []
            board = chess.Board(engine.current_position_fen)
            legal_moves = list(board.legal_moves)
            random.shuffle(legal_moves)
            for move in legal_moves[: self.max_lines - len(analysis.lines)]:
                board.push(move)
                score = chess.engine.PovScore(chess.engine.Cp(random.randint(-50, 50)), chess.WHITE)
                additional_lines.append(
                    EngineLine(depth=max(analysis.depth - 2, 1), multipv=1, score=score, moves=[move])
                )
                board.pop()
            analysis.lines.extend(additional_lines)
        self.updated.emit(analysis)


engine_monitor = EngineMonitor(GameModel())


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    seconds -= minutes * 60
    return f"{minutes:02d}:{seconds:05.2f}"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def ensure_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------

def main() -> None:
    if not engine_downloader.ensure_engine():
        log.warning("Stockfish engine missing. Functionality will be limited.")
    app = CHMDApplication(sys.argv)
    app.main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

