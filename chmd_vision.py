import collections
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from chmd_core import (
    APP_PATHS,
    CAPTURE_INDEX,
    CAPTURE_INTERVAL,
    CAPTURE_SCHEDULE,
    COMMENTARY,
    DIAGNOSTICS,
    ENGINE_INTERVAL,
    EngineResult,
    EngineCommand,
    ENGINE_QUEUE,
    EVENT_BUS,
    LOGGER,
    NOTES,
    SETTINGS,
    TASK_SCHEDULER,
    TRAINER_STATE,
    describe_move,
    explain_hint,
    ensure_directory,
    validate_fen,
)


# --------------------------------------------------------------------------------------
# Capture sources
# --------------------------------------------------------------------------------------


class CaptureError(RuntimeError):
    pass


@dataclass
class CaptureFrame:
    image: np.ndarray
    timestamp: float


class CaptureSource:
    def read(self) -> CaptureFrame:
        raise NotImplementedError

    def release(self) -> None:
        pass


class WebcamCapture(CaptureSource):
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise CaptureError(f"Unable to open webcam index {index}")
        LOGGER.info("WebcamCapture initialized for index %d", index)

    def read(self) -> CaptureFrame:
        ret, frame = self._capture.read()
        if not ret:
            raise CaptureError("Failed to read frame from webcam")
        return CaptureFrame(frame, time.time())

    def release(self) -> None:
        self._capture.release()
        LOGGER.info("WebcamCapture released")


class ScreenCapture(CaptureSource):
    def __init__(self) -> None:
        self._screen = None

    def read(self) -> CaptureFrame:
        if self._screen is None:
            from mss import mss

            self._screen = mss()
        monitor = self._screen.monitors[1]
        frame = np.array(self._screen.grab(monitor))[:, :, :3]
        return CaptureFrame(frame, time.time())

    def release(self) -> None:
        if self._screen:
            self._screen.close()
        LOGGER.info("ScreenCapture released")


# --------------------------------------------------------------------------------------
# Board detection heuristics
# --------------------------------------------------------------------------------------


@dataclass
class BoardDetection:
    board_rect: Tuple[int, int, int, int]
    perspective: np.ndarray
    heatmap: np.ndarray
    confidence: float


def _largest_contour(contours: Iterable[np.ndarray]) -> Optional[np.ndarray]:
    max_area = 0
    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            best = contour
    return best


def detect_board_region(frame: np.ndarray) -> Optional[BoardDetection]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = _largest_contour(contours)
    if contour is None:
        return None
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) != 4:
        return None
    pts = approx.reshape(4, 2)
    rect = _order_corners(pts)
    (tl, tr, br, bl) = rect
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(frame, M, (max_width, max_height))
    heatmap = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)[:, :, 2]
    confidence = min(1.0, cv2.contourArea(contour) / float(frame.shape[0] * frame.shape[1]))
    x, y, w, h = cv2.boundingRect(contour)
    return BoardDetection((x, y, w, h), M, heatmap, confidence)


def _order_corners(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


# --------------------------------------------------------------------------------------
# Board grid extraction
# --------------------------------------------------------------------------------------


@dataclass
class BoardGrid:
    tiles: List[np.ndarray]
    tile_size: Tuple[int, int]
    board_image: np.ndarray


def extract_board_grid(frame: np.ndarray, detection: BoardDetection, grid_size: int = 8) -> Optional[BoardGrid]:
    warp = cv2.warpPerspective(frame, detection.perspective, (detection.heatmap.shape[1], detection.heatmap.shape[0]))
    h, w = warp.shape[:2]
    tile_h = h // grid_size
    tile_w = w // grid_size
    tiles: List[np.ndarray] = []
    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * tile_h
            y2 = (row + 1) * tile_h
            x1 = col * tile_w
            x2 = (col + 1) * tile_w
            tile = warp[y1:y2, x1:x2]
            tiles.append(tile)
    return BoardGrid(tiles, (tile_w, tile_h), warp)


# --------------------------------------------------------------------------------------
# Piece recognition using color histograms and template matching
# --------------------------------------------------------------------------------------


PIECE_LABELS = [
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K",
    "p",
    "n",
    "b",
    "r",
    "q",
    "k",
]


@dataclass
class PieceTemplate:
    label: str
    image: np.ndarray
    histogram: np.ndarray


class TemplateLibrary:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.templates: Dict[str, List[PieceTemplate]] = collections.defaultdict(list)
        ensure = Path(path)
        if not ensure.exists():
            ensure.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        for file in self.path.glob("*.png"):
            label = file.stem.split("_")[0]
            image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            histogram = self._compute_histogram(image)
            self.templates[label].append(PieceTemplate(label, image, histogram))
        LOGGER.info("Loaded %d templates", sum(len(v) for v in self.templates.values()))

    def _compute_histogram(self, image: np.ndarray) -> np.ndarray:
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def match(self, tile: np.ndarray) -> Tuple[str, float]:
        if tile.size == 0:
            return "", 0.0
        hist = self._compute_histogram(tile)
        best_label = ""
        best_score = 0.0
        for label, templates in self.templates.items():
            for template in templates:
                score = cv2.compareHist(hist, template.histogram, cv2.HISTCMP_CORREL)
                if score > best_score:
                    best_score = score
                    best_label = label
        return best_label, best_score

    def add_template(self, label: str, image: np.ndarray) -> None:
        histogram = self._compute_histogram(image)
        index = len(self.templates[label]) + 1
        path = self.path / f"{label}_{index}.png"
        cv2.imwrite(str(path), image)
        self.templates[label].append(PieceTemplate(label, image, histogram))
        LOGGER.info("Template for %s added at %s", label, path)


TEMPLATE_LIBRARY = TemplateLibrary(APP_PATHS.root / "templates")


# --------------------------------------------------------------------------------------
# Piece classifier orchestrating template matching and heuristics
# --------------------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    fen: str
    confidence_map: List[float]
    captures: List[str]
    anomalies: List[str]


class PieceClassifier:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.last_confidence = 0.0

    def classify(self, grid: BoardGrid) -> ClassificationResult:
        labels: List[List[str]] = [[""] * 8 for _ in range(8)]
        confidences: List[float] = []
        captures: List[str] = []
        anomalies: List[str] = []
        for index, tile in enumerate(grid.tiles):
            row = index // 8
            col = index % 8
            label, score = TEMPLATE_LIBRARY.match(tile)
            confidences.append(score)
            if score < 0.35:
                anomalies.append(f"Low confidence tile {row},{col}")
            labels[row][col] = label if score > 0.25 else ""
        fen_board = []
        for row in labels:
            empty = 0
            row_parts: List[str] = []
            for item in row:
                if not item:
                    empty += 1
                else:
                    if empty:
                        row_parts.append(str(empty))
                        empty = 0
                    row_parts.append(item)
            if empty:
                row_parts.append(str(empty))
            fen_board.append("".join(row_parts))
        fen = "/".join(fen_board) + " w KQkq - 0 1"
        if not validate_fen(fen):
            anomalies.append("FEN validation failed, falling back to previous state")
            fen = TRAINER_STATE.current_fen or fen
        return ClassificationResult(fen, confidences, captures, anomalies)


PIECE_CLASSIFIER = PieceClassifier()


# --------------------------------------------------------------------------------------
# Board stabilizer to reduce jitter between frames
# --------------------------------------------------------------------------------------


class BoardStabilizer:
    def __init__(self, history: int = 6) -> None:
        self.history = collections.deque(maxlen=history)

    def update(self, fen: str) -> str:
        self.history.append(fen)
        counts = collections.Counter(self.history)
        best, _ = counts.most_common(1)[0]
        return best


BOARD_STABILIZER = BoardStabilizer()


# --------------------------------------------------------------------------------------
# Evaluation overlay data aggregator
# --------------------------------------------------------------------------------------


class OverlayAggregator:
    def __init__(self) -> None:
        self.last_overlay = None

    def update(self, result: ClassificationResult) -> None:
        annotation = COMMENTARY.annotate(sum(result.confidence_map) / max(len(result.confidence_map), 1))
        NOTES.add(result.fen, annotation)
        self.last_overlay = annotation

    def get(self) -> Optional[str]:
        return self.last_overlay


OVERLAY_AGGREGATOR = OverlayAggregator()


# --------------------------------------------------------------------------------------
# Capture coordinator
# --------------------------------------------------------------------------------------


class CaptureCoordinator:
    def __init__(self) -> None:
        self.source: Optional[CaptureSource] = None
        self.lock = threading.Lock()
        self.active = False

    def _create_source(self) -> CaptureSource:
        mode = SETTINGS.get("capture_source", "screen")
        LOGGER.info("Creating capture source for %s", mode)
        if mode == "webcam":
            return WebcamCapture(CAPTURE_INDEX.index)
        return ScreenCapture()

    def start(self) -> None:
        with self.lock:
            if self.active:
                return
            try:
                self.source = self._create_source()
                self.active = True
                LOGGER.info("CaptureCoordinator started")
            except CaptureError:
                LOGGER.exception("Failed to start capture source")
                self.active = False

    def stop(self) -> None:
        with self.lock:
            if not self.active:
                return
            if self.source:
                self.source.release()
            self.active = False
            LOGGER.info("CaptureCoordinator stopped")

    def capture_loop(self) -> None:
        if not self.active:
            return
        if not CAPTURE_SCHEDULE.should_capture():
            return
        try:
            frame = self.source.read()
        except CaptureError:
            LOGGER.exception("Capture read failed")
            return
        detection = detect_board_region(frame.image)
        if not detection:
            DIAGNOSTICS.set("capture_confidence", 0.0, "No board detected")
            return
        grid = extract_board_grid(frame.image, detection)
        if not grid:
            return
        classification = PIECE_CLASSIFIER.classify(grid)
        stable_fen = BOARD_STABILIZER.update(classification.fen)
        TRAINER_STATE.set_fen(stable_fen)
        OVERLAY_AGGREGATOR.update(classification)
        DIAGNOSTICS.set("capture_confidence", detection.confidence, "Board confidence")
        DIAGNOSTICS.set("classification_mean", float(np.mean(classification.confidence_map)), "Classification mean")


CAPTURE_COORDINATOR = CaptureCoordinator()


# --------------------------------------------------------------------------------------
# Event-driven integration
# --------------------------------------------------------------------------------------


def _on_capture_request(topic: str, payload: Dict[str, object]) -> None:
    CAPTURE_COORDINATOR.capture_loop()


EVENT_BUS.subscribe("capture.request", _on_capture_request)


# --------------------------------------------------------------------------------------
# Calibration routines
# --------------------------------------------------------------------------------------


@dataclass
class CalibrationSample:
    fen: str
    image: np.ndarray
    timestamp: float


class CalibrationSession:
    def __init__(self) -> None:
        self.samples: List[CalibrationSample] = []
        self.running = False

    def begin(self) -> None:
        self.samples.clear()
        self.running = True

    def add_sample(self, fen: str, image: np.ndarray) -> None:
        if not self.running:
            return
        self.samples.append(CalibrationSample(fen, image, time.time()))

    def finish(self, path: Path) -> None:
        if not self.samples:
            return
        ensure_directory(path)
        for index, sample in enumerate(self.samples, start=1):
            cv2.imwrite(str(path / f"sample_{index}.png"), sample.image)
        LOGGER.info("Calibration session saved %d samples", len(self.samples))
        self.running = False


CALIBRATION_SESSION = CalibrationSession()


# --------------------------------------------------------------------------------------
# Lighting normalization
# --------------------------------------------------------------------------------------


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    merged = cv2.merge([l, a, b])
    normalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return normalized


# --------------------------------------------------------------------------------------
# Drift correction across frames
# --------------------------------------------------------------------------------------


class DriftCorrector:
    def __init__(self) -> None:
        self.previous_frame: Optional[np.ndarray] = None

    def correct(self, frame: np.ndarray) -> np.ndarray:
        if self.previous_frame is None:
            self.previous_frame = frame
            return frame
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        h, w = flow.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + flow[..., 0]).astype(np.float32)
        map_y = (map_y + flow[..., 1]).astype(np.float32)
        corrected = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        self.previous_frame = corrected
        return corrected


DRIFT_CORRECTOR = DriftCorrector()


# --------------------------------------------------------------------------------------
# Stabilized capture pipeline
# --------------------------------------------------------------------------------------


class StabilizedCapture:
    def __init__(self, coordinator: CaptureCoordinator) -> None:
        self.coordinator = coordinator
        self.lock = threading.Lock()
        self.samples: List[float] = []

    def process(self) -> None:
        if not self.coordinator.active:
            return
        frame = self.coordinator.source.read()
        corrected = DRIFT_CORRECTOR.correct(frame.image)
        detection = detect_board_region(corrected)
        if not detection:
            return
        grid = extract_board_grid(corrected, detection)
        if not grid:
            return
        normalized_tiles = [normalize_lighting(tile) for tile in grid.tiles]
        grid = BoardGrid(normalized_tiles, grid.tile_size, grid.board_image)
        classification = PIECE_CLASSIFIER.classify(grid)
        stable = BOARD_STABILIZER.update(classification.fen)
        TRAINER_STATE.set_fen(stable)
        self.samples.append(sum(classification.confidence_map) / max(len(classification.confidence_map), 1))
        if len(self.samples) > 120:
            self.samples = self.samples[-120:]
        DIAGNOSTICS.set("stabilized_confidence", float(np.mean(self.samples)), "Stabilized confidence")


STABILIZED_CAPTURE = StabilizedCapture(CAPTURE_COORDINATOR)


# --------------------------------------------------------------------------------------
# Multi-threaded capture worker
# --------------------------------------------------------------------------------------


class CaptureWorker(threading.Thread):
    def __init__(self, coordinator: CaptureCoordinator) -> None:
        super().__init__(name="CaptureWorker", daemon=True)
        self.coordinator = coordinator
        self.running = True

    def run(self) -> None:
        while self.running:
            start = time.time()
            try:
                self.coordinator.capture_loop()
            except Exception:
                LOGGER.exception("Capture worker failure")
            elapsed = time.time() - start
            CAPTURE_INTERVAL.record_load(elapsed / max(CAPTURE_INTERVAL.base_interval, 1e-3))
            time.sleep(max(0.01, CAPTURE_INTERVAL.current_interval - elapsed))

    def stop(self) -> None:
        self.running = False


CAPTURE_WORKER = CaptureWorker(CAPTURE_COORDINATOR)


# --------------------------------------------------------------------------------------
# Piece heat map generation
# --------------------------------------------------------------------------------------


def generate_piece_heatmap(grid: BoardGrid) -> np.ndarray:
    heatmap = np.zeros(grid.board_image.shape[:2], dtype=np.float32)
    for index, tile in enumerate(grid.tiles):
        row = index // 8
        col = index % 8
        score = float(np.mean(tile)) / 255.0
        y1 = row * grid.tile_size[1]
        y2 = (row + 1) * grid.tile_size[1]
        x1 = col * grid.tile_size[0]
        x2 = (col + 1) * grid.tile_size[0]
        heatmap[y1:y2, x1:x2] = score
    heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)
    return heatmap


# --------------------------------------------------------------------------------------
# Motion detection around the board for auto-pausing
# --------------------------------------------------------------------------------------


class MotionDetector:
    def __init__(self) -> None:
        self.previous = None
        self.threshold = 25.0

    def update(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous is None:
            self.previous = gray
            return 0.0
        diff = cv2.absdiff(self.previous, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion = float(np.sum(thresh) / (255.0 * thresh.size))
        self.previous = gray
        return motion


MOTION_DETECTOR = MotionDetector()


# --------------------------------------------------------------------------------------
# Auto pause/resume logic based on motion
# --------------------------------------------------------------------------------------


def auto_pause_logic(frame: np.ndarray) -> None:
    motion = MOTION_DETECTOR.update(frame)
    DIAGNOSTICS.set("motion", motion, "Motion level")
    if motion > 0.15:
        EVENT_BUS.publish("game.pause", {"reason": "motion"})
    elif motion < 0.05:
        EVENT_BUS.publish("game.resume", {"reason": "motion"})


# --------------------------------------------------------------------------------------
# Utility for drawing overlays (diagnostic only)
# --------------------------------------------------------------------------------------


def draw_board_overlay(frame: np.ndarray, detection: BoardDetection) -> np.ndarray:
    x, y, w, h = detection.board_rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    blend = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
    return blend


# --------------------------------------------------------------------------------------
# Frame recorder for debugging
# --------------------------------------------------------------------------------------


class FrameRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_directory(path)
        self.index = 0

    def record(self, frame: np.ndarray) -> None:
        self.index += 1
        filename = self.path / f"frame_{self.index:05d}.png"
        cv2.imwrite(str(filename), frame)


FRAME_RECORDER = FrameRecorder(APP_PATHS.capture_dir)


# --------------------------------------------------------------------------------------
# Capture pipeline entry points
# --------------------------------------------------------------------------------------


def start_capture_pipeline() -> None:
    CAPTURE_COORDINATOR.start()
    CAPTURE_WORKER.start()


def stop_capture_pipeline() -> None:
    CAPTURE_WORKER.stop()
    CAPTURE_COORDINATOR.stop()


# --------------------------------------------------------------------------------------
# Stockfish integration handshake
# --------------------------------------------------------------------------------------


def request_engine_analysis(fen: str) -> None:
    ENGINE_QUEUE.submit(
        EngineCommand(
            command="analyze",
            payload={"fen": fen, "multi_pv": SETTINGS.get("multi_pv", 3), "depth": SETTINGS.get("analysis_depth", 18)},
        )
    )


# --------------------------------------------------------------------------------------
# Engine response handling for overlays
# --------------------------------------------------------------------------------------


def handle_engine_result(result: EngineResult) -> None:
    if not validate_fen(result.fen):
        return
    DIAGNOSTICS.set("engine_depth", float(result.depth), "Engine depth")
    DIAGNOSTICS.set("engine_eval", result.evaluation, "Engine eval")
    overlay_text = explain_hint(result.fen, result.best_move, result.evaluation)
    NOTES.add(result.fen, overlay_text)
    EVENT_BUS.publish(
        "overlay.update",
        {
            "fen": result.fen,
            "best_move": result.best_move,
            "pv": result.pv_lines,
            "evaluation": result.evaluation,
            "depth": result.depth,
        },
    )


EVENT_BUS.subscribe(
    "engine.result",
    lambda topic, payload: handle_engine_result(payload["result"]) if "result" in payload else None,
)


# --------------------------------------------------------------------------------------
# Arrow overlay geometry helpers
# --------------------------------------------------------------------------------------


def algebraic_to_point(square: str, size: Tuple[int, int]) -> Tuple[int, int]:
    files = "abcdefgh"
    ranks = "12345678"
    file_index = files.index(square[0])
    rank_index = ranks.index(square[1])
    x = int((file_index + 0.5) * size[0] / 8)
    y = int((7 - rank_index + 0.5) * size[1] / 8)
    return x, y


def draw_best_move_arrow(image: np.ndarray, move: str, color: Tuple[int, int, int]) -> np.ndarray:
    if len(move) < 4:
        return image
    start = move[:2]
    end = move[2:4]
    h, w = image.shape[:2]
    start_pt = algebraic_to_point(start, (w, h))
    end_pt = algebraic_to_point(end, (w, h))
    overlay = image.copy()
    cv2.arrowedLine(overlay, start_pt, end_pt, color, 3, tipLength=0.3)
    return cv2.addWeighted(image, 0.75, overlay, 0.25, 0)


# --------------------------------------------------------------------------------------
# Overlay builder hooking capture with engine data
# --------------------------------------------------------------------------------------


@dataclass
class OverlayData:
    frame: np.ndarray
    fen: str
    move: str
    evaluation: float
    pv_lines: List[str]


class OverlayBuilder:
    def __init__(self) -> None:
        self.last_frame: Optional[np.ndarray] = None
        self.last_overlay: Optional[OverlayData] = None

    def build(self, frame: np.ndarray, detection: BoardDetection, result: Optional[EngineResult]) -> Optional[OverlayData]:
        if result is None:
            return None
        board_image = cv2.warpPerspective(frame, detection.perspective, (detection.heatmap.shape[1], detection.heatmap.shape[0]))
        arrow = draw_best_move_arrow(board_image, result.best_move, (0, 255, 0))
        text = describe_move(result.best_move)
        cv2.putText(
            arrow,
            f"Eval {result.evaluation:.2f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            arrow,
            text,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        self.last_overlay = OverlayData(arrow, result.fen, result.best_move, result.evaluation, result.pv_lines)
        return self.last_overlay

    def last(self) -> Optional[OverlayData]:
        return self.last_overlay


OVERLAY_BUILDER = OverlayBuilder()


# --------------------------------------------------------------------------------------
# Engine polling loop to fetch results for overlays
# --------------------------------------------------------------------------------------


class EngineResultPoller(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="EngineResultPoller", daemon=True)
        self.running = True

    def run(self) -> None:
        while self.running:
            result = ENGINE_QUEUE.poll_result(timeout=0.1)
            if result:
                EVENT_BUS.publish("engine.result", {"result": result})
                handle_engine_result(result)
            time.sleep(ENGINE_INTERVAL.current_interval)

    def stop(self) -> None:
        self.running = False


ENGINE_POLLER = EngineResultPoller()


# --------------------------------------------------------------------------------------
# Capture diagnostics overlay
# --------------------------------------------------------------------------------------


def annotate_capture(frame: np.ndarray, detection: Optional[BoardDetection], info: str) -> np.ndarray:
    overlay = frame.copy()
    if detection:
        x, y, w, h = detection.board_rect
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            overlay,
            f"Conf {detection.confidence:.2f}",
            (x + 10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        info,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)


# --------------------------------------------------------------------------------------
# Capture analysis log
# --------------------------------------------------------------------------------------


@dataclass
class CaptureAnalysis:
    timestamp: float
    fen: str
    confidence: float
    commentary: str


class CaptureAnalysisLog:
    def __init__(self) -> None:
        self.entries: List[CaptureAnalysis] = []
        self.maxlen = 200

    def add(self, fen: str, confidence: float) -> None:
        comment = COMMENTARY.annotate(confidence * 2 - 1)
        self.entries.append(CaptureAnalysis(time.time(), fen, confidence, comment))
        if len(self.entries) > self.maxlen:
            self.entries = self.entries[-self.maxlen :]

    def recent(self, seconds: float = 60.0) -> List[CaptureAnalysis]:
        cutoff = time.time() - seconds
        return [entry for entry in self.entries if entry.timestamp >= cutoff]


CAPTURE_ANALYSIS_LOG = CaptureAnalysisLog()


# --------------------------------------------------------------------------------------
# Capture smoothing filters
# --------------------------------------------------------------------------------------


class ConfidenceFilter:
    def __init__(self) -> None:
        self.value = 0.0

    def update(self, confidence: float) -> float:
        self.value = 0.8 * self.value + 0.2 * confidence
        return self.value


CONFIDENCE_FILTER = ConfidenceFilter()


# --------------------------------------------------------------------------------------
# Capture to engine dispatcher
# --------------------------------------------------------------------------------------


def dispatch_capture_to_engine(fen: str, confidence: float) -> None:
    if confidence < 0.25:
        return
    request_engine_analysis(fen)


EVENT_BUS.subscribe(
    "trainer.fen.update",
    lambda topic, payload: dispatch_capture_to_engine(payload["fen"], payload.get("confidence", 1.0))
    if "fen" in payload
    else None,
)


# --------------------------------------------------------------------------------------
# Capture statistics exported for UI
# --------------------------------------------------------------------------------------


@dataclass
class CaptureStatistics:
    captures: int = 0
    avg_confidence: float = 0.0
    board_confidence: float = 0.0


class CaptureStatsTracker:
    def __init__(self) -> None:
        self.stats = CaptureStatistics()
        self.history: List[float] = []

    def record(self, confidence: float, board_confidence: float) -> None:
        self.stats.captures += 1
        self.history.append(confidence)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        self.stats.avg_confidence = float(np.mean(self.history)) if self.history else 0.0
        self.stats.board_confidence = board_confidence

    def snapshot(self) -> CaptureStatistics:
        return self.stats


CAPTURE_STATS = CaptureStatsTracker()


# --------------------------------------------------------------------------------------
# Integration with trainer state notifications
# --------------------------------------------------------------------------------------


def on_trainer_status(status) -> None:
    if not status.current_fen:
        return
    CAPTURE_ANALYSIS_LOG.add(status.current_fen, status.overlay.evaluation if status.overlay else 0.5)


TRAINER_STATE.subscribe(on_trainer_status)


# --------------------------------------------------------------------------------------
# High frequency capture queue
# --------------------------------------------------------------------------------------


class CaptureQueue:
    def __init__(self) -> None:
        self.queue: "collections.deque[CaptureFrame]" = collections.deque(maxlen=4)
        self.lock = threading.Lock()

    def push(self, frame: CaptureFrame) -> None:
        with self.lock:
            self.queue.append(frame)

    def pop(self) -> Optional[CaptureFrame]:
        with self.lock:
            if not self.queue:
                return None
            return self.queue.popleft()


CAPTURE_QUEUE = CaptureQueue()


# --------------------------------------------------------------------------------------
# High frequency capture worker using queue
# --------------------------------------------------------------------------------------


class HighFrequencyCaptureWorker(threading.Thread):
    def __init__(self, coordinator: CaptureCoordinator) -> None:
        super().__init__(name="HighFreqCapture", daemon=True)
        self.coordinator = coordinator
        self.running = True

    def run(self) -> None:
        while self.running:
            if not self.coordinator.active:
                time.sleep(0.1)
                continue
            try:
                frame = self.coordinator.source.read()
            except Exception:
                LOGGER.exception("High frequency capture failed")
                continue
            CAPTURE_QUEUE.push(frame)
            time.sleep(0.05)

    def stop(self) -> None:
        self.running = False


HIGH_FREQ_WORKER = HighFrequencyCaptureWorker(CAPTURE_COORDINATOR)


# --------------------------------------------------------------------------------------
# Queue consumer to process frames asynchronously
# --------------------------------------------------------------------------------------


class QueueConsumer(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="CaptureQueueConsumer", daemon=True)
        self.running = True

    def run(self) -> None:
        while self.running:
            frame = CAPTURE_QUEUE.pop()
            if frame is None:
                time.sleep(0.05)
                continue
            detection = detect_board_region(frame.image)
            if not detection:
                continue
            grid = extract_board_grid(frame.image, detection)
            if not grid:
                continue
            classification = PIECE_CLASSIFIER.classify(grid)
            stable_fen = BOARD_STABILIZER.update(classification.fen)
            TRAINER_STATE.set_fen(stable_fen)
            confidence = sum(classification.confidence_map) / max(len(classification.confidence_map), 1)
            CAPTURE_STATS.record(confidence, detection.confidence)
            dispatch_capture_to_engine(stable_fen, confidence)

    def stop(self) -> None:
        self.running = False


CAPTURE_QUEUE_CONSUMER = QueueConsumer()


# --------------------------------------------------------------------------------------
# Entry points for asynchronous capture subsystem
# --------------------------------------------------------------------------------------


def start_async_capture() -> None:
    CAPTURE_COORDINATOR.start()
    if not CAPTURE_WORKER.is_alive():
        CAPTURE_WORKER.start()
    if not HIGH_FREQ_WORKER.is_alive():
        HIGH_FREQ_WORKER.start()
    if not CAPTURE_QUEUE_CONSUMER.is_alive():
        CAPTURE_QUEUE_CONSUMER.start()
    if not ENGINE_POLLER.is_alive():
        ENGINE_POLLER.start()


def stop_async_capture() -> None:
    CAPTURE_WORKER.stop()
    HIGH_FREQ_WORKER.stop()
    CAPTURE_QUEUE_CONSUMER.stop()
    ENGINE_POLLER.stop()
    CAPTURE_COORDINATOR.stop()


__all__ = [
    "start_capture_pipeline",
    "stop_capture_pipeline",
    "start_async_capture",
    "stop_async_capture",
    "CAPTURE_COORDINATOR",
    "CAPTURE_STATS",
]
