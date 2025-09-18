import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from chmd_core import (
    APP_PATHS,
    CLOCK,
    DIAGNOSTICS,
    ENGINE_QUEUE,
    EVENT_BUS,
    HINT_CACHE,
    LOGGER,
    NOTES,
    PGN_SERIALIZER,
    PROFILE_MANAGER,
    SETTINGS,
    TRAINER_STATE,
    TRAINING_PLANNER,
    VARIATION_TREE,
    COMMENTARY,
    DIFFICULTY_MODEL,
    EVALUATIONS,
    HOTKEYS,
    SCOREBOARD,
    STREAK_TRACKER,
    THEME_MANAGER,
    TRAINER_STATE,
    PROFILER,
    dump_diagnostics,
    export_history_as_json,
    import_history_from_json,
)
from chmd_core import MoveRecord, TrainerStatus
from chmd_vision import CAPTURE_COORDINATOR, CAPTURE_STATS


# --------------------------------------------------------------------------------------
# Utility widgets
# --------------------------------------------------------------------------------------


class LiquidFrame(QtWidgets.QFrame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setStyleSheet(
            """
            LiquidFrame {
                background-color: rgba(32, 32, 48, 160);
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 12px;
            }
            """
        )


class GlassButton(QtWidgets.QPushButton):
    def __init__(
        self,
        text: str,
        on_click: Optional[Callable[[], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedHeight(32)
        self.setStyleSheet(
            """
            QPushButton {
                color: #f5f5f5;
                background-color: rgba(255, 255, 255, 40);
                border: 1px solid rgba(255, 255, 255, 70);
                border-radius: 10px;
                padding: 6px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 80);
            }
            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 110);
            }
            """
        )
        if on_click is not None:
            self.clicked.connect(on_click)  # type: ignore[arg-type]


class OverlayListWidget(QtWidgets.QListWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setFixedHeight(110)
        self.setWordWrap(True)
        self.setStyleSheet(
            """
            QListWidget {
                background: transparent;
                color: #e0f7fa;
                font-size: 13px;
                border: none;
            }
            QListWidget::item {
                padding: 4px;
            }
            """
        )


# --------------------------------------------------------------------------------------
# Evaluation bar widget
# --------------------------------------------------------------------------------------


class EvaluationBar(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.evaluation = 0.0
        self.trend = 0.0
        self.setMinimumWidth(50)
        self.setMaximumWidth(80)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

    def set_evaluation(self, evaluation: float, trend: float) -> None:
        self.evaluation = evaluation
        self.trend = trend
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        rect = self.rect()
        gradient = QtGui.QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0.0, QtGui.QColor("#f5f5f5"))
        gradient.setColorAt(1.0, QtGui.QColor("#212121"))
        painter.fillRect(rect, gradient)
        value = max(-5.0, min(5.0, self.evaluation))
        ratio = (value + 5.0) / 10.0
        fill_height = rect.height() * ratio
        bar_rect = QtCore.QRectF(rect.left(), rect.bottom() - fill_height, rect.width(), fill_height)
        painter.fillRect(bar_rect, QtGui.QColor("#4caf50"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff")))
        painter.drawText(rect.adjusted(0, 5, 0, -5), QtCore.Qt.AlignCenter, f"{self.evaluation:+.2f}")
        arrow_y = rect.bottom() - fill_height
        painter.setPen(QtGui.QPen(QtGui.QColor("#2196f3"), 2))
        painter.drawLine(rect.left(), int(arrow_y), rect.right(), int(arrow_y))


# --------------------------------------------------------------------------------------
# Move history widget
# --------------------------------------------------------------------------------------


class MoveHistoryWidget(QtWidgets.QTreeWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setColumnCount(3)
        self.setHeaderLabels(["Move", "SAN", "Eval"])
        self.setAlternatingRowColors(True)
        self.setRootIsDecorated(False)
        self.setStyleSheet(
            """
            QTreeWidget {
                background-color: rgba(20, 20, 30, 160);
                color: #f5f5f5;
                border: none;
            }
            QTreeWidget::item {
                padding: 4px;
            }
            """
        )
        self.setColumnWidth(0, 60)
        self.setColumnWidth(1, 140)
        self.setColumnWidth(2, 60)

    def update_history(self, records: List[MoveRecord]) -> None:
        self.clear()
        for index, record in enumerate(records, start=1):
            item = QtWidgets.QTreeWidgetItem([str(index), record.san, f"{record.evaluation:+.2f}" if record.evaluation else ""])
            self.addTopLevelItem(item)
            if record.comment:
                note = QtWidgets.QTreeWidgetItem(["", record.comment, ""])
                item.addChild(note)
        self.expandAll()


# --------------------------------------------------------------------------------------
# Variation view
# --------------------------------------------------------------------------------------


class VariationWidget(QtWidgets.QTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumHeight(120)
        self.setStyleSheet(
            """
            QTextEdit {
                background: transparent;
                color: #b3e5fc;
                border: none;
                font-size: 12px;
            }
            """
        )

    def update_lines(self, lines: List[str]) -> None:
        self.setPlainText("\n".join(lines))


# --------------------------------------------------------------------------------------
# FEN display widget
# --------------------------------------------------------------------------------------


class FENDisplay(QtWidgets.QLineEdit):
    def __init__(self) -> None:
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet(
            """
            QLineEdit {
                background-color: rgba(0, 0, 0, 90);
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 6px;
                color: #ffffff;
                font-family: "Fira Code";
                font-size: 11px;
                padding: 4px;
            }
            """
        )


# --------------------------------------------------------------------------------------
# Overlay window always-on-top
# --------------------------------------------------------------------------------------


class OverlayWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__(flags=QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        self.setWindowFlag(QtCore.Qt.WindowDoesNotAcceptFocus)
        self.resize(360, 200)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.overlay_title = QtWidgets.QLabel("CHMD Overlay")
        self.overlay_title.setStyleSheet("color: #f5f5f5; font-size: 16px; font-weight: 600;")
        layout.addWidget(self.overlay_title)
        self.best_move_label = QtWidgets.QLabel("Best move")
        self.best_move_label.setStyleSheet("color: #ffffff; font-size: 24px; font-weight: bold;")
        layout.addWidget(self.best_move_label)
        self.evaluation_label = QtWidgets.QLabel("Evaluation")
        layout.addWidget(self.evaluation_label)
        self.pv_list = OverlayListWidget()
        layout.addWidget(self.pv_list)
        self.opacity = SETTINGS.get("overlay_opacity", 0.85)
        self.setWindowOpacity(self.opacity)

    def update_overlay(self, status: TrainerStatus) -> None:
        overlay = status.overlay
        if not overlay:
            self.best_move_label.setText("Waiting for analysis…")
            self.pv_list.clear()
            return
        self.best_move_label.setText(overlay.best_move)
        self.evaluation_label.setText(f"Eval: {overlay.evaluation:+.2f} depth {overlay.depth}")
        self.pv_list.clear()
        for line in overlay.pv_lines:
            item = QtWidgets.QListWidgetItem(line)
            self.pv_list.addItem(item)

    def toggle_visibility(self) -> None:
        self.setVisible(not self.isVisible())


# --------------------------------------------------------------------------------------
# Control panel
# --------------------------------------------------------------------------------------


class ControlPanel(LiquidFrame):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        self.hint_button = GlassButton("Toggle Hints")
        self.save_button = GlassButton("Save Game")
        self.load_button = GlassButton("Load Last")
        self.tactics_button = GlassButton("New Tactic")
        self.settings_button = GlassButton("Settings")
        self.overlay_button = GlassButton("Overlay")
        layout.addWidget(self.hint_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.load_button)
        layout.addWidget(self.tactics_button)
        layout.addWidget(self.settings_button)
        layout.addWidget(self.overlay_button)


# --------------------------------------------------------------------------------------
# Diagnostics panel
# --------------------------------------------------------------------------------------


class DiagnosticsPanel(LiquidFrame):
    def __init__(self) -> None:
        super().__init__()
        self.layout = QtWidgets.QFormLayout(self)
        self.layout.setLabelAlignment(QtCore.Qt.AlignLeft)
        self.layout.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.labels: Dict[str, QtWidgets.QLabel] = {}

    def update_metrics(self) -> None:
        snapshot = DIAGNOSTICS.get_snapshot()
        for key, (value, label) in snapshot.items():
            widget = self.labels.get(key)
            if not widget:
                widget = QtWidgets.QLabel()
                widget.setStyleSheet("color: #bbdefb; font-size: 11px;")
                self.labels[key] = widget
                self.layout.addRow(f"{label}:", widget)
            widget.setText(f"{value:.3f}")


# --------------------------------------------------------------------------------------
# Move explanation widget
# --------------------------------------------------------------------------------------


class MoveExplanationWidget(QtWidgets.QTextBrowser):
    def __init__(self) -> None:
        super().__init__()
        self.setStyleSheet(
            """
            QTextBrowser {
                background-color: rgba(15, 15, 30, 180);
                border: none;
                color: #f5f5f5;
                font-size: 12px;
            }
            """
        )
        self.setMaximumHeight(140)

    def add_entry(self, text: str) -> None:
        self.append(text)
        self.moveCursor(QtGui.QTextCursor.End)


# --------------------------------------------------------------------------------------
# Evaluation graph placeholder
# --------------------------------------------------------------------------------------


class EvaluationGraph(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.points: List[float] = []
        self.setMinimumHeight(120)
        self.setStyleSheet("background: rgba(0, 0, 0, 120);")

    def append(self, value: float) -> None:
        self.points.append(value)
        if len(self.points) > 120:
            self.points = self.points[-120:]
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor(0, 0, 0, 140))
        if len(self.points) < 2:
            return
        step = rect.width() / (len(self.points) - 1)
        path = QtGui.QPainterPath()
        max_eval = max(1.0, max(abs(p) for p in self.points))
        for index, value in enumerate(self.points):
            x = rect.left() + index * step
            normalized = (value / (2 * max_eval)) + 0.5
            y = rect.bottom() - normalized * rect.height()
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        pen = QtGui.QPen(QtGui.QColor("#4caf50"), 2)
        painter.setPen(pen)
        painter.drawPath(path)


# --------------------------------------------------------------------------------------
# Chessboard view using QGraphicsView
# --------------------------------------------------------------------------------------


class ChessBoardView(QtWidgets.QGraphicsView):
    square_clicked = QtCore.pyqtSignal(str)
    square_dragged = QtCore.pyqtSignal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setStyleSheet("background: rgba(10, 10, 20, 200);")
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.square_size = 80
        self.board_group = QtWidgets.QGraphicsItemGroup()
        self.scene.addItem(self.board_group)
        self.piece_group = QtWidgets.QGraphicsItemGroup()
        self.scene.addItem(self.piece_group)
        self._selected_square: Optional[str] = None
        self._drag_start: Optional[str] = None
        self.draw_board()

    def draw_board(self) -> None:
        self.scene.clear()
        self.board_group = QtWidgets.QGraphicsItemGroup()
        self.scene.addItem(self.board_group)
        self.piece_group = QtWidgets.QGraphicsItemGroup()
        self.scene.addItem(self.piece_group)
        theme = THEME_MANAGER.active_theme
        light = QtGui.QColor(theme.board_light)
        dark = QtGui.QColor(theme.board_dark)
        for rank in range(8):
            for file in range(8):
                rect = QtCore.QRectF(file * self.square_size, rank * self.square_size, self.square_size, self.square_size)
                color = light if (rank + file) % 2 == 0 else dark
                item = self.scene.addRect(rect, brush=QtGui.QBrush(color))
                self.board_group.addToGroup(item)
        self.setSceneRect(0, 0, self.square_size * 8, self.square_size * 8)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        size = min(self.width(), self.height())
        self.square_size = size // 8
        self.setSceneRect(0, 0, self.square_size * 8, self.square_size * 8)
        self.draw_board()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        position = self.mapToScene(event.pos())
        file = int(position.x() // self.square_size)
        rank = int(position.y() // self.square_size)
        if 0 <= file < 8 and 0 <= rank < 8:
            square = chr(ord('a') + file) + str(8 - rank)
            self._selected_square = square
            self.square_clicked.emit(square)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._selected_square is None:
            return
        position = self.mapToScene(event.pos())
        file = max(0, min(7, int(position.x() // self.square_size)))
        rank = max(0, min(7, int(position.y() // self.square_size)))
        square = chr(ord('a') + file) + str(8 - rank)
        self._drag_start = self._drag_start or self._selected_square
        if square != self._drag_start:
            self.square_dragged.emit(self._drag_start, square)
            self._drag_start = square
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._selected_square and self._drag_start:
            self.square_dragged.emit(self._selected_square, self._drag_start)
        self._selected_square = None
        self._drag_start = None
        super().mouseReleaseEvent(event)


# --------------------------------------------------------------------------------------
# Settings dialog
# --------------------------------------------------------------------------------------


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        layout = QtWidgets.QFormLayout(self)
        self.engine_path = QtWidgets.QLineEdit(SETTINGS.get("engine_path", "stockfish"))
        self.engine_threads = QtWidgets.QSpinBox()
        self.engine_threads.setRange(1, 16)
        self.engine_threads.setValue(SETTINGS.get("engine_threads", 4))
        self.multi_pv = QtWidgets.QSpinBox()
        self.multi_pv.setRange(1, 5)
        self.multi_pv.setValue(SETTINGS.get("multi_pv", 3))
        self.overlay_opacity = QtWidgets.QDoubleSpinBox()
        self.overlay_opacity.setRange(0.1, 1.0)
        self.overlay_opacity.setSingleStep(0.05)
        self.overlay_opacity.setValue(SETTINGS.get("overlay_opacity", 0.85))
        layout.addRow("Engine Path", self.engine_path)
        layout.addRow("Threads", self.engine_threads)
        layout.addRow("MultiPV", self.multi_pv)
        layout.addRow("Overlay Opacity", self.overlay_opacity)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:  # type: ignore[override]
        SETTINGS.set("engine_path", self.engine_path.text())
        SETTINGS.set("engine_threads", self.engine_threads.value())
        SETTINGS.set("multi_pv", self.multi_pv.value())
        SETTINGS.set("overlay_opacity", self.overlay_opacity.value())
        super().accept()


# --------------------------------------------------------------------------------------
# Trainer main window
# --------------------------------------------------------------------------------------


class BaseTrainerMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CHMD Trainer")
        self.setMinimumSize(1280, 720)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        self.overlay_window = OverlayWindow()
        self.overlay_window.hide()
        layout = QtWidgets.QGridLayout(central_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)

        self.board_view = ChessBoardView()
        layout.addWidget(self.board_view, 0, 0, 3, 1)

        self.eval_bar = EvaluationBar()
        layout.addWidget(self.eval_bar, 0, 1, 1, 1)

        self.move_history = MoveHistoryWidget()
        layout.addWidget(self.move_history, 1, 1, 1, 1)

        self.variation_widget = VariationWidget()
        layout.addWidget(self.variation_widget, 2, 1, 1, 1)

        self.fen_display = FENDisplay()
        layout.addWidget(self.fen_display, 3, 0, 1, 2)

        self.move_explanation = MoveExplanationWidget()
        layout.addWidget(self.move_explanation, 4, 0, 1, 2)

        self.evaluation_graph = EvaluationGraph()
        layout.addWidget(self.evaluation_graph, 5, 0, 1, 2)

        self.control_panel = ControlPanel()
        layout.addWidget(self.control_panel, 0, 2, 2, 1)

        self.diagnostics_panel = DiagnosticsPanel()
        layout.addWidget(self.diagnostics_panel, 2, 2, 2, 1)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self.update_status_widgets)
        self.status_timer.start(500)

        self._install_connections()
        self._register_hotkeys()
        TRAINER_STATE.subscribe(self.on_trainer_status)
        EVENT_BUS.subscribe("overlay.update", self.on_overlay_update)
        EVENT_BUS.subscribe("game.pause", self.on_game_pause)
        EVENT_BUS.subscribe("game.resume", self.on_game_resume)

    def _install_connections(self) -> None:
        self.control_panel.hint_button.clicked.connect(self.overlay_window.toggle_visibility)
        self.control_panel.overlay_button.clicked.connect(self.overlay_window.toggle_visibility)
        self.control_panel.save_button.clicked.connect(self.save_game)
        self.control_panel.load_button.clicked.connect(self.load_game)
        self.control_panel.tactics_button.clicked.connect(self.start_tactic)
        self.control_panel.settings_button.clicked.connect(self.open_settings)
        self.board_view.square_clicked.connect(self.on_square_clicked)
        self.board_view.square_dragged.connect(self.on_square_dragged)

    def _register_hotkeys(self) -> None:
        HOTKEYS.register("Space", "Toggle overlay", self.overlay_window.toggle_visibility)
        HOTKEYS.register("Ctrl+S", "Save game", self.save_game)
        HOTKEYS.register("Ctrl+L", "Load last game", self.load_game)
        HOTKEYS.register("Ctrl+T", "Start tactic", self.start_tactic)

    # ----------------------------------------------------------------------------------
    # UI callbacks
    # ----------------------------------------------------------------------------------

    def on_square_clicked(self, square: str) -> None:
        self.status_bar.showMessage(f"Clicked {square}")

    def on_square_dragged(self, start: str, end: str) -> None:
        self.status_bar.showMessage(f"Dragged {start}->{end}")

    def on_overlay_update(self, topic: str, payload: Dict[str, object]) -> None:
        overlay = HINT_CACHE.get(payload.get("fen", ""))
        if overlay:
            self.overlay_window.best_move_label.setText(overlay.best_move)
        pv_lines = payload.get("pv", [])
        if isinstance(pv_lines, list):
            self.overlay_window.pv_list.clear()
            for line in pv_lines:
                self.overlay_window.pv_list.addItem(str(line))

    def on_game_pause(self, topic: str, payload: Dict[str, object]) -> None:
        self.status_bar.showMessage("Paused due to motion")

    def on_game_resume(self, topic: str, payload: Dict[str, object]) -> None:
        self.status_bar.showMessage("Resumed")

    def update_status_widgets(self) -> None:
        self.diagnostics_panel.update_metrics()
        stats = CAPTURE_STATS.snapshot()
        self.status_bar.showMessage(
            f"Captures: {stats.captures} avg {stats.avg_confidence:.2f} board {stats.board_confidence:.2f}"
        )
        self.eval_bar.set_evaluation(EVALUATIONS.last(), EVALUATIONS.trend())
        self.evaluation_graph.append(EVALUATIONS.last())

    def on_trainer_status(self, status: TrainerStatus) -> None:
        self.fen_display.setText(status.current_fen)
        self.overlay_window.update_overlay(status)
        self.move_history.update_history(TRAINER_STATE.history.records)

    def save_game(self) -> None:
        history = TRAINER_STATE.history
        PGN_SERIALIZER.save(history, APP_PATHS.default_pgn)
        export_history_as_json(history, APP_PATHS.default_pgn.with_suffix(".json"))
        self.status_bar.showMessage("Game saved")

    def load_game(self) -> None:
        try:
            history = import_history_from_json(APP_PATHS.default_pgn.with_suffix(".json"))
        except FileNotFoundError:
            self.status_bar.showMessage("No saved game")
            return
        TRAINER_STATE.history = history
        self.status_bar.showMessage("Game loaded")

    def start_tactic(self) -> None:
        task = TRAINING_PLANNER.random_task()
        TRAINER_STATE.set_fen(task.fen)
        self.move_explanation.add_entry(f"New tactic: {task.description}")

    def open_settings(self) -> None:
        dialog = SettingsDialog(self)
        if dialog.exec_():
            self.overlay_window.setWindowOpacity(SETTINGS.get("overlay_opacity", 0.85))


# --------------------------------------------------------------------------------------
# Background worker to pump events into UI
# --------------------------------------------------------------------------------------


class StatusDispatcher(QtCore.QObject):
    status_updated = QtCore.pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        TRAINER_STATE.subscribe(self._on_status)

    def _on_status(self, status: TrainerStatus) -> None:
        self.status_updated.emit(status)


STATUS_DISPATCHER = StatusDispatcher()


# --------------------------------------------------------------------------------------
# Overlay manager for top-level toggles
# --------------------------------------------------------------------------------------


class OverlayManager:
    def __init__(self, overlay: OverlayWindow) -> None:
        self.overlay = overlay
        self.visible = False

    def toggle(self) -> None:
        self.visible = not self.visible
        self.overlay.setVisible(self.visible)


# --------------------------------------------------------------------------------------
# Diagnostics exporter thread
# --------------------------------------------------------------------------------------


class DiagnosticsExporter(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="DiagnosticsExporter", daemon=True)
        self.running = True

    def run(self) -> None:
        while self.running:
            dump_diagnostics(APP_PATHS.root / "diagnostics.json")
            time.sleep(10.0)

    def stop(self) -> None:
        self.running = False


DIAGNOSTICS_EXPORTER = DiagnosticsExporter()


# --------------------------------------------------------------------------------------
# Menu bar
# --------------------------------------------------------------------------------------


class MenuBar(QtWidgets.QMenuBar):
    def __init__(self, window: BaseTrainerMainWindow) -> None:
        super().__init__(window)
        file_menu = self.addMenu("File")
        save_action = file_menu.addAction("Save Game")
        save_action.triggered.connect(window.save_game)
        load_action = file_menu.addAction("Load Game")
        load_action.triggered.connect(window.load_game)
        export_action = file_menu.addAction("Export Diagnostics")
        export_action.triggered.connect(lambda: dump_diagnostics(APP_PATHS.root / "manual_dump.json"))
        view_menu = self.addMenu("View")
        overlay_action = view_menu.addAction("Toggle Overlay")
        overlay_action.triggered.connect(window.overlay_window.toggle_visibility)
        tools_menu = self.addMenu("Tools")
        settings_action = tools_menu.addAction("Settings")
        settings_action.triggered.connect(window.open_settings)


# --------------------------------------------------------------------------------------
# Application controller binding UI with backend
# --------------------------------------------------------------------------------------


class ApplicationController:
    def __init__(self) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.window = TrainerMainWindow()
        self.window.setMenuBar(MenuBar(self.window))
        self.overlay_manager = OverlayManager(self.window.overlay_window)
        self.status_thread = threading.Thread(target=self._status_loop, name="StatusLoop", daemon=True)
        self.running = False

    def _status_loop(self) -> None:
        while self.running:
            PROFILER.start("ui_status_loop")
            EVENT_BUS.publish("ui.refresh")
            PROFILER.stop("ui_status_loop")
            time.sleep(0.5)

    def start(self) -> None:
        self.running = True
        if not DIAGNOSTICS_EXPORTER.is_alive():
            DIAGNOSTICS_EXPORTER.start()
        self.status_thread.start()
        self.window.show()
        self.overlay_manager.overlay.hide()
        self.app.exec_()

    def stop(self) -> None:
        self.running = False
        DIAGNOSTICS_EXPORTER.stop()


__all__ = ["ApplicationController", "TrainerMainWindow", "OverlayWindow"]

# --------------------------------------------------------------------------------------
# Scoreboard widget
# --------------------------------------------------------------------------------------


class ScoreboardWidget(LiquidFrame):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.title = QtWidgets.QLabel("Leaderboard")
        self.title.setStyleSheet("color: #f5f5f5; font-weight: bold;")
        layout.addWidget(self.title)
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setStyleSheet(
            """
            QListWidget {
                background: transparent;
                color: #ffffff;
                border: none;
            }
            QListWidget::item {
                padding: 4px;
            }
            """
        )
        layout.addWidget(self.list_widget)
        self.refresh()

    def refresh(self) -> None:
        self.list_widget.clear()
        for rank, (player, score) in enumerate(SCOREBOARD.top(), start=1):
            item = QtWidgets.QListWidgetItem(f"{rank}. {player} {score:.1f}")
            self.list_widget.addItem(item)


# --------------------------------------------------------------------------------------
# Clock widget
# --------------------------------------------------------------------------------------


class ClockWidget(LiquidFrame):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        self.white_label = QtWidgets.QLabel("White 05:00")
        self.black_label = QtWidgets.QLabel("Black 05:00")
        for label in (self.white_label, self.black_label):
            label.setStyleSheet("color: #f5f5f5; font-size: 18px;")
            layout.addWidget(label)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(1000)

    def _format(self, seconds: float) -> str:
        minutes = int(seconds) // 60
        sec = int(seconds) % 60
        return f"{minutes:02d}:{sec:02d}"

    def _update(self) -> None:
        self.white_label.setText(f"White {self._format(CLOCK.white_time)}")
        self.black_label.setText(f"Black {self._format(CLOCK.black_time)}")


# --------------------------------------------------------------------------------------
# Hotkey helper dialog
# --------------------------------------------------------------------------------------


class HotkeyDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Hotkeys")
        layout = QtWidgets.QVBoxLayout(self)
        table = QtWidgets.QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["Key", "Action"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for key, description in HOTKEYS.list_bindings():
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(description))
        layout.addWidget(table)
        close_button = GlassButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)


# --------------------------------------------------------------------------------------
# Profile widget
# --------------------------------------------------------------------------------------


class ProfileWidget(LiquidFrame):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QFormLayout(self)
        layout.setLabelAlignment(QtCore.Qt.AlignLeft)
        profile = PROFILE_MANAGER.profile
        layout.addRow("Name", QtWidgets.QLabel(profile.name))
        layout.addRow("Games", QtWidgets.QLabel(str(profile.games_played)))
        layout.addRow("Wins", QtWidgets.QLabel(str(profile.wins)))
        layout.addRow("Losses", QtWidgets.QLabel(str(profile.losses)))
        layout.addRow("Draws", QtWidgets.QLabel(str(profile.draws)))
        layout.addRow("High", QtWidgets.QLabel(str(profile.highest_rating)))
        layout.addRow("Low", QtWidgets.QLabel(str(profile.lowest_rating)))


# --------------------------------------------------------------------------------------
# Hint ribbon widget
# --------------------------------------------------------------------------------------


class HintRibbon(LiquidFrame):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.label = QtWidgets.QLabel("Hints ready")
        self.label.setStyleSheet("color: #ffe082; font-size: 14px;")
        layout.addWidget(self.label)

    def update_hint(self, text: str) -> None:
        self.label.setText(text)


# --------------------------------------------------------------------------------------
# Tactics streak widget
# --------------------------------------------------------------------------------------


class StreakWidget(LiquidFrame):
    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.current_label = QtWidgets.QLabel("Current streak: 0")
        self.best_label = QtWidgets.QLabel("Best streak: 0")
        for label in (self.current_label, self.best_label):
            label.setStyleSheet("color: #c5e1a5; font-size: 14px;")
            layout.addWidget(label)
        EVENT_BUS.subscribe("tactic.result", self._on_result)

    def _on_result(self, topic: str, payload: Dict[str, object]) -> None:
        correct = payload.get("correct", False)
        STREAK_TRACKER.record(bool(correct))
        self.current_label.setText(f"Current streak: {STREAK_TRACKER.current}")
        self.best_label.setText(f"Best streak: {STREAK_TRACKER.best}")


# --------------------------------------------------------------------------------------
# Overlay inspector window for debugging
# --------------------------------------------------------------------------------------


class OverlayInspector(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Overlay Inspector")
        layout = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        close_btn = GlassButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        EVENT_BUS.subscribe("overlay.update", self._on_update)

    def _on_update(self, topic: str, payload: Dict[str, object]) -> None:
        lines = [f"{key}: {value}" for key, value in payload.items()]
        self.text.setPlainText("\n".join(lines))


# --------------------------------------------------------------------------------------
# Extended main window integration of new widgets
# --------------------------------------------------------------------------------------


class TrainerMainWindow(BaseTrainerMainWindow):  # type: ignore[misc]
    def __init__(self) -> None:  # type: ignore[override]
        super().__init__()
        self.scoreboard_widget = ScoreboardWidget()
        self.clock_widget = ClockWidget()
        self.profile_widget = ProfileWidget()
        self.hint_ribbon = HintRibbon()
        self.streak_widget = StreakWidget()
        self.overlay_inspector = OverlayInspector(self)
        self.additional_layout = QtWidgets.QVBoxLayout()
        self.additional_layout.addWidget(self.scoreboard_widget)
        self.additional_layout.addWidget(self.clock_widget)
        self.additional_layout.addWidget(self.profile_widget)
        self.additional_layout.addWidget(self.hint_ribbon)
        self.additional_layout.addWidget(self.streak_widget)
        self.centralWidget().layout().addLayout(self.additional_layout, 3, 2, 2, 1)
        self.statusBar().addPermanentWidget(GlassButton("Hotkeys", self.open_hotkeys))
        self.statusBar().addPermanentWidget(GlassButton("Overlay Inspector", self.overlay_inspector.show))

    def open_hotkeys(self) -> None:
        dialog = HotkeyDialog(self)
        dialog.exec_()

    def on_trainer_status(self, status: TrainerStatus) -> None:  # type: ignore[override]
        super().on_trainer_status(status)
        self.scoreboard_widget.refresh()
        hint = NOTES.get(status.current_fen)
        if hint:
            self.hint_ribbon.update_hint(hint)


# --------------------------------------------------------------------------------------
# Entry point convenience function
# --------------------------------------------------------------------------------------


def run_ui() -> None:
    controller = ApplicationController()
    controller.start()


__all__.append("run_ui")

# --------------------------------------------------------------------------------------
# Status toast notifications
# --------------------------------------------------------------------------------------


class StatusToast(QtWidgets.QLabel):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("", parent)
        self.setStyleSheet("background: rgba(0,0,0,160); color: white; padding: 6px 12px; border-radius: 8px;")
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)

    def show_message(self, text: str, duration: int = 2000) -> None:
        self.setText(text)
        self.adjustSize()
        if self.parent():
            geo = self.parent().geometry()
            self.move(geo.width() - self.width() - 40, 40)
        self.show()
        self.timer.start(duration)


__all__.append("StatusToast")
