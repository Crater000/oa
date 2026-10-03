import csv
import os
import sys
from collections import deque
from datetime import datetime

import mss
import numpy as np
from PIL import Image

from PySide6.QtCore import (
    Qt,
    QRect,
    QPoint,
    QSize,
    QTimer,
)

from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPainter,
    QColor,
)

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QRubberBand,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFrame,
    QScrollArea,
)

from blitz_signal import (
    analyze_structure,
    get_candle_data,
)

from chart_reader import (
    detect_colors,
    clean_mask,
    find_candle_candidates,
    merge_close_candidates,
    filter_candidates,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

INTERVAL_SECONDS = 10
CHART_CHANGE_THRESHOLD = 0.8
CONFIRMATION_COUNT = 3

DATA_DIR = "data"

SCREENSHOT_PATH = os.path.join(
    DATA_DIR,
    "grafico.png"
)

CSV_PATH = os.path.join(
    DATA_DIR,
    "blitz_signals.csv"
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ============================================================
# COLORES
# ============================================================

BG = "#080b12"
CARD = "#10141d"
CARD_2 = "#151a24"
BORDER = "#252c38"

TEXT = "#f5f5f7"
MUTED = "#8e96a3"

GREEN = "#30d158"
GREEN_DARK = "#163722"

RED = "#ff453a"
RED_DARK = "#421b1a"

YELLOW = "#ffbd2e"
BLUE = "#4da3ff"


# ============================================================
# CARD
# ============================================================

class Card(QFrame):

    def __init__(
        self,
        parent=None
    ):
        super().__init__(
            parent
        )

        self.setObjectName(
            "Card"
        )

        self.setStyleSheet(
            f"""
            QFrame#Card {{
                background-color: {CARD};
                border: 1px solid {BORDER};
                border-radius: 16px;
            }}
            """
        )


# ============================================================
# BOTONES MAC
# ============================================================

class MacButton(QPushButton):

    def __init__(
        self,
        color,
        hover_color,
        parent=None
    ):
        super().__init__(
            parent
        )

        self.setFixedSize(
            13,
            13
        )

        self.setCursor(
            Qt.PointingHandCursor
        )

        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {color};
                border: none;
                border-radius: 6px;
                padding: 0px;
                margin: 0px;
            }}

            QPushButton:hover {{
                background-color: {hover_color};
            }}
            """
        )


# ============================================================
# SELECTOR DE GRÁFICO
# ============================================================

class ScreenSelector(QWidget):

    def __init__(
        self,
        callback
    ):
        super().__init__()

        self.callback = callback

        self.start = QPoint()

        self.selecting = False

        self.rubber_band = QRubberBand(
            QRubberBand.Rectangle,
            self
        )

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )

        self.setAttribute(
            Qt.WA_TranslucentBackground
        )

        self.setCursor(
            Qt.CrossCursor
        )

        with mss.mss() as sct:

            monitor = sct.monitors[1]

        self.monitor_left = (
            monitor["left"]
        )

        self.monitor_top = (
            monitor["top"]
        )

        self.setGeometry(
            monitor["left"],
            monitor["top"],
            monitor["width"],
            monitor["height"]
        )

        # ----------------------------------------------------
        # TEXTO PRINCIPAL
        # ----------------------------------------------------

        self.instruction = QLabel(
            "SELECCIONA EL ÁREA DEL GRÁFICO",
            self
        )

        self.instruction.setAlignment(
            Qt.AlignCenter
        )

        self.instruction.setStyleSheet(
            """
            QLabel {
                color: white;
                background-color: rgba(10,14,22,235);
                border: 1px solid rgba(255,255,255,45);
                border-radius: 12px;
                padding: 12px 25px;
                font-size: 18px;
                font-weight: 800;
            }
            """
        )

        self.instruction.adjustSize()

        # ----------------------------------------------------
        # SUBTÍTULO
        # ----------------------------------------------------

        self.sub_instruction = QLabel(
            "Mantén presionado el botón izquierdo y arrastra",
            self
        )

        self.sub_instruction.setAlignment(
            Qt.AlignCenter
        )

        self.sub_instruction.setStyleSheet(
            """
            QLabel {
                color: rgba(255,255,255,200);
                background-color: rgba(10,14,22,220);
                border-radius: 9px;
                padding: 7px 15px;
                font-size: 11px;
            }
            """
        )

        self.sub_instruction.adjustSize()

        # ----------------------------------------------------
        # TAMAÑO
        # ----------------------------------------------------

        self.size_label = QLabel(
            "",
            self
        )

        self.size_label.setAlignment(
            Qt.AlignCenter
        )

        self.size_label.setStyleSheet(
            """
            QLabel {
                color: white;
                background-color: rgba(10,14,22,235);
                border: 1px solid rgba(255,255,255,45);
                border-radius: 8px;
                padding: 5px 10px;
                font-size: 10px;
                font-weight: 700;
            }
            """
        )

        self.size_label.hide()

    # ========================================================
    # FONDO
    # ========================================================

    def paintEvent(
        self,
        event
    ):

        painter = QPainter(
            self
        )

        painter.fillRect(
            self.rect(),
            QColor(
                5,
                8,
                14,
                175
            )
        )

        painter.end()

    # ========================================================
    # PRESS
    # ========================================================

    def mousePressEvent(
        self,
        event
    ):

        if event.button() != Qt.LeftButton:
            return

        self.selecting = True

        self.start = event.pos()

        self.rubber_band.setGeometry(
            QRect(
                self.start,
                QSize(
                    0,
                    0
                )
            )
        )

        self.rubber_band.show()

        self.size_label.show()

    # ========================================================
    # MOVE
    # ========================================================

    def mouseMoveEvent(
        self,
        event
    ):

        if not self.selecting:
            return

        rect = QRect(
            self.start,
            event.pos()
        ).normalized()

        self.rubber_band.setGeometry(
            rect
        )

        self.update_size_label(
            rect
        )

    # ========================================================
    # RELEASE
    # ========================================================

    def mouseReleaseEvent(
        self,
        event
    ):

        if event.button() != Qt.LeftButton:
            return

        if not self.selecting:
            return

        self.selecting = False

        rect = QRect(
            self.start,
            event.pos()
        ).normalized()

        if (
            rect.width() >= 150
            and rect.height() >= 120
        ):

            final_rect = QRect(
                self.monitor_left + rect.x(),
                self.monitor_top + rect.y(),
                rect.width(),
                rect.height()
            )

            self.callback(
                final_rect
            )

        self.close()

    # ========================================================
    # TAMAÑO
    # ========================================================

    def update_size_label(
        self,
        rect
    ):

        width = rect.width()

        height = rect.height()

        self.size_label.setText(
            f"{width} × {height} px"
        )

        self.size_label.adjustSize()

        x = rect.right() + 10
        y = rect.bottom() + 10

        if (
            x + self.size_label.width()
            > self.width()
        ):

            x = (
                rect.left()
                - self.size_label.width()
                - 10
            )

        if (
            y + self.size_label.height()
            > self.height()
        ):

            y = (
                rect.top()
                - self.size_label.height()
                - 10
            )

        self.size_label.move(
            x,
            y
        )

    # ========================================================
    # RESIZE
    # ========================================================

    def resizeEvent(
        self,
        event
    ):

        if hasattr(
            self,
            "instruction"
        ):

            self.instruction.move(
                (
                    self.width()
                    - self.instruction.width()
                ) // 2,
                35
            )

        if hasattr(
            self,
            "sub_instruction"
        ):

            self.sub_instruction.move(
                (
                    self.width()
                    - self.sub_instruction.width()
                ) // 2,
                83
            )

        super().resizeEvent(
            event
        )


# ============================================================
# MONITOR
# ============================================================

class BlitzMonitor(QWidget):

    RESIZE_MARGIN = 10

    def __init__(
        self
    ):
        super().__init__()

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.Window
        )

        self.setAttribute(
            Qt.WA_TranslucentBackground
        )

        self.setWindowTitle(
            "IQ AI — BTC BLITZ"
        )

        # ----------------------------------------------------
        # TAMANHO
        # ----------------------------------------------------

        self.setMinimumSize(
            700,
            500
        )

        self.resize(
            1250,
            800
        )

        # ----------------------------------------------------
        # MOVIMIENTO
        # ----------------------------------------------------

        self.dragging = False

        self.drag_position = QPoint()

        # ----------------------------------------------------
        # REDIMENSIONAMIENTO
        # ----------------------------------------------------

        self.resizing = False

        self.resize_edges = 0

        self.resize_start_geometry = QRect()

        self.resize_start_position = QPoint()

        # ----------------------------------------------------
        # MONITOR
        # ----------------------------------------------------

        self.running = False

        self.chart_rect = None

        self.selector = None

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.process_capture
        )

        self.previous_mask = None

        self.confirmation_history = deque(
            maxlen=CONFIRMATION_COUNT
        )

        self.last_signal = (
            "NO OPERAR"
        )

        self.last_score = 0

        self.last_movement = 0

        self.last_candle_change = 0

        self.last_candles = []

        self.history = []

        self.analysis_count = 0

        self.last_analysis_time = None

        # ----------------------------------------------------
        # ESTILO
        # ----------------------------------------------------

        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: {BG};
                color: {TEXT};
                font-family: "Segoe UI";
            }}

            QWidget#MainWindow {{
                background-color: {BG};
                border: 1px solid #303746;
                border-radius: 18px;
            }}

            QLabel {{
                background: transparent;
            }}

            QPushButton {{
                border: none;
                border-radius: 9px;
                padding: 10px 16px;
                font-size: 12px;
                font-weight: 600;
            }}

            QPushButton:hover {{
                background-color: #202735;
            }}

            QPushButton:pressed {{
                background-color: #2a3241;
            }}

            QScrollBar:vertical {{
                background: transparent;
                width: 7px;
            }}

            QScrollBar::handle:vertical {{
                background: #303746;
                border-radius: 3px;
            }}

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            """
        )

        self.build_ui()

    # ========================================================
    # UI
    # ========================================================

    def build_ui(
        self
    ):

        outer = QVBoxLayout(
            self
        )

        outer.setContentsMargins(
            1,
            1,
            1,
            1
        )

        outer.setSpacing(
            0
        )

        self.main_window = QWidget()

        self.main_window.setObjectName(
            "MainWindow"
        )

        outer.addWidget(
            self.main_window
        )

        main = QVBoxLayout(
            self.main_window
        )

        main.setContentsMargins(
            18,
            12,
            18,
            14
        )

        main.setSpacing(
            13
        )

        # ====================================================
        # TOP BAR
        # ====================================================

        top_bar = QWidget()

        top_bar.setFixedHeight(
            38
        )

        top_layout = QHBoxLayout(
            top_bar
        )

        top_layout.setContentsMargins(
            5,
            0,
            3,
            0
        )

        top_layout.setSpacing(
            7
        )

        self.close_button = MacButton(
            "#ff5f57",
            "#ff7b75"
        )

        self.minimize_button = MacButton(
            "#ffbd2e",
            "#ffd36a"
        )

        self.maximize_button = MacButton(
            "#28c840",
            "#55d968"
        )

        self.close_button.clicked.connect(
            self.close
        )

        self.minimize_button.clicked.connect(
            self.showMinimized
        )

        self.maximize_button.clicked.connect(
            self.toggle_maximize
        )

        top_layout.addWidget(
            self.close_button
        )

        top_layout.addWidget(
            self.minimize_button
        )

        top_layout.addWidget(
            self.maximize_button
        )

        top_layout.addSpacing(
            10
        )

        title = QLabel(
            "IQ AI"
        )

        title.setStyleSheet(
            f"""
            color: {TEXT};
            font-size: 13px;
            font-weight: 700;
            """
        )

        top_layout.addWidget(
            title
        )

        version = QLabel(
            "BTC BLITZ"
        )

        version.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 10px;
            font-weight: 600;
            """
        )

        top_layout.addWidget(
            version
        )

        top_layout.addStretch()

        self.live_label = QLabel(
            "● OFFLINE"
        )

        self.live_label.setStyleSheet(
            f"""
            color: {MUTED};
            background-color: {CARD_2};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 5px 11px;
            font-size: 9px;
            font-weight: 700;
            """
        )

        top_layout.addWidget(
            self.live_label
        )

        main.addWidget(
            top_bar
        )

        self.title_bar = top_bar

        # ====================================================
        # LINE
        # ====================================================

        line = QFrame()

        line.setFrameShape(
            QFrame.HLine
        )

        line.setStyleSheet(
            f"color: {BORDER};"
        )

        main.addWidget(
            line
        )

        # ====================================================
        # CONTENT
        # ====================================================

        content = QHBoxLayout()

        content.setSpacing(
            13
        )

        # ====================================================
        # LEFT
        # ====================================================

        left = QVBoxLayout()

        left.setSpacing(
            13
        )

        # ----------------------------------------------------
        # MARKET
        # ----------------------------------------------------

        market_card = Card()

        market_layout = QHBoxLayout(
            market_card
        )

        market_layout.setContentsMargins(
            17,
            13,
            17,
            13
        )

        market_info = QVBoxLayout()

        market_title = QLabel(
            "BTC / USDT"
        )

        market_title.setStyleSheet(
            f"""
            font-size: 15px;
            font-weight: 700;
            color: {TEXT};
            """
        )

        market_subtitle = QLabel(
            "Bitcoin • análisis visual"
        )

        market_subtitle.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 10px;
            """
        )

        market_info.addWidget(
            market_title
        )

        market_info.addWidget(
            market_subtitle
        )

        market_layout.addLayout(
            market_info
        )

        market_layout.addStretch()

        self.market_status = QLabel(
            "AGUARDANDO CAPTURA"
        )

        self.market_status.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 10px;
            font-weight: 600;
            """
        )

        market_layout.addWidget(
            self.market_status
        )

        left.addWidget(
            market_card
        )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        chart_card = Card()

        chart_layout = QVBoxLayout(
            chart_card
        )

        chart_layout.setContentsMargins(
            14,
            14,
            14,
            14
        )

        chart_header = QHBoxLayout()

        chart_title = QLabel(
            "CAPTURA DEL GRÁFICO"
        )

        chart_title.setStyleSheet(
            f"""
            color: {TEXT};
            font-size: 11px;
            font-weight: 700;
            """
        )

        chart_header.addWidget(
            chart_title
        )

        chart_header.addStretch()

        self.capture_info = QLabel(
            "Sin captura"
        )

        self.capture_info.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 9px;
            """
        )

        chart_header.addWidget(
            self.capture_info
        )

        chart_layout.addLayout(
            chart_header
        )

        self.preview = QLabel()

        self.preview.setMinimumHeight(
            180
        )

        self.preview.setAlignment(
            Qt.AlignCenter
        )

        self.preview.setStyleSheet(
            f"""
            background-color: #070a10;
            border: 1px solid {BORDER};
            border-radius: 11px;
            color: {MUTED};
            """
        )

        self.preview.setText(
            "Selecciona la zona del gráfico"
        )

        chart_layout.addWidget(
            self.preview
        )

        left.addWidget(
            chart_card,
            1
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = QHBoxLayout()

        metrics.setSpacing(
            9
        )

        self.score_card, self.score_value = (
            self.create_metric(
                "SCORE",
                "0"
            )
        )

        self.movement_card, self.movement_value = (
            self.create_metric(
                "MOVIMIENTO",
                "0"
            )
        )

        self.change_card, self.change_value = (
            self.create_metric(
                "CANDLE CHANGE",
                "0%"
            )
        )

        metrics.addWidget(
            self.score_card
        )

        metrics.addWidget(
            self.movement_card
        )

        metrics.addWidget(
            self.change_card
        )

        left.addLayout(
            metrics
        )

        # ====================================================
        # RIGHT
        # ====================================================

        right = QVBoxLayout()

        right.setSpacing(
            13
        )

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        signal_card = Card()

        signal_layout = QVBoxLayout(
            signal_card
        )

        signal_layout.setContentsMargins(
            18,
            18,
            18,
            18
        )

        signal_title = QLabel(
            "LECTURA ACTUAL"
        )

        signal_title.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 10px;
            font-weight: 700;
            """
        )

        signal_layout.addWidget(
            signal_title
        )

        self.signal_label = QLabel(
            "NO OPERAR"
        )

        self.signal_label.setAlignment(
            Qt.AlignCenter
        )

        self.signal_label.setMinimumHeight(
            80
        )

        self.signal_label.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 29px;
            font-weight: 800;
            """
        )

        signal_layout.addWidget(
            self.signal_label
        )

        self.experimental_label = QLabel(
            "SEÑAL EXPERIMENTAL"
        )

        self.experimental_label.setAlignment(
            Qt.AlignCenter
        )

        self.experimental_label.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 9px;
            font-weight: 600;
            """
        )

        signal_layout.addWidget(
            self.experimental_label
        )

        confirmation_title = QLabel(
            "CONFIRMACIÓN"
        )

        confirmation_title.setAlignment(
            Qt.AlignCenter
        )

        confirmation_title.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 9px;
            font-weight: 700;
            margin-top: 13px;
            """
        )

        signal_layout.addWidget(
            confirmation_title
        )

        self.confirmation_label = QLabel(
            "0 / 3"
        )

        self.confirmation_label.setAlignment(
            Qt.AlignCenter
        )

        self.confirmation_label.setStyleSheet(
            f"""
            color: {YELLOW};
            font-size: 23px;
            font-weight: 800;
            """
        )

        signal_layout.addWidget(
            self.confirmation_label
        )

        dots = QHBoxLayout()

        dots.setAlignment(
            Qt.AlignCenter
        )

        self.confirmation_dots = []

        for i in range(3):

            dot = QLabel(
                "●"
            )

            dot.setAlignment(
                Qt.AlignCenter
            )

            dot.setStyleSheet(
                f"""
                color: {BORDER};
                font-size: 20px;
                """
            )

            self.confirmation_dots.append(
                dot
            )

            dots.addWidget(
                dot
            )

            if i < 2:

                separator = QLabel(
                    "—"
                )

                separator.setStyleSheet(
                    f"color: {BORDER};"
                )

                dots.addWidget(
                    separator
                )

        signal_layout.addLayout(
            dots
        )

        self.waiting_label = QLabel(
            "Esperando confirmación..."
        )

        self.waiting_label.setAlignment(
            Qt.AlignCenter
        )

        self.waiting_label.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 10px;
            """
        )

        signal_layout.addWidget(
            self.waiting_label
        )

        right.addWidget(
            signal_card
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history_card = Card()

        history_layout = QVBoxLayout(
            history_card
        )

        history_layout.setContentsMargins(
            15,
            15,
            15,
            15
        )

        history_title = QLabel(
            "HISTORIAL"
        )

        history_title.setStyleSheet(
            f"""
            color: {TEXT};
            font-size: 11px;
            font-weight: 700;
            """
        )

        history_layout.addWidget(
            history_title
        )

        self.history_scroll = QScrollArea()

        self.history_scroll.setWidgetResizable(
            True
        )

        self.history_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff
        )

        self.history_scroll.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background: transparent;
            }
            """
        )

        self.history_container = QWidget()

        self.history_layout = QVBoxLayout(
            self.history_container
        )

        self.history_layout.setContentsMargins(
            0,
            7,
            0,
            0
        )

        self.history_layout.setSpacing(
            5
        )

        self.history_scroll.setWidget(
            self.history_container
        )

        history_layout.addWidget(
            self.history_scroll
        )

        right.addWidget(
            history_card,
            1
        )

        # ----------------------------------------------------
        # CONTROLS
        # ----------------------------------------------------

        controls = QHBoxLayout()

        self.select_button = QPushButton(
            "⚙  GRÁFICO"
        )

        self.select_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {CARD_2};
                color: {TEXT};
                border: 1px solid {BORDER};
            }}

            QPushButton:hover {{
                background-color: #202735;
            }}
            """
        )

        self.select_button.clicked.connect(
            self.select_chart
        )

        self.start_button = QPushButton(
            "▶  INICIAR"
        )

        self.start_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {GREEN};
                color: #061008;
            }}

            QPushButton:hover {{
                background-color: #4ade70;
            }}
            """
        )

        self.start_button.clicked.connect(
            self.start_monitor
        )

        self.stop_button = QPushButton(
            "■  DETENER"
        )

        self.stop_button.setEnabled(
            False
        )

        self.stop_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {RED_DARK};
                color: {RED};
                border: 1px solid #652321;
            }}

            QPushButton:hover {{
                background-color: #56201e;
            }}
            """
        )

        self.stop_button.clicked.connect(
            self.stop_monitor
        )

        controls.addWidget(
            self.select_button
        )

        controls.addWidget(
            self.start_button
        )

        controls.addWidget(
            self.stop_button
        )

        right.addLayout(
            controls
        )

        # ----------------------------------------------------
        # JUNTAR
        # ----------------------------------------------------

        content.addLayout(
            left,
            2
        )

        content.addLayout(
            right,
            1
        )

        main.addLayout(
            content,
            1
        )

        # ====================================================
        # FOOTER
        # ====================================================

        footer = QHBoxLayout()

        self.status_label = QLabel(
            "● Sistema listo"
        )

        self.status_label.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 9px;
            """
        )

        footer.addWidget(
            self.status_label
        )

        footer.addStretch()

        self.last_analysis_label = QLabel(
            "Último análisis: —"
        )

        self.last_analysis_label.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 9px;
            """
        )

        footer.addWidget(
            self.last_analysis_label
        )

        main.addLayout(
            footer
        )

    # ========================================================
    # MÉTRICA
    # ========================================================

    def create_metric(
        self,
        title,
        value
    ):

        card = Card()

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            14,
            10,
            14,
            10
        )

        title_label = QLabel(
            title
        )

        title_label.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 8px;
            font-weight: 700;
            """
        )

        value_label = QLabel(
            value
        )

        value_label.setStyleSheet(
            f"""
            color: {TEXT};
            font-size: 20px;
            font-weight: 800;
            """
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            value_label
        )

        return (
            card,
            value_label
        )

    # ========================================================
    # MAXIMIZAR
    # ========================================================

    def toggle_maximize(
        self
    ):

        if self.isMaximized():

            self.showNormal()

        else:

            self.showMaximized()

    # ========================================================
    # MOUSE PRESS
    # ========================================================

    def mousePressEvent(
        self,
        event
    ):

        if event.button() == Qt.LeftButton:

            pos = event.position().toPoint()

            edges = self.get_resize_edges(
                pos
            )

            if edges:

                self.resizing = True

                self.resize_edges = edges

                self.resize_start_geometry = (
                    self.geometry()
                )

                self.resize_start_position = (
                    event.globalPosition().toPoint()
                )

                event.accept()

                return

            if self.is_in_title_bar(
                pos
            ):

                self.dragging = True

                self.drag_position = (
                    event.globalPosition().toPoint()
                    - self.frameGeometry().topLeft()
                )

                event.accept()

                return

        super().mousePressEvent(
            event
        )

    # ========================================================
    # MOUSE MOVE
    # ========================================================

    def mouseMoveEvent(
        self,
        event
    ):

        pos = event.position().toPoint()

        if self.resizing:

            self.perform_resize(
                event.globalPosition().toPoint()
            )

            event.accept()

            return

        if self.dragging:

            self.move(
                event.globalPosition().toPoint()
                - self.drag_position
            )

            event.accept()

            return

        edges = self.get_resize_edges(
            pos
        )

        if edges == 0:

            self.setCursor(
                Qt.ArrowCursor
            )

        elif edges in (
            1,
            2
        ):

            self.setCursor(
                Qt.SizeHorCursor
            )

        elif edges in (
            4,
            8
        ):

            self.setCursor(
                Qt.SizeVerCursor
            )

        elif edges in (
            5,
            10
        ):

            self.setCursor(
                Qt.SizeFDiagCursor
            )

        elif edges in (
            6,
            9
        ):

            self.setCursor(
                Qt.SizeBDiagCursor
            )

        super().mouseMoveEvent(
            event
        )

    # ========================================================
    # MOUSE RELEASE
    # ========================================================

    def mouseReleaseEvent(
        self,
        event
    ):

        self.dragging = False

        self.resizing = False

        self.resize_edges = 0

        self.setCursor(
            Qt.ArrowCursor
        )

        super().mouseReleaseEvent(
            event
        )

    # ========================================================
    # BORDES
    # ========================================================

    def get_resize_edges(
        self,
        pos
    ):

        margin = (
            self.RESIZE_MARGIN
        )

        rect = self.rect()

        edges = 0

        if pos.x() <= margin:

            edges |= 1

        elif (
            pos.x()
            >= rect.width()
            - margin
        ):

            edges |= 2

        if pos.y() <= margin:

            edges |= 4

        elif (
            pos.y()
            >= rect.height()
            - margin
        ):

            edges |= 8

        return edges

    # ========================================================
    # REDIMENSIONAR
    # ========================================================

    def perform_resize(
        self,
        global_pos
    ):

        delta = (
            global_pos
            - self.resize_start_position
        )

        geometry = QRect(
            self.resize_start_geometry
        )

        minimum_width = (
            self.minimumWidth()
        )

        minimum_height = (
            self.minimumHeight()
        )

        if self.resize_edges & 1:

            new_left = (
                geometry.left()
                + delta.x()
            )

            if (
                geometry.right()
                - new_left
                + 1
                >= minimum_width
            ):

                geometry.setLeft(
                    new_left
                )

        if self.resize_edges & 2:

            new_right = (
                geometry.right()
                + delta.x()
            )

            if (
                new_right
                - geometry.left()
                + 1
                >= minimum_width
            ):

                geometry.setRight(
                    new_right
                )

        if self.resize_edges & 4:

            new_top = (
                geometry.top()
                + delta.y()
            )

            if (
                geometry.bottom()
                - new_top
                + 1
                >= minimum_height
            ):

                geometry.setTop(
                    new_top
                )

        if self.resize_edges & 8:

            new_bottom = (
                geometry.bottom()
                + delta.y()
            )

            if (
                new_bottom
                - geometry.top()
                + 1
                >= minimum_height
            ):

                geometry.setBottom(
                    new_bottom
                )

        self.setGeometry(
            geometry
        )

    # ========================================================
    # TITLE BAR
    # ========================================================

    def is_in_title_bar(
        self,
        pos
    ):

        if not hasattr(
            self,
            "title_bar"
        ):

            return False

        return self.title_bar.geometry().contains(
            pos
        )

    # ========================================================
    # SELECCIONAR GRÁFICO
    # ========================================================

    def select_chart(
        self
    ):

        self.status_label.setText(
            "● Selecciona el área del gráfico..."
        )

        self.status_label.setStyleSheet(
            f"""
            color: {BLUE};
            font-size: 9px;
            """
        )

        self.market_status.setText(
            "SELECCIONANDO GRÁFICO..."
        )

        self.selector = ScreenSelector(
            self.set_chart_rect
        )

        self.selector.show()

    # ========================================================
    # GUARDAR ÁREA
    # ========================================================

    def set_chart_rect(
        self,
        rect
    ):

        self.chart_rect = rect

        self.status_label.setText(
            "● Área del gráfico configurada"
        )

        self.status_label.setStyleSheet(
            f"""
            color: {GREEN};
            font-size: 9px;
            """
        )

        self.market_status.setText(
            "GRÁFICO: "
            f"{rect.width()} × "
            f"{rect.height()}"
        )

    # ========================================================
    # INICIAR
    # ========================================================

    def start_monitor(
        self
    ):

        if self.chart_rect is None:

            self.status_label.setText(
                "● Primero selecciona el gráfico"
            )

            self.status_label.setStyleSheet(
                f"""
                color: {YELLOW};
                font-size: 9px;
                """
            )

            return

        self.running = True

        self.previous_mask = None

        self.confirmation_history.clear()

        self.timer.start(
            INTERVAL_SECONDS * 1000
        )

        self.start_button.setEnabled(
            False
        )

        self.stop_button.setEnabled(
            True
        )

        self.live_label.setText(
            "● LIVE"
        )

        self.live_label.setStyleSheet(
            f"""
            color: {GREEN};
            background-color: {GREEN_DARK};
            border: 1px solid #23613a;
            border-radius: 12px;
            padding: 5px 11px;
            font-size: 9px;
            font-weight: 700;
            """
        )

        self.status_label.setText(
            "● Monitor activo"
        )

        self.status_label.setStyleSheet(
            f"""
            color: {GREEN};
            font-size: 9px;
            """
        )

        self.process_capture()

    # ========================================================
    # DETENER
    # ========================================================

    def stop_monitor(
        self
    ):

        self.running = False

        self.timer.stop()

        self.start_button.setEnabled(
            True
        )

        self.stop_button.setEnabled(
            False
        )

        self.live_label.setText(
            "● OFFLINE"
        )

        self.live_label.setStyleSheet(
            f"""
            color: {MUTED};
            background-color: {CARD_2};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 5px 11px;
            font-size: 9px;
            font-weight: 700;
            """
        )

        self.status_label.setText(
            "● Monitor detenido"
        )

        self.status_label.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 9px;
            """
        )

    # ========================================================
    # CAPTURAR
    # ========================================================

    def capture_chart(
        self
    ):

        if self.chart_rect is None:

            return None

        x = self.chart_rect.x()

        y = self.chart_rect.y()

        width = (
            self.chart_rect.width()
        )

        height = (
            self.chart_rect.height()
        )

        with mss.mss() as sct:

            monitor = {
                "left": x,
                "top": y,
                "width": width,
                "height": height
            }

            screenshot = sct.grab(
                monitor
            )

            img = Image.frombytes(
                "RGB",
                screenshot.size,
                screenshot.rgb
            )

        img.save(
            SCREENSHOT_PATH
        )

        return img

    # ========================================================
    # PREVIEW
    # ========================================================

    def update_preview(
        self,
        img
    ):

        if img is None:

            return

        img = img.convert(
            "RGB"
        )

        data = img.tobytes(
            "raw",
            "RGB"
        )

        qimage = QImage(
            data,
            img.width,
            img.height,
            img.width * 3,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(
            qimage
        )

        pixmap = pixmap.scaled(
            self.preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.preview.setPixmap(
            pixmap
        )

        self.capture_info.setText(
            f"{img.width} × {img.height}"
        )

    # ========================================================
    # PROCESAR CAPTURA
    # ========================================================

    def process_capture(
        self
    ):

        if not self.running:

            return

        try:

            # ------------------------------------------------
            # 1. CAPTURA
            # ------------------------------------------------

            img = self.capture_chart()

            if img is None:

                return

            self.update_preview(
                img
            )

            frame = np.array(
                img
            )

            # ------------------------------------------------
            # 2. DETECTAR COLORES
            #
            # IMPORTANTE:
            # detect_colors devuelve:
            # green_mask, red_mask
            # ------------------------------------------------

            green_mask, red_mask = (
                detect_colors(
                    frame
                )
            )

            # ------------------------------------------------
            # 3. LIMPIAR CADA MÁSCARA
            # ------------------------------------------------

            green_mask = clean_mask(
                green_mask
            )

            red_mask = clean_mask(
                red_mask
            )

            # ------------------------------------------------
            # 4. MÁSCARA TOTAL
            # ------------------------------------------------

            current_mask = (
                green_mask
                | red_mask
            )

            current_mask = (
                current_mask
                .astype(
                    np.uint8
                )
            )

            # ------------------------------------------------
            # 5. CAMBIO DEL GRÁFICO
            # ------------------------------------------------

            candle_change = 100.0

            if (
                self.previous_mask is not None
                and
                current_mask.shape
                ==
                self.previous_mask.shape
            ):

                old = (
                    self.previous_mask
                    > 0
                )

                new = (
                    current_mask
                    > 0
                )

                changed = (
                    np.logical_xor(
                        old,
                        new
                    )
                )

                total = (
                    np.count_nonzero(
                        old | new
                    )
                )

                if total > 0:

                    candle_change = (
                        np.count_nonzero(
                            changed
                        )
                        / total
                    ) * 100

            self.previous_mask = (
                current_mask.copy()
            )

            self.last_candle_change = (
                candle_change
            )

            self.change_value.setText(
                f"{candle_change:.1f}%"
            )

            # ------------------------------------------------
            # 6. SI NO CAMBIÓ NADA
            # ------------------------------------------------

            if (
                candle_change
                <
                CHART_CHANGE_THRESHOLD
                and
                self.analysis_count > 0
            ):

                self.market_status.setText(
                    "SIN CAMBIO RELEVANTE"
                )

                return

            # ------------------------------------------------
            # 7. BUSCAR CANDIDATOS
            #
            # IMPORTANTE:
            # ahora pasamos green y red
            # por separado.
            # ------------------------------------------------

            candidates = (
                find_candle_candidates(
                    green_mask,
                    red_mask
                )
            )

            candidates = (
                merge_close_candidates(
                    candidates
                )
            )

            candidates = (
                filter_candidates(
                    candidates
                )
            )

            # ------------------------------------------------
            # 8. CONSTRUIR VELAS
            #
            # get_candle_data necesita:
            # green
            # red
            # x
            # ------------------------------------------------

            candles = []

            for candidate in candidates:

                candle = get_candle_data(
                    green_mask,
                    red_mask,
                    candidate["x"]
                )

                if candle is not None:

                    candles.append(
                        candle
                    )

            # ------------------------------------------------
            # 9. COMPROBAR VELAS
            # ------------------------------------------------

            if len(candles) < 3:

                self.market_status.setText(
                    f"Solo {len(candles)} velas"
                )

                return

            self.last_candles = (
                candles
            )

            # La última vela puede estar en formación.
            # No la usamos para el análisis.
            if len(candles) >= 4:

                closed_candles = (
                    candles[:-1]
                )

            else:

                closed_candles = (
                    candles
                )

            # ------------------------------------------------
            # 10. ANALIZAR ESTRUCTURA
            # ------------------------------------------------

            result = analyze_structure(
                closed_candles
            )

            signal = "NO OPERAR"

            score = 0

            movement = 0

            reasons = []

            green_count = 0

            red_count = 0

            if isinstance(
                result,
                dict
            ):

                signal = result.get(
                    "signal",
                    "NO OPERAR"
                )

                score = result.get(
                    "score",
                    0
                )

                movement = result.get(
                    "movement",
                    0
                )

                reasons = result.get(
                    "reasons",
                    []
                )

                green_count = result.get(
                    "green_count",
                    0
                )

                red_count = result.get(
                    "red_count",
                    0
                )

            # ------------------------------------------------
            # 11. NORMALIZAR SEÑAL
            # ------------------------------------------------

            signal_text = str(
                signal
            ).upper()

            if "SUBIDA" in signal_text:

                signal = "SUBIDA"

            elif "BAJADA" in signal_text:

                signal = "BAJADA"

            else:

                signal = "NO OPERAR"

            try:

                score = float(
                    score
                )

            except Exception:

                score = 0

            try:

                movement = float(
                    movement
                )

            except Exception:

                movement = 0

            self.last_signal = (
                signal
            )

            self.last_score = (
                score
            )

            self.last_movement = (
                movement
            )

            self.analysis_count += 1

            now = datetime.now()

            self.last_analysis_time = (
                now
            )

            self.last_analysis_label.setText(
                "Último análisis: "
                + now.strftime(
                    "%H:%M:%S"
                )
            )

            # ------------------------------------------------
            # 12. MÉTRICAS
            # ------------------------------------------------

            if score > 0:

                self.score_value.setText(
                    f"+{score:g}"
                )

            else:

                self.score_value.setText(
                    f"{score:g}"
                )

            if movement > 0:

                self.movement_value.setText(
                    f"+{movement:g}"
                )

            else:

                self.movement_value.setText(
                    f"{movement:g}"
                )

            # ------------------------------------------------
            # 13. CONFIRMACIÓN
            # ------------------------------------------------

            if signal in (
                "SUBIDA",
                "BAJADA"
            ):

                self.confirmation_history.append(
                    signal
                )

            else:

                self.confirmation_history.clear()

            confirmation = len(
                self.confirmation_history
            )

            confirmed = (
                confirmation
                >=
                CONFIRMATION_COUNT
                and
                len(
                    set(
                        self.confirmation_history
                    )
                )
                == 1
            )

            # ------------------------------------------------
            # 14. MOSTRAR
            # ------------------------------------------------

            if confirmed:

                confirmed_signal = (
                    self.confirmation_history[-1]
                )

                self.show_signal(
                    confirmed_signal,
                    score,
                    movement,
                    confirmation,
                    True
                )

                self.add_history(
                    confirmed_signal,
                    score,
                    True
                )

            else:

                self.show_signal(
                    signal,
                    score,
                    movement,
                    confirmation,
                    False
                )

                if signal != "NO OPERAR":

                    self.add_history(
                        signal,
                        score,
                        False
                    )

            # ------------------------------------------------
            # 15. CSV
            # ------------------------------------------------

            self.save_csv(
                signal,
                score,
                movement,
                confirmation,
                confirmed
            )

            # ------------------------------------------------
            # 16. ESTADO
            # ------------------------------------------------

            self.market_status.setText(
                f"{len(candles)} velas | "
                f"Verdes {green_count} | "
                f"Rojas {red_count}"
            )

        except Exception as error:

            self.status_label.setText(
                f"● Error: {str(error)[:90]}"
            )

            self.status_label.setStyleSheet(
                f"""
                color: {RED};
                font-size: 9px;
                """
            )

    # ========================================================
    # MOSTRAR SEÑAL
    # ========================================================

    def show_signal(
        self,
        signal,
        score,
        movement,
        confirmation,
        confirmed
    ):

        self.confirmation_label.setText(
            f"{confirmation} / 3"
        )

        if signal == "NO OPERAR":

            self.signal_label.setText(
                "NO OPERAR"
            )

            self.signal_label.setStyleSheet(
                f"""
                color: {MUTED};
                font-size: 29px;
                font-weight: 800;
                """
            )

            self.waiting_label.setText(
                "Esperando una dirección clara..."
            )

            self.confirmation_label.setStyleSheet(
                f"""
                color: {YELLOW};
                font-size: 23px;
                font-weight: 800;
                """
            )

        elif signal == "SUBIDA":

            self.signal_label.setText(
                "SUBIDA"
            )

            self.signal_label.setStyleSheet(
                f"""
                color: {GREEN};
                font-size: 29px;
                font-weight: 800;
                """
            )

            if confirmed:

                self.waiting_label.setText(
                    "✓ SUBIDA CONFIRMADA"
                )

                self.waiting_label.setStyleSheet(
                    f"""
                    color: {GREEN};
                    font-size: 10px;
                    font-weight: 800;
                    """
                )

            else:

                self.waiting_label.setText(
                    "Esperando confirmación..."
                )

            self.confirmation_label.setStyleSheet(
                f"""
                color: {GREEN};
                font-size: 23px;
                font-weight: 800;
                """
            )

        elif signal == "BAJADA":

            self.signal_label.setText(
                "BAJADA"
            )

            self.signal_label.setStyleSheet(
                f"""
                color: {RED};
                font-size: 29px;
                font-weight: 800;
                """
            )

            if confirmed:

                self.waiting_label.setText(
                    "✓ BAJADA CONFIRMADA"
                )

                self.waiting_label.setStyleSheet(
                    f"""
                    color: {RED};
                    font-size: 10px;
                    font-weight: 800;
                    """
                )

            else:

                self.waiting_label.setText(
                    "Esperando confirmación..."
                )

            self.confirmation_label.setStyleSheet(
                f"""
                color: {RED};
                font-size: 23px;
                font-weight: 800;
                """
            )

        # ----------------------------------------------------
        # PUNTOS
        # ----------------------------------------------------

        for i, dot in enumerate(
            self.confirmation_dots
        ):

            if i < confirmation:

                if signal == "SUBIDA":

                    color = GREEN

                elif signal == "BAJADA":

                    color = RED

                else:

                    color = YELLOW

                dot.setStyleSheet(
                    f"""
                    color: {color};
                    font-size: 20px;
                    """
                )

            else:

                dot.setStyleSheet(
                    f"""
                    color: {BORDER};
                    font-size: 20px;
                    """
                )

    # ========================================================
    # HISTORIAL
    # ========================================================

    def add_history(
        self,
        signal,
        score,
        confirmed
    ):

        now = datetime.now().strftime(
            "%H:%M:%S"
        )

        item = QFrame()

        item.setStyleSheet(
            f"""
            QFrame {{
                background-color: {CARD_2};
                border-radius: 8px;
            }}
            """
        )

        layout = QHBoxLayout(
            item
        )

        layout.setContentsMargins(
            9,
            6,
            9,
            6
        )

        time_label = QLabel(
            now
        )

        time_label.setStyleSheet(
            f"""
            color: {MUTED};
            font-size: 8px;
            """
        )

        signal_label = QLabel(
            signal
        )

        if signal == "SUBIDA":

            color = GREEN

        elif signal == "BAJADA":

            color = RED

        else:

            color = MUTED

        signal_label.setStyleSheet(
            f"""
            color: {color};
            font-size: 9px;
            font-weight: 800;
            """
        )

        score_label = QLabel(
            f"Score {score:g}"
        )

        score_label.setStyleSheet(
            f"""
            color: {TEXT};
            font-size: 8px;
            """
        )

        layout.addWidget(
            time_label
        )

        layout.addWidget(
            signal_label
        )

        layout.addStretch()

        layout.addWidget(
            score_label
        )

        if confirmed:

            confirmed_label = QLabel(
                "CONFIRMADA"
            )

            confirmed_label.setStyleSheet(
                f"""
                color: {GREEN};
                font-size: 7px;
                font-weight: 800;
                """
            )

            layout.addWidget(
                confirmed_label
            )

        self.history_layout.insertWidget(
            0,
            item
        )

        self.history.append(
            {
                "time": now,
                "signal": signal,
                "score": score,
                "confirmed": confirmed
            }
        )

        while (
            self.history_layout.count()
            > 12
        ):

            item_to_remove = (
                self.history_layout.takeAt(
                    self.history_layout.count()
                    - 1
                )
            )

            if item_to_remove.widget():

                item_to_remove.widget().deleteLater()

    # ========================================================
    # GUARDAR CSV
    # ========================================================

    def save_csv(
        self,
        signal,
        score,
        movement,
        confirmation,
        confirmed
    ):

        exists = os.path.exists(
            CSV_PATH
        )

        with open(
            CSV_PATH,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file
            )

            if not exists:

                writer.writerow(
                    [
                        "fecha",
                        "hora",
                        "señal",
                        "score",
                        "movimiento",
                        "confirmacion",
                        "confirmada"
                    ]
                )

            now = datetime.now()

            writer.writerow(
                [
                    now.strftime(
                        "%Y-%m-%d"
                    ),
                    now.strftime(
                        "%H:%M:%S"
                    ),
                    signal,
                    score,
                    movement,
                    confirmation,
                    confirmed
                ]
            )

    # ========================================================
    # CERRAR
    # ========================================================

    def closeEvent(
        self,
        event
    ):

        self.timer.stop()

        event.accept()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    app.setApplicationName(
        "IQ AI"
    )

    window = BlitzMonitor()

    window.show()

    sys.exit(
        app.exec()
    )