import csv
import os
import sys
from collections import deque
from datetime import datetime

import mss
import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QRect, QPoint, QSize, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QRubberBand,
    QComboBox,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFrame,
    QScrollArea,
)

from chart_reader import (
    detect_colors,
    clean_mask,
    find_candle_candidates,
    merge_close_candidates,
    filter_candidates,
)


# ============================================================
# CONFIG
# ============================================================

INTERVAL_SECONDS = 2
CHART_CHANGE_THRESHOLD = 0.8
CONFIRMATION_COUNT = 3
CANDLE_INTERVAL_SECONDS = 60
DATA_DIR = "data"
SCREENSHOT_PATH = os.path.join(DATA_DIR, "grafico.png")
CSV_PATH = os.path.join(DATA_DIR, "blitz_signals.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# COLORS
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
# SIGNAL LOGIC - copied from the user's blitz_signal.py
# ============================================================

def get_candle_data(green, red, x):
    height, width = green.shape

    left = max(0, x - 4)
    right = min(width - 1, x + 4)

    green_region = green[:, left:right + 1]
    red_region = red[:, left:right + 1]
    combined = green_region | red_region

    row_strength = combined.sum(axis=1)
    rows = np.where(row_strength > 0)[0]

    if len(rows) < 5:
        return None

    high = int(rows.min())
    low = int(rows.max())
    total_height = low - high + 1

    green_pixels = int(green_region.sum())
    red_pixels = int(red_region.sum())

    color = "VERDE" if green_pixels >= red_pixels else "ROJA"

    max_width = int(row_strength.max())
    if max_width < 2:
        return None

    body_threshold = max(2, int(max_width * 0.55))
    body_rows = np.where(row_strength >= body_threshold)[0]

    if len(body_rows) == 0:
        return None

    groups = []
    start = None
    previous = None

    for y in body_rows:
        y = int(y)

        if start is None:
            start = y
            previous = y
            continue

        if y == previous + 1:
            previous = y
        else:
            groups.append((start, previous))
            start = y
            previous = y

    if start is not None:
        groups.append((start, previous))

    best_group = None
    best_score = -1

    for group_start, group_end in groups:
        score = int(row_strength[group_start:group_end + 1].sum())
        if score > best_score:
            best_score = score
            best_group = (group_start, group_end)

    if best_group is None:
        return None

    body_top, body_bottom = best_group
    body_height = body_bottom - body_top + 1
    upper_wick = body_top - high
    lower_wick = low - body_bottom

    if color == "VERDE":
        open_y = body_bottom
        close_y = body_top
    else:
        open_y = body_top
        close_y = body_bottom

    return {
        "x": x,
        "color": color,
        "high": high,
        "low": low,
        "open": open_y,
        "close": close_y,
        "body": body_height,
        "height": total_height,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
    }


def analyze_structure(candles):
    if len(candles) < 10:
        return {
            "signal": "NO OPERAR",
            "score": 0,
            "reasons": ["No hay suficientes velas."],
            "green_count": 0,
            "red_count": 0,
            "movement": 0,
        }

    recent = candles[-12:]
    score = 0
    reasons = []

    green_count = sum(c["color"] == "VERDE" for c in recent)
    red_count = sum(c["color"] == "ROJA" for c in recent)

    if green_count >= 8:
        score += 2
        reasons.append("Predominio de velas verdes.")
    elif red_count >= 8:
        score -= 2
        reasons.append("Predominio de velas rojas.")

    strong_green = 0
    strong_red = 0

    for candle in recent:
        if candle["height"] <= 0:
            continue

        body_ratio = candle["body"] / candle["height"]

        if body_ratio >= 0.55:
            if candle["color"] == "VERDE":
                strong_green += 1
            else:
                strong_red += 1

    if strong_green >= 5:
        score += 1
        reasons.append("Hay varias velas verdes con cuerpos fuertes.")

    if strong_red >= 5:
        score -= 1
        reasons.append("Hay varias velas rojas con cuerpos fuertes.")

    lower_rejections = 0
    upper_rejections = 0

    for candle in recent:
        body = max(candle["body"], 1)

        if candle["lower_wick"] > body * 0.8:
            lower_rejections += 1

        if candle["upper_wick"] > body * 0.8:
            upper_rejections += 1

    if lower_rejections >= 3:
        score += 1
        reasons.append("Se detectan varios rechazos de precios bajos.")

    if upper_rejections >= 3:
        score -= 1
        reasons.append("Se detectan varios rechazos de precios altos.")

    first = recent[0]
    last = recent[-1]
    movement = first["close"] - last["close"]

    if movement > 15:
        score += 2
        reasons.append("Movimiento visual ascendente.")
    elif movement < -15:
        score -= 2
        reasons.append("Movimiento visual descendente.")

    last_five = recent[-5:]
    green_recent = sum(c["color"] == "VERDE" for c in last_five)
    red_recent = sum(c["color"] == "ROJA" for c in last_five)

    if green_recent >= 4:
        score += 1
        reasons.append("Impulso alcista reciente.")
    elif red_recent >= 4:
        score -= 1
        reasons.append("Impulso bajista reciente.")

    if score >= 4:
        signal = "SUBIDA"
    elif score <= -4:
        signal = "BAJADA"
    else:
        signal = "NO OPERAR"

    return {
        "signal": signal,
        "score": score,
        "reasons": reasons,
        "green_count": green_count,
        "red_count": red_count,
        "movement": movement,
    }


# ============================================================
# UI HELPERS
# ============================================================

class Card(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setStyleSheet(
            f"""
            QFrame#Card {{
                background-color: {CARD};
                border: 1px solid {BORDER};
                border-radius: 16px;
            }}
            """
        )


class MacButton(QPushButton):
    def __init__(self, color, hover_color, parent=None):
        super().__init__(parent)
        self.setFixedSize(13, 13)
        self.setCursor(Qt.PointingHandCursor)
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
# SCREEN SELECTOR
# ============================================================

class ScreenSelector(QWidget):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.start = QPoint()
        self.selecting = False

        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)

        with mss.mss() as sct:
            monitor = sct.monitors[1]

        self.monitor_left = monitor["left"]
        self.monitor_top = monitor["top"]

        self.setGeometry(
            monitor["left"],
            monitor["top"],
            monitor["width"],
            monitor["height"],
        )

        self.rubber_band = QRubberBand(
            QRubberBand.Rectangle,
            self,
        )

        self.rubber_band.setStyleSheet(
            """
            QRubberBand {
                border: 2px solid #4da3ff;
                background: rgba(77, 163, 255, 35);
            }
            """
        )

        self.instruction = QLabel(
            "SELECCIONA EL ÁREA DEL GRÁFICO",
            self,
        )
        self.instruction.setAlignment(Qt.AlignCenter)
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

        self.sub_instruction = QLabel(
            "Mantén presionado el botón izquierdo y arrastra",
            self,
        )
        self.sub_instruction.setAlignment(Qt.AlignCenter)
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

        self.size_label = QLabel("", self)
        self.size_label.setAlignment(Qt.AlignCenter)
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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(5, 8, 14, 175))
        painter.end()

    def resizeEvent(self, event):
        self.instruction.move(
            (self.width() - self.instruction.width()) // 2,
            35,
        )
        self.sub_instruction.move(
            (self.width() - self.sub_instruction.width()) // 2,
            83,
        )
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        self.selecting = True
        self.start = event.pos()
        self.rubber_band.setGeometry(QRect(self.start, QSize(0, 0)))
        self.rubber_band.show()
        self.size_label.show()

    def mouseMoveEvent(self, event):
        if not self.selecting:
            return

        rect = QRect(self.start, event.pos()).normalized()
        self.rubber_band.setGeometry(rect)
        self.update_size_label(rect)

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or not self.selecting:
            return

        self.selecting = False
        rect = QRect(self.start, event.pos()).normalized()

        if rect.width() >= 150 and rect.height() >= 120:
            final_rect = QRect(
                self.monitor_left + rect.x(),
                self.monitor_top + rect.y(),
                rect.width(),
                rect.height(),
            )
            self.callback(final_rect)

        self.close()

    def update_size_label(self, rect):
        self.size_label.setText(
            f"{rect.width()} × {rect.height()} px"
        )
        self.size_label.adjustSize()

        x = rect.right() + 10
        y = rect.bottom() + 10

        if x + self.size_label.width() > self.width():
            x = rect.left() - self.size_label.width() - 10

        if y + self.size_label.height() > self.height():
            y = rect.top() - self.size_label.height() - 10

        self.size_label.move(x, y)


# ============================================================
# MAIN WINDOW
# ============================================================

class BlitzMonitor(QWidget):
    RESIZE_MARGIN = 10

    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("IQ AI — BTC BLITZ")
        self.setMinimumSize(700, 500)
        self.resize(1250, 800)

        self.dragging = False
        self.drag_position = QPoint()
        self.resizing = False
        self.resize_edges = 0
        self.resize_start_geometry = QRect()
        self.resize_start_position = QPoint()

        self.running = False
        self.chart_rect = None
        self.selector = None
        self.previous_mask = None
        self.confirmation_history = deque(maxlen=CONFIRMATION_COUNT)
        self.last_signal = "NO OPERAR"
        self.last_score = 0
        self.last_movement = 0
        self.last_candle_change = 0
        self.history = []
        self.analysis_count = 0

        # Timing de entrada: usa el reloj local para ubicar la vela actual.
        self.confirmation_timestamp = None
        self.confirmed_direction = None
        self.previous_confirmed = False
        self.last_candle_slot = None
        self.decision_timestamp = None
        self.decision_direction = None

        self.clock_timer = QTimer(self)
        self.clock_timer.setInterval(1000)
        self.clock_timer.timeout.connect(self.update_timing_view)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_capture)

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
            QLabel {{ background: transparent; }}
            QPushButton {{
                border: none;
                border-radius: 9px;
                padding: 10px 16px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background-color: #202735; }}
            QPushButton:pressed {{ background-color: #2a3241; }}
            QScrollBar:vertical {{
                background: transparent;
                width: 7px;
            }}
            QScrollBar::handle:vertical {{
                background: #303746;
                border-radius: 3px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height: 0px; }}
            """
        )

        self.build_ui()

    def build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)

        self.main_window = QWidget()
        self.main_window.setObjectName("MainWindow")
        outer.addWidget(self.main_window)

        main = QVBoxLayout(self.main_window)
        main.setContentsMargins(18, 12, 18, 14)
        main.setSpacing(13)

        # ---------------- Top bar ----------------
        top_bar = QWidget()
        top_bar.setFixedHeight(38)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(5, 0, 3, 0)
        top_layout.setSpacing(7)

        self.close_button = MacButton("#ff5f57", "#ff7b75")
        self.minimize_button = MacButton("#ffbd2e", "#ffd36a")
        self.maximize_button = MacButton("#28c840", "#55d968")

        self.close_button.clicked.connect(self.close)
        self.minimize_button.clicked.connect(self.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximize)

        top_layout.addWidget(self.close_button)
        top_layout.addWidget(self.minimize_button)
        top_layout.addWidget(self.maximize_button)
        top_layout.addSpacing(10)

        title = QLabel("IQ AI")
        title.setStyleSheet(
            f"color: {TEXT}; font-size: 13px; font-weight: 700;"
        )
        top_layout.addWidget(title)

        version = QLabel("BTC BLITZ")
        version.setStyleSheet(
            f"color: {MUTED}; font-size: 10px; font-weight: 600;"
        )
        top_layout.addWidget(version)
        top_layout.addStretch()

        self.live_label = QLabel("● OFFLINE")
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
        top_layout.addWidget(self.live_label)

        main.addWidget(top_bar)
        self.title_bar = top_bar

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {BORDER};")
        main.addWidget(line)

        # ---------------- Main content ----------------
        content = QHBoxLayout()
        content.setSpacing(13)

        left = QVBoxLayout()
        left.setSpacing(13)

        market_card = Card()
        market_layout = QHBoxLayout(market_card)
        market_layout.setContentsMargins(17, 13, 17, 13)

        market_info = QVBoxLayout()
        market_title = QLabel("BTC / USDT")
        market_title.setStyleSheet(
            f"font-size: 15px; font-weight: 700; color: {TEXT};"
        )
        market_subtitle = QLabel("Bitcoin • análisis visual")
        market_subtitle.setStyleSheet(
            f"color: {MUTED}; font-size: 10px;"
        )
        market_info.addWidget(market_title)
        market_info.addWidget(market_subtitle)
        market_layout.addLayout(market_info)
        market_layout.addStretch()

        self.market_status = QLabel("AGUARDANDO CAPTURA")
        self.market_status.setStyleSheet(
            f"color: {MUTED}; font-size: 10px; font-weight: 600;"
        )
        market_layout.addWidget(self.market_status)
        left.addWidget(market_card)

        chart_card = Card()
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(14, 14, 14, 14)

        chart_header = QHBoxLayout()
        chart_title = QLabel("CAPTURA DEL GRÁFICO")
        chart_title.setStyleSheet(
            f"color: {TEXT}; font-size: 11px; font-weight: 700;"
        )
        chart_header.addWidget(chart_title)
        chart_header.addStretch()

        self.capture_info = QLabel("Sin captura")
        self.capture_info.setStyleSheet(
            f"color: {MUTED}; font-size: 9px;"
        )
        chart_header.addWidget(self.capture_info)
        chart_layout.addLayout(chart_header)

        self.preview = QLabel("Selecciona la zona del gráfico")
        self.preview.setMinimumHeight(180)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(
            f"""
            background-color: #070a10;
            border: 1px solid {BORDER};
            border-radius: 11px;
            color: {MUTED};
            """
        )
        chart_layout.addWidget(self.preview)
        left.addWidget(chart_card, 1)

        metrics = QHBoxLayout()
        metrics.setSpacing(9)
        self.score_card, self.score_value = self.create_metric("SCORE", "0")
        self.movement_card, self.movement_value = self.create_metric("MOVIMIENTO", "0")
        self.change_card, self.change_value = self.create_metric("CANDLE CHANGE", "0%")
        metrics.addWidget(self.score_card)
        metrics.addWidget(self.movement_card)
        metrics.addWidget(self.change_card)
        left.addLayout(metrics)

        right = QVBoxLayout()
        right.setSpacing(13)

        signal_card = Card()
        signal_layout = QVBoxLayout(signal_card)
        signal_layout.setContentsMargins(18, 18, 18, 18)

        signal_title = QLabel("LECTURA DE LA PRÓXIMA VELA")
        signal_title.setStyleSheet(
            f"color: {MUTED}; font-size: 10px; font-weight: 700;"
        )
        signal_layout.addWidget(signal_title)

        self.signal_label = QLabel("NO OPERAR")
        self.signal_label.setAlignment(Qt.AlignCenter)
        self.signal_label.setMinimumHeight(80)
        self.signal_label.setStyleSheet(
            f"color: {MUTED}; font-size: 29px; font-weight: 800;"
        )
        signal_layout.addWidget(self.signal_label)

        experimental_label = QLabel("SEÑAL EXPERIMENTAL")
        experimental_label.setAlignment(Qt.AlignCenter)
        experimental_label.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-weight: 600;"
        )
        signal_layout.addWidget(experimental_label)

        confirmation_title = QLabel("DECISIÓN AL CIERRE")
        confirmation_title.setAlignment(Qt.AlignCenter)
        confirmation_title.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-weight: 700; margin-top: 13px;"
        )
        signal_layout.addWidget(confirmation_title)

        self.confirmation_label = QLabel("ESPERANDO")
        self.confirmation_label.setAlignment(Qt.AlignCenter)
        self.confirmation_label.setStyleSheet(
            f"color: {YELLOW}; font-size: 23px; font-weight: 800;"
        )
        signal_layout.addWidget(self.confirmation_label)

        dots = QHBoxLayout()
        dots.setAlignment(Qt.AlignCenter)
        self.confirmation_dots = []
        for i in range(3):
            dot = QLabel("●")
            dot.setAlignment(Qt.AlignCenter)
            dot.setStyleSheet(f"color: {BORDER}; font-size: 20px;")
            self.confirmation_dots.append(dot)
            dots.addWidget(dot)
            if i < 2:
                separator = QLabel("—")
                separator.setStyleSheet(f"color: {BORDER};")
                dots.addWidget(separator)
        signal_layout.addLayout(dots)

        self.waiting_label = QLabel("Espera al cierre de la vela para recibir la dirección de la siguiente.")
        self.waiting_label.setAlignment(Qt.AlignCenter)
        self.waiting_label.setStyleSheet(
            f"color: {MUTED}; font-size: 10px;"
        )
        signal_layout.addWidget(self.waiting_label)
        right.addWidget(signal_card)

        # ---------------- Timing de entrada ----------------
        timing_card = Card()
        timing_layout = QVBoxLayout(timing_card)
        timing_layout.setContentsMargins(15, 14, 15, 14)
        timing_layout.setSpacing(8)

        timing_header = QHBoxLayout()
        timing_title = QLabel("TIMING DE ENTRADA")
        timing_title.setStyleSheet(
            f"color: {TEXT}; font-size: 11px; font-weight: 700;"
        )
        timing_header.addWidget(timing_title)
        timing_header.addStretch()

        interval_label = QLabel("VELA")
        interval_label.setStyleSheet(
            f"color: {MUTED}; font-size: 8px; font-weight: 700;"
        )
        timing_header.addWidget(interval_label)

        self.candle_interval_combo = QComboBox()
        self.candle_interval_combo.addItem("15 s", 15)
        self.candle_interval_combo.addItem("30 s", 30)
        self.candle_interval_combo.addItem("1 min", 60)
        self.candle_interval_combo.addItem("2 min", 120)
        self.candle_interval_combo.addItem("3 min", 180)
        self.candle_interval_combo.addItem("5 min", 300)
        self.candle_interval_combo.setCurrentIndex(2)
        self.candle_interval_combo.setFixedWidth(78)
        self.candle_interval_combo.setStyleSheet(
            f"""
            QComboBox {{
                background-color: {CARD_2};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 4px 8px;
                font-size: 9px;
                font-weight: 700;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 18px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {CARD_2};
                color: {TEXT};
                border: 1px solid {BORDER};
                selection-background-color: #202735;
            }}
            """
        )
        self.candle_interval_combo.currentIndexChanged.connect(
            self.update_timing_view
        )
        timing_header.addWidget(self.candle_interval_combo)
        timing_layout.addLayout(timing_header)

        timing_values = QHBoxLayout()
        timing_values.setSpacing(8)

        self.current_candle_value = self.create_timing_value(
            timing_values, "VELA ACTUAL", "00:00"
        )
        self.next_candle_value = self.create_timing_value(
            timing_values, "SIGUIENTE", "00:00"
        )
        self.confirmed_age_value = self.create_timing_value(
            timing_values, "CONFIRMACIÓN", "—"
        )
        timing_layout.addLayout(timing_values)

        self.timing_status = QLabel(
            "Esperando una señal confirmada..."
        )
        self.timing_status.setAlignment(Qt.AlignCenter)
        self.timing_status.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-weight: 700;"
        )
        timing_layout.addWidget(self.timing_status)

        self.timing_hint = QLabel(
            "La señal se calcula cuando termina una vela. La captura se actualiza cada 2 s."
        )
        self.timing_hint.setAlignment(Qt.AlignCenter)
        self.timing_hint.setWordWrap(True)
        self.timing_hint.setStyleSheet(
            f"color: {MUTED}; font-size: 8px;"
        )
        timing_layout.addWidget(self.timing_hint)

        right.addWidget(timing_card)

        history_card = Card()
        history_layout = QVBoxLayout(history_card)
        history_layout.setContentsMargins(15, 15, 15, 15)

        history_title = QLabel("HISTORIAL")
        history_title.setStyleSheet(
            f"color: {TEXT}; font-size: 11px; font-weight: 700;"
        )
        history_layout.addWidget(history_title)

        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history_scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
        )

        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setContentsMargins(0, 7, 0, 0)
        self.history_layout.setSpacing(5)
        self.history_scroll.setWidget(self.history_container)
        history_layout.addWidget(self.history_scroll)
        right.addWidget(history_card, 1)

        controls = QHBoxLayout()

        self.select_button = QPushButton("⚙  GRÁFICO")
        self.select_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {CARD_2};
                color: {TEXT};
                border: 1px solid {BORDER};
            }}
            QPushButton:hover {{ background-color: #202735; }}
            """
        )
        self.select_button.clicked.connect(self.select_chart)

        self.start_button = QPushButton("▶  INICIAR")
        self.start_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {GREEN};
                color: #061008;
            }}
            QPushButton:hover {{ background-color: #4ade70; }}
            """
        )
        self.start_button.clicked.connect(self.start_monitor)

        self.stop_button = QPushButton("■  DETENER")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {RED_DARK};
                color: {RED};
                border: 1px solid #652321;
            }}
            QPushButton:hover {{ background-color: #56201e; }}
            """
        )
        self.stop_button.clicked.connect(self.stop_monitor)

        controls.addWidget(self.select_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        right.addLayout(controls)

        content.addLayout(left, 2)
        content.addLayout(right, 1)
        main.addLayout(content, 1)

        footer = QHBoxLayout()
        self.status_label = QLabel("● Sistema listo")
        self.status_label.setStyleSheet(
            f"color: {MUTED}; font-size: 9px;"
        )
        footer.addWidget(self.status_label)
        footer.addStretch()

        self.last_analysis_label = QLabel("Último análisis: —")
        self.last_analysis_label.setStyleSheet(
            f"color: {MUTED}; font-size: 9px;"
        )
        footer.addWidget(self.last_analysis_label)
        main.addLayout(footer)

    def create_metric(self, title, value):
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"color: {MUTED}; font-size: 8px; font-weight: 700;"
        )

        value_label = QLabel(value)
        value_label.setStyleSheet(
            f"color: {TEXT}; font-size: 20px; font-weight: 800;"
        )

        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return card, value_label

    def create_timing_value(self, parent_layout, title, value):
        box = QFrame()
        box.setStyleSheet(
            f"QFrame {{ background-color: {CARD_2}; border-radius: 8px; }}"
        )
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(7, 6, 7, 6)
        box_layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            f"color: {MUTED}; font-size: 7px; font-weight: 700;"
        )

        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet(
            f"color: {TEXT}; font-size: 14px; font-weight: 800;"
        )

        box_layout.addWidget(title_label)
        box_layout.addWidget(value_label)
        parent_layout.addWidget(box, 1)
        return value_label

    def get_candle_interval(self):
        if not hasattr(self, "candle_interval_combo"):
            return CANDLE_INTERVAL_SECONDS
        value = self.candle_interval_combo.currentData()
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = CANDLE_INTERVAL_SECONDS
        return max(1, value)

    @staticmethod
    def format_seconds(seconds):
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes:02d}:{seconds:02d}"

    def get_candle_slot(self):
        interval = self.get_candle_interval()
        now = datetime.now()
        total_seconds = (
            now.hour * 3600
            + now.minute * 60
            + now.second
            + now.microsecond / 1_000_000
        )
        return int(total_seconds // interval)

    def update_timing_view(self):
        if not hasattr(self, "current_candle_value"):
            return

        interval = self.get_candle_interval()
        now = datetime.now()

        total_seconds = (
            now.hour * 3600
            + now.minute * 60
            + now.second
            + now.microsecond / 1_000_000
        )
        elapsed = int(total_seconds % interval)
        remaining = max(0, interval - elapsed)
        if elapsed == 0:
            remaining = interval

        self.current_candle_value.setText(self.format_seconds(elapsed))
        self.next_candle_value.setText(self.format_seconds(remaining))

        if self.decision_timestamp is not None and self.decision_direction:
            age = max(0, int((now - self.decision_timestamp).total_seconds()))
            self.confirmed_age_value.setText(self.format_seconds(age))
            direction_color = GREEN if self.decision_direction == "SUBIDA" else RED

            if age <= 4:
                status = f"NUEVA DECISIÓN · {self.decision_direction}"
                status_color = direction_color
            elif remaining <= 3:
                status = f"{self.decision_direction} · PRÓXIMA VELA EN {self.format_seconds(remaining)}"
                status_color = YELLOW
            else:
                status = f"{self.decision_direction} · SEÑAL PARA ESTA VELA"
                status_color = direction_color

            self.timing_status.setText(status)
            self.timing_status.setStyleSheet(
                f"color: {status_color}; font-size: 9px; font-weight: 800;"
            )
            self.timing_hint.setText(
                f"La decisión se calculó al cerrar la vela anterior · "
                f"señal vigente hace {self.format_seconds(age)}"
            )
            return

        self.confirmed_age_value.setText("—")
        self.timing_status.setText("Esperando el cierre de la vela...")
        self.timing_status.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-weight: 700;"
        )
        self.timing_hint.setText(
            f"La decisión para la siguiente vela aparece cuando termina la actual. "
            f"Faltan {self.format_seconds(remaining)}."
        )

    # ========================================================
    # WINDOW CONTROLS
    # ========================================================

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            edges = self.get_resize_edges(pos)

            if edges:
                self.resizing = True
                self.resize_edges = edges
                self.resize_start_geometry = self.geometry()
                self.resize_start_position = event.globalPosition().toPoint()
                event.accept()
                return

            if self.is_in_title_bar(pos):
                self.dragging = True
                self.drag_position = (
                    event.globalPosition().toPoint()
                    - self.frameGeometry().topLeft()
                )
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()

        if self.resizing:
            self.perform_resize(event.globalPosition().toPoint())
            event.accept()
            return

        if self.dragging:
            self.move(
                event.globalPosition().toPoint()
                - self.drag_position
            )
            event.accept()
            return

        edges = self.get_resize_edges(pos)

        if edges == 0:
            self.setCursor(Qt.ArrowCursor)
        elif edges in (1, 2):
            self.setCursor(Qt.SizeHorCursor)
        elif edges in (4, 8):
            self.setCursor(Qt.SizeVerCursor)
        elif edges in (5, 10):
            self.setCursor(Qt.SizeFDiagCursor)
        elif edges in (6, 9):
            self.setCursor(Qt.SizeBDiagCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.resizing = False
        self.resize_edges = 0
        self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def get_resize_edges(self, pos):
        margin = self.RESIZE_MARGIN
        rect = self.rect()
        edges = 0

        if pos.x() <= margin:
            edges |= 1
        elif pos.x() >= rect.width() - margin:
            edges |= 2

        if pos.y() <= margin:
            edges |= 4
        elif pos.y() >= rect.height() - margin:
            edges |= 8

        return edges

    def perform_resize(self, global_pos):
        delta = global_pos - self.resize_start_position
        geometry = QRect(self.resize_start_geometry)

        min_w = self.minimumWidth()
        min_h = self.minimumHeight()

        if self.resize_edges & 1:
            new_left = geometry.left() + delta.x()
            if geometry.right() - new_left + 1 >= min_w:
                geometry.setLeft(new_left)

        if self.resize_edges & 2:
            new_right = geometry.right() + delta.x()
            if new_right - geometry.left() + 1 >= min_w:
                geometry.setRight(new_right)

        if self.resize_edges & 4:
            new_top = geometry.top() + delta.y()
            if geometry.bottom() - new_top + 1 >= min_h:
                geometry.setTop(new_top)

        if self.resize_edges & 8:
            new_bottom = geometry.bottom() + delta.y()
            if new_bottom - geometry.top() + 1 >= min_h:
                geometry.setBottom(new_bottom)

        self.setGeometry(geometry)

    def is_in_title_bar(self, pos):
        return hasattr(self, "title_bar") and self.title_bar.geometry().contains(pos)

    # ========================================================
    # SELECT CHART
    # ========================================================

    def select_chart(self):
        self.status_label.setText("● Selecciona el área del gráfico...")
        self.status_label.setStyleSheet(f"color: {BLUE}; font-size: 9px;")
        self.market_status.setText("SELECCIONANDO GRÁFICO...")

        self.selector = ScreenSelector(self.set_chart_rect)
        self.selector.show()
        self.selector.raise_()
        self.selector.activateWindow()

    def set_chart_rect(self, rect):
        self.chart_rect = rect
        self.status_label.setText("● Área del gráfico configurada")
        self.status_label.setStyleSheet(f"color: {GREEN}; font-size: 9px;")
        self.market_status.setText(
            f"GRÁFICO: {rect.width()} × {rect.height()}"
        )

    # ========================================================
    # START / STOP
    # ========================================================

    def start_monitor(self):
        if self.chart_rect is None:
            self.status_label.setText("● Primero selecciona el gráfico")
            self.status_label.setStyleSheet(f"color: {YELLOW}; font-size: 9px;")
            return

        self.running = True
        self.previous_mask = None
        self.confirmation_history.clear()
        self.confirmation_timestamp = None
        self.confirmed_direction = None
        self.previous_confirmed = False
        self.decision_timestamp = None
        self.decision_direction = None
        # Forzamos un primer análisis para mostrar la última vela cerrada.
        self.last_candle_slot = self.get_candle_slot() - 1
        self.timer.start(INTERVAL_SECONDS * 1000)
        self.clock_timer.start()
        self.update_timing_view()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.live_label.setText("● LIVE")
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

        self.status_label.setText("● Monitor activo")
        self.status_label.setStyleSheet(f"color: {GREEN}; font-size: 9px;")
        self.process_capture()

    def stop_monitor(self):
        self.running = False
        self.timer.stop()
        self.clock_timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        self.live_label.setText("● OFFLINE")
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

        self.status_label.setText("● Monitor detenido")
        self.status_label.setStyleSheet(f"color: {MUTED}; font-size: 9px;")

    # ========================================================
    # CAPTURE
    # ========================================================

    def capture_chart(self):
        if self.chart_rect is None:
            return None

        rect = self.chart_rect

        with mss.mss() as sct:
            monitor = {
                "left": rect.x(),
                "top": rect.y(),
                "width": rect.width(),
                "height": rect.height(),
            }
            screenshot = sct.grab(monitor)
            img = Image.frombytes(
                "RGB",
                screenshot.size,
                screenshot.rgb,
            )

        img.save(SCREENSHOT_PATH)
        return img

    def update_preview(self, img):
        if img is None:
            return

        img = img.convert("RGB")
        data = img.tobytes("raw", "RGB")

        qimage = QImage(
            data,
            img.width,
            img.height,
            img.width * 3,
            QImage.Format_RGB888,
        )

        pixmap = QPixmap.fromImage(qimage).scaled(
            self.preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.preview.setPixmap(pixmap)
        self.capture_info.setText(f"{img.width} × {img.height}")

    # ========================================================
    # PROCESS CAPTURE
    # ========================================================

    def process_capture(self):
        if not self.running:
            return

        try:
            img = self.capture_chart()
            if img is None:
                return

            self.update_preview(img)
            frame = np.array(img)

            green_mask, red_mask = detect_colors(frame)
            green_mask = clean_mask(green_mask)
            red_mask = clean_mask(red_mask)

            current_mask = (green_mask | red_mask).astype(np.uint8)

            candle_change = 100.0
            if (
                self.previous_mask is not None
                and current_mask.shape == self.previous_mask.shape
            ):
                old = self.previous_mask > 0
                new = current_mask > 0
                changed = np.logical_xor(old, new)
                total = np.count_nonzero(old | new)
                if total > 0:
                    candle_change = (
                        np.count_nonzero(changed) / total
                    ) * 100

            self.previous_mask = current_mask.copy()
            self.last_candle_change = candle_change
            self.change_value.setText(f"{candle_change:.1f}%")

            # ----------------------------------------------------
            # PUNTO CLAVE: solo tomamos una decisión cuando cambia
            # la vela. La captura cada 2 s mantiene la app al día,
            # pero la señal se genera UNA VEZ por cada cierre.
            # ----------------------------------------------------
            current_slot = self.get_candle_slot()
            new_candle = current_slot != self.last_candle_slot
            if not new_candle:
                self.market_status.setText("Vigilando la vela actual...")
                self.update_timing_view()
                return

            self.last_candle_slot = current_slot

            candidates = find_candle_candidates(
                green_mask,
                red_mask,
            )
            candidates = merge_close_candidates(candidates)
            candidates = filter_candidates(candidates)

            candles = []
            for candidate in candidates:
                candle = get_candle_data(
                    green_mask,
                    red_mask,
                    candidate["x"],
                )
                if candle is not None:
                    candles.append(candle)

            if len(candles) < 10:
                self.market_status.setText(
                    f"Cierre detectado, pero solo hay {len(candles)} velas"
                )
                self.decision_timestamp = None
                self.decision_direction = None
                self.show_signal("NO OPERAR", True)
                return

            # La última vela de la captura es la nueva vela que acaba
            # de empezar. Por tanto, candles[:-1] contiene la vela
            # recién cerrada y las anteriores.
            closed_candles = candles[:-1]
            result = analyze_structure(closed_candles)

            signal = result.get("signal", "NO OPERAR")
            score = result.get("score", 0)
            movement = result.get("movement", 0)
            green_count = result.get("green_count", 0)
            red_count = result.get("red_count", 0)

            signal_text = str(signal).upper()
            if "SUBIDA" in signal_text:
                signal = "SUBIDA"
            elif "BAJADA" in signal_text:
                signal = "BAJADA"
            else:
                signal = "NO OPERAR"

            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0

            try:
                movement = float(movement)
            except (TypeError, ValueError):
                movement = 0

            self.analysis_count += 1
            self.last_signal = signal
            self.last_score = score
            self.last_movement = movement

            self.last_analysis_label.setText(
                "Último cierre analizado: "
                + datetime.now().strftime("%H:%M:%S")
            )

            self.score_value.setText(
                f"+{score:g}" if score > 0 else f"{score:g}"
            )
            self.movement_value.setText(
                f"+{movement:g}" if movement > 0 else f"{movement:g}"
            )

            # Una única decisión para la PRÓXIMA vela.
            if signal in ("SUBIDA", "BAJADA"):
                self.decision_timestamp = datetime.now()
                self.decision_direction = signal
            else:
                self.decision_timestamp = None
                self.decision_direction = None

            self.update_timing_view()
            self.show_signal(signal, True)

            if signal != "NO OPERAR":
                self.add_history(
                    signal,
                    score,
                    True,
                )

            self.save_csv(
                signal,
                score,
                movement,
                1 if signal != "NO OPERAR" else 0,
                signal != "NO OPERAR",
            )

            self.market_status.setText(
                f"CIERRE ANALIZADO · {len(closed_candles)} velas "
                f"| Verdes {green_count} | Rojas {red_count}"
            )

        except Exception as error:
            self.status_label.setText(
                f"● Error: {str(error)[:100]}"
            )
            self.status_label.setStyleSheet(
                f"color: {RED}; font-size: 9px;"
            )

    # ========================================================
    # SIGNAL UI
    # ========================================================

    def show_signal(self, signal, is_new_candle):
        if signal == "NO OPERAR":
            self.signal_label.setText("NO OPERAR")
            self.signal_label.setStyleSheet(
                f"color: {MUTED}; font-size: 29px; font-weight: 800;"
            )
            self.confirmation_label.setText("ESPERAR")
            self.confirmation_label.setStyleSheet(
                f"color: {YELLOW}; font-size: 20px; font-weight: 800;"
            )
            self.waiting_label.setText(
                "La vela terminó sin una dirección suficientemente clara."
            )
            self.waiting_label.setStyleSheet(
                f"color: {MUTED}; font-size: 10px; font-weight: 600;"
            )
        elif signal == "SUBIDA":
            self.signal_label.setText("PRÓXIMA VELA: SUBIDA")
            self.signal_label.setStyleSheet(
                f"color: {GREEN}; font-size: 23px; font-weight: 800;"
            )
            self.confirmation_label.setText("LISTA")
            self.confirmation_label.setStyleSheet(
                f"color: {GREEN}; font-size: 20px; font-weight: 800;"
            )
            self.waiting_label.setText(
                "✓ VELA CERRADA · OPCIÓN ALCISTA PARA LA SIGUIENTE VELA · REVISAR ENTRADA"
            )
            self.waiting_label.setStyleSheet(
                f"color: {GREEN}; font-size: 9px; font-weight: 800;"
            )
        elif signal == "BAJADA":
            self.signal_label.setText("PRÓXIMA VELA: BAJADA")
            self.signal_label.setStyleSheet(
                f"color: {RED}; font-size: 23px; font-weight: 800;"
            )
            self.confirmation_label.setText("LISTA")
            self.confirmation_label.setStyleSheet(
                f"color: {RED}; font-size: 20px; font-weight: 800;"
            )
            self.waiting_label.setText(
                "✓ VELA CERRADA · OPCIÓN BAJISTA PARA LA SIGUIENTE VELA · REVISAR ENTRADA"
            )
            self.waiting_label.setStyleSheet(
                f"color: {RED}; font-size: 9px; font-weight: 800;"
            )

        for dot in self.confirmation_dots:
            if signal == "SUBIDA":
                color = GREEN
            elif signal == "BAJADA":
                color = RED
            else:
                color = BORDER
            dot.setStyleSheet(f"color: {color}; font-size: 20px;")

    # ========================================================
    # HISTORY
    # ========================================================

    def add_history(self, signal, score, confirmed):
        now = datetime.now().strftime("%H:%M:%S")

        item = QFrame()
        item.setStyleSheet(
            f"QFrame {{ background-color: {CARD_2}; border-radius: 8px; }}"
        )

        layout = QHBoxLayout(item)
        layout.setContentsMargins(9, 6, 9, 6)

        time_label = QLabel(now)
        time_label.setStyleSheet(
            f"color: {MUTED}; font-size: 8px;"
        )

        signal_label = QLabel(signal)
        color = GREEN if signal == "SUBIDA" else RED if signal == "BAJADA" else MUTED
        signal_label.setStyleSheet(
            f"color: {color}; font-size: 9px; font-weight: 800;"
        )

        score_label = QLabel(f"Score {score:g}")
        score_label.setStyleSheet(
            f"color: {TEXT}; font-size: 8px;"
        )

        layout.addWidget(time_label)
        layout.addWidget(signal_label)
        layout.addStretch()
        layout.addWidget(score_label)

        if confirmed:
            confirmed_label = QLabel("CONFIRMADA")
            confirmed_label.setStyleSheet(
                f"color: {GREEN}; font-size: 7px; font-weight: 800;"
            )
            layout.addWidget(confirmed_label)

        self.history_layout.insertWidget(0, item)
        self.history.append(
            {
                "time": now,
                "signal": signal,
                "score": score,
                "confirmed": confirmed,
            }
        )

        while self.history_layout.count() > 12:
            old = self.history_layout.takeAt(
                self.history_layout.count() - 1
            )
            if old.widget():
                old.widget().deleteLater()

    # ========================================================
    # CSV
    # ========================================================

    def save_csv(
        self,
        signal,
        score,
        movement,
        confirmation,
        confirmed,
    ):
        exists = os.path.exists(CSV_PATH)

        with open(
            CSV_PATH,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            if not exists:
                writer.writerow(
                    [
                        "fecha",
                        "hora",
                        "señal",
                        "score",
                        "movimiento",
                        "confirmacion",
                        "confirmada",
                    ]
                )

            now = datetime.now()
            writer.writerow(
                [
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S"),
                    signal,
                    score,
                    movement,
                    confirmation,
                    confirmed,
                ]
            )

    def closeEvent(self, event):
        self.timer.stop()
        self.clock_timer.stop()
        event.accept()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("IQ AI")
    window = BlitzMonitor()
    window.show()
    sys.exit(app.exec())
