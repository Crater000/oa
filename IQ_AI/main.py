import os
import sys

import mss
from PIL import Image
from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QRubberBand,
    QVBoxLayout,
    QWidget,
)


class SelectionOverlay(QWidget):
    def __init__(self, screenshot, callback):
        super().__init__()

        self.screenshot = screenshot
        self.callback = callback
        self.start_point = None
        self.end_point = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )

        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setCursor(Qt.CursorShape.CrossCursor)

        # Mostrar la captura de pantalla como fondo
        self.background = QLabel(self)
        self.background.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.screen_width = self.width()
        self.screen_height = self.height()

        self.update_background()

        self.selection = QRubberBand(
            QRubberBand.Shape.Rectangle,
            self
        )

    def update_background(self):
        image = QImage(
            self.screenshot.tobytes("raw", "RGB"),
            self.screenshot.width,
            self.screenshot.height,
            self.screenshot.width * 3,
            QImage.Format.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(image)

        self.background.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        self.background.resize(self.size())
        self.background.lower()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.position().toPoint()

            self.selection.setGeometry(
                QRect(self.start_point, QPoint())
            )

            self.selection.show()

    def mouseMoveEvent(self, event):
        if self.start_point is not None:
            current = event.position().toPoint()

            rectangle = QRect(
                self.start_point,
                current
            ).normalized()

            self.selection.setGeometry(rectangle)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.end_point = event.position().toPoint()

            rectangle = QRect(
                self.start_point,
                self.end_point
            ).normalized()

            if rectangle.width() > 30 and rectangle.height() > 30:
                self.callback(rectangle)

            self.close()


class IQAIWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "IQ AI - Analizador de Mercado"
        )

        self.resize(950, 700)

        self.data_dir = os.path.join(
            os.getcwd(),
            "data"
        )

        os.makedirs(
            self.data_dir,
            exist_ok=True
        )

        self.setup_ui()

    def setup_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        title = QLabel("IQ AI ANALYZER")

        title.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        title.setStyleSheet(
            "font-size: 28px;"
            "font-weight: bold;"
            "padding: 10px;"
        )

        main_layout.addWidget(title)

        # Selección de activo y temporalidad
        controls = QGridLayout()

        asset_label = QLabel("Activo:")

        self.asset = QComboBox()

        self.asset.addItems([
            "BTC/USD",
            "EUR/USD",
            "GBP/USD",
            "USD/JPY",
        ])

        timeframe_label = QLabel(
            "Temporalidad:"
        )

        self.timeframe = QComboBox()

        self.timeframe.addItems([
            "1 minuto",
            "5 minutos",
            "15 minutos",
            "1 hora",
        ])

        controls.addWidget(
            asset_label,
            0,
            0
        )

        controls.addWidget(
            self.asset,
            0,
            1
        )

        controls.addWidget(
            timeframe_label,
            0,
            2
        )

        controls.addWidget(
            self.timeframe,
            0,
            3
        )

        main_layout.addLayout(controls)

        # Indicadores
        self.analysis_box = QLabel(
            "Tendencia:     --\n"
            "RSI:           --\n"
            "MACD:          --\n"
            "EMA:           --\n"
            "Volatilidad:   --"
        )

        self.analysis_box.setStyleSheet(
            "font-size: 20px;"
            "padding: 18px;"
            "border: 1px solid #555;"
            "border-radius: 8px;"
        )

        main_layout.addWidget(
            self.analysis_box
        )

        # Señal
        self.signal = QLabel(
            "ESPERANDO ANÁLISIS"
        )

        self.signal.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.signal.setStyleSheet(
            "font-size: 26px;"
            "font-weight: bold;"
            "padding: 20px;"
            "border: 2px solid #555;"
            "border-radius: 10px;"
        )

        main_layout.addWidget(
            self.signal
        )

        self.confidence = QLabel(
            "Confianza: --"
        )

        self.expiration = QLabel(
            "Tiempo sugerido: --"
        )

        for label in (
            self.confidence,
            self.expiration
        ):
            label.setAlignment(
                Qt.AlignmentFlag.AlignCenter
            )

            label.setStyleSheet(
                "font-size: 19px;"
            )

            main_layout.addWidget(label)

        # Botones
        buttons = QGridLayout()

        select_button = QPushButton(
            "SELECCIONAR GRÁFICO"
        )

        select_button.clicked.connect(
            self.select_graph_area
        )

        analyze_button = QPushButton(
            "ANALIZAR CAPTURA"
        )

        analyze_button.clicked.connect(
            self.analyze_capture
        )

        buttons.addWidget(
            select_button,
            0,
            0
        )

        buttons.addWidget(
            analyze_button,
            0,
            1
        )

        main_layout.addLayout(buttons)

        # Vista previa
        self.preview = QLabel(
            "Aquí aparecerá el gráfico seleccionado"
        )

        self.preview.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.preview.setMinimumHeight(
            200
        )

        self.preview.setStyleSheet(
            "border: 1px dashed #777;"
            "padding: 10px;"
        )

        main_layout.addWidget(
            self.preview
        )

        self.status = QLabel(
            "Estado: listo"
        )

        self.status.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        main_layout.addWidget(
            self.status
        )

    def select_graph_area(self):

        self.hide()

        QApplication.processEvents()

        # Captura de toda la pantalla
        with mss.mss() as screenshot_tool:

            monitor = screenshot_tool.monitors[1]

            shot = screenshot_tool.grab(
                monitor
            )

            screenshot = Image.frombytes(
                "RGB",
                shot.size,
                shot.rgb
            )

        self.overlay = SelectionOverlay(
            screenshot,
            lambda rect: self.save_selection(
                screenshot,
                rect
            )
        )

        self.overlay.showFullScreen()

    def save_selection(
        self,
        screenshot,
        rect
    ):

        screen_width = self.overlay.width()
        screen_height = self.overlay.height()

        image_width, image_height = (
            screenshot.size
        )

        scale_x = (
            image_width / screen_width
        )

        scale_y = (
            image_height / screen_height
        )

        left = int(
            rect.left() * scale_x
        )

        top = int(
            rect.top() * scale_y
        )

        right = int(
            rect.right() * scale_x
        )

        bottom = int(
            rect.bottom() * scale_y
        )

        selected = screenshot.crop(
            (
                left,
                top,
                right,
                bottom
            )
        )

        filename = os.path.join(
            self.data_dir,
            "grafico.png"
        )

        selected.save(filename)

        self.show()

        self.show_preview(selected)

        self.status.setText(
            "Estado: gráfico seleccionado correctamente."
        )

    def show_preview(self, image):

        image = image.convert("RGB")

        qimage = QImage(
            image.tobytes("raw", "RGB"),
            image.width,
            image.height,
            image.width * 3,
            QImage.Format.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(
            qimage
        )

        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def analyze_capture(self):

        capture_path = os.path.join(
            self.data_dir,
            "grafico.png"
        )

        if not os.path.exists(
            capture_path
        ):

            self.status.setText(
                "Primero selecciona el gráfico."
            )

            return

        self.signal.setText(
            "ANÁLISIS PENDIENTE"
        )

        self.confidence.setText(
            "Confianza: --"
        )

        self.expiration.setText(
            "Tiempo sugerido: --"
        )

        self.status.setText(
            "Gráfico encontrado. "
            "Preparado para el módulo de análisis."
        )


if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = IQAIWindow()

    window.show()

    sys.exit(
        app.exec()
    )