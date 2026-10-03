import csv
import os
import sys
import time
from collections import deque
from datetime import datetime

import mss
import numpy as np
from PIL import Image, ImageDraw

# ML se carga de forma diferida para que la ventana abra rápido.
joblib = None
RandomForestClassifier = None
ExtraTreesClassifier = None
LogisticRegression = None
StandardScaler = None
make_pipeline = None
accuracy_score = None
SKLEARN_OK = None


def ensure_ml_imports():
    global joblib, RandomForestClassifier, ExtraTreesClassifier
    global LogisticRegression, StandardScaler, make_pipeline
    global accuracy_score, SKLEARN_OK

    if SKLEARN_OK is not None:
        return SKLEARN_OK

    try:
        import joblib as _joblib
        from sklearn.ensemble import (
            RandomForestClassifier as _RandomForestClassifier,
            ExtraTreesClassifier as _ExtraTreesClassifier,
        )
        from sklearn.linear_model import LogisticRegression as _LogisticRegression
        from sklearn.preprocessing import StandardScaler as _StandardScaler
        from sklearn.pipeline import make_pipeline as _make_pipeline
        from sklearn.metrics import accuracy_score as _accuracy_score

        joblib = _joblib
        RandomForestClassifier = _RandomForestClassifier
        ExtraTreesClassifier = _ExtraTreesClassifier
        LogisticRegression = _LogisticRegression
        StandardScaler = _StandardScaler
        make_pipeline = _make_pipeline
        accuracy_score = _accuracy_score
        SKLEARN_OK = True
    except Exception:
        joblib = None
        RandomForestClassifier = None
        ExtraTreesClassifier = None
        LogisticRegression = None
        StandardScaler = None
        make_pipeline = None
        accuracy_score = None
        SKLEARN_OK = False

    return SKLEARN_OK

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
    assess_detection_quality,
)


# ============================================================
# CONFIG
# ============================================================

CAPTURE_INTERVAL_MS = 500  # análisis cada 0,5 s
PREVIEW_INTERVAL_MS = 1000  # refresco visual más liviano; no afecta al análisis
MIN_DETECTION_QUALITY = 0.50  # debajo de esto la captura NO alimenta la IA
CHART_CHANGE_THRESHOLD = 0.8
CONFIRMATION_COUNT = 3
CANDLE_INTERVAL_SECONDS = 60
SIGNAL_VALID_SECONDS = 10
DATA_DIR = "data"
SCREENSHOT_PATH = os.path.join(DATA_DIR, "grafico.png")
CSV_PATH = os.path.join(DATA_DIR, "blitz_signals.csv")
PREDICTION_RESULTS_PATH = os.path.join(DATA_DIR, "prediction_results.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# IA DE APRENDIZAJE DE PATRONES
# ============================================================

DATASET_PATH = os.path.join(DATA_DIR, "ia_market_patterns.csv")
MODEL_PATH = os.path.join(DATA_DIR, "ia_market_model.joblib")
MIN_TRAIN_SAMPLES = 60
RETRAIN_EVERY = 5
WINDOW = 12
AI_TRUST_MIN_SAMPLES = 100
AI_TRUST_MIN_VALIDATION = 0.55
AI_TRUST_MIN_CONFIDENCE = 0.60
AI_TRUST_MIN_MODEL_AGREEMENT = 2.0 / 3.0

# Motor secuencial para predecir la vela SIGUIENTE mientras la actual
# todavía está abierta.
LIVE_PREDICTION_THRESHOLD = 0.07
FINAL_PREDICTION_THRESHOLD = 0.09
FINAL_SECONDS = 5
MIN_FINAL_STABILITY = 0.55
CALIBRATION_MIN_RESULTS = 12
MIN_ACTIVE_METHODS = 0  # ya NO bloquea la decisión por conteo
PREDICTION_HISTORY_SIZE = 12

class MarketPatternAI:
    """
    IA de patrones con ENSAMBLE de modelos.

    Aprende de las últimas 12 velas cerradas y predice la siguiente vela.
    Usa tres modelos distintos (Random Forest, Extra Trees y Regresión
    Logística) y valida cronológicamente: entrena con datos anteriores y
    prueba con datos posteriores.

    IMPORTANTE: una predicción puede mostrarse aunque la IA todavía NO esté
    autorizada a participar en la decisión final. Para participar debe tener
    suficientes muestras, validación, confianza y acuerdo entre modelos.
    """

    def __init__(self):
        self.pending_features = None
        self.pending_time = None
        self.models = {}
        self.model_scores = {}
        self.sample_count = self._count_samples()
        self.last_validation_accuracy = None
        self.new_samples_since_train = 0
        self.live_predictions = 0
        self.live_correct = 0
        self.model_checked = False

    # ------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------

    @staticmethod
    def extract_features(candles):
        if len(candles) < WINDOW:
            return None

        recent = candles[-WINDOW:]
        heights = np.array(
            [max(float(c["height"]), 1.0) for c in recent],
            dtype=float,
        )
        median_height = max(float(np.median(heights)), 1.0)

        features = []

        for candle in recent:
            direction = 1.0 if candle["color"] == "VERDE" else -1.0
            height = max(float(candle["height"]), 1.0)
            body_ratio = float(candle["body"]) / height
            upper_ratio = float(candle["upper_wick"]) / height
            lower_ratio = float(candle["lower_wick"]) / height
            size_ratio = height / median_height

            features.extend([
                direction,
                body_ratio,
                upper_ratio,
                lower_ratio,
                size_ratio,
            ])

        closes = np.array(
            [float(c["close"]) for c in recent],
            dtype=float,
        )

        # Y menor = precio visualmente más alto.
        price_moves = closes[:-1] - closes[1:]
        normalized_moves = price_moves / median_height

        green_ratio = sum(c["color"] == "VERDE" for c in recent) / WINDOW
        mean_body = float(np.mean([
            float(c["body"]) / max(float(c["height"]), 1.0)
            for c in recent
        ]))
        last3_move = float(np.sum(normalized_moves[-3:])) if len(normalized_moves) >= 3 else 0.0
        last5_move = float(np.sum(normalized_moves[-5:])) if len(normalized_moves) >= 5 else 0.0
        total_move = float((closes[0] - closes[-1]) / median_height)

        color_values = np.array(
            [1 if c["color"] == "VERDE" else -1 for c in recent],
            dtype=float,
        )
        last5_color_sum = float(np.sum(color_values[-5:]))

        streak = 1.0
        last_color = recent[-1]["color"]
        for candle in reversed(recent[:-1]):
            if candle["color"] == last_color:
                streak += 1.0
            else:
                break
        streak = streak if last_color == "VERDE" else -streak

        features.extend([
            green_ratio,
            mean_body,
            last3_move,
            last5_move,
            total_move,
            last5_color_sum,
            streak,
        ])

        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def _feature_count():
        return WINDOW * 5 + 7

    # ------------------------------------------------------------
    # STORAGE
    # ------------------------------------------------------------

    def _count_samples(self):
        if not os.path.exists(DATASET_PATH):
            return 0
        try:
            with open(DATASET_PATH, "r", encoding="utf-8", newline="") as f:
                return max(sum(1 for _ in f) - 1, 0)
        except Exception:
            return 0

    def _ensure_header(self):
        if os.path.exists(DATASET_PATH):
            return
        with open(DATASET_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp"]
                + [f"f{i}" for i in range(self._feature_count())]
                + ["target"]
            )

    def add_labeled_sample(self, features, target):
        if features is None or target not in ("VERDE", "ROJA"):
            return

        self._ensure_header()
        with open(DATASET_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [datetime.now().isoformat(timespec="seconds")]
                + [f"{float(x):.8f}" for x in features]
                + [target]
            )

        self.sample_count += 1
        self.new_samples_since_train += 1

    def _load_dataset(self):
        if not os.path.exists(DATASET_PATH):
            return None, None

        rows = []
        targets = []
        try:
            with open(DATASET_PATH, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                feature_keys = [f"f{i}" for i in range(self._feature_count())]
                for row in reader:
                    target = row.get("target")
                    if target not in ("VERDE", "ROJA"):
                        continue
                    try:
                        vector = [float(row[key]) for key in feature_keys]
                    except Exception:
                        continue
                    rows.append(vector)
                    targets.append(target)
        except Exception:
            return None, None

        if not rows:
            return None, None

        return np.asarray(rows, dtype=np.float32), np.asarray(targets)

    # ------------------------------------------------------------
    # MODELOS / ENSAMBLE
    # ------------------------------------------------------------

    @staticmethod
    def _build_models():
        return {
            "RF": RandomForestClassifier(
                n_estimators=180,
                max_depth=7,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "ET": ExtraTreesClassifier(
                n_estimators=220,
                max_depth=7,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=84,
                n_jobs=-1,
            ),
            "LR": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=700,
                    class_weight="balanced",
                    random_state=126,
                ),
            ),
        }

    @staticmethod
    def _model_probability(model, X):
        probs = model.predict_proba(X)
        classes = list(model.classes_)
        if "VERDE" not in classes or "ROJA" not in classes:
            return None
        iv = classes.index("VERDE")
        ir = classes.index("ROJA")
        return np.column_stack([probs[:, iv], probs[:, ir]])

    @staticmethod
    def _ensemble_probabilities(models, scores, X):
        pieces = []
        weights = []

        for name, model in models.items():
            try:
                p = MarketPatternAI._model_probability(model, X)
                if p is None:
                    continue
                # Modelos con validación mejor reciben más peso, pero ninguno
                # puede dominar completamente al resto.
                acc = float(scores.get(name, 0.50) or 0.50)
                weight = max(0.15, min(1.0, 0.25 + (acc - 0.50) * 2.5))
                pieces.append(p)
                weights.append(weight)
            except Exception:
                continue

        if not pieces:
            return None

        stacked = np.stack(pieces, axis=0)
        weights = np.asarray(weights, dtype=float)
        weighted = np.tensordot(weights, stacked, axes=(0, 0)) / max(weights.sum(), 1e-9)
        return weighted

    def _trust_state(self, confidence, model_agreement):
        accuracy = self.last_validation_accuracy

        reasons = []
        if self.sample_count < AI_TRUST_MIN_SAMPLES:
            reasons.append(f"muestras {self.sample_count}/{AI_TRUST_MIN_SAMPLES}")
        if accuracy is None or accuracy < AI_TRUST_MIN_VALIDATION:
            pct = 0 if accuracy is None else int(round(accuracy * 100))
            reasons.append(f"validación {pct}%<{int(AI_TRUST_MIN_VALIDATION*100)}%")
        if confidence is None or confidence < AI_TRUST_MIN_CONFIDENCE:
            pct = 0 if confidence is None else int(round(confidence * 100))
            reasons.append(f"confianza {pct}%<{int(AI_TRUST_MIN_CONFIDENCE*100)}%")
        if model_agreement is None or model_agreement < AI_TRUST_MIN_MODEL_AGREEMENT:
            pct = 0 if model_agreement is None else int(round(model_agreement * 100))
            reasons.append(f"acuerdo modelos {pct}%<{int(AI_TRUST_MIN_MODEL_AGREEMENT*100)}%")

        return len(reasons) == 0, reasons

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    def load_model(self):
        if not ensure_ml_imports() or not os.path.exists(MODEL_PATH):
            self.model_checked = True
            return

        try:
            payload = joblib.load(MODEL_PATH)

            if isinstance(payload, dict) and "models" in payload:
                self.models = payload.get("models", {}) or {}
                self.model_scores = payload.get("model_scores", {}) or {}
                self.last_validation_accuracy = payload.get("validation_accuracy")
            elif isinstance(payload, dict) and payload.get("model") is not None:
                # Compatibilidad con el Random Forest de versiones anteriores.
                self.models = {"RF-legacy": payload.get("model")}
                self.model_scores = {"RF-legacy": float(payload.get("validation_accuracy") or 0.50)}
                self.last_validation_accuracy = payload.get("validation_accuracy")
            elif payload is not None:
                self.models = {"RF-legacy": payload}
                self.model_scores = {"RF-legacy": 0.50}
        except Exception:
            self.models = {}
            self.model_scores = {}
        finally:
            self.model_checked = True

    def train(self, force=False):
        if not self.model_checked:
            self.load_model()

        if not ensure_ml_imports():
            return {
                "active": False,
                "status": "sklearn no instalado",
                "accuracy": None,
            }

        if self.sample_count < MIN_TRAIN_SAMPLES:
            return {
                "active": False,
                "status": f"aprendiendo {self.sample_count}/{MIN_TRAIN_SAMPLES}",
                "accuracy": self.last_validation_accuracy,
            }

        if not force and self.models and self.new_samples_since_train < RETRAIN_EVERY:
            return {
                "active": True,
                "status": "ensamble activo",
                "accuracy": self.last_validation_accuracy,
            }

        X, y = self._load_dataset()
        if X is None or len(X) < MIN_TRAIN_SAMPLES:
            return {
                "active": False,
                "status": f"aprendiendo {self.sample_count}/{MIN_TRAIN_SAMPLES}",
                "accuracy": self.last_validation_accuracy,
            }

        if len(np.unique(y)) < 2:
            return {
                "active": False,
                "status": "esperando ejemplos de ambos sentidos",
                "accuracy": None,
            }

        # Validación temporal: nunca mezclamos el futuro dentro del train.
        split = max(int(len(X) * 0.80), 1)
        if split >= len(X):
            split = len(X) - 1

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if len(X_test) < 5 or len(np.unique(y_train)) < 2:
            return {
                "active": False,
                "status": "faltan datos para validar cronológicamente",
                "accuracy": None,
            }

        validation_models = self._build_models()
        validation_scores = {}
        fitted_validation = {}

        for name, model in validation_models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                validation_scores[name] = float(accuracy_score(y_test, pred))
                fitted_validation[name] = model
            except Exception:
                continue

        if not fitted_validation:
            return {
                "active": False,
                "status": "no se pudo entrenar el ensamble",
                "accuracy": None,
            }

        ensemble_probs = self._ensemble_probabilities(
            fitted_validation,
            validation_scores,
            X_test,
        )

        ensemble_accuracy = None
        if ensemble_probs is not None:
            ensemble_pred = np.where(
                ensemble_probs[:, 0] >= ensemble_probs[:, 1],
                "VERDE",
                "ROJA",
            )
            ensemble_accuracy = float(accuracy_score(y_test, ensemble_pred))

        # Entrenamiento final con todos los datos después de validar.
        final_models = self._build_models()
        fitted_final = {}
        for name, model in final_models.items():
            try:
                model.fit(X, y)
                fitted_final[name] = model
            except Exception:
                continue

        self.models = fitted_final
        self.model_scores = validation_scores
        self.last_validation_accuracy = ensemble_accuracy
        self.new_samples_since_train = 0

        try:
            joblib.dump(
                {
                    "version": 2,
                    "models": self.models,
                    "model_scores": self.model_scores,
                    "validation_accuracy": self.last_validation_accuracy,
                    "samples": self.sample_count,
                },
                MODEL_PATH,
            )
        except Exception:
            pass

        return {
            "active": bool(self.models),
            "status": "ensamble activo" if self.models else "sin modelos",
            "accuracy": ensemble_accuracy,
        }

    # ------------------------------------------------------------
    # ONLINE LOOP
    # ------------------------------------------------------------

    def on_candle_close(self, closed_candles):
        info = {
            "active": False,
            "trusted": False,
            "participates": False,
            "signal": None,
            "confidence": None,
            "model_agreement": None,
            "model_votes": {},
            "status": "esperando datos",
            "samples": self.sample_count,
            "validation_accuracy": self.last_validation_accuracy,
            "trust_reasons": [],
        }

        # Etiquetar el patrón anterior con la vela recién cerrada.
        if self.pending_features is not None and closed_candles:
            target = closed_candles[-1].get("color")
            if target in ("VERDE", "ROJA"):
                self.add_labeled_sample(self.pending_features, target)
                self.pending_features = None
                self.pending_time = None

        train_info = self.train()
        info.update({
            "active": train_info["active"],
            "status": train_info["status"],
            "validation_accuracy": train_info["accuracy"],
            "samples": self.sample_count,
        })

        features = self.extract_features(closed_candles)
        if features is None:
            return info

        if self.models:
            model_votes = {}
            all_probabilities = []
            model_names = []

            for name, model in self.models.items():
                try:
                    p = self._model_probability(model, features.reshape(1, -1))
                    if p is None:
                        continue
                    p_green = float(p[0, 0])
                    p_red = float(p[0, 1])
                    vote = "SUBIDA" if p_green >= p_red else "BAJADA"
                    model_votes[name] = vote
                    all_probabilities.append([p_green, p_red])
                    model_names.append(name)
                except Exception:
                    continue

            ensemble = self._ensemble_probabilities(
                self.models,
                self.model_scores,
                features.reshape(1, -1),
            )

            if ensemble is not None and model_votes:
                p_green = float(ensemble[0, 0])
                p_red = float(ensemble[0, 1])
                signal = "SUBIDA" if p_green >= p_red else "BAJADA"
                confidence = max(p_green, p_red)
                agree_count = sum(v == signal for v in model_votes.values())
                model_agreement = agree_count / max(len(model_votes), 1)

                trusted, trust_reasons = self._trust_state(
                    confidence,
                    model_agreement,
                )

                info.update({
                    "signal": signal,
                    "confidence": confidence,
                    "model_agreement": model_agreement,
                    "model_votes": model_votes,
                    "trusted": trusted,
                    "participates": trusted,
                    "trust_reasons": trust_reasons,
                })

        # Guardar patrón actual para etiquetarlo con la próxima vela.
        self.pending_features = features
        self.pending_time = datetime.now()
        return info

    def predict_live(self, candles):
        """
        Predicción provisional usando la vela ACTUAL aunque todavía esté
        formándose.

        NO agrega muestras ni reentrena. El aprendizaje ocurre únicamente
        cuando una vela cierra mediante on_candle_close().
        """
        info = {
            "active": False,
            "trusted": False,
            "participates": False,
            "signal": None,
            "confidence": None,
            "model_agreement": None,
            "model_votes": {},
            "status": "aprendiendo",
            "samples": self.sample_count,
            "validation_accuracy": self.last_validation_accuracy,
            "trust_reasons": [],
        }

        if not self.model_checked:
            self.load_model()

        features = self.extract_features(candles)
        if features is None or not self.models:
            if self.sample_count < MIN_TRAIN_SAMPLES:
                info["status"] = (
                    f"aprendiendo {self.sample_count}/{MIN_TRAIN_SAMPLES}"
                )
            else:
                info["status"] = "modelo todavía no disponible"
            return info

        model_votes = {}

        for name, model in self.models.items():
            try:
                p = self._model_probability(
                    model,
                    features.reshape(1, -1),
                )
                if p is None:
                    continue

                p_green = float(p[0, 0])
                p_red = float(p[0, 1])
                model_votes[name] = (
                    "SUBIDA" if p_green >= p_red else "BAJADA"
                )
            except Exception:
                continue

        ensemble = self._ensemble_probabilities(
            self.models,
            self.model_scores,
            features.reshape(1, -1),
        )

        if ensemble is None or not model_votes:
            return info

        p_green = float(ensemble[0, 0])
        p_red = float(ensemble[0, 1])
        signal = "SUBIDA" if p_green >= p_red else "BAJADA"
        confidence = max(p_green, p_red)

        agree_count = sum(
            vote == signal
            for vote in model_votes.values()
        )
        model_agreement = agree_count / max(len(model_votes), 1)

        trusted, reasons = self._trust_state(
            confidence,
            model_agreement,
        )

        info.update({
            "active": True,
            "trusted": trusted,
            # La IA siempre aporta una lectura; el PESO final depende de
            # validación, muestras, confianza y acuerdo de modelos.
            "participates": True,
            "signal": signal,
            "confidence": confidence,
            "model_agreement": model_agreement,
            "model_votes": model_votes,
            "status": "lectura secuencial",
            "samples": self.sample_count,
            "validation_accuracy": self.last_validation_accuracy,
            "trust_reasons": reasons,
        })

        return info


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

def get_candle_data(green, red, x, left=None, right=None):
    height, width = green.shape

    # Usar la caja precisa del detector cuando está disponible.
    # El fallback ±4 mantiene compatibilidad con llamadas antiguas.
    if left is None:
        left = max(0, x - 4)
    else:
        left = max(0, int(left))

    if right is None:
        right = min(width - 1, x + 4)
    else:
        right = min(width - 1, int(right))

    if right < left:
        return None

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
# MOTOR DE CONFLUENCIA TÉCNICA
# ============================================================

class TechnicalConfluenceEngine:
    """
    Motor técnico para trabajar con las velas reconstruidas desde la imagen.

    IMPORTANTE:
    Los valores OHLC vienen de coordenadas de pantalla. Convertimos Y a
    "precio visual" multiplicando por -1 para que:
        precio más alto -> valor mayor
        precio más bajo -> valor menor

    Por eso EMA/RSI/MACD/etc. describen la geometría del gráfico capturado,
    no precios monetarios exactos. Aun así, son útiles para confluencia
    dentro de la misma captura si la escala visual es aproximadamente lineal.
    """

    METHOD_NAMES = [
        "Price Action",
        "Estructura",
        "Soporte/Resistencia",
        "Tendencia EMA",
        "Momentum",
        "Volatilidad",
        "Contexto",
    ]

    @staticmethod
    def _ema(values, period):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return np.asarray([], dtype=float)

        alpha = 2.0 / (period + 1.0)
        out = np.empty(len(values), dtype=float)
        out[0] = values[0]

        for i in range(1, len(values)):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]

        return out

    @staticmethod
    def _rsi_series(values, period=14):
        values = np.asarray(values, dtype=float)

        if len(values) < 2:
            return np.full(len(values), 50.0, dtype=float)

        delta = np.diff(values)
        gains = np.maximum(delta, 0.0)
        losses = np.maximum(-delta, 0.0)

        out = np.full(len(values), 50.0, dtype=float)

        if len(delta) < period:
            avg_gain = float(np.mean(gains)) if len(gains) else 0.0
            avg_loss = float(np.mean(losses)) if len(losses) else 0.0
            if avg_loss <= 1e-9:
                out[-1] = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                out[-1] = 100.0 - (100.0 / (1.0 + rs))
            return out

        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))

        if avg_loss <= 1e-9:
            out[period] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            out[period] = 100.0 - (100.0 / (1.0 + rs))

        for i in range(period + 1, len(values)):
            gain = gains[i - 1]
            loss = losses[i - 1]

            avg_gain = (
                (avg_gain * (period - 1)) + gain
            ) / period
            avg_loss = (
                (avg_loss * (period - 1)) + loss
            ) / period

            if avg_loss <= 1e-9:
                out[i] = 100.0 if avg_gain > 0 else 50.0
            else:
                rs = avg_gain / avg_loss
                out[i] = 100.0 - (100.0 / (1.0 + rs))

        # Rellenar el inicio con el primer RSI válido para evitar NaN.
        first_valid = min(period, len(out) - 1)
        out[:first_valid] = out[first_valid]
        return out

    @staticmethod
    def _atr_series(highs, lows, closes, period=14):
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)

        if len(closes) == 0:
            return np.asarray([], dtype=float)

        tr = np.zeros(len(closes), dtype=float)
        tr[0] = max(highs[0] - lows[0], 0.0)

        for i in range(1, len(closes)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )

        atr = np.zeros(len(closes), dtype=float)

        for i in range(len(closes)):
            start = max(0, i - period + 1)
            atr[i] = float(np.mean(tr[start:i + 1]))

        return atr

    @staticmethod
    def _stochastic_series(highs, lows, closes, period=14):
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)

        k = np.full(len(closes), 50.0, dtype=float)

        for i in range(len(closes)):
            start = max(0, i - period + 1)
            hh = float(np.max(highs[start:i + 1]))
            ll = float(np.min(lows[start:i + 1]))

            if hh - ll > 1e-9:
                k[i] = 100.0 * (closes[i] - ll) / (hh - ll)

        d = np.zeros(len(k), dtype=float)
        for i in range(len(k)):
            start = max(0, i - 2)
            d[i] = float(np.mean(k[start:i + 1]))

        return k, d

    @staticmethod
    def _slope(values, window):
        values = np.asarray(values, dtype=float)
        if len(values) < 2:
            return 0.0

        data = values[-min(window, len(values)):]
        if len(data) < 2:
            return 0.0

        x = np.arange(len(data), dtype=float)
        return float(np.polyfit(x, data, 1)[0])

    @staticmethod
    def _rolling_bb_width(closes, period=20):
        closes = np.asarray(closes, dtype=float)
        widths = np.full(len(closes), np.nan, dtype=float)

        if len(closes) < 2:
            return widths

        for i in range(len(closes)):
            start = max(0, i - period + 1)
            data = closes[start:i + 1]
            if len(data) < min(5, period):
                continue

            std = float(np.std(data))
            widths[i] = 4.0 * std

        return widths

    @staticmethod
    def _direction(score, threshold=0.75):
        if score >= threshold:
            return "SUBIDA"
        if score <= -threshold:
            return "BAJADA"
        return "NEUTRAL"

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    @classmethod
    def analyze(cls, candles):
        if len(candles) < 20:
            return {
                "signal": "NO OPERAR",
                "watch_direction": None,
                "entry_action": "SIN ENTRADA",
                "call_put": None,
                "bull_score": 0.0,
                "bear_score": 0.0,
                "score": 0.0,
                "movement": 0.0,
                "agreement": 0.0,
                "method_votes": {},
                "method_summary": "Faltan velas para indicadores técnicos.",
                "reasons": ["Se necesitan al menos 20 velas cerradas."],
                "indicators": {},
                "lateral": False,
            }

        # --------------------------------------------------------
        # CONVERSIÓN DE COORDENADAS DE PANTALLA A PRECIO VISUAL
        # --------------------------------------------------------
        opens = -np.asarray([float(c["open"]) for c in candles], dtype=float)
        closes = -np.asarray([float(c["close"]) for c in candles], dtype=float)
        highs = -np.asarray([float(c["high"]) for c in candles], dtype=float)
        lows = -np.asarray([float(c["low"]) for c in candles], dtype=float)

        body_sizes = np.abs(closes - opens)
        ranges = np.maximum(highs - lows, 1.0)

        n = len(closes)
        last = n - 1

        ema9 = cls._ema(closes, 9)
        ema20 = cls._ema(closes, 20)
        ema50 = cls._ema(closes, 50)

        rsi = cls._rsi_series(closes, 14)

        ema12 = cls._ema(closes, 12)
        ema26 = cls._ema(closes, 26)
        macd = ema12 - ema26
        macd_signal = cls._ema(macd, 9)
        macd_hist = macd - macd_signal

        stoch_k, stoch_d = cls._stochastic_series(
            highs,
            lows,
            closes,
            14,
        )

        atr_series = cls._atr_series(
            highs,
            lows,
            closes,
            14,
        )
        atr = max(float(atr_series[-1]), 1.0)

        bb_period = min(20, n)
        bb_data = closes[-bb_period:]
        bb_mid = float(np.mean(bb_data))
        bb_std = float(np.std(bb_data))
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std
        bb_width = max(bb_upper - bb_lower, 0.0)
        bb_width_series = cls._rolling_bb_width(closes, 20)

        recent_ranges = ranges[-min(20, len(ranges)):]
        median_range = max(float(np.median(recent_ranges)), 1.0)

        # Movimiento conserva la convención anterior:
        # positivo = visualmente subió.
        movement_window = min(12, n)
        movement = float(closes[-1] - closes[-movement_window])

        category_scores = {
            name: {"bull": 0.0, "bear": 0.0}
            for name in cls.METHOD_NAMES
        }
        reasons = []

        def add(method, direction, points, reason=None):
            if method not in category_scores:
                return

            points = float(max(points, 0.0))

            if direction == "SUBIDA":
                category_scores[method]["bull"] += points
            elif direction == "BAJADA":
                category_scores[method]["bear"] += points

            if reason:
                reasons.append(reason)

        # ========================================================
        # 1) PRICE ACTION
        # ========================================================
        o = opens[-1]
        c = closes[-1]
        h = highs[-1]
        l = lows[-1]

        po = opens[-2]
        pc = closes[-2]
        ph = highs[-2]
        pl = lows[-2]

        body = max(abs(c - o), 0.1)
        candle_range = max(h - l, 1.0)
        upper_wick = max(h - max(o, c), 0.0)
        lower_wick = max(min(o, c) - l, 0.0)
        body_ratio = body / candle_range

        bullish = c > o
        bearish = c < o

        prev_bullish = pc > po
        prev_bearish = pc < po

        # Envolvente.
        if (
            prev_bearish
            and bullish
            and min(o, c) <= min(po, pc)
            and max(o, c) >= max(po, pc)
            and body >= abs(pc - po) * 0.90
        ):
            add(
                "Price Action",
                "SUBIDA",
                2.2,
                "Envolvente alcista.",
            )

        if (
            prev_bullish
            and bearish
            and min(o, c) <= min(po, pc)
            and max(o, c) >= max(po, pc)
            and body >= abs(pc - po) * 0.90
        ):
            add(
                "Price Action",
                "BAJADA",
                2.2,
                "Envolvente bajista.",
            )

        # Martillo / pin bar.
        if (
            lower_wick >= body * 1.8
            and upper_wick <= body * 0.9
            and body_ratio <= 0.50
        ):
            add(
                "Price Action",
                "SUBIDA",
                1.7,
                "Rechazo inferior tipo martillo/pin bar.",
            )

        if (
            upper_wick >= body * 1.8
            and lower_wick <= body * 0.9
            and body_ratio <= 0.50
        ):
            add(
                "Price Action",
                "BAJADA",
                1.7,
                "Rechazo superior tipo estrella/pin bar.",
            )

        # Cuerpo fuerte.
        if body_ratio >= 0.62:
            if bullish:
                add(
                    "Price Action",
                    "SUBIDA",
                    1.1,
                    "Última vela con cuerpo alcista fuerte.",
                )
            elif bearish:
                add(
                    "Price Action",
                    "BAJADA",
                    1.1,
                    "Última vela con cuerpo bajista fuerte.",
                )

        # Outside bar.
        outside_bar = h >= ph and l <= pl
        if outside_bar:
            if bullish:
                add(
                    "Price Action",
                    "SUBIDA",
                    0.8,
                    "Outside bar con cierre alcista.",
                )
            elif bearish:
                add(
                    "Price Action",
                    "BAJADA",
                    0.8,
                    "Outside bar con cierre bajista.",
                )

        # Tres velas consecutivas.
        last3_colors = [candle["color"] for candle in candles[-3:]]
        if last3_colors == ["VERDE", "VERDE", "VERDE"]:
            add(
                "Price Action",
                "SUBIDA",
                0.9,
                "Tres cierres verdes consecutivos.",
            )
        elif last3_colors == ["ROJA", "ROJA", "ROJA"]:
            add(
                "Price Action",
                "BAJADA",
                0.9,
                "Tres cierres rojos consecutivos.",
            )

        doji = body_ratio <= 0.12
        inside_bar = h <= ph and l >= pl

        # ========================================================
        # 2) ESTRUCTURA DE MERCADO
        # ========================================================
        slope5 = cls._slope(closes, 5) / atr
        slope10 = cls._slope(closes, 10) / atr
        slope20 = cls._slope(closes, 20) / atr

        if slope10 >= 0.10:
            add(
                "Estructura",
                "SUBIDA",
                1.4,
                "Pendiente de 10 velas alcista.",
            )
        elif slope10 <= -0.10:
            add(
                "Estructura",
                "BAJADA",
                1.4,
                "Pendiente de 10 velas bajista.",
            )

        if slope20 >= 0.06:
            add(
                "Estructura",
                "SUBIDA",
                1.0,
                "Estructura de 20 velas ascendente.",
            )
        elif slope20 <= -0.06:
            add(
                "Estructura",
                "BAJADA",
                1.0,
                "Estructura de 20 velas descendente.",
            )

        # Máximos/mínimos crecientes vs decrecientes.
        recent6_h = highs[-6:]
        recent6_l = lows[-6:]

        old_high = float(np.mean(recent6_h[:3]))
        new_high = float(np.mean(recent6_h[3:]))
        old_low = float(np.mean(recent6_l[:3]))
        new_low = float(np.mean(recent6_l[3:]))

        higher_highs = new_high > old_high + atr * 0.10
        higher_lows = new_low > old_low + atr * 0.10
        lower_highs = new_high < old_high - atr * 0.10
        lower_lows = new_low < old_low - atr * 0.10

        if higher_highs and higher_lows:
            add(
                "Estructura",
                "SUBIDA",
                1.5,
                "Máximos y mínimos recientes crecientes.",
            )

        if lower_highs and lower_lows:
            add(
                "Estructura",
                "BAJADA",
                1.5,
                "Máximos y mínimos recientes decrecientes.",
            )

        # ========================================================
        # 3) SOPORTE / RESISTENCIA
        # ========================================================
        sr_lookback = min(20, n - 1)
        prior_highs = highs[-(sr_lookback + 1):-1]
        prior_lows = lows[-(sr_lookback + 1):-1]

        resistance = float(np.max(prior_highs))
        support = float(np.min(prior_lows))
        sr_tolerance = max(atr * 0.45, median_range * 0.28)

        near_support = abs(l - support) <= sr_tolerance
        near_resistance = abs(h - resistance) <= sr_tolerance

        # Rebote/rechazo.
        if near_support and lower_wick >= body * 0.8 and c > l + candle_range * 0.45:
            add(
                "Soporte/Resistencia",
                "SUBIDA",
                1.7,
                "Rechazo alcista cerca de soporte.",
            )

        if near_resistance and upper_wick >= body * 0.8 and c < h - candle_range * 0.45:
            add(
                "Soporte/Resistencia",
                "BAJADA",
                1.7,
                "Rechazo bajista cerca de resistencia.",
            )

        # Ruptura.
        if c > resistance + atr * 0.12 and bullish and body_ratio >= 0.45:
            add(
                "Soporte/Resistencia",
                "SUBIDA",
                2.0,
                "Ruptura alcista de resistencia.",
            )

        if c < support - atr * 0.12 and bearish and body_ratio >= 0.45:
            add(
                "Soporte/Resistencia",
                "BAJADA",
                2.0,
                "Ruptura bajista de soporte.",
            )

        # Ruptura falsa.
        if h > resistance + atr * 0.08 and c < resistance:
            add(
                "Soporte/Resistencia",
                "BAJADA",
                1.3,
                "Posible falsa ruptura alcista.",
            )

        if l < support - atr * 0.08 and c > support:
            add(
                "Soporte/Resistencia",
                "SUBIDA",
                1.3,
                "Posible falsa ruptura bajista.",
            )

        # Retesteo aproximado.
        if len(closes) >= 3:
            prev_close = closes[-2]
            prev2_close = closes[-3]

            if (
                prev2_close <= resistance
                and prev_close > resistance
                and l <= resistance + sr_tolerance
                and c > resistance
            ):
                add(
                    "Soporte/Resistencia",
                    "SUBIDA",
                    1.2,
                    "Ruptura + retesteo alcista.",
                )

            if (
                prev2_close >= support
                and prev_close < support
                and h >= support - sr_tolerance
                and c < support
            ):
                add(
                    "Soporte/Resistencia",
                    "BAJADA",
                    1.2,
                    "Ruptura + retesteo bajista.",
                )

        # ========================================================
        # 4) TENDENCIA - EMA 9 / 20 / 50
        # ========================================================
        e9 = float(ema9[-1])
        e20 = float(ema20[-1])
        e50 = float(ema50[-1])

        if e9 > e20:
            add(
                "Tendencia EMA",
                "SUBIDA",
                1.1,
                "EMA 9 por encima de EMA 20.",
            )
        elif e9 < e20:
            add(
                "Tendencia EMA",
                "BAJADA",
                1.1,
                "EMA 9 por debajo de EMA 20.",
            )

        # EMA50 solo pesa de verdad cuando existen suficientes velas.
        if n >= 50:
            if e9 > e20 > e50:
                add(
                    "Tendencia EMA",
                    "SUBIDA",
                    1.5,
                    "Alineación EMA 9 > 20 > 50.",
                )
            elif e9 < e20 < e50:
                add(
                    "Tendencia EMA",
                    "BAJADA",
                    1.5,
                    "Alineación EMA 9 < 20 < 50.",
                )
        else:
            if c > e9 > e20:
                add(
                    "Tendencia EMA",
                    "SUBIDA",
                    0.7,
                    "Precio visual sobre EMA 9 y EMA 20.",
                )
            elif c < e9 < e20:
                add(
                    "Tendencia EMA",
                    "BAJADA",
                    0.7,
                    "Precio visual bajo EMA 9 y EMA 20.",
                )

        ema9_slope = cls._slope(ema9, 5) / atr
        ema20_slope = cls._slope(ema20, 5) / atr

        if ema9_slope > 0.05 and ema20_slope > 0.02:
            add(
                "Tendencia EMA",
                "SUBIDA",
                0.8,
                "EMAs con pendiente alcista.",
            )
        elif ema9_slope < -0.05 and ema20_slope < -0.02:
            add(
                "Tendencia EMA",
                "BAJADA",
                0.8,
                "EMAs con pendiente bajista.",
            )

        # ========================================================
        # 5) MOMENTUM - RSI / MACD / ESTOCÁSTICO / DIVERGENCIA
        # ========================================================
        current_rsi = float(rsi[-1])

        if current_rsi >= 55.0:
            add(
                "Momentum",
                "SUBIDA",
                0.9,
                f"RSI alcista ({current_rsi:.0f}).",
            )
        elif current_rsi <= 45.0:
            add(
                "Momentum",
                "BAJADA",
                0.9,
                f"RSI bajista ({current_rsi:.0f}).",
            )

        current_macd = float(macd[-1])
        current_macd_signal = float(macd_signal[-1])
        current_hist = float(macd_hist[-1])

        if current_macd > current_macd_signal and current_hist > 0:
            add(
                "Momentum",
                "SUBIDA",
                1.0,
                "MACD con momentum alcista.",
            )
        elif current_macd < current_macd_signal and current_hist < 0:
            add(
                "Momentum",
                "BAJADA",
                1.0,
                "MACD con momentum bajista.",
            )

        if len(macd_hist) >= 2:
            hist_delta = macd_hist[-1] - macd_hist[-2]
            if current_hist > 0 and hist_delta > 0:
                add(
                    "Momentum",
                    "SUBIDA",
                    0.5,
                    "Histograma MACD acelerando al alza.",
                )
            elif current_hist < 0 and hist_delta < 0:
                add(
                    "Momentum",
                    "BAJADA",
                    0.5,
                    "Histograma MACD acelerando a la baja.",
                )

        current_k = float(stoch_k[-1])
        current_d = float(stoch_d[-1])

        if current_k > current_d + 2.0 and current_k < 85.0:
            add(
                "Momentum",
                "SUBIDA",
                0.7,
                "Estocástico favorable a subida.",
            )
        elif current_k < current_d - 2.0 and current_k > 15.0:
            add(
                "Momentum",
                "BAJADA",
                0.7,
                "Estocástico favorable a bajada.",
            )

        # Cruce desde extremos.
        if len(stoch_k) >= 2:
            if (
                stoch_k[-2] <= stoch_d[-2]
                and current_k > current_d
                and current_k <= 35.0
            ):
                add(
                    "Momentum",
                    "SUBIDA",
                    0.8,
                    "Cruce estocástico desde zona baja.",
                )

            if (
                stoch_k[-2] >= stoch_d[-2]
                and current_k < current_d
                and current_k >= 65.0
            ):
                add(
                    "Momentum",
                    "BAJADA",
                    0.8,
                    "Cruce estocástico desde zona alta.",
                )

        # Divergencia aproximada sobre dos bloques.
        if n >= 16:
            left_slice = slice(-16, -8)
            right_slice = slice(-8, None)

            old_low_idx_local = int(np.argmin(closes[left_slice]))
            new_low_idx_local = int(np.argmin(closes[right_slice]))
            old_high_idx_local = int(np.argmax(closes[left_slice]))
            new_high_idx_local = int(np.argmax(closes[right_slice]))

            old_low_idx = n - 16 + old_low_idx_local
            new_low_idx = n - 8 + new_low_idx_local
            old_high_idx = n - 16 + old_high_idx_local
            new_high_idx = n - 8 + new_high_idx_local

            if (
                closes[new_low_idx] < closes[old_low_idx] - atr * 0.10
                and rsi[new_low_idx] > rsi[old_low_idx] + 3.0
            ):
                add(
                    "Momentum",
                    "SUBIDA",
                    1.0,
                    "Divergencia alcista RSI aproximada.",
                )

            if (
                closes[new_high_idx] > closes[old_high_idx] + atr * 0.10
                and rsi[new_high_idx] < rsi[old_high_idx] - 3.0
            ):
                add(
                    "Momentum",
                    "BAJADA",
                    1.0,
                    "Divergencia bajista RSI aproximada.",
                )

        # ========================================================
        # 6) VOLATILIDAD - BOLLINGER / ATR
        # ========================================================
        atr_reference = max(
            float(np.median(atr_series[-min(20, n):])),
            1.0,
        )
        atr_ratio = atr / atr_reference

        valid_bb_widths = bb_width_series[
            np.isfinite(bb_width_series)
        ]
        previous_bb = valid_bb_widths[-6:-1] if len(valid_bb_widths) >= 6 else valid_bb_widths[:-1]

        bb_expanding = False
        bb_squeeze = False

        if len(previous_bb) >= 2:
            bb_reference = max(float(np.median(previous_bb)), 1.0)
            bb_expanding = bb_width > bb_reference * 1.12
            bb_squeeze = bb_width < bb_reference * 0.78

        if bb_expanding:
            if c > bb_mid and slope5 > 0:
                add(
                    "Volatilidad",
                    "SUBIDA",
                    0.9,
                    "Bandas de Bollinger expandiendo al alza.",
                )
            elif c < bb_mid and slope5 < 0:
                add(
                    "Volatilidad",
                    "BAJADA",
                    0.9,
                    "Bandas de Bollinger expandiendo a la baja.",
                )

        # Reacción en bandas.
        if l <= bb_lower + atr * 0.15 and lower_wick >= body * 0.8:
            add(
                "Volatilidad",
                "SUBIDA",
                0.8,
                "Rechazo cerca de banda inferior.",
            )

        if h >= bb_upper - atr * 0.15 and upper_wick >= body * 0.8:
            add(
                "Volatilidad",
                "BAJADA",
                0.8,
                "Rechazo cerca de banda superior.",
            )

        if atr_ratio >= 1.18 and body_ratio >= 0.55:
            if bullish:
                add(
                    "Volatilidad",
                    "SUBIDA",
                    0.5,
                    "ATR elevado con impulso alcista.",
                )
            elif bearish:
                add(
                    "Volatilidad",
                    "BAJADA",
                    0.5,
                    "ATR elevado con impulso bajista.",
                )

        # ========================================================
        # 7) CONTEXTO MULTIVELA
        # ========================================================
        slope_votes_bull = sum(
            value > threshold
            for value, threshold in (
                (slope5, 0.10),
                (slope10, 0.08),
                (slope20, 0.05),
            )
        )
        slope_votes_bear = sum(
            value < -threshold
            for value, threshold in (
                (slope5, 0.10),
                (slope10, 0.08),
                (slope20, 0.05),
            )
        )

        if slope_votes_bull >= 2:
            add(
                "Contexto",
                "SUBIDA",
                1.2,
                "Ventanas 5/10/20 alineadas al alza.",
            )

        if slope_votes_bear >= 2:
            add(
                "Contexto",
                "BAJADA",
                1.2,
                "Ventanas 5/10/20 alineadas a la baja.",
            )

        last10 = candles[-10:]
        greens10 = sum(candle["color"] == "VERDE" for candle in last10)
        reds10 = sum(candle["color"] == "ROJA" for candle in last10)

        if greens10 >= 7:
            add(
                "Contexto",
                "SUBIDA",
                0.8,
                "Predominio verde en últimas 10.",
            )
        elif reds10 >= 7:
            add(
                "Contexto",
                "BAJADA",
                0.8,
                "Predominio rojo en últimas 10.",
            )

        # ========================================================
        # DETECCIÓN DE LATERALIZACIÓN / INCERTIDUMBRE
        # ========================================================
        ema_gap = abs(e9 - e20) / atr
        lateral = (
            abs(slope10) < 0.055
            and ema_gap < 0.30
            and 46.0 <= current_rsi <= 54.0
        )

        # Doji + inside bar o squeeze reduce convicción.
        uncertainty_penalty = 0.0

        if doji:
            uncertainty_penalty += 0.7
            reasons.append("Doji: indecisión en la última vela.")

        if inside_bar:
            uncertainty_penalty += 0.5
            reasons.append("Inside bar: compresión de precio.")

        if bb_squeeze:
            uncertainty_penalty += 0.6
            reasons.append("Bollinger en compresión.")

        if lateral:
            uncertainty_penalty += 1.2
            reasons.append("Mercado visual lateral.")

        # ========================================================
        # NORMALIZAR PUNTOS POR MÉTODO Y VOTAR
        # ========================================================
        caps = {
            "Price Action": 3.2,
            "Estructura": 3.4,
            "Soporte/Resistencia": 3.5,
            "Tendencia EMA": 3.3,
            "Momentum": 3.6,
            "Volatilidad": 2.2,
            "Contexto": 2.2,
        }

        bull_total = 0.0
        bear_total = 0.0
        method_votes = {}

        for name in cls.METHOD_NAMES:
            cap = caps[name]
            bull_value = min(category_scores[name]["bull"], cap)
            bear_value = min(category_scores[name]["bear"], cap)

            category_scores[name]["bull"] = bull_value
            category_scores[name]["bear"] = bear_value

            bull_total += bull_value
            bear_total += bear_value

            diff = bull_value - bear_value
            method_votes[name] = cls._direction(diff, threshold=0.55)

        # Penalizar únicamente al lado ganador, no sumar artificialmente
        # puntos al sentido opuesto.
        raw_difference = bull_total - bear_total

        if raw_difference > 0:
            effective_difference = raw_difference - uncertainty_penalty
        elif raw_difference < 0:
            effective_difference = raw_difference + uncertainty_penalty
        else:
            effective_difference = 0.0

        directional_votes = [
            vote
            for vote in method_votes.values()
            if vote in ("SUBIDA", "BAJADA")
        ]

        bull_votes = sum(v == "SUBIDA" for v in directional_votes)
        bear_votes = sum(v == "BAJADA" for v in directional_votes)

        winner_direction = None
        winner_votes = 0
        loser_votes = 0
        winner_score = 0.0

        if effective_difference > 0:
            winner_direction = "SUBIDA"
            winner_votes = bull_votes
            loser_votes = bear_votes
            winner_score = bull_total
        elif effective_difference < 0:
            winner_direction = "BAJADA"
            winner_votes = bear_votes
            loser_votes = bull_votes
            winner_score = bear_total

        vote_count = max(len(directional_votes), 1)
        agreement = winner_votes / vote_count if winner_direction else 0.0
        margin = abs(effective_difference)

        # ========================================================
        # DECISIÓN: ENTRAR / ESPERAR / SIN ENTRADA
        # ========================================================
        signal = "NO OPERAR"
        watch_direction = None
        entry_action = "SIN ENTRADA"
        call_put = None

        strong_setup = (
            winner_direction is not None
            and winner_score >= 6.5
            and margin >= 3.0
            and winner_votes >= 4
            and agreement >= 0.64
            and loser_votes <= 2
            and not lateral
        )

        medium_setup = (
            winner_direction is not None
            and winner_score >= 4.5
            and margin >= 1.8
            and winner_votes >= 3
            and agreement >= 0.55
        )

        if strong_setup:
            signal = winner_direction
            watch_direction = winner_direction
            entry_action = "ENTRAR PRÓXIMA VELA"
            call_put = "CALL" if winner_direction == "SUBIDA" else "PUT"

        elif medium_setup:
            watch_direction = winner_direction
            entry_action = "ESPERAR 1 VELA"
            call_put = "CALL" if winner_direction == "SUBIDA" else "PUT"

        # Resumen compacto de métodos.
        arrow_map = {
            "SUBIDA": "↑",
            "BAJADA": "↓",
            "NEUTRAL": "—",
        }
        short_names = {
            "Price Action": "PA",
            "Estructura": "EST",
            "Soporte/Resistencia": "S/R",
            "Tendencia EMA": "EMA",
            "Momentum": "MOM",
            "Volatilidad": "VOL",
            "Contexto": "CTX",
        }

        parts = []
        for name in cls.METHOD_NAMES:
            parts.append(
                f"{short_names[name]} {arrow_map[method_votes[name]]}"
            )

        method_summary = " | ".join(parts)

        indicators = {
            "rsi": current_rsi,
            "macd_hist": current_hist,
            "stoch_k": current_k,
            "stoch_d": current_d,
            "ema9": e9,
            "ema20": e20,
            "ema50": e50,
            "atr": atr,
            "atr_ratio": atr_ratio,
            "bb_mid": bb_mid,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "support": support,
            "resistance": resistance,
            "slope5": slope5,
            "slope10": slope10,
            "slope20": slope20,
        }

        return {
            "signal": signal,
            "watch_direction": watch_direction,
            "entry_action": entry_action,
            "call_put": call_put,
            "bull_score": bull_total,
            "bear_score": bear_total,
            "score": effective_difference,
            "movement": movement,
            "agreement": agreement,
            "bull_votes": bull_votes,
            "bear_votes": bear_votes,
            "method_votes": method_votes,
            "method_summary": method_summary,
            "reasons": reasons,
            "indicators": indicators,
            "lateral": lateral,
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

        # Rendimiento: captura MSS persistente + previews desacoplados del análisis.
        self.sct = None
        self._last_preview_update = 0.0
        self._processing_capture = False

        # Timing de entrada: usa el reloj local para ubicar la vela actual.
        self.confirmation_timestamp = None
        self.confirmed_direction = None
        self.previous_confirmed = False
        self.last_candle_slot = None
        self.decision_timestamp = None
        self.decision_direction = None
        self.live_direction = "ESPERANDO"
        self.live_strength = "—"
        self.last_entry_action = "SIN ENTRADA"
        self.watch_direction = None
        self.last_call_put = None
        self.last_ai_relation = "IGNORADA"
        self.last_confluence_count = 0
        self.last_active_methods = 0
        self.last_agreement = 0.0
        self.last_signal_strength = "—"
        self.details_visible = False
        self.last_detection_quality = 0.0
        self.last_detection_spacing = 0.0

        # Predicción secuencial de la próxima vela.
        self.next_prediction = "SIN SEÑAL"
        self.next_prediction_strength = 0
        self.next_prediction_stability = 0
        self.next_combined_score = 0.0
        self.next_ai_weight = 0.0
        self.next_active_methods = 0
        self.prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)
        self.prediction_score_history = deque(maxlen=PREDICTION_HISTORY_SIZE)
        self.last_candles_snapshot = []

        # Evaluación objetiva de la predicción FINAL.
        # pending_final_prediction: decisión creada en los últimos segundos
        # de la vela actual y destinada a la vela siguiente.
        # active_candle_prediction: decisión que corresponde a la vela que
        # está abierta ahora y que solo se evalúa cuando esa vela cierre.
        self.pending_final_prediction = None
        self.active_candle_prediction = None
        self.prediction_results = deque(maxlen=500)
        self.load_prediction_results()

        self.market_ai = MarketPatternAI()

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
        market_card.setVisible(False)
        left.addWidget(market_card)

        chart_card = Card()
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.setContentsMargins(14, 14, 14, 14)
        chart_layout.setSpacing(10)

        chart_header = QHBoxLayout()
        chart_title = QLabel("ANÁLISIS VISUAL EN RAM")
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

        # Interfaz limpia: ocultamos el encabezado y dejamos únicamente
        # el gráfico de detección.
        chart_title.setVisible(False)
        self.capture_info.setVisible(False)

        previews = QHBoxLayout()
        previews.setSpacing(10)

        # -------- Gráfico limpio --------
        live_box = QVBoxLayout()
        live_box.setSpacing(6)

        live_title_row = QHBoxLayout()
        live_title = QLabel("GRÁFICO EN VIVO")
        live_title.setStyleSheet(
            f"color: {TEXT}; font-size: 9px; font-weight: 800;"
        )
        live_title_row.addWidget(live_title)
        live_title_row.addStretch()

        self.live_preview_info = QLabel("RAM")
        self.live_preview_info.setStyleSheet(
            f"color: {MUTED}; font-size: 8px;"
        )
        live_title_row.addWidget(self.live_preview_info)
        live_box.addLayout(live_title_row)

        self.preview = QLabel("Selecciona la zona del gráfico")
        self.preview.setMinimumHeight(200)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(
            f"""
            background-color: #070a10;
            border: 1px solid {BORDER};
            border-radius: 11px;
            color: {MUTED};
            """
        )
        live_box.addWidget(self.preview, 1)
        self.preview.setVisible(False)
        live_title.setVisible(False)
        self.live_preview_info.setVisible(False)

        # -------- Detección amarilla --------
        detection_box = QVBoxLayout()
        detection_box.setSpacing(6)

        detection_title_row = QHBoxLayout()
        detection_title = QLabel("")
        detection_title.setStyleSheet(
            f"color: {TEXT}; font-size: 9px; font-weight: 800;"
        )
        detection_title_row.addWidget(detection_title)
        detection_title_row.addStretch()

        self.detection_info = QLabel("0 velas · V 0 · R 0")
        self.detection_info.setStyleSheet(
            f"color: {MUTED}; font-size: 8px;"
        )
        detection_title_row.addWidget(self.detection_info)
        detection_box.addLayout(detection_title_row)

        self.detection_preview = QLabel("Esperando detección")
        self.detection_preview.setMinimumHeight(420)
        self.detection_preview.setAlignment(Qt.AlignCenter)
        self.detection_preview.setStyleSheet(
            f"""
            background-color: #070a10;
            border: 1px solid {BORDER};
            border-radius: 11px;
            color: {MUTED};
            """
        )
        detection_box.addWidget(self.detection_preview, 1)

        # Solo añadimos el panel con rectángulos amarillos.
        # El panel limpio NO se añade al layout visible.
        previews.addLayout(detection_box, 1)
        chart_layout.addLayout(previews, 1)

        left.addWidget(chart_card, 1)

        metrics = QHBoxLayout()
        metrics.setSpacing(9)
        self.score_card, self.score_value = self.create_metric("SCORE", "0")
        self.movement_card, self.movement_value = self.create_metric("MOVIMIENTO", "0")
        self.change_card, self.change_value = self.create_metric("CANDLE CHANGE", "0%")
        metrics.addWidget(self.score_card)
        metrics.addWidget(self.movement_card)
        metrics.addWidget(self.change_card)
        self.score_card.setVisible(False)
        self.movement_card.setVisible(False)
        self.change_card.setVisible(False)
        left.addLayout(metrics)

        right = QVBoxLayout()
        right.setSpacing(13)

        signal_card = Card()
        signal_layout = QVBoxLayout(signal_card)
        signal_layout.setContentsMargins(18, 16, 18, 16)
        signal_layout.setSpacing(8)

        signal_title = QLabel("PRÓXIMA VELA")
        signal_title.setStyleSheet(
            f"color: {MUTED}; font-size: 10px; font-weight: 800;"
        )
        signal_layout.addWidget(signal_title)

        self.signal_label = QLabel("SIN SEÑAL")
        self.signal_label.setAlignment(Qt.AlignCenter)
        self.signal_label.setMinimumHeight(92)
        self.signal_label.setWordWrap(True)
        self.signal_label.setStyleSheet(
            f"color: {MUTED}; font-size: 34px; font-weight: 900;"
        )
        signal_layout.addWidget(self.signal_label)

        self.quick_meta_label = QLabel("FUERZA — · ESTABILIDAD —")
        self.quick_meta_label.setAlignment(Qt.AlignCenter)
        self.quick_meta_label.setStyleSheet(
            f"color: {TEXT}; font-size: 11px; font-weight: 800;"
        )
        signal_layout.addWidget(self.quick_meta_label)

        self.validity_label = QLabel("PREDICCIÓN EN FORMACIÓN")
        self.validity_label.setAlignment(Qt.AlignCenter)
        self.validity_label.setStyleSheet(
            f"color: {YELLOW}; font-size: 10px; font-weight: 900;"
        )
        signal_layout.addWidget(self.validity_label)

        self.live_candle_label = QLabel("VELA ACTUAL: ESPERANDO")
        self.live_candle_label.setAlignment(Qt.AlignCenter)
        self.live_candle_label.setStyleSheet(
            f"color: {MUTED}; font-size: 12px; font-weight: 800;"
        )
        signal_layout.addWidget(self.live_candle_label)

        self.details_button = QPushButton("DETALLES")
        self.details_button.setCursor(Qt.PointingHandCursor)
        self.details_button.setFixedHeight(30)
        self.details_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {CARD_2};
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 5px 12px;
                font-size: 9px;
                font-weight: 800;
            }}
            QPushButton:hover {{
                background-color: #202735;
            }}
            """
        )
        self.details_button.clicked.connect(self.toggle_details)
        signal_layout.addWidget(self.details_button)

        self.details_panel = QFrame()
        self.details_panel.setStyleSheet(
            f"""
            QFrame {{
                background-color: #0c1018;
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            """
        )
        details_layout = QVBoxLayout(self.details_panel)
        details_layout.setContentsMargins(10, 9, 10, 9)
        details_layout.setSpacing(6)

        self.ai_label = QLabel("IA: recopilando patrones...")
        self.ai_label.setAlignment(Qt.AlignCenter)
        self.ai_label.setWordWrap(True)
        self.ai_label.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-weight: 700;"
        )
        details_layout.addWidget(self.ai_label)

        self.methods_label = QLabel("MÉTODOS: esperando cierre...")
        self.methods_label.setAlignment(Qt.AlignCenter)
        self.methods_label.setWordWrap(True)
        self.methods_label.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-weight: 700;"
        )
        details_layout.addWidget(self.methods_label)

        self.details_metrics_label = QLabel(
            "Score —   ·   Movimiento —   ·   Cambio —"
        )
        self.details_metrics_label.setAlignment(Qt.AlignCenter)
        self.details_metrics_label.setWordWrap(True)
        self.details_metrics_label.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-weight: 700;"
        )
        details_layout.addWidget(self.details_metrics_label)

        self.accuracy_label = QLabel("PRECISIÓN REAL: todavía sin resultados")
        self.accuracy_label.setAlignment(Qt.AlignCenter)
        self.accuracy_label.setWordWrap(True)
        self.accuracy_label.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-weight: 800;"
        )
        details_layout.addWidget(self.accuracy_label)
        self.update_prediction_accuracy_ui()

        self.confirmation_label = QLabel("ESPERANDO")
        self.confirmation_label.setAlignment(Qt.AlignCenter)
        self.confirmation_label.setStyleSheet(
            f"color: {YELLOW}; font-size: 15px; font-weight: 800;"
        )
        details_layout.addWidget(self.confirmation_label)

        dots = QHBoxLayout()
        dots.setAlignment(Qt.AlignCenter)
        self.confirmation_dots = []
        for i in range(3):
            dot = QLabel("●")
            dot.setAlignment(Qt.AlignCenter)
            dot.setStyleSheet(f"color: {BORDER}; font-size: 15px;")
            self.confirmation_dots.append(dot)
            dots.addWidget(dot)
            if i < 2:
                separator = QLabel("—")
                separator.setStyleSheet(f"color: {BORDER};")
                dots.addWidget(separator)
        details_layout.addLayout(dots)

        self.waiting_label = QLabel("Detalles técnicos disponibles después del cierre.")
        self.waiting_label.setAlignment(Qt.AlignCenter)
        self.waiting_label.setWordWrap(True)
        self.waiting_label.setStyleSheet(
            f"color: {MUTED}; font-size: 9px;"
        )
        details_layout.addWidget(self.waiting_label)

        self.details_panel.setVisible(False)
        signal_layout.addWidget(self.details_panel)

        right.addWidget(signal_card)

        # ---------------- Timing de entrada ----------------
        timing_card = Card()
        timing_layout = QVBoxLayout(timing_card)
        timing_layout.setContentsMargins(15, 14, 15, 14)
        timing_layout.setSpacing(8)

        timing_header = QHBoxLayout()
        timing_title = QLabel("VELA")
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
            timing_values, "CIERRA EN", "00:00"
        )
        self.confirmed_age_value = self.create_timing_value(
            timing_values, "CONFIRMACIÓN", "—"
        )
        self.confirmed_age_value.parentWidget().setVisible(False)
        timing_layout.addLayout(timing_values)

        self.timing_status = QLabel(
            "Esperando una señal confirmada..."
        )
        self.timing_status.setAlignment(Qt.AlignCenter)
        self.timing_status.setStyleSheet(
            f"color: {MUTED}; font-size: 9px; font-weight: 700;"
        )
        self.timing_status.setVisible(False)
        timing_layout.addWidget(self.timing_status)

        self.timing_hint = QLabel("")
        self.timing_hint.setAlignment(Qt.AlignCenter)
        self.timing_hint.setWordWrap(True)
        self.timing_hint.setStyleSheet(
            f"color: {MUTED}; font-size: 8px;"
        )
        self.timing_hint.setVisible(False)
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
        history_card.setVisible(False)
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

        content.addLayout(left, 5)
        content.addLayout(right, 2)
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
        self.last_analysis_label.setVisible(False)
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
        self.confirmed_age_value.setText("—")

        if self.next_prediction in ("SUBIDA", "BAJADA"):
            color = GREEN if self.next_prediction == "SUBIDA" else RED
            if remaining <= FINAL_SECONDS:
                self.timing_status.setText(
                    f"FINAL · PRÓXIMA {self.next_prediction} · "
                    f"{remaining}s"
                )
            else:
                self.timing_status.setText(
                    f"EN FORMACIÓN · PRÓXIMA {self.next_prediction} · "
                    f"{remaining}s"
                )
            self.timing_status.setStyleSheet(
                f"color: {color}; font-size: 9px; font-weight: 800;"
            )
        else:
            self.timing_status.setText(
                f"SIN SEÑAL · faltan {self.format_seconds(remaining)}"
            )
            self.timing_status.setStyleSheet(
                f"color: {MUTED}; font-size: 9px; font-weight: 700;"
            )

    def toggle_details(self):
        self.details_visible = not self.details_visible
        self.details_panel.setVisible(self.details_visible)
        self.details_button.setText(
            "OCULTAR" if self.details_visible else "DETALLES"
        )

    @staticmethod
    def _strength_from_confluence(winner_methods, agreement):
        if winner_methods >= 5 and agreement >= 0.75:
            return "ALTA"
        if winner_methods >= 4:
            return "MEDIA"
        if winner_methods >= 3:
            return "BAJA"
        return "—"

    def update_quick_summary(self):
        if not hasattr(self, "quick_meta_label"):
            return

        if self.last_active_methods <= 0:
            self.quick_meta_label.setText("CONFLUENCIA —/7 · FUERZA —")
            return

        self.quick_meta_label.setText(
            f"CONFLUENCIA {self.last_confluence_count}/7 · "
            f"FUERZA {self.last_signal_strength}"
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
        self.live_direction = "ESPERANDO"
        self.live_strength = "—"
        # Forzamos un primer análisis para mostrar la última vela cerrada.
        self.last_candle_slot = self.get_candle_slot()
        if self.sct is None:
            self.sct = mss.mss()
        self._last_preview_update = 0.0
        self.timer.start(CAPTURE_INTERVAL_MS)
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

        self.status_label.setText("● Monitor activo · análisis en RAM · sin guardar imágenes")
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
        if hasattr(self, "live_candle_label"):
            self.live_candle_label.setText("VELA ACTUAL: DETENIDA")
            self.live_candle_label.setStyleSheet(
                f"color: {MUTED}; font-size: 13px; font-weight: 800;"
            )

        self.last_candles_snapshot = []
        self.prediction_history.clear()
        self.prediction_score_history.clear()
        self.next_prediction = "SIN SEÑAL"
        self.next_prediction_strength = 0
        self.next_prediction_stability = 0

        self.pending_final_prediction = None
        self.active_candle_prediction = None
        self.last_detection_quality = 0.0
        self.last_detection_spacing = 0.0
    # ========================================================
    # CAPTURE
    # ========================================================

    def capture_chart(self):
        if self.chart_rect is None:
            return None

        rect = self.chart_rect

        if self.sct is None:
            self.sct = mss.mss()

        monitor = {
            "left": rect.x(),
            "top": rect.y(),
            "width": rect.width(),
            "height": rect.height(),
        }
        screenshot = self.sct.grab(monitor)
        img = Image.frombytes(
            "RGB",
            screenshot.size,
            screenshot.rgb,
        )

        # IMPORTANTE: análisis 100% en RAM. No guardamos grafico.png.
        return img

    @staticmethod
    def pil_to_pixmap(img, target_size):
        if img is None:
            return QPixmap()

        rgb = img.convert("RGB")
        data = rgb.tobytes("raw", "RGB")

        qimage = QImage(
            data,
            rgb.width,
            rgb.height,
            rgb.width * 3,
            QImage.Format_RGB888,
        ).copy()

        return QPixmap.fromImage(qimage).scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )

    def update_preview(self, img):
        if img is None:
            return

        self.preview.setPixmap(
            self.pil_to_pixmap(img, self.preview.size())
        )
        self.capture_info.setText(
            f"RAM · {img.width} × {img.height} · 0 imágenes guardadas"
        )
        self.live_preview_info.setText(
            f"{img.width} × {img.height}"
        )

    def update_detection_preview(self, img, candidates):
        """
        Dibuja en RAM la captura principal visible y añade una caja
        amarilla y V/R sobre cada vela detectada.

        No guarda PNG ni archivos temporales.
        """
        if img is None:
            return

        debug = img.convert("RGB").copy()
        draw = ImageDraw.Draw(debug)

        green_count = 0
        red_count = 0

        for number, candle in enumerate(candidates, start=1):
            color = candle.get("color", "")
            if color == "VERDE":
                green_count += 1
                short = "V"
            elif color == "ROJA":
                red_count += 1
                short = "R"
            else:
                short = "?"

            x = int(candle.get("x", 0))
            left = int(candle.get("left", max(0, x - 4)))
            right = int(candle.get("right", min(debug.width - 1, x + 4)))
            top = int(candle.get("top", 0))
            bottom = int(candle.get("bottom", debug.height - 1))

            left = max(0, min(left, debug.width - 1))
            right = max(0, min(right, debug.width - 1))
            top = max(0, min(top, debug.height - 1))
            bottom = max(0, min(bottom, debug.height - 1))

            if right < left or bottom < top:
                continue

            draw.rectangle(
                [(left, top), (right, bottom)],
                outline=(255, 255, 0),
                width=2,
            )

            draw.text(
                (left, max(0, top - 11)),
                f"{number}{short}",
                fill=(255, 255, 0),
            )

        self.detection_preview.setPixmap(
            self.pil_to_pixmap(
                debug,
                self.detection_preview.size(),
            )
        )

        self.detection_info.setText(
            f"{len(candidates)} velas · V {green_count} · R {red_count}"
        )

    # ========================================================
    # DECISION DIRECTIONAL AL CIERRE
    # ========================================================

    @staticmethod
    def analyze_next_direction(candles):
        """
        Motor de confluencia:
        Price Action + estructura + soporte/resistencia + EMA + momentum
        + volatilidad + contexto multivela.

        La IA aprendida sigue mostrándose por separado por ahora.
        """
        return TechnicalConfluenceEngine.analyze(candles)

    @staticmethod
    def analyze_live_candle(candles):
        """
        Lee la vela que todavía se está formando.
        Esto es independiente de la decisión tomada al cierre anterior.
        """
        if not candles:
            return {"signal": "ESPERANDO", "score": 0.0, "strength": "—"}

        current = candles[-1]
        score = 2.0 if current["color"] == "VERDE" else -2.0

        body = max(current["body"], 1)
        body_ratio = current["body"] / max(current["height"], 1)

        if body_ratio >= 0.35:
            score += 1.0 if current["color"] == "VERDE" else -1.0

        if len(candles) >= 2:
            previous = candles[-2]
            # Y menor = precio visualmente más alto.
            delta = previous["close"] - current["close"]
            if delta > 4:
                score += 1.5
            elif delta < -4:
                score -= 1.5

        if current["lower_wick"] > body * 1.2:
            score += 0.75
        if current["upper_wick"] > body * 1.2:
            score -= 0.75

        if score >= 2.0:
            signal = "SUBIDA"
        elif score <= -2.0:
            signal = "BAJADA"
        else:
            signal = "MIXTA"

        magnitude = abs(score)
        if magnitude >= 4:
            strength = "ALTA"
        elif magnitude >= 2.5:
            strength = "MEDIA"
        else:
            strength = "BAJA"

        return {
            "signal": signal,
            "score": score,
            "strength": strength,
        }

    @staticmethod
    def _clamp01(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def empirical_signal_reliability(
        self,
        direction,
        strength_hint=50.0,
        stability_hint=50.0,
    ):
        """
        Precisión empírica de señales parecidas ya cerradas.

        Da más peso a resultados recientes y a señales de fuerza/estabilidad
        parecidas. Usa un prior 50/50 para no sobreajustar pocas muestras.
        """
        if direction not in ("SUBIDA", "BAJADA"):
            return 0.50, 0.0

        rows = [
            row for row in list(self.prediction_results)[-180:]
            if row.get("predicted") == direction
            and row.get("result") in ("ACIERTO", "FALLO")
        ]

        if not rows:
            return 0.50, 0.0

        weighted_correct = 0.0
        weighted_total = 0.0
        total_rows = len(rows)

        for index, row in enumerate(rows):
            age = total_rows - 1 - index
            recency_weight = 0.985 ** age

            row_strength = self._safe_float(
                row.get("strength"), 50.0
            )
            row_stability = self._safe_float(
                row.get("stability"), 50.0
            )

            strength_distance = abs(
                row_strength - float(strength_hint)
            )
            stability_distance = abs(
                row_stability - float(stability_hint)
            )

            profile_weight = (
                np.exp(-strength_distance / 38.0)
                * np.exp(-stability_distance / 42.0)
            )

            weight = float(
                recency_weight * (0.35 + 0.65 * profile_weight)
            )
            weighted_total += weight

            if row.get("result") == "ACIERTO":
                weighted_correct += weight

        prior_each_side = 6.0
        reliability = (
            weighted_correct + prior_each_side
        ) / (
            weighted_total + prior_each_side * 2.0
        )

        return float(
            np.clip(reliability, 0.25, 0.75)
        ), float(weighted_total)

    def _sequence_metrics(self, current_score):
        """
        Resume varias lecturas de la vela actual para que la decisión final
        no dependa de una sola captura de 0,5 s.
        """
        previous = list(self.prediction_score_history)[-7:]
        scores = previous + [float(current_score)]
        arr = np.asarray(scores, dtype=float)

        if len(arr) == 1:
            smoothed = float(arr[-1])
            volatility = 0.0
            sign_consistency = 1.0 if abs(arr[-1]) > 0.02 else 0.0
            maturity = 1.0 / 6.0
            return smoothed, volatility, sign_consistency, maturity

        weights = np.linspace(0.65, 1.35, len(arr))
        smoothed = float(np.average(arr, weights=weights))
        volatility = float(np.std(arr))

        reference_sign = (
            1 if smoothed > 0.02
            else -1 if smoothed < -0.02
            else 0
        )
        directional = arr[np.abs(arr) > 0.02]

        if reference_sign == 0 or len(directional) == 0:
            sign_consistency = 0.0
        else:
            signs = np.sign(directional)
            sign_consistency = float(
                np.mean(signs == reference_sign)
            )

        maturity = min(1.0, len(arr) / 6.0)

        return smoothed, volatility, sign_consistency, maturity

    def combine_methods_and_ai_live(
        self,
        technical_result,
        ai_info,
        seconds_remaining,
    ):
        """
        Motor adaptativo MÉTODOS + IA + SECUENCIA + RESULTADOS REALES.

        No usa "X de 7 = decisión". El conteo de métodos es diagnóstico.
        """
        technical_score = float(
            technical_result.get("score", 0.0) or 0.0
        )
        method_votes = technical_result.get("method_votes", {}) or {}

        directional = [
            vote for vote in method_votes.values()
            if vote in ("SUBIDA", "BAJADA")
        ]
        active_methods = len(directional)
        bull_methods = sum(v == "SUBIDA" for v in directional)
        bear_methods = sum(v == "BAJADA" for v in directional)

        if active_methods:
            winner_methods = max(bull_methods, bear_methods)
            method_agreement = winner_methods / active_methods
        else:
            winner_methods = 0
            method_agreement = 0.0

        technical_direction = float(
            np.tanh(technical_score / 5.5)
        )

        if active_methods > 0:
            technical_quality = 0.58 + 0.42 * method_agreement
        else:
            technical_quality = 0.42

        technical_component = (
            technical_direction * technical_quality
        )

        ai_signal = ai_info.get("signal")
        ai_confidence = ai_info.get("confidence")
        ai_accuracy = ai_info.get("validation_accuracy")
        ai_agreement = ai_info.get("model_agreement")
        ai_samples = int(ai_info.get("samples") or 0)

        ai_component = 0.0
        ai_weight = 0.0
        ai_reliability = 0.0

        if (
            ai_signal in ("SUBIDA", "BAJADA")
            and ai_confidence is not None
        ):
            ai_sign = 1.0 if ai_signal == "SUBIDA" else -1.0
            ai_edge = max(
                0.0,
                (float(ai_confidence) - 0.50) * 2.0,
            )

            validation = (
                0.50
                if ai_accuracy is None
                else float(ai_accuracy)
            )
            validation_quality = self._clamp01(
                (validation - 0.50) / 0.18
            )
            sample_quality = self._clamp01(
                (ai_samples - MIN_TRAIN_SAMPLES)
                / max(350 - MIN_TRAIN_SAMPLES, 1)
            )
            agreement_quality = self._clamp01(
                0.0 if ai_agreement is None else ai_agreement
            )

            ai_reliability = (
                0.52 * validation_quality
                + 0.18 * sample_quality
                + 0.30 * agreement_quality
            )

            ai_weight = 0.04 + 0.34 * ai_reliability
            ai_weight = max(0.04, min(0.38, ai_weight))

            ai_component = (
                ai_sign
                * ai_edge
                * (0.30 + 0.70 * ai_reliability)
            )

        technical_weight = 1.0 - ai_weight

        raw_score = (
            technical_component * technical_weight
            + ai_component * ai_weight
        )
        raw_score = max(-1.0, min(1.0, raw_score))

        if (
            abs(technical_component) >= 0.18
            and abs(ai_component) >= 0.18
            and np.sign(technical_component) != np.sign(ai_component)
        ):
            raw_score *= 0.72

        detection_quality = self._clamp01(
            getattr(self, "last_detection_quality", 0.0)
        )
        quality_factor = 0.82 + 0.18 * self._clamp01(
            (detection_quality - 0.50) / 0.50
        )
        raw_score *= quality_factor

        (
            smoothed_score,
            score_volatility,
            sign_consistency,
            maturity,
        ) = self._sequence_metrics(raw_score)

        combined_score = (
            0.60 * raw_score
            + 0.40 * smoothed_score
        )

        if (
            abs(raw_score) > 0.04
            and abs(smoothed_score) > 0.04
            and np.sign(raw_score) != np.sign(smoothed_score)
        ):
            combined_score *= 0.68

        volatility_penalty = self._clamp01(
            score_volatility / 0.24
        )
        combined_score *= (
            1.0 - 0.24 * volatility_penalty
        )

        combined_score = max(
            -1.0,
            min(1.0, combined_score),
        )

        self.prediction_score_history.append(raw_score)

        threshold = (
            FINAL_PREDICTION_THRESHOLD
            if seconds_remaining <= FINAL_SECONDS
            else LIVE_PREDICTION_THRESHOLD
        )

        provisional_direction = (
            "SUBIDA"
            if combined_score > 0
            else "BAJADA"
            if combined_score < 0
            else "SIN SEÑAL"
        )

        preliminary_strength = int(round(
            min(100.0, abs(combined_score) * 220.0)
        ))
        preliminary_stability = int(round(
            100.0
            * self._clamp01(
                sign_consistency
                * maturity
                * (1.0 - 0.30 * volatility_penalty)
            )
        ))

        historical_reliability, historical_samples = (
            self.empirical_signal_reliability(
                provisional_direction,
                preliminary_strength,
                preliminary_stability,
            )
        )

        if historical_samples >= CALIBRATION_MIN_RESULTS:
            if historical_reliability < 0.46:
                threshold *= 1.24
            elif historical_reliability < 0.50:
                threshold *= 1.10
            elif historical_reliability >= 0.60:
                threshold *= 0.92

        if combined_score >= threshold:
            direction = "SUBIDA"
        elif combined_score <= -threshold:
            direction = "BAJADA"
        else:
            direction = "SIN SEÑAL"

        if combined_score > 0.02:
            raw_direction = "SUBIDA"
        elif combined_score < -0.02:
            raw_direction = "BAJADA"
        else:
            raw_direction = "NEUTRAL"

        self.prediction_history.append(raw_direction)

        history = [
            item for item in self.prediction_history
            if item in ("SUBIDA", "BAJADA")
        ]

        if direction in ("SUBIDA", "BAJADA") and history:
            same = sum(item == direction for item in history)
            direction_consistency = same / len(history)
        elif history:
            dominant = max(
                history.count("SUBIDA"),
                history.count("BAJADA"),
            )
            direction_consistency = dominant / len(history)
        else:
            direction_consistency = 0.0

        stability = self._clamp01(
            (
                0.62 * direction_consistency
                + 0.38 * sign_consistency
            )
            * maturity
            * (1.0 - 0.28 * volatility_penalty)
        )

        if (
            seconds_remaining <= FINAL_SECONDS
            and direction in ("SUBIDA", "BAJADA")
            and stability < MIN_FINAL_STABILITY
        ):
            direction = "SIN SEÑAL"

        edge = self._clamp01(
            (abs(combined_score) - threshold)
            / max(0.34 - threshold, 0.08)
        )

        history_quality = (
            historical_reliability
            if historical_samples >= CALIBRATION_MIN_RESULTS
            else 0.50
        )

        strength = int(round(
            100.0 * self._clamp01(
                0.52 * edge
                + 0.24 * stability
                + 0.14 * detection_quality
                + 0.10 * self._clamp01(
                    (history_quality - 0.35) / 0.35
                )
            )
        ))

        return {
            "direction": direction,
            "combined_score": combined_score,
            "raw_score": raw_score,
            "smoothed_score": smoothed_score,
            "score_volatility": score_volatility,
            "strength": strength,
            "stability": int(round(stability * 100.0)),
            "active_methods": active_methods,
            "winner_methods": winner_methods,
            "method_agreement": method_agreement,
            "technical_component": technical_component,
            "technical_score": technical_score,
            "ai_component": ai_component,
            "ai_weight": ai_weight,
            "ai_reliability": ai_reliability,
            "historical_reliability": historical_reliability,
            "historical_samples": historical_samples,
            "threshold": threshold,
            "ai_info": ai_info,
            "method_summary": technical_result.get(
                "method_summary",
                "",
            ),
            "movement": float(
                technical_result.get("movement", 0.0) or 0.0
            ),
            "bull_score": float(
                technical_result.get("bull_score", 0.0) or 0.0
            ),
            "bear_score": float(
                technical_result.get("bear_score", 0.0) or 0.0
            ),
        }

    def update_next_prediction_ui(
        self,
        prediction,
        seconds_remaining,
    ):
        direction = prediction["direction"]
        strength = prediction["strength"]
        stability = prediction["stability"]
        active_methods = prediction["active_methods"]
        ai_info = prediction["ai_info"]
        ai_signal = ai_info.get("signal")
        ai_confidence = ai_info.get("confidence")
        ai_accuracy = ai_info.get("validation_accuracy")
        ai_weight = prediction["ai_weight"]

        self.next_prediction = direction
        self.next_prediction_strength = strength
        self.next_prediction_stability = stability
        self.next_combined_score = prediction["combined_score"]
        self.next_ai_weight = ai_weight
        self.next_active_methods = active_methods

        final_phase = seconds_remaining <= FINAL_SECONDS

        if direction == "SUBIDA":
            color = GREEN
            self.signal_label.setText("SUBIDA ↑")
        elif direction == "BAJADA":
            color = RED
            self.signal_label.setText("BAJADA ↓")
        else:
            color = MUTED
            self.signal_label.setText("SIN SEÑAL")

        self.signal_label.setStyleSheet(
            f"color: {color}; font-size: 26px; font-weight: 900;"
        )

        quality_pct = int(round(self.last_detection_quality * 100.0))
        self.quick_meta_label.setText(
            f"FUERZA {strength}/100 · "
            f"ESTABILIDAD {stability}% · "
            f"LECTURA {quality_pct}%"
        )

        if final_phase:
            self.validity_label.setText(
                f"PREDICCIÓN FINAL · {seconds_remaining}s"
            )
            self.validity_label.setStyleSheet(
                f"color: {color if direction != 'SIN SEÑAL' else YELLOW}; "
                "font-size: 10px; font-weight: 900;"
            )
        else:
            self.validity_label.setText(
                f"PREDICCIÓN EN FORMACIÓN · {seconds_remaining}s"
            )
            self.validity_label.setStyleSheet(
                f"color: {TEXT}; font-size: 10px; font-weight: 800;"
            )

        if ai_signal in ("SUBIDA", "BAJADA") and ai_confidence is not None:
            acc_text = (
                "—"
                if ai_accuracy is None
                else f"{ai_accuracy * 100:.0f}%"
            )
            self.ai_label.setText(
                f"IA: {ai_signal} · "
                f"confianza {ai_confidence * 100:.0f}% · "
                f"validación {acc_text} · "
                f"peso en decisión {ai_weight * 100:.0f}% · "
                f"muestras {ai_info.get('samples', 0)}"
            )
            ai_color = GREEN if ai_signal == "SUBIDA" else RED
            self.ai_label.setStyleSheet(
                f"color: {ai_color}; font-size: 9px; font-weight: 800;"
            )
        else:
            self.update_ai_label(ai_info)

        hist_rel = prediction.get("historical_reliability", 0.50)
        hist_samples = prediction.get("historical_samples", 0.0)

        calibration_text = (
            f"calibración {hist_rel * 100:.0f}%"
            if hist_samples >= CALIBRATION_MIN_RESULTS
            else "calibración: reuniendo resultados"
        )

        self.methods_label.setText(
            f"MÉTODOS: score técnico {prediction['technical_score']:+.2f} · "
            f"Alcista {prediction['bull_score']:.1f} · "
            f"Bajista {prediction['bear_score']:.1f} · "
            f"activos {active_methods}/7\n"
            f"{prediction['method_summary']} · "
            f"MÉTODOS + IA = {direction} · {calibration_text}"
        )

        self.methods_label.setStyleSheet(
            f"color: {color}; font-size: 9px; font-weight: 700;"
        )

        self.score_value.setText(
            f"{prediction['combined_score'] * 100:+.0f}"
        )
        self.movement_value.setText(
            f"{prediction['movement']:+.0f}"
        )

        if hasattr(self, "details_metrics_label"):
            self.details_metrics_label.setText(
                f"Score {prediction['combined_score'] * 100:+.0f}   ·   "
                f"Movimiento {prediction['movement']:+.0f}   ·   "
                f"Cambio {self.change_value.text()}"
            )

    # ========================================================
    # MEDICIÓN REAL DE PREDICCIONES
    # ========================================================

    def load_prediction_results(self):
        self.prediction_results.clear()

        if not os.path.exists(PREDICTION_RESULTS_PATH):
            return

        try:
            with open(
                PREDICTION_RESULTS_PATH,
                "r",
                newline="",
                encoding="utf-8",
            ) as file:
                reader = csv.DictReader(file)
                rows = list(reader)

            for row in rows[-500:]:
                predicted = row.get("predicted")
                actual = row.get("actual")
                result = row.get("result")

                if (
                    predicted in ("SUBIDA", "BAJADA")
                    and actual in ("SUBIDA", "BAJADA")
                    and result in ("ACIERTO", "FALLO")
                ):
                    self.prediction_results.append({
                        "predicted": predicted,
                        "actual": actual,
                        "result": result,
                        "strength": row.get("strength", ""),
                        "stability": row.get("stability", ""),
                        "score": row.get("score", ""),
                        "ai_weight": row.get("ai_weight", ""),
                        "active_methods": row.get("active_methods", ""),
                        "timestamp": row.get("timestamp", ""),
                    })
        except Exception:
            # Un CSV viejo o incompleto nunca debe impedir abrir la app.
            self.prediction_results.clear()

    def record_prediction_result(self, prediction, actual_direction):
        predicted = prediction.get("direction")

        if predicted not in ("SUBIDA", "BAJADA"):
            return

        if actual_direction not in ("SUBIDA", "BAJADA"):
            return

        result = (
            "ACIERTO"
            if predicted == actual_direction
            else "FALLO"
        )

        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predicted": predicted,
            "actual": actual_direction,
            "result": result,
            "strength": int(prediction.get("strength", 0) or 0),
            "stability": int(prediction.get("stability", 0) or 0),
            "score": float(prediction.get("score", 0.0) or 0.0),
            "ai_weight": float(prediction.get("ai_weight", 0.0) or 0.0),
            "active_methods": int(
                prediction.get("active_methods", 0) or 0
            ),
        }

        self.prediction_results.append(row)

        exists = os.path.exists(PREDICTION_RESULTS_PATH)
        try:
            with open(
                PREDICTION_RESULTS_PATH,
                "a",
                newline="",
                encoding="utf-8",
            ) as file:
                fieldnames = [
                    "timestamp",
                    "predicted",
                    "actual",
                    "result",
                    "strength",
                    "stability",
                    "score",
                    "ai_weight",
                    "active_methods",
                ]
                writer = csv.DictWriter(
                    file,
                    fieldnames=fieldnames,
                )
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception:
            pass

        self.update_prediction_accuracy_ui()

    def update_prediction_accuracy_ui(self):
        if not hasattr(self, "accuracy_label"):
            return

        rows = list(self.prediction_results)
        total = len(rows)

        if total == 0:
            self.accuracy_label.setText(
                "PRECISIÓN REAL: todavía sin resultados"
            )
            self.accuracy_label.setStyleSheet(
                f"color: {MUTED}; font-size: 9px; font-weight: 800;"
            )
            return

        correct = sum(
            row.get("result") == "ACIERTO"
            for row in rows
        )
        overall = correct / total * 100.0

        recent = rows[-20:]
        recent_correct = sum(
            row.get("result") == "ACIERTO"
            for row in recent
        )
        recent_accuracy = (
            recent_correct / len(recent) * 100.0
            if recent
            else 0.0
        )

        up_rows = [
            row for row in rows
            if row.get("predicted") == "SUBIDA"
        ]
        down_rows = [
            row for row in rows
            if row.get("predicted") == "BAJADA"
        ]

        up_accuracy = (
            sum(r.get("result") == "ACIERTO" for r in up_rows)
            / len(up_rows) * 100.0
            if up_rows
            else None
        )
        down_accuracy = (
            sum(r.get("result") == "ACIERTO" for r in down_rows)
            / len(down_rows) * 100.0
            if down_rows
            else None
        )

        up_text = (
            "—"
            if up_accuracy is None
            else f"{up_accuracy:.0f}%"
        )
        down_text = (
            "—"
            if down_accuracy is None
            else f"{down_accuracy:.0f}%"
        )

        self.accuracy_label.setText(
            f"PRECISIÓN REAL: {correct}/{total} · {overall:.1f}%"
            f"   |   últimas {len(recent)}: {recent_accuracy:.1f}%"
            f"\nSUBIDA {up_text}   ·   BAJADA {down_text}"
        )

        accuracy_color = (
            GREEN if overall >= 55.0
            else YELLOW if overall >= 48.0
            else RED
        )
        self.accuracy_label.setStyleSheet(
            f"color: {accuracy_color}; font-size: 9px; font-weight: 800;"
        )

    @staticmethod
    def candle_result_direction(candles):
        if not candles:
            return None

        color = candles[-1].get("color")
        if color == "VERDE":
            return "SUBIDA"
        if color == "ROJA":
            return "BAJADA"
        return None

    # ========================================================
    # IA DE PATRONES
    # ========================================================

    def update_ai_label(self, info):
        if not hasattr(self, "ai_label"):
            return

        samples = int(info.get("samples") or 0)
        signal = info.get("signal")
        confidence = info.get("confidence")
        accuracy = info.get("validation_accuracy")
        model_agreement = info.get("model_agreement")
        trusted = bool(info.get("trusted", False))
        status = info.get("status") or ""
        votes = info.get("model_votes") or {}
        reasons = info.get("trust_reasons") or []

        if signal in ("SUBIDA", "BAJADA") and confidence is not None:
            direction_color = GREEN if signal == "SUBIDA" else RED
            conf_pct = confidence * 100.0
            acc_pct = None if accuracy is None else accuracy * 100.0
            agree_pct = 0.0 if model_agreement is None else model_agreement * 100.0

            vote_text = " ".join(
                f"{name}:{'↑' if vote == 'SUBIDA' else '↓'}"
                for name, vote in votes.items()
            )

            if trusted:
                text = (
                    f"IA ENSAMBLE: {signal} · confianza {conf_pct:.0f}% · "
                    f"acuerdo modelos {agree_pct:.0f}% · muestras {samples}"
                )
                if acc_pct is not None:
                    text += f" · validación {acc_pct:.0f}%"
                if vote_text:
                    text += f"\n{vote_text} · PARTICIPA EN LA DECISIÓN"
                self.ai_label.setStyleSheet(
                    f"color: {direction_color}; font-size: 9px; font-weight: 800;"
                )
            else:
                reason_text = " · ".join(reasons[:3]) if reasons else "todavía sin confianza suficiente"
                text = (
                    f"IA APRENDIENDO: lectura {signal} · confianza {conf_pct:.0f}% · "
                    f"acuerdo modelos {agree_pct:.0f}% · muestras {samples}"
                )
                if acc_pct is not None:
                    text += f" · validación {acc_pct:.0f}%"
                text += f"\nNO PARTICIPA: {reason_text}"
                self.ai_label.setStyleSheet(
                    f"color: {YELLOW}; font-size: 9px; font-weight: 800;"
                )

            self.ai_label.setText(text)
            return

        if samples < MIN_TRAIN_SAMPLES:
            self.ai_label.setText(
                f"IA APRENDIENDO: {samples}/{MIN_TRAIN_SAMPLES} patrones · "
                "ensamble todavía no disponible"
            )
            self.ai_label.setStyleSheet(
                f"color: {YELLOW}; font-size: 9px; font-weight: 700;"
            )
        else:
            text = f"IA: {status} · muestras {samples}"
            if accuracy is not None:
                text += f" · validación {accuracy * 100:.0f}%"
            text += " · NO PARTICIPA todavía"
            self.ai_label.setText(text)
            self.ai_label.setStyleSheet(
                f"color: {MUTED}; font-size: 9px; font-weight: 700;"
            )

    @staticmethod
    def combine_technical_and_ai(result, ai_info):
        """
        Combina métodos técnicos + IA sin dejar que una IA débil mande.

        Reglas conservadoras:
        - IA no confiable: no altera la decisión técnica.
        - IA confiable y de acuerdo: refuerza, pero no inventa una entrada.
        - IA confiable y en contra de una entrada técnica: rebaja a ESPERAR.
        - Si métodos estaban en ESPERAR e IA confiable coincide, sigue ESPERAR;
          se exige que los métodos técnicos por sí solos alcancen ENTRAR.
        """
        combined = dict(result)
        entry_action = combined.get("entry_action", "SIN ENTRADA")
        watch_direction = combined.get("watch_direction")
        signal = combined.get("signal", "NO OPERAR")

        ai_signal = ai_info.get("signal")
        ai_trusted = bool(ai_info.get("trusted", False))

        combined["ai_relation"] = "IGNORADA"
        combined["final_reason"] = "Decisión basada en métodos técnicos."

        if not ai_trusted or ai_signal not in ("SUBIDA", "BAJADA"):
            return combined

        # Si la IA ya es confiable, sí puede frenar una contradicción.
        if watch_direction in ("SUBIDA", "BAJADA"):
            if ai_signal == watch_direction:
                combined["ai_relation"] = "ACUERDO"
                combined["final_reason"] = "Métodos e IA confiable coinciden."
                return combined

            combined["ai_relation"] = "CONFLICTO"
            combined["final_reason"] = "IA confiable contradice los métodos: se evita entrada inmediata."

            # Nunca mantener una entrada fuerte si una IA ya validada contradice.
            if entry_action == "ENTRAR PRÓXIMA VELA":
                combined["entry_action"] = "ESPERAR 1 VELA"
                combined["signal"] = "NO OPERAR"
                combined["watch_direction"] = watch_direction
            return combined

        # Si los métodos no tienen dirección, la IA sola NO crea una entrada.
        combined["ai_relation"] = "SOLO IA"
        combined["final_reason"] = "La IA tiene dirección, pero falta confluencia técnica."
        return combined

    # ========================================================
    # PROCESS CAPTURE
    # ========================================================

    def process_capture(self):
        if not self.running or self._processing_capture:
            return

        self._processing_capture = True

        try:
            img = self.capture_chart()
            if img is None:
                return

            # El análisis sigue a 0,5 s. Los dos paneles visuales se
            # actualizan juntos cada 1 s para no cargar la interfaz.
            now_perf = time.monotonic()

            frame = np.asarray(img)

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

            # Detectar velas en TODAS las capturas para poder leer también
            # la vela que todavía está formándose.
            candidates = find_candle_candidates(
                green_mask,
                red_mask,
            )
            candidates = merge_close_candidates(candidates)
            candidates = filter_candidates(candidates)

            detection_quality = assess_detection_quality(
                candidates,
                frame_shape=green_mask.shape,
                details=True,
            )
            self.last_detection_quality = float(
                detection_quality.get("quality", 0.0) or 0.0
            )
            self.last_detection_spacing = float(
                detection_quality.get("spacing", 0.0) or 0.0
            )

            # Mismo frame, mismo instante:
            # izquierda = captura limpia; derecha = lectura del detector.
            if (now_perf - self._last_preview_update) * 1000 >= PREVIEW_INTERVAL_MS:
                self.update_preview(img)
                self.update_detection_preview(img, candidates)

                q_pct = int(round(self.last_detection_quality * 100.0))
                spacing_text = (
                    f"{self.last_detection_spacing:.1f}px"
                    if self.last_detection_spacing > 0
                    else "—"
                )
                base_info = self.detection_info.text().split(" · Q ")[0]
                self.detection_info.setText(
                    f"{base_info} · Q {q_pct}% · escala {spacing_text}"
                )

                self._last_preview_update = now_perf

            # Una lectura visual mala no debe generar señales ni ejemplos
            # de entrenamiento. Así evitamos que la IA aprenda errores del
            # detector cuando el zoom/captura todavía se está estabilizando.
            if self.last_detection_quality < MIN_DETECTION_QUALITY:
                q_pct = int(round(self.last_detection_quality * 100.0))
                self.market_status.setText(
                    f"LECTURA INESTABLE · {q_pct}% · recalibrando escala"
                )
                self.live_candle_label.setText(
                    "VELA ACTUAL: LECTURA INESTABLE"
                )
                self.signal_label.setText("SIN SEÑAL")
                self.validity_label.setText("RECALIBRANDO GRÁFICO")
                return

            candles = []
            for candidate in candidates:
                candle = get_candle_data(
                    green_mask,
                    red_mask,
                    candidate["x"],
                    candidate.get("left"),
                    candidate.get("right"),
                )
                if candle is not None:
                    candles.append(candle)

            if len(candles) < 10:
                self.market_status.setText(
                    f"Detectando velas... {len(candles)}"
                )
                self.live_candle_label.setText(
                    "VELA ACTUAL: ESPERANDO DATOS"
                )
                self.update_ai_label({
                    "samples": self.market_ai.sample_count,
                    "signal": None,
                    "status": "recopilando patrones",
                    "validation_accuracy": self.market_ai.last_validation_accuracy,
                })
                self.update_timing_view()
                return

            # ----------------------------------------------------
            # 1) LECTURA SIMULTÁNEA DE LA VELA ACTUAL
            # ----------------------------------------------------
            live = self.analyze_live_candle(candles)
            self.live_direction = live["signal"]
            self.live_strength = live["strength"]

            if live["signal"] == "SUBIDA":
                self.live_candle_label.setText(
                    f"VELA ACTUAL: SUBIDA · {live['strength']}"
                )
                self.live_candle_label.setStyleSheet(
                    f"color: {GREEN}; font-size: 13px; font-weight: 800;"
                )
            elif live["signal"] == "BAJADA":
                self.live_candle_label.setText(
                    f"VELA ACTUAL: BAJADA · {live['strength']}"
                )
                self.live_candle_label.setStyleSheet(
                    f"color: {RED}; font-size: 13px; font-weight: 800;"
                )
            else:
                self.live_candle_label.setText(
                    "VELA ACTUAL: MIXTA"
                )
                self.live_candle_label.setStyleSheet(
                    f"color: {YELLOW}; font-size: 13px; font-weight: 800;"
                )

            # ----------------------------------------------------
            # 2) APRENDIZAJE AL CIERRE + PREDICCIÓN SECUENCIAL
            # ----------------------------------------------------
            interval = self.get_candle_interval()
            now = datetime.now()
            total_seconds = (
                now.hour * 3600
                + now.minute * 60
                + now.second
                + now.microsecond / 1_000_000
            )
            elapsed = int(total_seconds % interval)
            seconds_remaining = max(0, interval - elapsed)
            if elapsed == 0:
                seconds_remaining = interval

            current_slot = self.get_candle_slot()
            new_candle = current_slot != self.last_candle_slot

            if new_candle:
                # La última vela de la captura ya es la nueva en formación.
                # Todas las anteriores están cerradas y pueden etiquetar IA.
                self.last_candle_slot = current_slot

                # Usar la última captura del slot ANTERIOR. Así la IA aprende
                # con la vela realmente cerrada, no con la nueva que apenas
                # comienza a aparecer.
                if len(self.last_candles_snapshot) >= WINDOW:
                    closed_candles = [
                        dict(c) for c in self.last_candles_snapshot
                    ]
                else:
                    closed_candles = candles[:-1]

                # Primero evaluamos la predicción que correspondía a la
                # vela que acaba de cerrar.
                actual_direction = self.candle_result_direction(
                    closed_candles
                )

                if (
                    self.active_candle_prediction is not None
                    and actual_direction in ("SUBIDA", "BAJADA")
                ):
                    self.record_prediction_result(
                        self.active_candle_prediction,
                        actual_direction,
                    )

                # La predicción FINAL creada durante la vela que acaba de
                # cerrar ahora pasa a ser la predicción de la nueva vela.
                self.active_candle_prediction = (
                    dict(self.pending_final_prediction)
                    if self.pending_final_prediction is not None
                    else None
                )
                self.pending_final_prediction = None

                self.prediction_history.clear()
                self.prediction_score_history.clear()

                # APRENDER SIEMPRE al cierre:
                # 1) etiqueta el patrón anterior con la vela recién cerrada;
                # 2) agrega una muestra;
                # 3) reentrena cuando corresponde;
                # 4) prepara el patrón cerrado para etiquetarlo en el
                #    próximo cierre.
                ai_close_info = self.market_ai.on_candle_close(
                    closed_candles
                )
                self.update_ai_label(ai_close_info)

                self.last_analysis_label.setText(
                    "Último aprendizaje: "
                    + datetime.now().strftime("%H:%M:%S")
                )

            # PREDICCIÓN DE LA PRÓXIMA VELA usando también la vela que está
            # abierta ahora. Se recalcula cada 0,5 s.
            technical_live = self.analyze_next_direction(candles)
            ai_live = self.market_ai.predict_live(candles)

            prediction = self.combine_methods_and_ai_live(
                technical_live,
                ai_live,
                seconds_remaining,
            )

            self.update_next_prediction_ui(
                prediction,
                seconds_remaining,
            )

            self.analysis_count += 1
            self.last_signal = (
                prediction["direction"]
                if prediction["direction"] in ("SUBIDA", "BAJADA")
                else "NO OPERAR"
            )
            self.last_score = (
                prediction["combined_score"] * 100.0
            )
            self.last_movement = prediction["movement"]

            # Estado breve; los detalles quedan detrás de VER DETALLES.
            ai_signal = ai_live.get("signal") or "—"
            self.market_status.setText(
                f"{prediction['direction']} · "
                f"{prediction['strength']}/100 · "
                f"IA {ai_signal}"
            )

            # Guardar la decisión FINAL destinada a la vela siguiente.
            # Se actualiza en cada lectura de los últimos segundos para que
            # la última lectura válida sea la que realmente cuenta.
            if seconds_remaining <= FINAL_SECONDS:
                if (
                    prediction["direction"] in ("SUBIDA", "BAJADA")
                    and prediction["stability"] >= int(MIN_FINAL_STABILITY * 100)
                ):
                    self.pending_final_prediction = {
                        "direction": prediction["direction"],
                        "strength": prediction["strength"],
                        "stability": prediction["stability"],
                        "score": prediction["combined_score"],
                        "ai_weight": prediction["ai_weight"],
                        "active_methods": prediction["active_methods"],
                        "timestamp": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                    self.decision_direction = prediction["direction"]
                    self.decision_timestamp = datetime.now()
                else:
                    # Si la lectura final pierde estabilidad o pasa a
                    # SIN SEÑAL, no conservamos una predicción vieja.
                    self.pending_final_prediction = None
                    self.decision_direction = None
                    self.decision_timestamp = None
            else:
                self.decision_direction = None
                self.decision_timestamp = None

            # Guardar esta captura como referencia para el próximo cierre.
            self.last_candles_snapshot = [
                dict(c) for c in candles
            ]

            self.update_timing_view()

        except Exception as error:
            self.status_label.setText(
                f"● Error: {error}"
            )
            self.status_label.setStyleSheet(
                f"color: {RED}; font-size: 9px;"
            )
        finally:
            self._processing_capture = False

    # ========================================================
    # SIGNAL UI
    # ========================================================

    def show_signal(
        self,
        signal,
        is_new_candle,
        entry_action=None,
        watch_direction=None,
        call_put=None,
        final_reason=None,
    ):
        entry_action = entry_action or "SIN ENTRADA"

        if signal == "NO OPERAR":
            if entry_action == "ESPERAR 1 VELA" and watch_direction:
                direction_color = GREEN if watch_direction == "SUBIDA" else RED
                self.signal_label.setText(f"ESPERAR · {watch_direction}")
                self.signal_label.setStyleSheet(
                    f"color: {YELLOW}; font-size: 24px; font-weight: 800;"
                )
                self.confirmation_label.setText(
                    f"SESGO {watch_direction}"
                )
                self.confirmation_label.setStyleSheet(
                    f"color: {direction_color}; font-size: 18px; font-weight: 800;"
                )
                self.waiting_label.setText(
                    final_reason or
                    "Hay dirección, pero falta confluencia para una entrada inmediata."
                )
                self.waiting_label.setStyleSheet(
                    f"color: {YELLOW}; font-size: 9px; font-weight: 700;"
                )
            else:
                self.signal_label.setText("SIN ENTRADA")
                self.signal_label.setStyleSheet(
                    f"color: {MUTED}; font-size: 29px; font-weight: 800;"
                )
                self.confirmation_label.setText("SIN CONFLUENCIA")
                self.confirmation_label.setStyleSheet(
                    f"color: {MUTED}; font-size: 18px; font-weight: 800;"
                )
                self.waiting_label.setText(
                    final_reason or
                    "No hay suficiente acuerdo entre métodos para una entrada."
                )
                self.waiting_label.setStyleSheet(
                    f"color: {MUTED}; font-size: 10px; font-weight: 600;"
                )

        elif signal == "SUBIDA":
            self.signal_label.setText("SUBIDA")
            self.signal_label.setStyleSheet(
                f"color: {GREEN}; font-size: 22px; font-weight: 800;"
            )
            self.confirmation_label.setText("CALL · CONFLUENCIA")
            self.confirmation_label.setStyleSheet(
                f"color: {GREEN}; font-size: 18px; font-weight: 800;"
            )
            self.waiting_label.setText(
                final_reason or
                "Cierre analizado con confluencia técnica suficiente."
            )
            self.waiting_label.setStyleSheet(
                f"color: {GREEN}; font-size: 9px; font-weight: 800;"
            )

        elif signal == "BAJADA":
            self.signal_label.setText("BAJADA")
            self.signal_label.setStyleSheet(
                f"color: {RED}; font-size: 22px; font-weight: 800;"
            )
            self.confirmation_label.setText("PUT · CONFLUENCIA")
            self.confirmation_label.setStyleSheet(
                f"color: {RED}; font-size: 18px; font-weight: 800;"
            )
            self.waiting_label.setText(
                final_reason or
                "Cierre analizado con confluencia técnica suficiente."
            )
            self.waiting_label.setStyleSheet(
                f"color: {RED}; font-size: 9px; font-weight: 800;"
            )

        for dot in self.confirmation_dots:
            if signal == "SUBIDA":
                color = GREEN
            elif signal == "BAJADA":
                color = RED
            elif entry_action == "ESPERAR 1 VELA":
                color = YELLOW
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
        if self.sct is not None:
            try:
                self.sct.close()
            except Exception:
                pass
            self.sct = None
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
