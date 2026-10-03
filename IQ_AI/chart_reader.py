import os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

IMAGE_PATH = os.path.join("data", "grafico.png")
DEBUG_PATH = os.path.join("data", "debug_candles.png")

# Compatibilidad con el monitor actual.
ROI_LEFT = 25
ROI_TOP = 70
ROI_RIGHT = 680
ROI_BOTTOM = 535

# Estado pequeño para que la escala no "salte" entre capturas.
# Si detectamos un cambio real de zoom, se actualiza inmediatamente.
_SCALE_STATE = {
    "spacing": None,
    "frame_width": None,
}


# ============================================================
# COLOR
# ============================================================

def load_image():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"No se encontró: {IMAGE_PATH}")
    return Image.open(IMAGE_PATH).convert("RGB")


def detect_colors(image):
    """
    Detector de color v11.

    Mantiene umbrales suficientemente amplios para las velas de IQ Option,
    pero exige dominancia clara del canal verde o rojo para evitar grises,
    textos y rejilla.
    """
    arr = np.asarray(image)

    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("La imagen debe ser RGB.")

    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = max_c - min_c

    green = (
        (g >= 68)
        & (g - r >= 15)
        & (g - b >= 7)
        & (sat >= 24)
    )

    red = (
        (r >= 112)
        & (r - g >= 28)
        & (r - b >= 20)
        & (sat >= 36)
    )

    return green.astype(bool), red.astype(bool)


# ============================================================
# LIMPIEZA DE UI
# ============================================================

def _longest_run_1d(values):
    idx = np.flatnonzero(values)

    if len(idx) == 0:
        return 0
    if len(idx) == 1:
        return 1

    gaps = np.diff(idx)
    starts = np.r_[0, np.where(gaps > 1)[0] + 1]
    ends = np.r_[np.where(gaps > 1)[0], len(idx) - 1]
    lengths = idx[ends] - idx[starts] + 1

    return int(lengths.max())


def _groups_from_active(active):
    labeled, count = ndimage.label(active.astype(np.uint8))
    result = []

    for label_id in range(1, count + 1):
        xs = np.where(labeled == label_id)[0]

        if len(xs) == 0:
            continue

        result.append((int(xs.min()), int(xs.max())))

    return result


def _remove_tiny_components(mask, min_pixels=2):
    labeled, count = ndimage.label(mask.astype(bool))

    if count == 0:
        return mask.astype(bool)

    sizes = ndimage.sum(
        mask,
        labeled,
        index=np.arange(1, count + 1),
    )

    keep = np.where(sizes >= min_pixels)[0] + 1

    if len(keep) == 0:
        return np.zeros_like(mask, dtype=bool)

    return np.isin(labeled, keep)


def _remove_long_vertical_ui_lines(mask):
    """
    Quita líneas de expiración / cursores verticales de IQ Option.

    Una vela puede tener una mecha larga, pero una línea de interfaz suele
    atravesar gran parte de toda la altura del gráfico y ser muy estrecha.
    """
    mask = mask.astype(bool).copy()

    if mask.ndim != 2 or mask.size == 0:
        return mask

    height, width = mask.shape
    suspicious = np.zeros(width, dtype=bool)

    min_run = max(42, int(round(height * 0.50)))
    min_pixels = max(46, int(round(height * 0.54)))

    for x in range(width):
        column = mask[:, x]

        if int(column.sum()) < min_pixels:
            continue

        if _longest_run_1d(column) >= min_run:
            suspicious[x] = True

    groups = _groups_from_active(suspicious)
    max_ui_width = max(7, int(round(width * 0.025)))

    for left, right in groups:
        group_width = right - left + 1

        if group_width > max_ui_width:
            continue

        # La línea de expiración puede tener borde/sombra.
        pad = max(6, int(round(width * 0.012)))
        erase_left = max(0, left - pad)
        erase_right = min(width - 1, right + pad)

        mask[:, erase_left:erase_right + 1] = False

    return mask


def _remove_long_horizontal_ui_lines(mask):
    """
    Quita líneas horizontales largas de precio / temporizador.

    Se eliminan solo trazos muy largos respecto al ancho del frame para no
    borrar cuerpos horizontales o dojis normales.
    """
    mask = mask.astype(bool).copy()

    if mask.ndim != 2 or mask.size == 0:
        return mask

    height, width = mask.shape
    suspicious = np.zeros(height, dtype=bool)

    min_run = max(80, int(round(width * 0.30)))
    min_pixels = max(90, int(round(width * 0.34)))

    for y in range(height):
        row = mask[y, :]

        if int(row.sum()) < min_pixels:
            continue

        if _longest_run_1d(row) >= min_run:
            suspicious[y] = True

    groups = _groups_from_active(suspicious)

    for top, bottom in groups:
        group_height = bottom - top + 1

        if group_height > max(5, int(round(height * 0.018))):
            continue

        pad = max(1, int(round(height * 0.004)))
        erase_top = max(0, top - pad)
        erase_bottom = min(height - 1, bottom + pad)

        mask[erase_top:erase_bottom + 1, :] = False

    return mask


def clean_mask(mask):
    """
    Limpieza v11.1.

    Orden importante:
    1) quita líneas de interfaz;
    2) elimina ruido diminuto;
    3) repara microcortes verticales de las mechas/cuerpos;
    4) vuelve a quitar UI por seguridad.
    """
    mask = mask.astype(bool)
    mask = _remove_long_vertical_ui_lines(mask)
    mask = _remove_long_horizontal_ui_lines(mask)
    mask = _remove_tiny_components(mask, min_pixels=2)

    mask = ndimage.binary_closing(
        mask,
        structure=np.ones((3, 1), dtype=bool),
    )

    mask = _remove_long_vertical_ui_lines(mask)
    mask = _remove_long_horizontal_ui_lines(mask)

    return mask.astype(bool)


# ============================================================
# ESCALA Y GRID ADAPTATIVO
# ============================================================

def _vertical_evidence(mask):
    height, width = mask.shape

    longest = np.zeros(width, dtype=np.int16)
    pixels = mask.sum(axis=0).astype(np.int16)

    for x in range(width):
        longest[x] = _longest_run_1d(mask[:, x])

    # Continuidad vertical pesa más que la cantidad bruta de píxeles.
    evidence = (
        longest.astype(float) * 2.2
        + np.minimum(pixels, 16).astype(float)
    )

    return longest, pixels, evidence


def _seed_candidates(green, red):
    combined = green | red
    longest, pixels, evidence = _vertical_evidence(combined)

    # Una vela muy pequeña puede medir solo 2 px de alto.
    active = (
        (longest >= 2)
        | ((longest >= 1) & (pixels >= 3) & (evidence >= 5.0))
    )

    # Solo unir microcortes horizontales.
    active = ndimage.binary_closing(
        active,
        structure=np.ones(2, dtype=bool),
    )

    groups = _groups_from_active(active)
    frame_width = green.shape[1]

    seeds = []
    max_group_width = max(44, int(round(frame_width * 0.085)))

    for left, right in groups:
        group_width = right - left + 1

        if group_width > max_group_width:
            continue

        local = evidence[left:right + 1]

        if len(local) == 0:
            continue

        weights = local + 1e-6
        xs = np.arange(left, right + 1, dtype=float)
        center = int(round(np.average(xs, weights=weights)))

        seeds.append({
            "x": center,
            "left": int(left),
            "right": int(right),
            "seed_width": int(group_width),
            "strength": float(local.max(initial=0.0)),
        })

    return sorted(seeds, key=lambda item: item["x"]), evidence


def _estimate_spacing_raw(seeds):
    """
    Obtiene el paso horizontal entre velas a partir de los centros detectados.

    Los gaps grandes pueden representar una o más velas que no fueron
    detectadas, por eso se favorecen las distancias pequeñas/centrales.
    """
    if len(seeds) < 3:
        return None

    xs = np.asarray(
        sorted({float(seed["x"]) for seed in seeds}),
        dtype=float,
    )

    if len(xs) < 3:
        return None

    diffs = np.diff(xs)
    diffs = diffs[(diffs >= 2.0) & (diffs <= 150.0)]

    if len(diffs) < 2:
        return None

    diffs = np.sort(diffs)

    # Evitar que gaps de 2x/3x por velas omitidas inflen la escala.
    pool_size = max(2, int(np.ceil(len(diffs) * 0.62)))
    base_pool = diffs[:pool_size]
    base = float(np.median(base_pool))

    compatible = diffs[
        (diffs >= base * 0.60)
        & (diffs <= base * 1.48)
    ]

    spacing = (
        float(np.median(compatible))
        if len(compatible) >= 2
        else base
    )

    if not np.isfinite(spacing):
        return None

    return float(np.clip(spacing, 2.0, 150.0))


def _stabilize_spacing(raw_spacing, seeds, frame_width):
    """
    Evita jitter entre capturas, pero detecta un zoom real inmediatamente.
    """
    global _SCALE_STATE

    widths = np.asarray(
        [
            max(1.0, float(seed.get("seed_width", 1)))
            for seed in seeds
        ],
        dtype=float,
    )

    fallback = (
        float(np.median(widths)) * 2.15
        if len(widths)
        else 10.0
    )

    raw = (
        fallback
        if raw_spacing is None
        else float(raw_spacing)
    )

    raw = float(np.clip(raw, 3.0, 150.0))

    old = _SCALE_STATE.get("spacing")
    old_width = _SCALE_STATE.get("frame_width")

    if (
        old is None
        or old_width is None
        or abs(frame_width - old_width) > max(3, old_width * 0.08)
    ):
        stabilized = raw
    else:
        ratio = raw / max(old, 1e-6)

        # Cambio claro de zoom: adoptar la nueva escala inmediatamente.
        if ratio < 0.88 or ratio > 1.14:
            stabilized = raw
        else:
            # Misma escala: suavizar pequeñas oscilaciones del detector.
            stabilized = old * 0.72 + raw * 0.28

    _SCALE_STATE["spacing"] = float(stabilized)
    _SCALE_STATE["frame_width"] = int(frame_width)

    return float(stabilized)


def _dedupe_seeds(seeds, spacing):
    if not seeds:
        return []

    result = []

    for seed in sorted(seeds, key=lambda item: item["x"]):
        if not result:
            result.append(seed)
            continue

        gap = seed["x"] - result[-1]["x"]
        duplicate_limit = max(1.0, spacing * 0.36)

        if gap <= duplicate_limit:
            if seed["strength"] > result[-1]["strength"]:
                result[-1] = seed
        else:
            result.append(seed)

    return result


def _fit_grid(seeds, spacing, evidence):
    """
    Construye una rejilla horizontal regular.

    Esto es la parte clave de v11:
    - una celda = una vela;
    - no permite que dos velas terminen dentro de la misma caja;
    - si el zoom cambia, spacing cambia y todas las celdas cambian juntas.
    """
    if not seeds:
        return [], spacing

    seeds = _dedupe_seeds(seeds, spacing)

    if len(seeds) == 1:
        return seeds, spacing

    xs = np.asarray([float(seed["x"]) for seed in seeds], dtype=float)

    indices = [0]

    for gap in np.diff(xs):
        step = max(1, int(round(gap / max(spacing, 1.0))))
        step = min(step, 6)
        indices.append(indices[-1] + step)

    indices = np.asarray(indices, dtype=float)

    # Reestimar spacing con la rejilla ya asignada.
    if len(indices) >= 2 and indices[-1] > indices[0]:
        slope = float(
            np.median(
                np.diff(xs) / np.maximum(np.diff(indices), 1.0)
            )
        )

        if np.isfinite(slope) and spacing * 0.70 <= slope <= spacing * 1.35:
            spacing = spacing * 0.45 + slope * 0.55

    offset = float(np.median(xs - indices * spacing))

    grid = []

    first_index = int(indices.min())
    last_index = int(indices.max())

    evidence_median = float(np.median(evidence[evidence > 0])) if np.any(evidence > 0) else 0.0

    for grid_index in range(first_index, last_index + 1):
        expected_x = offset + grid_index * spacing
        radius = max(1, int(round(spacing * 0.23)))

        lo = max(0, int(round(expected_x)) - radius)
        hi = min(len(evidence) - 1, int(round(expected_x)) + radius)

        if hi < lo:
            continue

        local = evidence[lo:hi + 1]

        if len(local) == 0:
            continue

        best_x = lo + int(np.argmax(local))
        strength = float(evidence[best_x])

        # Buscar la semilla observada que corresponde a esta celda.
        nearest = min(
            seeds,
            key=lambda seed: abs(seed["x"] - expected_x),
        )

        nearest_distance = abs(nearest["x"] - expected_x)
        observed = nearest_distance <= spacing * 0.34

        if observed:
            x = int(nearest["x"])
            left = int(nearest["left"])
            right = int(nearest["right"])
            strength = max(strength, float(nearest["strength"]))
            recovered = False
        else:
            # Recuperar solo si hay evidencia vertical real.
            min_strength = max(
                4.0,
                evidence_median * 0.72,
            )

            if strength < min_strength:
                continue

            half = max(1, int(round(spacing * 0.18)))
            x = int(best_x)
            left = max(0, x - half)
            right = min(len(evidence) - 1, x + half)
            recovered = True

        grid.append({
            "x": x,
            "grid_x": float(expected_x),
            "left": left,
            "right": right,
            "strength": float(strength),
            "recovered": recovered,
        })

    return sorted(grid, key=lambda item: item["x"]), float(spacing)


# ============================================================
# GEOMETRÍA DE VELA
# ============================================================

def _contiguous_groups(values, allowed_gap=1):
    if len(values) == 0:
        return []

    values = np.asarray(values, dtype=int)
    groups = []

    start = int(values[0])
    previous = int(values[0])

    for value in values[1:]:
        value = int(value)

        if value <= previous + allowed_gap:
            previous = value
        else:
            groups.append((start, previous))
            start = value
            previous = value

    groups.append((start, previous))
    return groups


def _candidate_from_cell(
    green,
    red,
    seed,
    spacing,
    previous_x=None,
    next_x=None,
):
    height, width = green.shape
    center = int(seed["x"])

    # Una caja amarilla tiene el MISMO ancho relativo para todas las velas
    # de la captura. Al cambiar zoom, spacing cambia y todas escalan juntas.
    target_box_width = max(3, int(round(spacing * 0.58)))
    half_box = target_box_width // 2

    # Límites seguros para no tocar la vela vecina.
    if previous_x is None:
        cell_left = max(0, int(round(center - spacing * 0.48)))
    else:
        cell_left = max(0, int(round((previous_x + center) / 2.0)) + 1)

    if next_x is None:
        cell_right = min(width - 1, int(round(center + spacing * 0.48)))
    else:
        cell_right = min(width - 1, int(round((center + next_x) / 2.0)))

    if cell_right < cell_left:
        return None

    box_left = max(cell_left, center - half_box)
    box_right = min(cell_right, box_left + target_box_width - 1)

    # Si tocamos el borde derecho, reconstruir desde ese borde.
    if box_right - box_left + 1 < target_box_width:
        box_left = max(cell_left, box_right - target_box_width + 1)

    region_green = green[:, box_left:box_right + 1]
    region_red = red[:, box_left:box_right + 1]
    combined = region_green | region_red

    if not combined.any():
        return None

    # Identificar el color dominante dentro de la celda.
    green_pixels_total = int(region_green.sum())
    red_pixels_total = int(region_red.sum())
    total_pixels = green_pixels_total + red_pixels_total

    if total_pixels < 2:
        return None

    color = (
        "VERDE"
        if green_pixels_total >= red_pixels_total
        else "ROJA"
    )

    color_mask = region_green if color == "VERDE" else region_red

    # Filas con señal de la vela.
    row_strength = color_mask.sum(axis=1).astype(float)
    any_strength = combined.sum(axis=1).astype(float)

    rows = np.where(any_strength > 0)[0]

    if len(rows) == 0:
        return None

    # El cuerpo se reconoce por ocupar una fracción del ancho de la caja.
    body_threshold = max(
        1,
        int(round((box_right - box_left + 1) * 0.28)),
    )

    body_rows = np.where(row_strength >= body_threshold)[0]

    if len(body_rows) == 0:
        # Doji / vela extremadamente fina.
        body_rows = np.where(row_strength >= 1)[0]

    if len(body_rows) == 0:
        return None

    groups = _contiguous_groups(
        body_rows,
        allowed_gap=max(1, int(round(spacing * 0.025))),
    )

    body_top, body_bottom = max(
        groups,
        key=lambda pair: float(
            row_strength[pair[0]:pair[1] + 1].sum()
        ),
    )

    body_height = int(body_bottom - body_top + 1)

    # --------------------------------------------------------
    # MECHA ROBUSTA v11.1
    # --------------------------------------------------------
    # Antes se tomaba CUALQUIER píxel verde/rojo que estuviera por encima
    # o debajo del cuerpo dentro de una distancia grande. Si había texto,
    # precio, iconos o elementos de IQ Option en la misma columna, la caja
    # amarilla podía crecer hasta casi toda la altura del gráfico.
    #
    # Ahora:
    # - usamos SOLO el color dominante de la vela;
    # - exigimos continuidad vertical con el cuerpo;
    # - permitimos únicamente microhuecos;
    # - limitamos la extensión máxima en función de la escala actual.
    wick_radius = max(1, int(round(spacing * 0.075)))
    wick_left = max(0, center - wick_radius)
    wick_right = min(width - 1, center + wick_radius)

    dominant_full = green if color == "VERDE" else red
    wick_mask = dominant_full[:, wick_left:wick_right + 1]
    wick_rows = np.where(wick_mask.any(axis=1))[0]

    top = int(body_top)
    bottom = int(body_bottom)

    if len(wick_rows):
        allowed_gap = max(1, int(round(spacing * 0.06)))
        wick_groups = _contiguous_groups(
            wick_rows,
            allowed_gap=allowed_gap,
        )

        # Nunca permitir que una "mecha" salte a un elemento lejano de UI.
        max_wick_distance = max(
            6,
            min(
                int(round(height * 0.20)),
                int(round(spacing * 4.5)),
                int(round(body_height * 4.0 + 10)),
            ),
        )
        join_gap = max(2, int(round(spacing * 0.12)))

        # Grupo principal: el que toca o está más cerca del cuerpo.
        connected_top = int(body_top)
        connected_bottom = int(body_bottom)

        # Expandir hacia arriba solo mediante grupos conectados/adyacentes.
        upper_groups = sorted(
            [
                (a, b) for a, b in wick_groups
                if b <= connected_top
                and connected_top - b <= max_wick_distance
            ],
            key=lambda pair: pair[1],
            reverse=True,
        )

        cursor = connected_top
        for a, b in upper_groups:
            if cursor - b <= join_gap:
                connected_top = min(connected_top, int(a))
                cursor = int(a)
            else:
                break

        # Expandir hacia abajo solo mediante grupos conectados/adyacentes.
        lower_groups = sorted(
            [
                (a, b) for a, b in wick_groups
                if a >= connected_bottom
                and a - connected_bottom <= max_wick_distance
            ],
            key=lambda pair: pair[0],
        )

        cursor = connected_bottom
        for a, b in lower_groups:
            if a - cursor <= join_gap:
                connected_bottom = max(connected_bottom, int(b))
                cursor = int(b)
            else:
                break

        top = max(
            int(body_top) - max_wick_distance,
            int(connected_top),
        )
        bottom = min(
            int(body_bottom) + max_wick_distance,
            int(connected_bottom),
        )

    total_height = max(1, int(bottom - top + 1))

    # Recalcular píxeles solo dentro de la altura de la vela.
    g_count = int(
        green[top:bottom + 1, box_left:box_right + 1].sum()
    )
    r_count = int(
        red[top:bottom + 1, box_left:box_right + 1].sum()
    )
    pixels = g_count + r_count

    if pixels < 2:
        return None

    dominance = abs(g_count - r_count) / max(pixels, 1)
    body_ratio = body_height / max(total_height, 1)

    # Calidad geométrica.
    pixel_factor = min(
        1.0,
        pixels / max(3.0, target_box_width * 1.25),
    )
    dominance_factor = min(1.0, dominance * 1.55)
    body_factor = min(1.0, 0.28 + body_ratio * 0.72)

    confidence = (
        0.36 * pixel_factor
        + 0.36 * dominance_factor
        + 0.28 * body_factor
    )

    # Semillas recuperadas se aceptan, pero con menor confianza.
    if seed.get("recovered", False):
        confidence *= 0.82

    return {
        "x": int(center),
        "left": int(box_left),
        "right": int(box_right),
        "top": int(top),
        "bottom": int(bottom),
        "body_top": int(body_top),
        "body_bottom": int(body_bottom),
        "height": int(total_height),
        "width": int(box_right - box_left + 1),
        "green_pixels": int(g_count),
        "red_pixels": int(r_count),
        "pixels": int(pixels),
        "color": color,
        "confidence": float(confidence),
        "spacing": float(spacing),
        "scale": float(spacing),
        "recovered": bool(seed.get("recovered", False)),
        "frame_width": int(width),
        "frame_height": int(height),
    }


# ============================================================
# API QUE USA EL MONITOR
# ============================================================

def find_candle_candidates(green, red):
    """
    DETECTOR v11.1 - GRID ADAPTATIVO

    La escala se obtiene DE LA CAPTURA ACTUAL.
    Todos los rectángulos amarillos usan el mismo ancho relativo:
        ancho_caja ~= 64% del espaciado entre velas.

    Si el usuario hace zoom:
    - cambia spacing;
    - cambian todas las cajas juntas;
    - sigue existiendo una sola caja por celda/vela.
    """
    seeds, evidence = _seed_candidates(green, red)

    if not seeds:
        return []

    raw_spacing = _estimate_spacing_raw(seeds)
    spacing = _stabilize_spacing(
        raw_spacing,
        seeds,
        green.shape[1],
    )

    grid_seeds, spacing = _fit_grid(
        seeds,
        spacing,
        evidence,
    )

    if not grid_seeds:
        return []

    centers = [int(seed["x"]) for seed in grid_seeds]
    candidates = []

    for index, seed in enumerate(grid_seeds):
        previous_x = (
            centers[index - 1]
            if index > 0
            else None
        )
        next_x = (
            centers[index + 1]
            if index < len(centers) - 1
            else None
        )

        candidate = _candidate_from_cell(
            green,
            red,
            seed,
            spacing,
            previous_x=previous_x,
            next_x=next_x,
        )

        if candidate is not None:
            candidates.append(candidate)

    return sorted(candidates, key=lambda item: item["x"])


def merge_close_candidates(candidates):
    """
    Seguridad final contra duplicados.

    No fusiona velas vecinas normales; solo cajas que prácticamente ocupan
    el mismo centro de rejilla.
    """
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda item: item["x"])
    result = []

    for candle in ordered:
        if not result:
            result.append(candle)
            continue

        previous = result[-1]
        spacing = min(
            float(candle.get("spacing", 10.0)),
            float(previous.get("spacing", 10.0)),
        )

        if candle["x"] - previous["x"] <= max(1, spacing * 0.30):
            if candle.get("confidence", 0.0) > previous.get("confidence", 0.0):
                result[-1] = candle
        else:
            result.append(candle)

    return result


def filter_candidates(candidates):
    """
    Filtro final v11.

    Se basa en proporciones de la escala detectada, no en tamaños absolutos.
    """
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda item: item["x"])

    spacing_values = [
        float(c.get("spacing", 0.0))
        for c in ordered
        if float(c.get("spacing", 0.0)) > 0
    ]

    spacing = (
        float(np.median(spacing_values))
        if spacing_values
        else 10.0
    )

    frame_height = max(
        int(c.get("frame_height", 0) or 0)
        for c in ordered
    )
    frame_width = max(
        int(c.get("frame_width", 0) or 0)
        for c in ordered
    )

    result = []

    min_box_width = max(2, int(round(spacing * 0.36)))
    max_box_width = max(4, int(round(spacing * 0.82)))
    max_height = max(
        30,
        int(round((frame_height or 500) * 0.90)),
    )

    for candle in ordered:
        width = int(candle.get("width", 0))
        height = int(candle.get("height", 0))
        pixels = int(candle.get("pixels", 0))
        confidence = float(candle.get("confidence", 0.0))
        top = int(candle.get("top", 0))

        if pixels < 2:
            continue

        if width < min_box_width or width > max_box_width:
            continue

        if height < 2 or height > max_height:
            continue

        if confidence < 0.22:
            continue

        # Iconos pegados al borde inferior.
        if frame_height > 0:
            near_bottom = top >= int(round(frame_height * 0.90))

            if near_bottom and height <= max(
                14,
                int(round(spacing * 0.50)),
            ):
                continue

        # Controles pegados al extremo derecho.
        if frame_width > 0:
            if candle["x"] >= frame_width - max(
                3,
                int(round(spacing * 0.28)),
            ):
                continue

        result.append(candle)

    return result


def assess_detection_quality(candidates, frame_shape=None, details=False):
    """
    Devuelve calidad 0..1 de la lectura visual.

    Mide:
    - confianza de las cajas;
    - regularidad del espaciado;
    - consistencia de la escala de cajas;
    - cantidad de velas disponibles.

    Esto permite al monitor NO alimentar la IA cuando el detector está
    claramente inestable.
    """
    if not candidates:
        result = {
            "quality": 0.0,
            "count": 0,
            "spacing_consistency": 0.0,
            "box_consistency": 0.0,
            "confidence": 0.0,
        }
        return result if details else 0.0

    ordered = sorted(candidates, key=lambda item: item["x"])
    count = len(ordered)

    confidences = np.asarray(
        [float(c.get("confidence", 0.0)) for c in ordered],
        dtype=float,
    )
    confidence_score = float(np.clip(np.median(confidences), 0.0, 1.0))

    spacings = np.asarray(
        [float(c.get("spacing", 0.0)) for c in ordered],
        dtype=float,
    )
    spacing = float(np.median(spacings[spacings > 0])) if np.any(spacings > 0) else 0.0

    xs = np.asarray([float(c["x"]) for c in ordered], dtype=float)

    if len(xs) >= 2 and spacing > 0:
        diffs = np.diff(xs)
        multiples = np.maximum(1.0, np.round(diffs / spacing))
        residual = np.abs(diffs - multiples * spacing) / max(spacing, 1e-6)
        spacing_consistency = float(
            np.clip(1.0 - np.median(residual) / 0.28, 0.0, 1.0)
        )
    else:
        spacing_consistency = 0.45

    widths = np.asarray(
        [float(c.get("width", 0.0)) for c in ordered],
        dtype=float,
    )

    if len(widths) >= 2 and np.median(widths) > 0:
        cv = float(np.std(widths) / np.median(widths))
        box_consistency = float(np.clip(1.0 - cv / 0.18, 0.0, 1.0))
    else:
        box_consistency = 0.70

    # 12 velas ya dan buena base visual; 18+ recibe máxima puntuación.
    count_score = float(np.clip((count - 4) / 14.0, 0.0, 1.0))

    quality = (
        0.35 * confidence_score
        + 0.30 * spacing_consistency
        + 0.20 * box_consistency
        + 0.15 * count_score
    )

    quality = float(np.clip(quality, 0.0, 1.0))

    result = {
        "quality": quality,
        "count": count,
        "spacing": spacing,
        "spacing_consistency": spacing_consistency,
        "box_consistency": box_consistency,
        "confidence": confidence_score,
    }

    return result if details else quality


def estimate_bias(candles):
    if not candles:
        return "SIN DATOS", 0.0

    green = sum(c.get("color") == "VERDE" for c in candles)
    red = sum(c.get("color") == "ROJA" for c in candles)
    total = green + red

    if total == 0:
        return "SIN DATOS", 0.0

    if green > red:
        return "SESGO ALCISTA", green / total

    if red > green:
        return "SESGO BAJISTA", red / total

    return "MIXTO", 0.5


def create_debug_image(image, candles, output_path=DEBUG_PATH):
    debug = image.convert("RGB").copy()
    draw = ImageDraw.Draw(debug)

    for index, candle in enumerate(candles, start=1):
        left = int(candle["left"])
        right = int(candle["right"])
        top = int(candle["top"])
        bottom = int(candle["bottom"])

        draw.rectangle(
            [(left, top), (right, bottom)],
            outline=(255, 255, 0),
            width=1,
        )

        short = "V" if candle.get("color") == "VERDE" else "R"

        draw.text(
            (left, max(0, top - 11)),
            f"{index}{short}",
            fill=(255, 255, 0),
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    debug.save(output_path)

    return debug


def print_candles(candles):
    quality = assess_detection_quality(candles, details=True)

    print(
        f"Velas: {len(candles)} | "
        f"escala: {quality.get('spacing', 0):.1f}px | "
        f"calidad: {quality['quality'] * 100:.0f}%"
    )

    for index, candle in enumerate(candles, start=1):
        print(
            index,
            candle.get("color"),
            f"x={candle.get('x')}",
            f"box={candle.get('width')}px",
            f"conf={candle.get('confidence', 0):.2f}",
        )


def main():
    image = load_image()
    arr = np.asarray(image)

    green, red = detect_colors(arr)
    green = clean_mask(green)
    red = clean_mask(red)

    candles = find_candle_candidates(green, red)
    candles = merge_close_candidates(candles)
    candles = filter_candidates(candles)

    print_candles(candles)
    create_debug_image(image, candles)

    print(f"Debug guardado en: {DEBUG_PATH}")


if __name__ == "__main__":
    main()
