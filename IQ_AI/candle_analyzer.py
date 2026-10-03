import os

import numpy as np
from PIL import Image

from chart_reader import (
    ROI_LEFT,
    ROI_TOP,
    ROI_RIGHT,
    ROI_BOTTOM,
    detect_colors,
    clean_mask,
    find_candle_candidates,
    merge_close_candidates,
    filter_candidates,
)


IMAGE_PATH = os.path.join(
    "data",
    "grafico.png"
)


def load_chart():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(
            f"No se encontró: {IMAGE_PATH}"
        )

    return Image.open(
        IMAGE_PATH
    ).convert("RGB")


def crop_chart(image):
    width, height = image.size

    right = min(
        ROI_RIGHT,
        width
    )

    bottom = min(
        ROI_BOTTOM,
        height
    )

    if ROI_LEFT >= right:
        raise ValueError(
            "ROI_LEFT no es válido."
        )

    if ROI_TOP >= bottom:
        raise ValueError(
            "ROI_TOP no es válido."
        )

    return image.crop(
        (
            ROI_LEFT,
            ROI_TOP,
            right,
            bottom
        )
    )


def analyze_single_candle(
    green,
    red,
    x
):
    """
    Analiza la misma vela detectada por chart_reader.
    """

    height, width = green.shape

    left = max(
        0,
        x - 4
    )

    right = min(
        width - 1,
        x + 4
    )

    green_region = green[
        :,
        left:right + 1
    ]

    red_region = red[
        :,
        left:right + 1
    ]

    combined = (
        green_region
        | red_region
    )

    row_strength = combined.sum(
        axis=1
    )

    colored_rows = np.where(
        row_strength > 0
    )[0]

    if len(colored_rows) < 5:
        return None

    high_y = int(
        colored_rows.min()
    )

    low_y = int(
        colored_rows.max()
    )

    total_height = (
        low_y
        - high_y
        + 1
    )

    green_pixels = int(
        green_region.sum()
    )

    red_pixels = int(
        red_region.sum()
    )

    if green_pixels >= red_pixels:
        color = "VERDE"
    else:
        color = "ROJA"

    # --------------------------------------------------------
    # Detectar el cuerpo.
    #
    # El cuerpo tiene normalmente mayor grosor horizontal
    # que una mecha.
    # --------------------------------------------------------

    max_width = int(
        row_strength.max()
    )

    if max_width < 2:
        return None

    threshold = max(
        2,
        int(max_width * 0.55)
    )

    body_rows = np.where(
        row_strength >= threshold
    )[0]

    if len(body_rows) == 0:
        return None

    # Agrupar filas consecutivas.
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
            groups.append(
                (
                    start,
                    previous
                )
            )

            start = y
            previous = y

    if start is not None:
        groups.append(
            (
                start,
                previous
            )
        )

    # Elegir el grupo de mayor densidad.
    best_group = None
    best_score = -1

    for group_start, group_end in groups:

        score = int(
            row_strength[
                group_start:group_end + 1
            ].sum()
        )

        if score > best_score:
            best_score = score
            best_group = (
                group_start,
                group_end
            )

    if best_group is None:
        return None

    body_top, body_bottom = best_group

    body_height = (
        body_bottom
        - body_top
        + 1
    )

    upper_wick = (
        body_top
        - high_y
    )

    lower_wick = (
        low_y
        - body_bottom
    )

    # Apertura/cierre aproximados en coordenadas de pantalla.
    if color == "VERDE":
        open_y = body_bottom
        close_y = body_top
    else:
        open_y = body_top
        close_y = body_bottom

    return {
        "x": int(x),
        "color": color,
        "high_y": high_y,
        "low_y": low_y,
        "open_y": int(open_y),
        "close_y": int(close_y),
        "body_top": int(body_top),
        "body_bottom": int(body_bottom),
        "body_height": int(body_height),
        "total_height": int(total_height),
        "upper_wick": int(upper_wick),
        "lower_wick": int(lower_wick),
        "pixel_width": max_width,
    }


def main():

    print(
        "=========================================="
    )

    print(
        "    IQ AI - CANDLE ANALYZER v3"
    )

    print(
        "=========================================="
    )

    print()

    try:
        original = load_chart()
        chart = crop_chart(original)

    except Exception as error:
        print(
            f"ERROR: {error}"
        )
        return

    print(
        f"Imagen original: "
        f"{original.size[0]} x {original.size[1]}"
    )

    print(
        f"Zona analizada: "
        f"{chart.size[0]} x {chart.size[1]}"
    )

    image = np.array(
        chart
    )

    green, red = detect_colors(
        image
    )

    green = clean_mask(
        green
    )

    red = clean_mask(
        red
    )

    # --------------------------------------------------------
    # Usar exactamente el mismo detector de chart_reader v5.
    # --------------------------------------------------------

    candidates = find_candle_candidates(
        green,
        red
    )

    candidates = merge_close_candidates(
        candidates
    )

    candidates = filter_candidates(
        candidates
    )

    print(
        f"Candidatos del detector: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # Convertir los candidatos a estructura de vela.
    # --------------------------------------------------------

    candles = []

    for candidate in candidates:

        candle = analyze_single_candle(
            green,
            red,
            candidate["x"]
        )

        if candle is None:
            continue

        candles.append(
            candle
        )

    print(
        f"Velas analizadas: "
        f"{len(candles)}"
    )

    # --------------------------------------------------------
    # Mostrar últimas velas.
    # --------------------------------------------------------

    print()
    print(
        "ESTRUCTURA DE LAS ÚLTIMAS VELAS"
    )
    print(
        "================================="
    )

    for number, candle in enumerate(
        candles[-25:],
        start=1
    ):

        print(
            f"{number:02d} | "
            f"{candle['color']:<6} | "
            f"altura={candle['total_height']:>3} | "
            f"cuerpo={candle['body_height']:>3} | "
            f"mecha_sup={candle['upper_wick']:>3} | "
            f"mecha_inf={candle['lower_wick']:>3}"
        )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    if candles:

        green_count = sum(
            c["color"] == "VERDE"
            for c in candles
        )

        red_count = sum(
            c["color"] == "ROJA"
            for c in candles
        )

        average_body = (
            sum(
                c["body_height"]
                for c in candles
            )
            / len(candles)
        )

        average_upper = (
            sum(
                c["upper_wick"]
                for c in candles
            )
            / len(candles)
        )

        average_lower = (
            sum(
                c["lower_wick"]
                for c in candles
            )
            / len(candles)
        )

        print()
        print(
            "================================="
        )

        print(
            "RESUMEN"
        )

        print(
            "================================="
        )

        print(
            f"Velas analizadas: {len(candles)}"
        )

        print(
            f"Verdes:          {green_count}"
        )

        print(
            f"Rojas:           {red_count}"
        )

        print(
            f"Cuerpo medio:    {average_body:.2f}px"
        )

        print(
            f"Mecha superior:  {average_upper:.2f}px"
        )

        print(
            f"Mecha inferior:  {average_lower:.2f}px"
        )

    print()
    print(
        "=========================================="
    )

    print(
        "ESTADO"
    )

    print(
        "=========================================="
    )

    print(
        "Candle Analyzer v3 funcionando."
    )

    print(
        "Todavía NO genera señales."
    )


if __name__ == "__main__":
    main()