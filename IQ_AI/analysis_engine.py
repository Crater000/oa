import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores técnicos sobre datos OHLC.
    Requiere columnas:
    open, high, low, close
    """

    data = df.copy()

    required = {"open", "high", "low", "close"}

    if not required.issubset(data.columns):
        raise ValueError(
            f"Faltan columnas. Necesarias: {required}"
        )

    # EMAs
    data["ema20"] = EMAIndicator(
        close=data["close"],
        window=20
    ).ema_indicator()

    data["ema50"] = EMAIndicator(
        close=data["close"],
        window=50
    ).ema_indicator()

    data["ema200"] = EMAIndicator(
        close=data["close"],
        window=200
    ).ema_indicator()

    # RSI
    data["rsi"] = RSIIndicator(
        close=data["close"],
        window=14
    ).rsi()

    # MACD
    macd = MACD(
        close=data["close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )

    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_hist"] = macd.macd_diff()

    # ATR
    atr = AverageTrueRange(
        high=data["high"],
        low=data["low"],
        close=data["close"],
        window=14
    )

    data["atr"] = atr.average_true_range()

    # Momentum simple
    data["momentum"] = data["close"].pct_change(5) * 100

    # Soporte y resistencia simples
    data["support"] = data["low"].rolling(20).min()
    data["resistance"] = data["high"].rolling(20).max()

    return data


def analyze_market(df: pd.DataFrame) -> dict:
    """
    Analiza la última vela disponible.
    """

    data = prepare_indicators(df).dropna()

    if len(data) < 50:
        raise ValueError(
            "No hay suficientes datos para analizar."
        )

    current = data.iloc[-1]

    score = 0
    reasons = []

    # -------------------------------------------------
    # 1. Tendencia mediante EMAs
    # -------------------------------------------------

    if (
        current["ema20"] > current["ema50"]
        and current["ema50"] > current["ema200"]
    ):
        score += 2
        reasons.append("Tendencia alcista por EMAs")

    elif (
        current["ema20"] < current["ema50"]
        and current["ema50"] < current["ema200"]
    ):
        score -= 2
        reasons.append("Tendencia bajista por EMAs")

    else:
        reasons.append("EMAs sin tendencia clara")

    # -------------------------------------------------
    # 2. RSI
    # -------------------------------------------------

    rsi = float(current["rsi"])

    if 50 < rsi < 70:
        score += 1
        reasons.append("RSI favorece impulso alcista")

    elif 30 < rsi < 50:
        score -= 1
        reasons.append("RSI favorece impulso bajista")

    elif rsi >= 70:
        reasons.append("RSI en sobrecompra")

    elif rsi <= 30:
        reasons.append("RSI en sobreventa")

    # -------------------------------------------------
    # 3. MACD
    # -------------------------------------------------

    if (
        current["macd"] > current["macd_signal"]
        and current["macd_hist"] > 0
    ):
        score += 1
        reasons.append("MACD positivo")

    elif (
        current["macd"] < current["macd_signal"]
        and current["macd_hist"] < 0
    ):
        score -= 1
        reasons.append("MACD negativo")

    # -------------------------------------------------
    # 4. Momentum
    # -------------------------------------------------

    momentum = float(current["momentum"])

    if momentum > 0:
        score += 1
        reasons.append("Momentum positivo")

    elif momentum < 0:
        score -= 1
        reasons.append("Momentum negativo")

    # -------------------------------------------------
    # 5. Distancia respecto a soporte/resistencia
    # -------------------------------------------------

    close = float(current["close"])
    support = float(current["support"])
    resistance = float(current["resistance"])

    if resistance > support:

        range_size = resistance - support

        position = (
            (close - support) / range_size
        )

        if position < 0.25:
            score += 1
            reasons.append(
                "Precio cerca del soporte"
            )

        elif position > 0.75:
            score -= 1
            reasons.append(
                "Precio cerca de la resistencia"
            )

    # -------------------------------------------------
    # 6. Volatilidad
    # -------------------------------------------------

    atr = float(current["atr"])

    if atr <= 0:
        volatility = "baja"

    else:
        atr_percent = (atr / close) * 100

        if atr_percent < 0.3:
            volatility = "baja"

        elif atr_percent < 1.0:
            volatility = "moderada"

        else:
            volatility = "alta"

    # -------------------------------------------------
    # Decisión
    # -------------------------------------------------

    if score >= 4:
        signal = "SUBIDA"

    elif score <= -4:
        signal = "BAJADA"

    else:
        signal = "NO OPERAR"

    # Confianza basada en la fuerza de la señal.
    # Esto NO representa una probabilidad real.
    confidence = min(
        95,
        max(
            50,
            50 + abs(score) * 8
        )
    )

    return {
        "signal": signal,
        "score": score,
        "confidence": confidence,
        "price": close,
        "rsi": rsi,
        "macd": float(current["macd"]),
        "ema20": float(current["ema20"]),
        "ema50": float(current["ema50"]),
        "ema200": float(current["ema200"]),
        "atr": atr,
        "volatility": volatility,
        "support": support,
        "resistance": resistance,
        "reasons": reasons,
    }


def generate_demo_data(rows: int = 300) -> pd.DataFrame:
    """
    Genera datos OHLC artificiales solamente
    para comprobar que el motor funciona.
    """

    rng = np.random.default_rng(42)

    returns = rng.normal(
        loc=0.0004,
        scale=0.008,
        size=rows
    )

    close = 80000 * np.exp(
        np.cumsum(returns)
    )

    open_price = np.roll(close, 1)

    open_price[0] = close[0]

    high = np.maximum(
        open_price,
        close
    ) * (
        1 + rng.uniform(
            0,
            0.003,
            rows
        )
    )

    low = np.minimum(
        open_price,
        close
    ) * (
        1 - rng.uniform(
            0,
            0.003,
            rows
        )
    )

    return pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
    })


if __name__ == "__main__":

    df = generate_demo_data()

    result = analyze_market(df)

    print()
    print("========== IQ AI ==========")
    print()
    print(f"Señal:       {result['signal']}")
    print(f"Puntuación:  {result['score']}")
    print(
        f"Confianza:   {result['confidence']:.0f}%"
    )
    print(
        f"Precio:      {result['price']:.2f}"
    )
    print(
        f"RSI:         {result['rsi']:.2f}"
    )
    print(
        f"MACD:        {result['macd']:.4f}"
    )
    print(
        f"EMA 20:      {result['ema20']:.2f}"
    )
    print(
        f"EMA 50:      {result['ema50']:.2f}"
    )
    print(
        f"EMA 200:     {result['ema200']:.2f}"
    )
    print(
        f"ATR:         {result['atr']:.2f}"
    )
    print(
        f"Volatilidad: {result['volatility']}"
    )

    print()
    print("Motivos:")

    for reason in result["reasons"]:
        print(f"- {reason}")