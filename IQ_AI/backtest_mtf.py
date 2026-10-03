import time
import requests
import pandas as pd

from analysis_engine import prepare_indicators


BINANCE_URL = "https://api.binance.com/api/v3/klines"


def get_historical_candles(
    symbol,
    interval,
    total_limit
):
    """
    Descarga datos históricos en bloques de máximo
    1000 velas por petición.
    """

    all_rows = []
    end_time = None

    while len(all_rows) < total_limit:

        remaining = total_limit - len(all_rows)

        request_limit = min(
            1000,
            remaining
        )

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": request_limit,
        }

        if end_time is not None:
            params["endTime"] = end_time

        response = requests.get(
            BINANCE_URL,
            params=params,
            timeout=15,
        )

        response.raise_for_status()

        batch = response.json()

        if not batch:
            break

        all_rows = batch + all_rows

        # Pedir el bloque anterior
        end_time = batch[0][0] - 1

        time.sleep(0.15)

        if len(batch) < request_limit:
            break

    all_rows = all_rows[-total_limit:]

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(
        all_rows,
        columns=columns
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True
    )

    # Eliminar posibles duplicados
    df = df.drop_duplicates(
        subset="timestamp"
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


def classify_row(row):
    """
    Aplica las mismas reglas principales
    utilizadas por analysis_engine.py.
    """

    score = 0

    # EMA
    if (
        row["ema20"] > row["ema50"]
        and row["ema50"] > row["ema200"]
    ):
        score += 2

    elif (
        row["ema20"] < row["ema50"]
        and row["ema50"] < row["ema200"]
    ):
        score -= 2

    # RSI
    rsi = float(row["rsi"])

    if 50 < rsi < 70:
        score += 1

    elif 30 < rsi < 50:
        score -= 1

    # MACD
    if (
        row["macd"] > row["macd_signal"]
        and row["macd_hist"] > 0
    ):
        score += 1

    elif (
        row["macd"] < row["macd_signal"]
        and row["macd_hist"] < 0
    ):
        score -= 1

    # Momentum
    momentum = float(row["momentum"])

    if momentum > 0:
        score += 1

    elif momentum < 0:
        score -= 1

    # Soporte / resistencia
    close = float(row["close"])
    support = float(row["support"])
    resistance = float(row["resistance"])

    if resistance > support:

        range_size = resistance - support

        position = (
            (close - support) / range_size
        )

        if position < 0.25:
            score += 1

        elif position > 0.75:
            score -= 1

    if score >= 4:
        signal = "SUBIDA"

    elif score <= -4:
        signal = "BAJADA"

    else:
        signal = "NO OPERAR"

    return signal, score


def main():

    print("==========================================")
    print("      IQ AI - BACKTEST MULTI-TIMEFRAME")
    print("==========================================\n")

    symbol = "BTCUSDT"

    print("Descargando 1m...")
    df_1m = get_historical_candles(
        symbol,
        "1m",
        2000
    )

    print("Descargando 5m...")
    df_5m = get_historical_candles(
        symbol,
        "5m",
        1000
    )

    print("Descargando 15m...")
    df_15m = get_historical_candles(
        symbol,
        "15m",
        500
    )

    print("\nDatos recibidos:")

    print(
        f"1m:  {len(df_1m)} velas"
    )

    print(
        f"5m:  {len(df_5m)} velas"
    )

    print(
        f"15m: {len(df_15m)} velas"
    )

    # Eliminar velas actualmente abiertas
    df_1m = df_1m.iloc[:-1].copy()
    df_5m = df_5m.iloc[:-1].copy()
    df_15m = df_15m.iloc[:-1].copy()

    print("\nCalculando indicadores...")

    ind_1m = prepare_indicators(
        df_1m
    )

    ind_5m = prepare_indicators(
        df_5m
    )

    ind_15m = prepare_indicators(
        df_15m
    )

    # Solo necesitamos columnas necesarias
    cols = [

 "close_time",
    "close",
    "ema20",
    "ema50",
    "ema200",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "momentum",
    "support",
    "resistance",

    ]

    ind_5m = ind_5m[cols].copy()

    ind_15m = ind_15m[cols].copy()

    print("\nAlineando temporalidades...")

    # Para cada vela de 1m buscamos la última vela
    # cerrada disponible de 5m y 15m.
    merged = pd.merge_asof(
        ind_1m.sort_values("close_time"),
        ind_5m.sort_values("close_time"),
        on="close_time",
        direction="backward",
        suffixes=("_1m", "_5m"),
    )

    merged = pd.merge_asof(
        merged.sort_values("close_time"),
        ind_15m.sort_values("close_time"),
        on="close_time",
        direction="backward",
        suffixes=("", "_15m"),
    )

    # Eliminar filas sin suficientes indicadores
    merged = merged.dropna().reset_index(
        drop=True
    )

    stats = {
        "SUBIDA": {
            "signals": 0,
            "wins": 0,
            "losses": 0,
        },
        "BAJADA": {
            "signals": 0,
            "wins": 0,
            "losses": 0,
        },
        "NO OPERAR": {
            "signals": 0,
        },
    }

    print(
        f"Velas disponibles para prueba: "
        f"{len(merged)}"
    )

    print("\nEjecutando backtest...\n")

    for i in range(
        len(merged) - 1
    ):

        row = merged.iloc[i]

        # Construimos una fila para cada temporalidad
        row_1m = pd.Series({
               "ema20": row["ema20_1m"],
               "ema50": row["ema50_1m"],
               "ema200": row["ema200_1m"],
               "rsi": row["rsi_1m"],
               "macd": row["macd_1m"],
               "macd_signal": row["macd_signal_1m"],
               "macd_hist": row["macd_hist_1m"],
               "momentum": row["momentum_1m"],
               "close": row["close_1m"],
               "support": row["support_1m"],
             "resistance": row["resistance_1m"],

        })

        row_5m = pd.Series({
              "ema20": row["ema20_5m"],
              "ema50": row["ema50_5m"],
              "ema200": row["ema200_5m"],
              "rsi": row["rsi_5m"],
              "macd": row["macd_5m"],
              "macd_signal": row["macd_signal_5m"],
              "macd_hist": row["macd_hist_5m"],
              "momentum": row["momentum_5m"],
              "close": row["close_5m"],
              "support": row["support_5m"],
              "resistance": row["resistance_5m"],

        })

        row_15m = pd.Series({
          "ema20": row["ema20"],
          "ema50": row["ema50"],
          "ema200": row["ema200"],
          "rsi": row["rsi"],
          "macd": row["macd"],
          "macd_signal": row["macd_signal"],
          "macd_hist": row["macd_hist"],
          "momentum": row["momentum"],
          "close": row["close"],
          "support": row["support"],
          "resistance": row["resistance"],
        })

        signal_1m, score_1m = classify_row(
            row_1m
        )

        signal_5m, score_5m = classify_row(
            row_5m
        )

        signal_15m, score_15m = classify_row(
            row_15m
        )

        # EXACTAMENTE la regla de nuestra
        # estrategia multi-timeframe actual.
        if (
            signal_1m == "SUBIDA"
            and signal_5m == "SUBIDA"
            and signal_15m == "SUBIDA"
        ):
            final_signal = "SUBIDA"

        elif (
            signal_1m == "BAJADA"
            and signal_5m == "BAJADA"
            and signal_15m == "BAJADA"
        ):
            final_signal = "BAJADA"

        else:
            final_signal = "NO OPERAR"

        if final_signal == "NO OPERAR":

            stats["NO OPERAR"]["signals"] += 1
            continue

        stats[final_signal]["signals"] += 1

        # Resultado en la siguiente vela de 1 minuto
               # Resultado en la siguiente vela de 1 minuto
        entry_price = float(
            merged.iloc[i]["close_1m"]
        )

        exit_price = float(
            merged.iloc[i + 1]["close_1m"]
        )

        change = (
            exit_price - entry_price
        )

        if final_signal == "SUBIDA":

            if change > 0:
                stats["SUBIDA"]["wins"] += 1
            else:
                stats["SUBIDA"]["losses"] += 1

        elif final_signal == "BAJADA":

            if change < 0:
                stats["BAJADA"]["wins"] += 1
            else:
                stats["BAJADA"]["losses"] += 1

    print("\n==========================================")
    print("RESULTADOS")
    print("==========================================\n")

    total_signals = 0
    total_wins = 0
    total_losses = 0

    for signal in [
        "SUBIDA",
        "BAJADA",
    ]:

        signals = stats[signal]["signals"]
        wins = stats[signal]["wins"]
        losses = stats[signal]["losses"]

        total_signals += signals
        total_wins += wins
        total_losses += losses

        accuracy = (
            (wins / signals) * 100
            if signals > 0
            else 0
        )

        print(signal)
        print(
            f"  Señales:    {signals}"
        )
        print(
            f"  Aciertos:   {wins}"
        )
        print(
            f"  Errores:    {losses}"
        )
        print(
            f"  Precisión:  {accuracy:.2f}%"
        )
        print()

    no_trade = stats[
        "NO OPERAR"
    ]["signals"]

    print("NO OPERAR")
    print(
        f"  Situaciones descartadas: "
        f"{no_trade}"
    )

    print(
        "\n=========================================="
    )

    if total_signals > 0:

        accuracy = (
            total_wins /
            total_signals
        ) * 100

        print(
            f"Señales totales: "
            f"{total_signals}"
        )

        print(
            f"Aciertos totales: "
            f"{total_wins}"
        )

        print(
            f"Errores totales: "
            f"{total_losses}"
        )

        print(
            f"Precisión total: "
            f"{accuracy:.2f}%"
        )

    print(
        "=========================================="
    )

    print(
        "\nEste backtest utiliza BTC/USDT de "
        "Binance como fuente de datos y "
        "evalúa la siguiente vela de 1 minuto."
    )


if __name__ == "__main__":
    main()