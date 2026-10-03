import pandas as pd

from analysis_engine import prepare_indicators
from backtest_mtf import (
    get_historical_candles,
    classify_row,
)


SYMBOL = "BTCUSDT"


def prepare_data():

    print("Descargando 1m...")
    df_1m = get_historical_candles(
        SYMBOL,
        "1m",
        5000
    )

    print("Descargando 5m...")
    df_5m = get_historical_candles(
        SYMBOL,
        "5m",
        2000
    )

    print("Descargando 15m...")
    df_15m = get_historical_candles(
        SYMBOL,
        "15m",
        1000
    )

    # Quitamos velas todavía abiertas
    df_1m = df_1m.iloc[:-1].copy()
    df_5m = df_5m.iloc[:-1].copy()
    df_15m = df_15m.iloc[:-1].copy()

    print()
    print("Calculando indicadores...")

    ind_1m = prepare_indicators(df_1m)
    ind_5m = prepare_indicators(df_5m)
    ind_15m = prepare_indicators(df_15m)

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

    print("Alineando temporalidades...")

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

    merged = merged.dropna().reset_index(drop=True)

    return merged


def get_signal(row):

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

    signal_1m, _ = classify_row(row_1m)
    signal_5m, _ = classify_row(row_5m)
    signal_15m, _ = classify_row(row_15m)

    if (
        signal_1m == "SUBIDA"
        and signal_5m == "SUBIDA"
        and signal_15m == "SUBIDA"
    ):
        return "SUBIDA"

    if (
        signal_1m == "BAJADA"
        and signal_5m == "BAJADA"
        and signal_15m == "BAJADA"
    ):
        return "BAJADA"

    return "NO OPERAR"


def run_backtest(merged, horizon):

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
        "NO OPERAR": 0,
    }

    last_index = len(merged) - horizon

    for i in range(last_index):

        row = merged.iloc[i]

        signal = get_signal(row)

        if signal == "NO OPERAR":
            stats["NO OPERAR"] += 1
            continue

        stats[signal]["signals"] += 1

        entry_price = float(
            merged.iloc[i]["close_1m"]
        )

        exit_price = float(
            merged.iloc[i + horizon]["close_1m"]
        )

        change = exit_price - entry_price

        if signal == "SUBIDA":

            if change > 0:
                stats["SUBIDA"]["wins"] += 1
            else:
                stats["SUBIDA"]["losses"] += 1

        elif signal == "BAJADA":

            if change < 0:
                stats["BAJADA"]["wins"] += 1
            else:
                stats["BAJADA"]["losses"] += 1

    return stats


def print_results(stats, horizon):

    total_signals = 0
    total_wins = 0
    total_losses = 0

    print()
    print("========================================")
    print(f"HORIZONTE: {horizon} MINUTO(S)")
    print("========================================")

    for signal in ("SUBIDA", "BAJADA"):

        signals = stats[signal]["signals"]
        wins = stats[signal]["wins"]
        losses = stats[signal]["losses"]

        total_signals += signals
        total_wins += wins
        total_losses += losses

        accuracy = (
            wins / signals * 100
            if signals > 0
            else 0
        )

        print()
        print(signal)
        print(f"  Señales:   {signals}")
        print(f"  Aciertos:  {wins}")
        print(f"  Errores:   {losses}")
        print(f"  Precisión: {accuracy:.2f}%")

    overall_accuracy = (
        total_wins / total_signals * 100
        if total_signals > 0
        else 0
    )

    print()
    print("TOTAL")
    print(f"  Señales:   {total_signals}")
    print(f"  Aciertos:  {total_wins}")
    print(f"  Errores:   {total_losses}")
    print(f"  Precisión: {overall_accuracy:.2f}%")
    print(
        f"  No operar: {stats['NO OPERAR']}"
    )


def main():

    print("========================================")
    print("     IQ AI - BACKTEST DE HORIZONTES")
    print("========================================")
    print()

    merged = prepare_data()

    print()
    print(
        f"Velas disponibles: {len(merged)}"
    )

    horizons = [1, 3, 5, 10]

    all_results = {}

    for horizon in horizons:

        stats = run_backtest(
            merged,
            horizon
        )

        all_results[horizon] = stats

        print_results(
            stats,
            horizon
        )

    print()
    print("========================================")
    print("RESUMEN COMPARATIVO")
    print("========================================")
    print()

    print(
        f"{'Horizonte':<12}"
        f"{'Señales':<10}"
        f"{'Aciertos':<10}"
        f"{'Errores':<10}"
        f"{'Precisión':<12}"
    )

    print("-" * 54)

    for horizon in horizons:

        stats = all_results[horizon]

        signals = (
            stats["SUBIDA"]["signals"]
            + stats["BAJADA"]["signals"]
        )

        wins = (
            stats["SUBIDA"]["wins"]
            + stats["BAJADA"]["wins"]
        )

        losses = (
            stats["SUBIDA"]["losses"]
            + stats["BAJADA"]["losses"]
        )

        accuracy = (
            wins / signals * 100
            if signals > 0
            else 0
        )

        print(
            f"{str(horizon) + ' min':<12}"
            f"{signals:<10}"
            f"{wins:<10}"
            f"{losses:<10}"
            f"{accuracy:.2f}%"
        )

    print()
    print(
        "Estos resultados son históricos y "
        "no garantizan resultados futuros."
    )


if __name__ == "__main__":
    main()