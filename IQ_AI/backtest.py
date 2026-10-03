import requests
import pandas as pd

from analysis_engine import analyze_market


BINANCE_URL = "https://api.binance.com/api/v3/klines"


def get_historical_candles(
    symbol="BTCUSDT",
    interval="1m",
    limit=1000
):
    """
    Descarga hasta 1000 velas históricas.
    """

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    response = requests.get(
        BINANCE_URL,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

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
        data,
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

    return df


def evaluate_signal(
    df,
    signal_index,
    horizon=1
):
    """
    Comprueba qué ocurrió después de la señal.

    horizon=1 significa que miramos la siguiente vela.
    """

    if signal_index + horizon >= len(df):
        return None

    entry_price = float(
        df.iloc[signal_index]["close"]
    )

    exit_price = float(
        df.iloc[
            signal_index + horizon
        ]["close"]
    )

    change = exit_price - entry_price

    return {
        "entry": entry_price,
        "exit": exit_price,
        "change": change,
    }


def main():

    print("======================================")
    print("        IQ AI - BACKTEST")
    print("======================================\n")

    print("Descargando datos históricos...")

    df = get_historical_candles(
        symbol="BTCUSDT",
        interval="1m",
        limit=1000
    )

    print(
        f"Velas recibidas: {len(df)}"
    )

    # Usamos únicamente velas cerradas.
    df = df.iloc[:-1].reset_index(drop=True)

    # Necesitamos suficientes velas para
    # calcular EMA 200.
    warmup = 250

    if len(df) <= warmup + 1:
        print(
            "No hay suficientes velas."
        )
        return

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

    print("\nEjecutando backtest...\n")

    for i in range(
        warmup,
        len(df) - 1
    ):

        historical = df.iloc[
            :i + 1
        ].copy()

        try:
            result = analyze_market(
                historical
            )

        except Exception:
            continue

        signal = result["signal"]

        if signal == "NO OPERAR":
            stats["NO OPERAR"]["signals"] += 1
            continue

        stats[signal]["signals"] += 1

        outcome = evaluate_signal(
            df,
            i,
            horizon=1
        )

        if outcome is None:
            continue

        change = outcome["change"]

        if signal == "SUBIDA":
            if change > 0:
                stats[signal]["wins"] += 1
            else:
                stats[signal]["losses"] += 1

        elif signal == "BAJADA":
            if change < 0:
                stats[signal]["wins"] += 1
            else:
                stats[signal]["losses"] += 1

    print("======================================")
    print("RESULTADOS")
    print("======================================\n")

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

        if signals > 0:
            accuracy = (
                wins / signals
            ) * 100
        else:
            accuracy = 0

        print(f"{signal}")
        print(f"  Señales: {signals}")
        print(f"  Aciertos: {wins}")
        print(f"  Errores: {losses}")
        print(
            f"  Precisión: {accuracy:.2f}%"
        )
        print()

    no_trade = stats["NO OPERAR"]["signals"]

    print("NO OPERAR")
    print(
        f"  Situaciones descartadas: {no_trade}"
    )

    print("\n======================================")

    if total_signals > 0:

        overall_accuracy = (
            total_wins / total_signals
        ) * 100

        print(
            f"Señales totales: {total_signals}"
        )

        print(
            f"Aciertos totales: {total_wins}"
        )

        print(
            f"Errores totales: {total_losses}"
        )

        print(
            f"Precisión total: "
            f"{overall_accuracy:.2f}%"
        )

    print("======================================")

    print(
        "\nEste resultado es un backtest "
        "histórico, no una garantía de "
        "resultados futuros."
    )


if __name__ == "__main__":
    main()