from market_data import get_candles
from analysis_engine import analyze_market


def analyze_timeframe(interval):
    df = get_candles(
        symbol="BTCUSDT",
        interval=interval,
        limit=500
    )

    # Quitamos la vela que todavía está abierta
    df = df.iloc[:-1].copy()

    return analyze_market(df)


def main():
    print("========== IQ AI MULTI-TIMEFRAME ==========\n")
    print("Analizando BTC/USDT...\n")

    results = {}

    for interval in ["15m", "5m", "1m"]:
        print(f"Analizando {interval}...")

        try:
            results[interval] = analyze_timeframe(interval)
        except Exception as error:
            print(f"Error en {interval}: {error}")
            return

    print("\n============================================")
    print("RESULTADOS")
    print("============================================\n")

    for interval in ["15m", "5m", "1m"]:
        result = results[interval]

        print(
            f"{interval:>3} | "
            f"{result['signal']:<10} | "
            f"score={result['score']:+d} | "
            f"RSI={result['rsi']:.2f} | "
            f"MACD={result['macd']:.2f}"
        )

    # Señales de cada temporalidad
    signal_15 = results["15m"]["signal"]
    signal_5 = results["5m"]["signal"]
    signal_1 = results["1m"]["signal"]

    bullish = [signal_15, signal_5, signal_1].count("SUBIDA")
    bearish = [signal_15, signal_5, signal_1].count("BAJADA")

    # Regla conservadora:
    # necesitamos que las 3 temporalidades coincidan.
    if bullish == 3:
        final_signal = "SUBIDA"

    elif bearish == 3:
        final_signal = "BAJADA"

    else:
        final_signal = "NO OPERAR"

    # Promedio de puntuaciones.
    # Esto es una métrica interna, NO una probabilidad.
    average_score = (
        results["15m"]["score"]
        + results["5m"]["score"]
        + results["1m"]["score"]
    ) / 3

    print("\n============================================")
    print("DECISIÓN FINAL")
    print("============================================\n")

    print(f"15 minutos : {signal_15}")
    print(f"5 minutos  : {signal_5}")
    print(f"1 minuto   : {signal_1}")
    print()
    print(f"Señal final: {final_signal}")
    print(f"Score medio: {average_score:.2f}")

    if final_signal == "SUBIDA":
        print("\n🟢 CONFIRMACIÓN ALCISTA")

    elif final_signal == "BAJADA":
        print("\n🔴 CONFIRMACIÓN BAJISTA")

    else:
        print("\n⚪ NO HAY CONFIRMACIÓN SUFICIENTE")

    print("\nIMPORTANTE:")
    print(
        "La señal es experimental y no representa "
        "una probabilidad garantizada de acierto."
    )


if __name__ == "__main__":
    main()