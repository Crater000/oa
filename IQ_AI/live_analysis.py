from market_data import get_candles
from analysis_engine import analyze_market


def main():
    print("Obteniendo mercado real...\n")

    df = get_candles(
        symbol="BTCUSDT",
        interval="1m",
        limit=500
    )

    # La última vela puede seguir formándose.
    # Analizamos la última vela cerrada.
    df = df.iloc[:-1].copy()

    result = analyze_market(df)

    print("========== IQ AI ==========\n")

    print(f"Activo:       BTC/USDT")
    print(f"Temporalidad: 1 minuto")
    print(f"Precio:       {result['price']:.2f}")
    print(f"Señal:        {result['signal']}")
    print(f"Puntuación:   {result['score']}")
    print(f"Confianza:    {result['confidence']:.0f}%")
    print(f"RSI:          {result['rsi']:.2f}")
    print(f"MACD:         {result['macd']:.4f}")
    print(f"EMA 20:       {result['ema20']:.2f}")
    print(f"EMA 50:       {result['ema50']:.2f}")
    print(f"EMA 200:      {result['ema200']:.2f}")
    print(f"ATR:          {result['atr']:.2f}")
    print(f"Volatilidad:  {result['volatility']}")
    print(f"Soporte:      {result['support']:.2f}")
    print(f"Resistencia:  {result['resistance']:.2f}")

    print("\nMotivos:")

    for reason in result["reasons"]:
        print(f"- {reason}")


if __name__ == "__main__":
    main()