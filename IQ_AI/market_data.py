import requests
import pandas as pd


BINANCE_URL = "https://api.binance.com/api/v3/klines"


def get_candles(
    symbol="BTCUSDT",
    interval="1m",
    limit=500
):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    response = requests.get(
        BINANCE_URL,
        params=params,
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            "No se recibieron datos del mercado."
        )

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

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:
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


if __name__ == "__main__":

    print("Obteniendo datos de BTC/USDT...")

    df = get_candles(
        symbol="BTCUSDT",
        interval="1m",
        limit=500
    )

    print()
    print("========== DATOS RECIBIDOS ==========")
    print()

    print(
        f"Velas recibidas: {len(df)}"
    )

    print(
        f"Primera vela: {df.iloc[0]['timestamp']}"
    )

    print(
        f"Última vela:  {df.iloc[-1]['timestamp']}"
    )

    print(
        f"Último precio: {df.iloc[-1]['close']:.2f}"
    )

    print()
    print("Últimas 5 velas:")
    print()

    print(
        df[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
            ]
        ].tail(5).to_string(
            index=False
        )
    )
    