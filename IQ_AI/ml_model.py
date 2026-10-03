import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from analysis_engine import prepare_indicators
from backtest_mtf import get_historical_candles


SYMBOL = "BTCUSDT"

# Horizonte del objetivo:
# 1 = próxima vela de 1 minuto
HORIZON = 1


def add_features(df, suffix):
    data = df.copy()

    # Indicadores existentes
    data = prepare_indicators(data)

    close = data["close"]

    # Distancia de EMAs
    data[f"ema20_dist_{suffix}"] = (
        (close - data["ema20"]) / close * 100
    )

    data[f"ema50_dist_{suffix}"] = (
        (close - data["ema50"]) / close * 100
    )

    data[f"ema200_dist_{suffix}"] = (
        (close - data["ema200"]) / close * 100
    )

    # ATR normalizado
    data[f"atr_pct_{suffix}"] = (
        data["atr"] / close * 100
    )

    # Distancia a soporte/resistencia
    data[f"support_dist_{suffix}"] = (
        (close - data["support"]) / close * 100
    )

    data[f"resistance_dist_{suffix}"] = (
        (data["resistance"] - close) / close * 100
    )

    # Estructura de la vela
    candle_range = (
        data["high"] - data["low"]
    ).replace(0, np.nan)

    body = (
        data["close"] - data["open"]
    )

    upper_wick = (
        data["high"]
        - data[["open", "close"]].max(axis=1)
    )

    lower_wick = (
        data[["open", "close"]].min(axis=1)
        - data["low"]
    )

    data[f"body_pct_{suffix}"] = (
        body / data["close"] * 100
    )

    data[f"range_pct_{suffix}"] = (
        candle_range / data["close"] * 100
    )

    data[f"upper_wick_pct_{suffix}"] = (
        upper_wick / data["close"] * 100
    )

    data[f"lower_wick_pct_{suffix}"] = (
        lower_wick / data["close"] * 100
    )

    data[f"volume_change_{suffix}"] = (
        data["volume"].pct_change(5) * 100
    )

    # Seleccionar las características
    feature_columns = [
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "momentum",
        f"ema20_dist_{suffix}",
        f"ema50_dist_{suffix}",
        f"ema200_dist_{suffix}",
        f"atr_pct_{suffix}",
        f"support_dist_{suffix}",
        f"resistance_dist_{suffix}",
        f"body_pct_{suffix}",
        f"range_pct_{suffix}",
        f"upper_wick_pct_{suffix}",
        f"lower_wick_pct_{suffix}",
        f"volume_change_{suffix}",
    ]

    return data, feature_columns


def build_dataset():

    print("Descargando datos 1m...")
    df_1m = get_historical_candles(
        SYMBOL,
        "1m",
        10000
    )

    print("Descargando datos 5m...")
    df_5m = get_historical_candles(
        SYMBOL,
        "5m",
        3000
    )

    print("Descargando datos 15m...")
    df_15m = get_historical_candles(
        SYMBOL,
        "15m",
        1000
    )

    # Retiramos la vela todavía abierta
    df_1m = df_1m.iloc[:-1].copy()
    df_5m = df_5m.iloc[:-1].copy()
    df_15m = df_15m.iloc[:-1].copy()

    print("\nCalculando características...")

    data_1m, features_1m = add_features(
        df_1m,
        "1m"
    )

    data_5m, features_5m = add_features(
        df_5m,
        "5m"
    )

    data_15m, features_15m = add_features(
        df_15m,
        "15m"
    )

    # Características de 5m y 15m
    data_5m = data_5m[
        [
            "close_time"
            + ""
        ] + features_5m
    ].copy()

    data_15m = data_15m[
        [
            "close_time"
            + ""
        ] + features_15m
    ].copy()

    print("Alineando temporalidades...")

    merged = pd.merge_asof(
        data_1m.sort_values("close_time"),
        data_5m.sort_values("close_time"),
        on="close_time",
        direction="backward",
        suffixes=("", "_5m"),
    )

    merged = pd.merge_asof(
        merged.sort_values("close_time"),
        data_15m.sort_values("close_time"),
        on="close_time",
        direction="backward",
        suffixes=("", "_15m"),
    )

    # Objetivo:
    # 1 si la próxima vela cierra más arriba
    # 0 si cierra igual o más abajo
    merged["future_close"] = (
        merged["close"].shift(-HORIZON)
    )

    merged["target"] = (
        merged["future_close"]
        > merged["close"]
    ).astype(int)

    all_features = (
        features_1m
        + [f"{x}_5m" for x in features_5m]
        + [f"{x}_15m" for x in features_15m]
    )

    dataset = merged[
        all_features + ["target"]
    ].copy()

    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan
    )

    dataset = dataset.dropna()

    return dataset, all_features


def evaluate_confidence(
    model,
    x_test,
    y_test,
    threshold
):
    probabilities = model.predict_proba(
        x_test
    )

    p_down = probabilities[:, 0]
    p_up = probabilities[:, 1]

    signals = []

    for down, up in zip(
        p_down,
        p_up
    ):
        if up >= threshold:
            signals.append(1)

        elif down >= threshold:
            signals.append(0)

        else:
            signals.append(-1)

    signals = np.array(signals)

    active = signals != -1

    if active.sum() == 0:
        return {
            "signals": 0,
            "accuracy": 0,
            "coverage": 0,
        }

    filtered_accuracy = accuracy_score(
        y_test[active],
        signals[active]
    )

    coverage = (
        active.sum()
        / len(y_test)
        * 100
    )

    return {
        "signals": int(active.sum()),
        "accuracy": filtered_accuracy * 100,
        "coverage": coverage,
    }


def main():

    print()
    print("==========================================")
    print("       IQ AI - MACHINE LEARNING")
    print("==========================================")
    print()

    dataset, feature_columns = build_dataset()

    print()
    print(
        f"Filas utilizables: {len(dataset)}"
    )

    print(
        f"Características: {len(feature_columns)}"
    )

    # División CRONOLÓGICA.
    # Nunca mezclamos pasado y futuro.
    split_index = int(
        len(dataset) * 0.75
    )

    train = dataset.iloc[
        :split_index
    ].copy()

    test = dataset.iloc[
        split_index:
    ].copy()

    x_train = train[
        feature_columns
    ]

    y_train = train[
        "target"
    ]

    x_test = test[
        feature_columns
    ]

    y_test = test[
        "target"
    ]

    print()
    print(
        f"Entrenamiento: {len(train)}"
    )

    print(
        f"Prueba inédita: {len(test)}"
    )

    print()
    print("Entrenando Random Forest...")

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=12,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train
    )

    # Predicción general
    y_pred = model.predict(
        x_test
    )

    overall_accuracy = (
        accuracy_score(
            y_test,
            y_pred
        )
        * 100
    )

    print()
    print("==========================================")
    print("RESULTADO GENERAL")
    print("==========================================")

    print(
        f"Precisión en datos inéditos: "
        f"{overall_accuracy:.2f}%"
    )

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    print()
    print("Matriz de confusión:")
    print(matrix)

    # Diferentes niveles de confianza
    print()
    print("==========================================")
    print("SEÑALES FILTRADAS POR CONFIANZA")
    print("==========================================")

    for threshold in [
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
    ]:

        result = evaluate_confidence(
            model,
            x_test,
            y_test,
            threshold
        )

        print()
        print(
            f"Umbral: {threshold:.2f}"
        )

        print(
            f"Señales: {result['signals']}"
        )

        print(
            f"Cobertura: "
            f"{result['coverage']:.2f}%"
        )

        print(
            f"Precisión: "
            f"{result['accuracy']:.2f}%"
        )

    # Importancia de características
    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns
    ).sort_values(
        ascending=False
    )

    print()
    print("==========================================")
    print("CARACTERÍSTICAS MÁS IMPORTANTES")
    print("==========================================")

    print(
        importance.head(15).to_string()
    )

    print()
    print(
        "NOTA: las probabilidades del modelo "
        "no representan una garantía de acierto."
    )


if __name__ == "__main__":
    main()