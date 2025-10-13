import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import random

# ==========================================================
# CONFIGURATION
# ==========================================================
SEQ_LEN = 60
BATCH_SIZE = 64
EPOCHS = 50
VAL_SPLIT = 0.1
DATA_PATH = "data/stocks.parquet"
MODEL_PATH = "models/gemini_final.keras"

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ==========================================================
# DATA PREPARATION
# ==========================================================
def simulate_data(n_tickers=25, n_days=400):
    """Simulates fake OHLCV data — replace with real Yahoo fetch."""
    tickers = [f"TICK{i}" for i in range(n_tickers)]
    df_list = []
    for t in tickers:
        df = pd.DataFrame({
            "ticker": t,
            "open": np.random.uniform(100, 200, n_days),
            "high": np.random.uniform(100, 210, n_days),
            "low": np.random.uniform(90, 190, n_days),
            "close": np.random.uniform(95, 205, n_days),
            "volume": np.random.uniform(1e5, 1e6, n_days),
        })
        df_list.append(df)
    return pd.concat(df_list)


def feature_engineer(df):
    """Applies technical features."""
    df["return"] = df.groupby("ticker")["close"].pct_change()
    df["vol_change"] = df.groupby("ticker")["volume"].pct_change()
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["sma_20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean())
    df["sma_50"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(50).mean())
    df["sma_ratio"] = df["sma_20"] / df["sma_50"]
    df.dropna(inplace=True)
    return df


def create_sequences(df, features):
    """Create sequences for supervised learning."""
    logging.info("Creating sequences...")
    seqs_X, seqs_y_dir, seqs_y_bk = [], [], []

    for ticker, tdf in df.groupby("ticker"):
        arr = tdf[features].values
        for i in range(len(arr) - SEQ_LEN - 1):
            seq = arr[i:i + SEQ_LEN]
            next_close = arr[i + SEQ_LEN, features.index("close")]
            prev_close = arr[i + SEQ_LEN - 1, features.index("close")]

            # Binary direction
            direction = int(next_close > prev_close)

            # Breakout (example: move > 2%)
            breakout = int(abs((next_close - prev_close) / prev_close) > 0.02)

            seqs_X.append(seq)
            seqs_y_dir.append(direction)
            seqs_y_bk.append(breakout)

    X = np.array(seqs_X)
    y_dir = np.array(seqs_y_dir)
    y_bk = np.array(seqs_y_bk)

    # --- Split ---
    n_total = len(X)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    X_train, X_val, X_test = X[:n_train], X[n_train:n_train + n_val], X[n_train + n_val:]
    y_dir_train, y_dir_val, y_dir_test = y_dir[:n_train], y_dir[n_train:n_train + n_val], y_dir[n_train + n_val:]
    y_bk_train, y_bk_val, y_bk_test = y_bk[:n_train], y_bk[n_train:n_train + n_val], y_bk[n_train + n_val:]

    logging.info(f"Sequences created: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    logging.info(f"Train shapes: X={X_train.shape}, y_dir={y_dir_train.shape}, y_bk={y_bk_train.shape}")

    return X_train, X_val, X_test, y_dir_train, y_dir_val, y_dir_test, y_bk_train, y_bk_val, y_bk_test


def scale_features(X_train, X_val, X_test):
    """Scales each feature consistently using MinMaxScaler."""
    n_samples, n_steps, n_features = X_train.shape
    scaler = MinMaxScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    scaler.fit(X_train_2d)
    X_train_scaled = scaler.transform(X_train_2d).reshape(n_samples, n_steps, n_features)
    X_val_scaled = scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def load_data():
    """Main data pipeline."""
    df = simulate_data()
    df = feature_engineer(df)
    features = ["open", "high", "low", "close", "volume", "return", "vol_change", "high_low_range", "sma_ratio"]
    X_train, X_val, X_test, y_dir_train, y_dir_val, y_dir_test, y_bk_train, y_bk_val, y_bk_test = create_sequences(df, features)
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)
    return X_train, X_val, X_test, y_dir_train, y_dir_val, y_dir_test, y_bk_train, y_bk_val, y_bk_test, scaler


# ==========================================================
# MODEL
# ==========================================================
def build_model(input_shape):
    inp = Input(shape=input_shape, name="input_seq")

    x = Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)

    x = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)

    # Separate heads
    head_dir = Dense(32, activation="relu")(x)
    head_dir = Dropout(0.3)(head_dir)
    out_dir = Dense(1, activation="sigmoid", name="direction")(head_dir)

    head_bk = Dense(32, activation="relu")(x)
    head_bk = Dropout(0.3)(head_bk)
    out_bk = Dense(1, activation="sigmoid", name="breakout")(head_bk)

    model = Model(inputs=inp, outputs=[out_dir, out_bk])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"direction": "binary_crossentropy", "breakout": "binary_crossentropy"},
        metrics={"direction": ["accuracy"], "breakout": ["accuracy"]}
    )

    model.summary(print_fn=logging.info)
    return model


# ==========================================================
# TRAINING
# ==========================================================
def train_model(model, X_train, y_dir_train, y_bk_train, X_val, y_dir_val, y_bk_val):
    class_weights_dir = compute_class_weight('balanced', classes=np.unique(y_dir_train), y=y_dir_train)
    class_weights_bk = compute_class_weight('balanced', classes=np.unique(y_bk_train), y=y_bk_train)
    cw = {"direction": dict(enumerate(class_weights_dir)), "breakout": dict(enumerate(class_weights_bk))}

    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    mc = ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)

    logging.info("🚀 Starting training...")
    history = model.fit(
        X_train,
        {"direction": y_dir_train, "breakout": y_bk_train},
        validation_data=(X_val, {"direction": y_dir_val, "breakout": y_bk_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=[es, mc],
        verbose=1,
    )
    return history


# ==========================================================
# MAIN
# ==========================================================
def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logging.info(f"✅ Using GPU: {gpus[0].name}")
    else:
        logging.info("⚙️  Using CPU")

    X_train, X_val, X_test, y_dir_train, y_dir_val, y_dir_test, y_bk_train, y_bk_val, y_bk_test, scaler = load_data()
    model = build_model((SEQ_LEN, X_train.shape[2]))
    history = train_model(model, X_train, y_dir_train, y_bk_train, X_val, y_dir_val, y_bk_val)

    model.save(MODEL_PATH)
    logging.info("✅ Training complete. Model saved.")


if __name__ == "__main__":
    main()
