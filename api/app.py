import numpy as np
from flask import Flask, jsonify, request
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras

app = Flask(__name__)

LABELS = ["setosa", "versicolor", "virginica"]
STEPS = ["idle", "data_loaded", "configured", "built", "trained"]

# Pipeline state — reset to "idle" on server start
_state = {
    "step": "idle",
    "X_train": None, "X_test": None,
    "y_train": None, "y_test": None,
    "scaler": None,
    "optimizer": None, "optimizer_name": None,
    "learning_rate": None, "epochs": None, "batch_size": None,
    "model": None,
    "last_metrics": None,
}


def _require(step: str):
    """Return a 409 response if the current step does not match `step`."""
    if _state["step"] != step:
        return jsonify({
            "error": f"pipeline step mismatch: expected '{step}', current is '{_state['step']}'",
            "current_step": _state["step"],
            "expected_step": step,
        }), 409
    return None


# ── Utility ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok", "step": _state["step"]})


@app.get("/status")
def status():
    idx = STEPS.index(_state["step"])
    return jsonify({
        "step": _state["step"],
        "next": STEPS[idx + 1] if idx < len(STEPS) - 1 else None,
        "last_metrics": _state["last_metrics"],
    })


# ── Step 1 ─────────────────────────────────────────────────────────────────────

@app.post("/data/load")
def data_load():
    if (err := _require("idle")):
        return err

    iris = load_iris()
    X, y = iris.data, iris.target.reshape(-1, 1)
    y_raw = iris.target  # label originali per stratify

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_raw
    )

    _state.update({
        "step": "data_loaded",
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler,
    })

    return jsonify({
        "step": "data_loaded",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": 4,
        "classes": 3,
    })


# ── Step 2 ─────────────────────────────────────────────────────────────────────

@app.post("/model/configure")
def model_configure():
    if (err := _require("data_loaded")):
        return err

    body = request.get_json(silent=True) or {}
    opt_name = body.get("optimizer", "adam").lower()
    lr = float(body.get("learning_rate", 0.001))
    epochs = int(body.get("epochs", 100))
    batch_size = int(body.get("batch_size", 16))

    if opt_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    else:
        return jsonify({"error": f"unknown optimizer '{opt_name}': use 'adam' or 'sgd'"}), 400

    _state.update({
        "step": "configured",
        "optimizer": optimizer,
        "optimizer_name": opt_name,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
    })

    return jsonify({
        "step": "configured",
        "optimizer": opt_name,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
    })


# ── Step 3 ─────────────────────────────────────────────────────────────────────

@app.post("/model/build")
def model_build():
    if (err := _require("configured")):
        return err

    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer=_state["optimizer"],
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    _state.update({"step": "built", "model": model})

    return jsonify({
        "step": "built",
        "architecture": "Input(4) → Dense(16, relu) → Dense(16, relu) → Dense(3, softmax)",
        "total_params": model.count_params(),
        "optimizer": _state["optimizer_name"],
    })


# ── Step 4 ─────────────────────────────────────────────────────────────────────

@app.post("/model/train")
def model_train():
    if (err := _require("built")):
        return err

    history = _state["model"].fit(
        _state["X_train"], _state["y_train"],
        epochs=_state["epochs"],
        batch_size=_state["batch_size"],
        validation_split=0.1,
        verbose=0,
    )

    loss, acc = _state["model"].evaluate(_state["X_test"], _state["y_test"], verbose=0)
    metrics = {
        "test_accuracy": round(float(acc), 4),
        "test_loss": round(float(loss), 4),
        "best_val_accuracy": round(float(max(history.history["val_accuracy"])), 4),
    }
    _state.update({"step": "trained", "last_metrics": metrics})

    return jsonify({"step": "trained", **metrics})


# ── Step 5 ─────────────────────────────────────────────────────────────────────

@app.post("/predict")
def predict():
    if (err := _require("trained")):
        return err

    body = request.get_json(silent=True)
    if not body or "features" not in body:
        return jsonify({"error": "missing 'features' field"}), 400

    features = body["features"]
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "'features' must be a list of 4 numbers"}), 400

    try:
        X = _state["scaler"].transform([features])
    except Exception:
        return jsonify({"error": "features must be numeric"}), 400

    probs = _state["model"].predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))

    return jsonify({
        "predicted_class": LABELS[idx],
        "probabilities": {label: round(float(p), 6) for label, p in zip(LABELS, probs)},
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
