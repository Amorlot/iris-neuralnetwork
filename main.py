import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras

tf.random.set_seed(42)

OPTIMIZERS = [
    keras.optimizers.SGD(learning_rate=0.01),
    keras.optimizers.Adam(learning_rate=0.001),
]

EPOCHS = 100
BATCH_SIZE = 16


def optimizer_name(opt) -> str:
    return opt.__class__.__name__


def build_model(optimizer, input_dim: int, num_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _plot_results(histories: dict):
    """Plot val_accuracy curves with a horizontal dashed line for the final test accuracy."""
    colors = {"SGD": "#e07b39", "Adam": "#4c72b0"}
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, (history, test_acc) in histories.items():
        color = colors.get(name, None)
        epochs = range(1, len(history.history["val_accuracy"]) + 1)
        ax.plot(epochs, history.history["val_accuracy"],
                label=f"{name} — val accuracy", color=color)
        ax.axhline(test_acc, linestyle="--", color=color,
                   label=f"{name} — test accuracy ({test_acc:.2%})")

    ax.set_title("Val Accuracy (per epoca) vs Test Accuracy finale")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    import os
    os.makedirs("output", exist_ok=True)
    path = "output/plot_val_vs_test.png"
    fig.savefig(path, dpi=150)
    print(f"\nGrafico salvato in {path}")
    plt.close(fig)


def main():
    iris = load_iris()
    X, y = iris.data, iris.target.reshape(-1, 1)
    y_raw = iris.target  # label originali per stratify

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_raw
    )

    results = []
    histories = {}

    for opt in OPTIMIZERS:
        name = optimizer_name(opt)
        print(f"\n{'='*50}")
        print(f"  Optimizer: {name}")
        print(f"{'='*50}")

        model = build_model(opt, input_dim=X_train.shape[1], num_classes=y.shape[1])
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=2,
        )

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        best_val_acc = max(history.history["val_accuracy"])
        results.append((name, acc, loss, best_val_acc))
        histories[name] = (history, acc)
        print(f"  => test acc={acc:.4f}  loss={loss:.4f}  best_val_acc={best_val_acc:.4f}")

    print(f"\n{'='*60}")
    print(f"{'OPTIMIZER':<22} {'TEST ACC':>10} {'TEST LOSS':>10} {'BEST VAL':>10}")
    print(f"{'-'*60}")
    results.sort(key=lambda r: r[1], reverse=True)
    for name, acc, loss, best_val in results:
        print(f"{name:<22} {acc:>10.4f} {loss:>10.4f} {best_val:>10.4f}")
    print(f"{'='*60}")

    _plot_results(histories)


if __name__ == "__main__":
    main()
