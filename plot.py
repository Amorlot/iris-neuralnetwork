import os

import matplotlib.pyplot as plt


def plot_val_vs_test(histories: dict):
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

    os.makedirs("output", exist_ok=True)
    path = "output/plot_val_vs_test.png"
    fig.savefig(path, dpi=150)
    print(f"\nGrafico salvato in {path}")
    plt.close(fig)
