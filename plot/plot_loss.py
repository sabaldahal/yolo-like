import json
import os
import matplotlib.pyplot as plt

with open("local/loss_history.json", "r") as f:
    history = json.load(f)

train = history["train"]
val = history["val"]

epochs = [e["epoch"] for e in train]

PLOT_DIR = "loss_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def save_loss_plot(loss_name, ylabel=None):
    train_vals = [e[loss_name] for e in train]
    val_vals = [e[loss_name] for e in val]

    plt.figure()
    plt.plot(epochs, train_vals, label="Train")
    plt.plot(epochs, val_vals, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel if ylabel else loss_name.capitalize())
    plt.title(f"{loss_name.capitalize()} Loss")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(PLOT_DIR, f"{loss_name}_loss.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")


save_loss_plot("total", "Total Loss")
save_loss_plot("box", "Box Loss")
save_loss_plot("object", "Objectness Loss")
save_loss_plot("no_object", "No-object Loss")
save_loss_plot("class", "Classification Loss")
