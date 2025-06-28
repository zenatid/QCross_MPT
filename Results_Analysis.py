import re, ast, json, os, textwrap, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = Path(r"C:\Users\DAVIDZEN\DQEC_CrossMPC_logical\Final_Results_QECCT\toric\Code_L_4\noise_model_independent\repetition_1\26_06_2025_23_51_55\logging.txt")
assert LOG_PATH.exists(), "Log file not found at expected location."
OUTPUT_DIR = LOG_PATH.parent

text = LOG_PATH.read_text()

# ---------------------------
# 1. Extract "Path to model/logs"
# ---------------------------
path_match = re.search(r"Path to model/logs:\s*(.+)", text)
model_path = path_match.group(1).strip() if path_match else "UNKNOWN"

# ---------------------------
# 2. Extract Namespace parameters
# ---------------------------
ns_match = re.search(r"Namespace\((.*?)\)", text, re.DOTALL)
ns_dict = {}
if ns_match:
    # Keep original string as backup
    ns_body = ns_match.group(1)

    # Split respecting parentheses by using regex
    entries = re.findall(r"(\w+)=([^,]+)(?=, \w+=|$)", ns_body)
    for key, val in entries:
        ns_dict[key] = val.strip()

# add the model/log path, overriding if already present
ns_dict["path_to_logs"] = model_path

# Save as CSV
param_df = pd.DataFrame(list(ns_dict.items()), columns=["parameter", "value"])
#param_csv_path = Path( "/mnt/data/namespace_params.csv")
param_csv_path = OUTPUT_DIR / "namespace_params.csv"
param_df.to_csv(param_csv_path, index=False)

# ---------------------------
# 3. Parse training metrics
# ---------------------------
# We'll collect metrics per epoch, averaging 250/500 batches when both exist.
epoch_data = {}

# Patterns for batch lines
batch_pattern = re.compile(
    r"Training epoch\s+(\d+),\s+Batch\s+(250|500)/500:\s+LR=([0-9eE\+\-\.]+),\s+Loss=([0-9eE\+\-\.]+)\s+BER=([0-9eE\+\-\.]+)\s+LER=([0-9eE\+\-\.]+)"
)

# Pattern for detailed loss line following batch line (***Loss=...)
loss_line_pattern = re.compile(r"\*\*\*Loss=([0-9eE\+\-\.]+)(.*)")

# Token pattern to capture key=value pairs after first loss
token_pattern = re.compile(r"([A-Za-z0-9_ ]+)=([0-9eE\+\-\.]+)")

lines = text.splitlines()
for i, line in enumerate(lines):
    batch_match = batch_pattern.search(line)
    if batch_match:
        epoch = int(batch_match.group(1))
        batch_id = batch_match.group(2)  # "250" or "500"
        lr = float(batch_match.group(3))
        loss_total = float(batch_match.group(4))
        ber = float(batch_match.group(5))
        ler = float(batch_match.group(6))

        # Read the next line for detailed losses (assuming it's right after)
        if i + 1 < len(lines):
            loss_line = lines[i + 1]
            loss_match = loss_line_pattern.search(loss_line)
        else:
            loss_match = None

        detailed_losses = {}
        if loss_match:
            detailed_losses["Loss"] = float(loss_match.group(1))
            remaining = loss_match.group(2)
            for key, val in token_pattern.findall(remaining):
                key = key.strip()
                detailed_losses[key] = float(val)

        # Initialize dicts
        if epoch not in epoch_data:
            epoch_data[epoch] = {"lr": [], "Loss": [], "BER": [], "LER": []}
        epoch_metrics = epoch_data[epoch]

        # Store values
        epoch_metrics["lr"].append(lr)
        epoch_metrics["Loss"].append(loss_total)
        epoch_metrics["BER"].append(ber)
        epoch_metrics["LER"].append(ler)

        # merge detailed_losses
        for k, v in detailed_losses.items():
            epoch_metrics.setdefault(k, []).append(v)

# Aggregate (average if multiple within epoch)
aggregated = {}
for epoch, metrics in epoch_data.items():
    aggregated[epoch] = {m: np.mean(vals) for m, vals in metrics.items()}

# Convert to DataFrame sorted by epoch
metrics_df = pd.DataFrame.from_dict(aggregated, orient="index").sort_index()

# Identify the metrics we'd like to plot (all except maybe lr)
plot_metrics = [c for c in metrics_df.columns if c.lower() not in {"lr"}]
plot_metrics.insert(0, "lr")  # ensure lr first for consistent ordering

# ---------------------------
# 4. Plot metrics
# ---------------------------
n_metrics = len(plot_metrics)
n_cols = 2
n_rows = math.ceil(n_metrics / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
axes = axes.flatten()

for idx, metric in enumerate(plot_metrics):
    axes[idx].plot(metrics_df.index, metrics_df[metric])
    axes[idx].set_title(metric)
    axes[idx].set_xlabel("Epoch")
    axes[idx].grid(True)

# Hide any unused subplots
for ax in axes[n_metrics:]:
    ax.axis("off")

fig.tight_layout()
#plots_path = Path("/mnt/data/log_metrics.png")
plots_path = OUTPUT_DIR / "log_metrics.png"
fig.savefig(plots_path, dpi=150)

# ---------------------------
# 5. Extract mask tensor and save image
# ---------------------------
mask_match = re.search(r"Mask:\s*tensor\((\[.*?])\)", text, re.DOTALL)
#mask_img_path = Path("/mnt/data/mask.png")
mask_img_path = OUTPUT_DIR / "mask.png"
if mask_match:
    mask_str = mask_match.group(1)
    # Replace True/False with Python literals (they already are) and eval safely
    mask_list = ast.literal_eval(mask_str)
    mask_arr = np.array(mask_list)
    # Squeeze singleton dimensions
    mask_arr = np.squeeze(mask_arr)
    # Ensure 2D for visualization
    while mask_arr.ndim > 2:
        mask_arr = mask_arr[0]
    plt.figure(figsize=(4, 4))
    plt.imshow(mask_arr, cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(mask_img_path, dpi=150)
    plt.close()
else:
    # create placeholder empty image
    plt.figure(figsize=(4, 4))
    plt.text(0.5, 0.5, "Mask not found", ha="center", va="center")
    plt.axis("off")
    plt.savefig(mask_img_path, dpi=150)
    plt.close()

# ---------------------------
# 6. Generate model diagram (markdown bullet list)
# ---------------------------
model_section_match = re.search(r"DataParallel\((.*?)# of Parameters:", text, re.DOTALL)
#diagram_md_path = Path("/mnt/data/model_diagram.md")
diagram_md_path = OUTPUT_DIR / "model_diagram.md"
if model_section_match:
    model_text = model_section_match.group(1)
    # Clean up indentation -> bullet hierarchy
    bullets = []
    for line in model_text.splitlines():
        # count leading spaces
        stripped = line.lstrip()
        if not stripped:
            continue
        indent = len(line) - len(stripped)
        level = indent // 2  # assume 2 spaces per indent
        bullets.append(f'{"  " * level}- {stripped}')
    diagram_md_content = f"# Model Diagram (hierarchical)\n\n" + "\n".join(bullets)
else:
    diagram_md_content = "# Model Diagram\n\nModel architecture not found in log."

diagram_md_path.write_text(diagram_md_content)

# ---------------------------
# Done. Provide simple verification output.
# ---------------------------
print("Generated files:")
print(plots_path)
print(param_csv_path)
print(mask_img_path)
print(diagram_md_path)


