import typing as t
from pathlib import Path

import torch

# Define paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MAP_DIR = ROOT_DIR / "maps"
IMG_DIR = ROOT_DIR / "img"
MODEL_WEIGHT_DIR = ROOT_DIR / "model_weights"

# IMG Paths
MANHATTAN_BASIC = "manhattan-full-basic.png"
MANHATTAN_COLORFUL = "manhattan-full-colorful.png"
MANHATTAN_SPARSE_COLORFUL = "manhattan-sparse-colorful.png"
MANHATTAN_SPARSE_COLORFUL_TEST = "manhattan-sparse-colorful-test.png"

# Map Names
MANHATTAN_RAW_GRAPH = "manhattan-raw.graphml"
MANHATTAN_PROCESSED_GRAPH = "manhattan-processed.graphml"
MANHATTAN_SPARSE_GRAPH = "manhattan-sparse-{}-nodes.graphml"

# WANDB
WANDB_TEAM_NAME = "rl-project-F25"
WANDB_PROJECT_NAME = "Waymo-Agent-Phase1"

# DEVICE
DEVICE_LITERAL = t.Literal["cuda", "mps", "xpu", "cpu"]  # extend to include "xla", "xpu" if needed


def pick_device() -> DEVICE_LITERAL:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    # If using Intel GPUs:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


DEVICE_TORCH_STR: DEVICE_LITERAL = pick_device()
DEVICE = torch.device(DEVICE_TORCH_STR)
