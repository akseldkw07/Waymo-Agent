# autoflake: off
# autoflake: skip_file
# isort: skip_file
from __future__ import annotations
import json
import os
import random
from turtle import bgcolor
import typing as t
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor

OX_PLOT_DEFAULTS = {
    "node_size": 5,
    "node_color": "white",
    "node_alpha": 0.5,
    "figsize": (10, 10),
    "bgcolor": "black",
    "show": True,
}
