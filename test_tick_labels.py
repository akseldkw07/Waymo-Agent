"""Test script to debug tick label visibility issues"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Create a simple plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot some dummy data
ax.plot([0, 1, 2, 3], [0, 1, 2, 3])

# Set background color similar to the map
ax.set_facecolor("silver")
fig.patch.set_facecolor("silver")


# Custom formatter similar to what we're using
def _fmt_x(raw: float, pos: int) -> str:
    x_norm = raw * 0.5  # dummy normalization
    return f"{raw:.4f}\n({x_norm:.2f})"


def _fmt_y(raw: float, pos: int) -> str:
    y_norm = raw * 0.5  # dummy normalization
    return f"{raw:.4f}\n({y_norm:.2f})"


ax.xaxis.set_major_formatter(FuncFormatter(_fmt_x))
ax.yaxis.set_major_formatter(FuncFormatter(_fmt_y))

# Apply the same styling
ax.tick_params(
    axis="both",
    which="major",
    labelsize=13,
    colors="black",
    labelcolor="black",
    length=8,
    width=2,
    pad=10,
)

# Add white background to tick labels
for label in ax.get_xticklabels():
    label.set_bbox(dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", linewidth=1.5))
for label in ax.get_yticklabels():
    label.set_bbox(dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black", linewidth=1.5))

# Add labels and title
ax.set_xlabel(
    "X Axis Test",
    fontsize=14,
    fontweight="bold",
    color="black",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", linewidth=2),
)
ax.set_ylabel(
    "Y Axis Test",
    fontsize=14,
    fontweight="bold",
    color="black",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", linewidth=2),
)

ax.set_title(
    "Test Plot - Tick Labels Visibility",
    fontsize=16,
    fontweight="bold",
    color="black",
    pad=20,
    bbox=dict(boxstyle="round,pad=0.8", facecolor="white", edgecolor="black", linewidth=2.5),
)

plt.tight_layout()
plt.savefig("/Users/Akseldkw/coding/Columbia/RL-Project/test_tick_labels.png", dpi=150, bbox_inches="tight")
print("Test plot saved to test_tick_labels.png")

# Now let's check what's in the actual environment
print("\n" + "=" * 60)
print("Checking actual render setup...")
print("=" * 60)

from waymo_agent.graph_env import RideShareEnv
from waymo_agent.data_classes import EnvConfig

config = EnvConfig(max_new_requests_per_step=0)
env = RideShareEnv(config=config)

# Try to render
fig, ax = env.render()

# Check what tick labels actually exist
print(f"\nNumber of X tick labels: {len(ax.get_xticklabels())}")
print(f"Number of Y tick labels: {len(ax.get_yticklabels())}")

print("\nX tick label properties:")
for i, label in enumerate(ax.get_xticklabels()[:3]):  # Just first 3
    print(f"  Label {i}: visible={label.get_visible()}, text='{label.get_text()}'")
    bbox = label.get_bbox_patch()
    if bbox:
        print(f"    Has bbox: visible={bbox.get_visible()}")

print("\nY tick label properties:")
for i, label in enumerate(ax.get_yticklabels()[:3]):  # Just first 3
    print(f"  Label {i}: visible={label.get_visible()}, text='{label.get_text()}'")
    bbox = label.get_bbox_patch()
    if bbox:
        print(f"    Has bbox: visible={bbox.get_visible()}")

plt.savefig("/Users/Akseldkw/coding/Columbia/RL-Project/test_env_render.png", dpi=150, bbox_inches="tight")
print("\nEnvironment render saved to test_env_render.png")
plt.close("all")
