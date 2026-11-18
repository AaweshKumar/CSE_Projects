"""Standalone Matplotlib 3D visualizer for hydrogenic orbitals.

Displays electron probability cloud in Bohr radii a0.
"""

import matplotlib.pyplot as plt
from maths import generate_point_cloud, get_energy, SPECTROSCOPIC_LETTERS


def render(n: int, l: int, m: int, Z: int = 1, num_points: int = 3000) -> None:
    """Render hydrogenic orbital using Matplotlib 3D scatter plot.

    Args:
        n, l, m: Quantum numbers.
        Z: Atomic number.
        num_points: Number of points to sample.
    """
    print(f"Generating point cloud for n={n}, l={l}, m={m}, Z={Z}...")
    cloud = generate_point_cloud(n, l, m, Z, num_points)

    x = cloud[:, 0]
    y = cloud[:, 1]
    z = cloud[:, 2]
    probability = cloud[:, 3]

    # Normalize probability to 0-1 for colormap, guarding against zero range
    p_min = probability.min()
    p_max = probability.max()
    denom = p_max - p_min
    if denom < 1e-15:
        denom = 1.0
    probability_normalized = (probability - p_min) / denom
    colors = plt.cm.plasma(probability_normalized)

    figure = plt.figure(figsize=(10, 8))
    figure.patch.set_facecolor("#020408")
    axes = figure.add_subplot(111, projection="3d")
    axes.set_facecolor("#020408")

    axes.scatter(x, y, z, c=colors, s=1.5, alpha=0.6, linewidths=0)
    axes.scatter([0], [0], [0], color="white", s=80, zorder=10)

    energy = get_energy(n, Z, unit="eV")
    letter = SPECTROSCOPIC_LETTERS[l] if l < len(SPECTROSCOPIC_LETTERS) else "?"
    orbital_label = f"{n}{letter}"
    axes.set_title(
        f"Orbital: {orbital_label}  |  n={n}, l={l}, m={m}  |  Z={Z}  |  E={energy:.2f} eV",
        color="white",
        fontsize=11,
        pad=15,
    )

    axes.xaxis.pane.fill = False
    axes.yaxis.pane.fill = False
    axes.zaxis.pane.fill = False
    axes.xaxis.pane.set_edgecolor("#112233")
    axes.yaxis.pane.set_edgecolor("#112233")
    axes.zaxis.pane.set_edgecolor("#112233")
    axes.tick_params(axis="x", colors="#334455")
    axes.tick_params(axis="y", colors="#334455")
    axes.tick_params(axis="z", colors="#334455")

    axes.set_xlabel("x (a₀)", color="#667788")
    axes.set_ylabel("y (a₀)", color="#667788")
    axes.set_zlabel("z (a₀)", color="#667788")

    plt.tight_layout()
    plt.show()
