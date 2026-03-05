import numpy as np
import matplotlib.pyplot as plt
from maths import generate_point_cloud, get_energy

def render(n, l, m, Z=1, num_points=3000):

    print(f"Generating point cloud for n={n}, l={l}, m={m}, Z={Z}...")
    cloud = generate_point_cloud(n, l, m, Z, num_points)

    # Split cloud array into coordinates and probability
    x           = cloud[:, 0]
    y           = cloud[:, 1]
    z           = cloud[:, 2]
    probability = cloud[:, 3]

    # Normalize probability to 0-1 so colormap understands it
    probability_normalized = (probability - probability.min()) / (probability.max() - probability.min())
    colors = plt.cm.cool(probability_normalized)

    # Create dark window and 3D plot area
    figure = plt.figure(figsize=(10, 8))
    figure.patch.set_facecolor('#020408')
    axes = figure.add_subplot(111, projection='3d')
    axes.set_facecolor('#020408')

    # Draw electron cloud and nucleus
    axes.scatter(x, y, z, c=colors, s=1.5, alpha=0.6, linewidths=0)
    axes.scatter([0], [0], [0], color='white', s=80, zorder=10)

    # Title
    energy        = get_energy(n, Z)
    orbital_names = {0:'s', 1:'p', 2:'d', 3:'f'}
    orbital_label = f"{n}{orbital_names.get(l, '?')}"
    axes.set_title(
        f"Orbital: {orbital_label}  |  n={n}, l={l}, m={m}  |  Z={Z}  |  E={energy:.2f} eV",
        color='white', fontsize=11, pad=15
    )

    # Style grid and axes
    axes.xaxis.pane.fill = False
    axes.yaxis.pane.fill = False
    axes.zaxis.pane.fill = False
    axes.xaxis.pane.set_edgecolor('#112233')
    axes.yaxis.pane.set_edgecolor('#112233')
    axes.zaxis.pane.set_edgecolor('#112233')
    axes.tick_params(axis='x', colors='#334455')
    axes.tick_params(axis='y', colors='#334455')
    axes.tick_params(axis='z', colors='#334455')

    plt.tight_layout()
    plt.show()

