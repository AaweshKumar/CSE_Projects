import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QSlider, QLabel, QPushButton, QComboBox, QGroupBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from maths import generate_point_cloud, get_energy


# List of elements the user can pick from
ELEMENTS = {
    1:'H',  2:'He', 3:'Li',  4:'Be', 5:'B',
    6:'C',  7:'N',  8:'O',   9:'F',  10:'Ne',
    26:'Fe', 29:'Cu', 47:'Ag', 79:'Au'
}

ORBITAL_SHAPES = {
    0: 'Sphere (s)',
    1: 'Dumbbell (p)',
    2: 'Clover (d)',
    3: 'Complex (f)'
}


class QuantumSimulatorApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Atom Simulator")
        self.setMinimumSize(1100, 700)

        # Current quantum state
        self.n = 1
        self.l = 0
        self.m = 0
        self.Z = 1
        self.num_points = 8000

        self.setup_ui()
        self.render_orbital()

    # ────────────────────────────────────────────
    # BUILD THE FULL UI
    # ────────────────────────────────────────────
    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left side = controls, Right side = 3D plot
        left_panel  = self.build_left_panel()
        right_panel = self.build_right_panel()

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

    # ────────────────────────────────────────────
    # LEFT PANEL — all controls
    # ────────────────────────────────────────────
    def build_left_panel(self):
        panel = QWidget()
        panel.setFixedWidth(250)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # Element dropdown
        element_group  = QGroupBox("Element")
        element_layout = QVBoxLayout(element_group)
        self.element_dropdown = QComboBox()
        for atomic_number, symbol in ELEMENTS.items():
            self.element_dropdown.addItem(f"{symbol}  (Z={atomic_number})", atomic_number)
        self.element_dropdown.currentIndexChanged.connect(self.on_element_changed)
        element_layout.addWidget(self.element_dropdown)

        # Quantum number sliders
        quantum_group  = QGroupBox("Quantum Numbers")
        quantum_layout = QVBoxLayout(quantum_group)
        quantum_layout.setSpacing(10)

        self.n_label  = QLabel("n = 1  (principal)")
        self.n_slider = self.make_slider(min=1, max=5, start=1)
        self.n_slider.valueChanged.connect(self.on_n_changed)

        self.l_label  = QLabel("l = 0  (angular)")
        self.l_slider = self.make_slider(min=0, max=0, start=0)
        self.l_slider.valueChanged.connect(self.on_l_changed)

        self.m_label  = QLabel("m = 0  (magnetic)")
        self.m_slider = self.make_slider(min=0, max=0, start=0)
        self.m_slider.valueChanged.connect(self.on_m_changed)

        for widget in [self.n_label, self.n_slider,
                       self.l_label, self.l_slider,
                       self.m_label, self.m_slider]:
            quantum_layout.addWidget(widget)

        # Info box
        info_group  = QGroupBox("Orbital Info")
        info_layout = QVBoxLayout(info_group)
        small_font  = QFont("Courier New", 10)

        self.label_orbital = QLabel("Orbital : 1s")
        self.label_energy  = QLabel("Energy  : -13.60 eV")
        self.label_shape   = QLabel("Shape   : Sphere (s)")
        self.label_rnodes  = QLabel("Radial nodes  : 0")
        self.label_anodes  = QLabel("Angular nodes : 0")

        for label in [self.label_orbital, self.label_energy,
                      self.label_shape, self.label_rnodes, self.label_anodes]:
            label.setFont(small_font)
            info_layout.addWidget(label)

        # Render button
        self.render_button = QPushButton("RENDER ORBITAL")
        self.render_button.setFixedHeight(40)
        self.render_button.clicked.connect(self.render_orbital)

        # Add everything to left panel
        layout.addWidget(element_group)
        layout.addWidget(quantum_group)
        layout.addWidget(info_group)
        layout.addWidget(self.render_button)
        layout.addStretch()

        return panel

    # ────────────────────────────────────────────
    # RIGHT PANEL — 3D matplotlib canvas
    # ────────────────────────────────────────────
    def build_right_panel(self):
        self.figure = plt.figure(facecolor='#050a10')
        self.canvas = FigureCanvas(self.figure)
        return self.canvas

    # ────────────────────────────────────────────
    # HELPER — create a horizontal slider
    # ────────────────────────────────────────────
    def make_slider(self, min, max, start):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min)
        slider.setMaximum(max)
        slider.setValue(start)
        return slider

    # ────────────────────────────────────────────
    # SLOTS — called when user changes something
    # ────────────────────────────────────────────
    def on_element_changed(self, index):
        self.Z = self.element_dropdown.itemData(index)
        self.update_info_panel()

    def on_n_changed(self, new_n):
        self.n = new_n
        self.n_label.setText(f"n = {new_n}  (principal)")

        # l cannot be >= n so clamp the l slider max
        self.l_slider.setMaximum(new_n - 1)
        if self.l >= new_n:
            self.l_slider.setValue(new_n - 1)

        self.update_info_panel()

    def on_l_changed(self, new_l):
        self.l = new_l
        self.l_label.setText(f"l = {new_l}  (angular)")

        # m must stay between -l and +l
        self.m_slider.setMinimum(-new_l)
        self.m_slider.setMaximum(new_l)
        if abs(self.m) > new_l:
            self.m_slider.setValue(0)

        self.update_info_panel()

    def on_m_changed(self, new_m):
        self.m = new_m
        self.m_label.setText(f"m = {new_m}  (magnetic)")
        self.update_info_panel()

    # ────────────────────────────────────────────
    # UPDATE the info panel text
    # ────────────────────────────────────────────
    def update_info_panel(self):
        orbital_letters = 'spldf'
        orbital_name    = f"{self.n}{orbital_letters[self.l]}"
        element_symbol  = ELEMENTS.get(self.Z, f"Z={self.Z}")
        energy          = get_energy(self.n, self.Z)
        shape           = ORBITAL_SHAPES.get(self.l, '?')
        radial_nodes    = self.n - self.l - 1
        angular_nodes   = self.l

        self.label_orbital.setText(f"Orbital : {element_symbol} {orbital_name}")
        self.label_energy.setText( f"Energy  : {energy:.2f} eV")
        self.label_shape.setText(  f"Shape   : {shape}")
        self.label_rnodes.setText( f"Radial nodes  : {radial_nodes}")
        self.label_anodes.setText( f"Angular nodes : {angular_nodes}")

    # ────────────────────────────────────────────
    # RENDER — generate cloud and draw it
    # ────────────────────────────────────────────
    def render_orbital(self):
        self.render_button.setEnabled(False)
        self.render_button.setText("Calculating...")
        QApplication.processEvents()  # refresh UI before heavy calculation

        # Generate point cloud from maths.py
        cloud = generate_point_cloud(self.n, self.l, self.m, self.Z, self.num_points)

        # Split into coordinates and probability
        x = cloud[:, 0] * self.Z   # scale by Z so shape stays visible
        y = cloud[:, 1] * self.Z
        z = cloud[:, 2] * self.Z
        probability = cloud[:, 3]

        # Normalize probability to 0-1 for the colormap
        prob_normalized = (probability - probability.min()) / (probability.max() - probability.min() + 1e-10)
        colors = plt.cm.plasma(prob_normalized)

        # Clear old plot and start fresh
        self.figure.clear()
        axes = self.figure.add_subplot(111, projection='3d')
        axes.set_facecolor('#050a10')

        # Draw the electron probability cloud
        axes.scatter(x, y, z, c=colors, s=1.2, alpha=0.5, linewidths=0)

        # Draw nucleus
        axes.scatter([0], [0], [0], color='white', s=100, zorder=10)

        # Build and set the title
        orbital_name   = f"{self.n}{'spldf'[self.l]}"
        element_symbol = ELEMENTS.get(self.Z, f"Z={self.Z}")
        energy         = get_energy(self.n, self.Z)
        axes.set_title(
            f"{element_symbol}  |  {orbital_name}  |  n={self.n}, l={self.l}, m={self.m}  |  E={energy:.2f} eV",
            color='white', fontsize=10
        )

        # Clean up grid walls
        for pane in [axes.xaxis.pane, axes.yaxis.pane, axes.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('#0a1a2a')
        axes.tick_params(axis='x', colors='#223344')
        axes.tick_params(axis='y', colors='#223344')
        axes.tick_params(axis='z', colors='#223344')

        self.canvas.draw()
        # Enable scroll to zoom
        def on_scroll(event):
            current_xlim = axes.get_xlim()
            current_ylim = axes.get_ylim()
            current_zlim = axes.get_zlim()

            # zoom in or out by 10% depending on scroll direction
            if event.button == 'up':
                scale = 0.9   # zoom in
            else:
                scale = 1.1   # zoom out

            axes.set_xlim([x * scale for x in current_xlim])
            axes.set_ylim([y * scale for y in current_ylim])
            axes.set_zlim([z * scale for z in current_zlim])
            self.canvas.draw()

        self.figure.canvas.mpl_connect('scroll_event', on_scroll)
        self.update_info_panel()

        self.render_button.setEnabled(True)
        self.render_button.setText("RENDER ORBITAL")


if __name__ == "__main__":
    app    = QApplication(sys.argv)
    app.setFont(QFont("Courier New", 10))
    window = QuantumSimulatorApp()
    window.show()
    sys.exit(app.exec_())
