"""Quantum Atom Simulator Desktop GUI Application (Phase U1 - U7 / Addendum v3).

Calm, restrained desktop-science-tool look:
- Palette: #0e1014 (bg), #14171c (panel), #1c2027 (raised), #2a2f38 (border),
           #d9dce1 (text), #8b929d (dim), #d4893a (muted amber accent).
- Font: Calibri 11pt across all widgets, menus, tooltips, dialogs, and plots.
- Fusion style with full dark QPalette: no white/light surfaces anywhere.
- Windows native dark title bar via DwmSetWindowAttribute.
- Section headers with 1px divider line; unboxed 3D scale bar and plasma legend.
- Pinned 'Render orbital' button (normal case, 38px, muted amber).
- High headroom (25%) on radial plot preventing peak label cutoff.
"""

from __future__ import annotations
import math
import os
import sys
import time
from typing import Any
import numpy as np

import matplotlib.cm as cm
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QEvent, QPointF, QSettings, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import (
    QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QSurfaceFormat
)
from PyQt5.QtWidgets import (
    QAction, QApplication, QComboBox, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMenuBar, QProgressBar, QPushButton,
    QScrollArea, QShortcut, QSlider, QSplitter, QStatusBar, QVBoxLayout, QWidget
)

from atomdata import ELEMENT_SYMBOLS, get_ground_state_config
from maths import HARTREE_EV, AU_TIME_S, get_energy, radial_wavefunction
from radial_plot import (
    find_mean_radius, find_most_probable_radius, find_radial_nodes, radial_plot_data
)
from sampling import get_hydrogenic_r999, sample_hydrogenic_orbital, sample_orbital
from scf import _SCF_CACHE, solve_atom, zeff_of_r
from superposition import (
    SuperpositionState, dipole_matrix_element, stark_n2_hydrogen, transition_info
)
from theme import (
    ACCENT_COLOR, ACCENT_HOVER, APP_STYLESHEET, BORDER_COLOR, BG_COLOR,
    FONT_FAMILY, FONT_SIZE, GRID_COLOR, PANEL_COLOR, RAISED_COLOR,
    TEXT_COLOR, TEXT_DIM, TICK_COLOR, TITLE_TEXT_COLOR, WARNING_COLOR,
    apply_dark_theme, apply_dark_title_bar, get_application_font
)
from validation_dialog import ValidationDialog

# Spectroscopic letters and shapes
ORBITAL_LETTERS = "spdfg"
ORBITAL_SHAPES = {
    0: "Sphere (s)",
    1: "Dumbbell (p)",
    2: "Clover (d)",
    3: "Complex (f)",
    4: "Complex (g)",
}

AVAILABLE_Z = list(range(1, 37)) + [47, 79]


def get_ion_symbol(Z: int) -> str:
    """Return one-electron ion symbol for hydrogenic mode, e.g. H, He+, Fe25+, Au78+."""
    sym = ELEMENT_SYMBOLS.get(Z, f"Z{Z}")
    charge = Z - 1
    if charge == 0:
        return sym
    elif charge == 1:
        return f"{sym}+"
    else:
        return f"{sym}{charge}+"


def create_plasma_colorbar_pixmap(width: int = 10, height: int = 70) -> QPixmap:
    """Render vertical plasma colormap gradient strip for 3D overlay legend."""
    img = QImage(width, height, QImage.Format_RGBA8888)
    for y in range(height):
        # 1.0 at top, 0.0 at bottom
        val = 1.0 - (float(y) / max(1.0, float(height - 1)))
        rgba = cm.plasma(val)
        c = QColor.fromRgbF(rgba[0], rgba[1], rgba[2], 0.95)
        for x in range(width):
            img.setPixelColor(x, y, c)
    return QPixmap.fromImage(img)


class ScaleBarOverlay(QWidget):
    """Unboxed minimal scale bar in the 3D viewport (plain line with label beneath)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = "1 a₀"
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFixedSize(65, 26)

    def set_step(self, step_text: str):
        self.text = step_text
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(TEXT_DIM), 1.2)
        p.setPen(pen)
        # 1px horizontal line with small vertical ticks at ends
        p.drawLine(6, 6, 52, 6)
        p.drawLine(6, 3, 6, 9)
        p.drawLine(52, 3, 52, 9)
        # Label beneath
        p.setFont(QFont(FONT_FAMILY, 9))
        p.setPen(QColor(TEXT_DIM))
        p.drawText(0, 11, 58, 14, Qt.AlignCenter, self.text)


class LegendOverlay(QWidget):
    """Unboxed minimal plasma colormap legend in the 3D viewport."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFixedSize(55, 125)
        self._pixmap = create_plasma_colorbar_pixmap(10, 68)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(QFont(FONT_FAMILY, 9))
        p.setPen(QColor(TEXT_DIM))
        # Header caption
        p.drawText(0, 0, 55, 14, Qt.AlignCenter, "|ψ|²")
        # 'high' label
        p.setFont(QFont(FONT_FAMILY, 8))
        p.drawText(0, 16, 55, 13, Qt.AlignCenter, "high")
        # Color gradient strip (width 10, height 68, centered at x=22, y=32)
        p.drawPixmap(22, 32, self._pixmap)
        # 'low' label
        p.drawText(0, 104, 55, 13, Qt.AlignCenter, "low")


class SCFWorkerThread(QThread):
    """Background worker thread for asynchronous SCF calculations."""
    finished_res = pyqtSignal(int, object)  # (request_id, SCFResult)

    def __init__(self, Z: int, request_id: int):
        super().__init__()
        self.Z = Z
        self.request_id = request_id

    def run(self):
        res = solve_atom(self.Z)
        self.finished_res.emit(self.request_id, res)


class SCFPrewarmThread(QThread):
    """Low-priority background thread to pre-warm common atoms in SCF cache."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        common_z = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 29, 36]
        for z in common_z:
            if self._stop_requested:
                break
            solve_atom(z)


class QuantumSimulatorApp(QMainWindow):
    """Calm desktop-science Quantum Atom Simulator application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantum Atom Simulator")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(APP_STYLESHEET)

        icon_path = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))), "app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.app_font = get_application_font()
        self.setFont(self.app_font)

        # Simulator State
        self.mode = "Hydrogenic"
        self.Z = 1
        self.num_points = 8000

        # Hydrogenic Quantum numbers
        self.n = 1
        self.l = 0
        self.m = 0

        # Superposition State
        self.sup_state_a = [1, 0, 0]
        self.sup_state_b = [2, 1, 0]
        self.sup_mix_angle = 45.0
        self.sup_period_s = 3.0
        self.sup_obj: SuperpositionState | None = None
        self.is_playing = True
        self.anim_time_s = 0.0
        self.last_anim_tick = time.perf_counter()

        # SCF State
        self.scf_res = None
        self.scf_selected_nl = (1, 0)
        self._scf_request_counter = 0
        self._current_scf_worker: SCFWorkerThread | None = None

        # Stark State
        self.stark_f_vm = 1.0e9
        self.stark_state_choice = "Upper state (+)"

        # Animation Timer
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(30)
        self.anim_timer.timeout.connect(self._on_anim_frame)

        # Debounced slider auto-render (150 ms)
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(150)
        self._debounce_timer.timeout.connect(self.render_orbital)

        # Setup menus, widgets, status bar, and shortcuts
        self._setup_menu_bar()
        self._setup_ui()
        self._setup_status_bar()
        self._setup_shortcuts()

        # Restore window geometry from QSettings
        self._restore_window_geometry()

        # Pre-warm common elements in background after entering event loop
        self._prewarm_thread: SCFPrewarmThread | None = None
        QTimer.singleShot(1500, self._start_prewarm)

        # Initial render
        self.render_orbital()

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_title_bar(self)

    def _start_prewarm(self):
        """Start prewarming after window is shown and idle."""
        self._prewarm_thread = SCFPrewarmThread(self)
        self._prewarm_thread.start()

    # ──────────────────────────────────────────────────────────────────────────
    # SETUP: MENU BAR, UI, STATUS BAR, SHORTCUTS
    # ──────────────────────────────────────────────────────────────────────────
    def _setup_menu_bar(self):
        menubar = self.menuBar()
        tools_menu = menubar.addMenu("Tools")

        val_action = QAction("Validation plots…", self)
        val_action.setShortcut(QKeySequence("Ctrl+V"))
        val_action.setStatusTip("Open NIST ionization energy and Moseley Kα validation dialog")
        val_action.triggered.connect(self._open_validation_dialog)
        tools_menu.addAction(val_action)

        reset_action = QAction("Reset 3D View", self)
        reset_action.setShortcut(QKeySequence("Home"))
        reset_action.setStatusTip("Reset 3D camera to default elevation and azimuth")
        reset_action.triggered.connect(self._reset_camera_view)
        tools_menu.addAction(reset_action)

    def _setup_ui(self):
        main_widget = QWidget(self)
        main_widget.setObjectName("centralWidget")
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left Column (300px fixed width, containing scroll area + pinned Render button)
        left_container = self._build_left_container()
        main_layout.addWidget(left_container)

        # Right Column (Vertical Splitter: 3D View ~78% + 2D Plot ~22%)
        right_container = self._build_right_container()
        main_layout.addWidget(right_container, stretch=1)

    def _setup_status_bar(self):
        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self.status_state_label = QLabel("Ready")
        self.status_state_label.setFont(QFont(FONT_FAMILY, 10))
        self.status_bar.addWidget(self.status_state_label, 1)

        self.status_progress = QProgressBar(self)
        self.status_progress.setRange(0, 0)
        self.status_progress.setFixedWidth(110)
        self.status_progress.setFixedHeight(12)
        self.status_progress.setVisible(False)
        self.status_bar.addWidget(self.status_progress)

        self.status_hint_label = QLabel("drag: rotate · scroll: zoom · double-click: reset")
        self.status_hint_label.setFont(QFont(FONT_FAMILY, 9.5))
        self.status_hint_label.setFont(QFont(FONT_FAMILY, 10))
        self.status_hint_label.setFont(QFont(FONT_FAMILY, 10))
        self.status_hint_label.setStyleSheet(f"color: {TEXT_DIM};")
        self.status_bar.addPermanentWidget(self.status_hint_label)

        self.status_perf_label = QLabel(f"{self.num_points} pts · 0 ms")
        self.status_perf_label.setFont(QFont(FONT_FAMILY, 10))
        self.status_perf_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-weight: 600;")
        self.status_bar.addPermanentWidget(self.status_perf_label)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("R"), self, self.render_orbital)
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_playback)
        QShortcut(QKeySequence(Qt.Key_Home), self, self._reset_camera_view)

    def _restore_window_geometry(self):
        settings = QSettings("QuantumAtomSimulator", "WindowGeometry")
        geom = settings.value("geometry")
        if geom is not None and isinstance(geom, (bytes, bytearray)):
            self.restoreGeometry(geom)
        else:
            screen = QApplication.primaryScreen().geometry()
            x = max(0, (screen.width() - 1280) // 2)
            y = max(0, (screen.height() - 800) // 2)
            self.setGeometry(x, y, 1280, 800)

    def closeEvent(self, event):
        if hasattr(self, "_prewarm_thread") and self._prewarm_thread is not None and self._prewarm_thread.isRunning():
            self._prewarm_thread.stop()
            self._prewarm_thread.quit()
            self._prewarm_thread.wait(300)
            self._prewarm_thread.wait(1000)
        if hasattr(self, "_current_scf_worker") and self._current_scf_worker is not None and self._current_scf_worker.isRunning():
            self._current_scf_worker.quit()
            self._current_scf_worker.wait(300)
            self._current_scf_worker.wait(1000)
        settings = QSettings("QuantumAtomSimulator", "WindowGeometry")
        settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)

    # ──────────────────────────────────────────────────────────────────────────
    # LEFT PANEL CONTROLS (T1, T3, T4)
    # ──────────────────────────────────────────────────────────────────────────
    def _build_left_container(self) -> QWidget:
        container = QWidget()
        container.setObjectName("leftPanelContainer")
        container.setFixedWidth(300)
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(10)

        # Frameless QScrollArea for the sections
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.viewport().setAutoFillBackground(True)

        inner_widget = QWidget()
        inner_widget.setObjectName("leftPanelInner")
        inner_layout = QVBoxLayout(inner_widget)
        inner_layout.setContentsMargins(0, 0, 4, 0)
        inner_layout.setSpacing(10)

        # 1. Mode Section Header
        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(0, 4, 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.setFont(self.app_font)
        self.mode_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.mode_combo.setMinimumContentsLength(18)
        self.mode_combo.addItems([
            "Hydrogenic",
            "SCF (multi-electron)",
            "Superposition",
            "Stark (n=2)",
        ])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        inner_layout.addWidget(mode_group)

        # 2. Element Section Header
        self.element_group = QGroupBox("Element")
        element_layout = QVBoxLayout(self.element_group)
        element_layout.setContentsMargins(0, 4, 0, 0)
        self.element_dropdown = QComboBox()
        self.element_dropdown.setFont(self.app_font)
        self.element_dropdown.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.element_dropdown.setMinimumContentsLength(18)
        self._populate_elements_dropdown()
        self.element_dropdown.currentIndexChanged.connect(self._on_element_changed)
        element_layout.addWidget(self.element_dropdown)
        inner_layout.addWidget(self.element_group)

        # 3. Quantum Numbers Section Header
        self.qn_group = QGroupBox("Quantum Numbers")
        self.qn_layout = QVBoxLayout(self.qn_group)
        self.qn_layout.setContentsMargins(0, 4, 0, 0)
        self.qn_layout.setSpacing(6)

        self._build_hydrogenic_controls()
        self._build_scf_controls()
        self._build_superposition_controls()
        self._build_stark_controls()
        inner_layout.addWidget(self.qn_group)

        # 4. Orbital Info Section Header (Clean QFormLayout, right-aligned keys)
        self.info_group = QGroupBox("Orbital Info")
        self.info_layout = QFormLayout(self.info_group)
        self.info_layout.setContentsMargins(0, 4, 0, 0)
        self.info_layout.setHorizontalSpacing(10)
        self.info_layout.setVerticalSpacing(4)
        self.info_layout.setLabelAlignment(Qt.AlignRight)
        self.info_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.info_rows: list[tuple[QLabel, QLabel]] = []
        for _ in range(7):
            lbl_k = QLabel("")
            lbl_k.setFont(self.app_font)
            lbl_k.setFixedWidth(68)
            lbl_k.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl_k.setStyleSheet(f"color: {TEXT_DIM};")
            lbl_v = QLabel("")
            lbl_v.setFont(self.app_font)
            lbl_v.setStyleSheet(f"color: {TEXT_COLOR};")
            lbl_v.setWordWrap(True)
            self.info_rows.append((lbl_k, lbl_v))
            self.info_layout.addRow(lbl_k, lbl_v)

        self.info_honesty_label = QLabel("")
        self.info_honesty_label.setFont(QFont(FONT_FAMILY, 9.5))
        self.info_honesty_label.setFont(QFont(FONT_FAMILY, 10))
        self.info_honesty_label.setFont(QFont(FONT_FAMILY, 10))
        self.info_honesty_label.setStyleSheet(f"color: {WARNING_COLOR}; padding-top: 4px;")
        self.info_honesty_label.setWordWrap(True)
        self.info_layout.addRow(self.info_honesty_label)

        inner_layout.addWidget(self.info_group)
        inner_layout.addStretch()

        self.scroll_area.setWidget(inner_widget)
        vbox.addWidget(self.scroll_area, stretch=1)

        # 5. Pinned 'Render orbital' button (38px high, normal case, solid muted accent)
        self.render_button = QPushButton("Render orbital")
        self.render_button.setObjectName("primaryButton")
        self.render_button.setFixedHeight(38)
        self.render_button.setFont(QFont(FONT_FAMILY, 11, QFont.DemiBold))
        self.render_button.clicked.connect(self.render_orbital)
        vbox.addWidget(self.render_button)

        return container

    def _build_hydrogenic_controls(self):
        self.hydro_widget = QWidget()
        layout = QVBoxLayout(self.hydro_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.n_label = QLabel("n = 1  (principal)")
        self.n_label.setWordWrap(True)
        self.n_slider = self._make_slider(min_val=1, max_val=5, start=1)
        self.n_slider.valueChanged.connect(self._on_n_changed)
        self.n_slider.sliderReleased.connect(self._on_slider_released)

        self.l_label = QLabel("l = 0  (s)")
        self.l_label.setWordWrap(True)
        self.l_slider = self._make_slider(min_val=0, max_val=0, start=0)
        self.l_slider.valueChanged.connect(self._on_l_changed)
        self.l_slider.sliderReleased.connect(self._on_slider_released)

        self.m_label = QLabel("m = 0  (magnetic)")
        self.m_label.setWordWrap(True)
        self.m_slider = self._make_slider(min_val=0, max_val=0, start=0)
        self.m_slider.valueChanged.connect(self._on_m_changed)
        self.m_slider.sliderReleased.connect(self._on_slider_released)

        for w in [self.n_label, self.n_slider, self.l_label, self.l_slider, self.m_label, self.m_slider]:
            w.setFont(self.app_font)
            layout.addWidget(w)

        self.qn_layout.addWidget(self.hydro_widget)

    def _build_scf_controls(self):
        self.scf_widget = QWidget()
        layout = QVBoxLayout(self.scf_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.scf_orb_label = QLabel("Subshell Orbital:")
        self.scf_orb_label.setWordWrap(True)
        self.scf_orb_combo = QComboBox()
        self.scf_orb_combo.setFont(self.app_font)
        self.scf_orb_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.scf_orb_combo.setMinimumContentsLength(18)
        self.scf_orb_combo.currentIndexChanged.connect(self._on_scf_orb_changed)

        self.scf_model_label = QLabel("Model: Spherically-averaged Xα central field")
        self.scf_model_label.setFont(QFont(FONT_FAMILY, 9.5))
        self.scf_model_label.setFont(QFont(FONT_FAMILY, 10))
        self.scf_model_label.setFont(QFont(FONT_FAMILY, 10))
        self.scf_model_label.setStyleSheet(f"color: {TEXT_DIM};")
        self.scf_model_label.setWordWrap(True)

        layout.addWidget(self.scf_orb_label)
        layout.addWidget(self.scf_orb_combo)
        layout.addWidget(self.scf_model_label)

        self.scf_widget.setVisible(False)
        self.qn_layout.addWidget(self.scf_widget)

    def _build_superposition_controls(self):
        self.sup_widget = QWidget()
        layout = QVBoxLayout(self.sup_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        preset_lbl = QLabel("Preset:")
        preset_lbl.setWordWrap(True)
        layout.addWidget(preset_lbl)

        self.preset_combo = QComboBox()
        self.preset_combo.setFont(self.app_font)
        self.preset_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.preset_combo.setMinimumContentsLength(18)
        self.preset_combo.addItems([
            "1s + 2pz",
            "2s + 2pz",
            "1s + 2s",
            "2pz + 3dz²",
        ])
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        layout.addWidget(self.preset_combo)

        # State A Controls
        self.sup_lbl_a = QLabel("State A: 1s (1,0,0)")
        self.sup_lbl_a.setWordWrap(True)
        self.sup_slider_na = self._make_slider(1, 4, 1)
        self.sup_slider_la = self._make_slider(0, 0, 0)
        self.sup_slider_ma = self._make_slider(0, 0, 0)

        # State B Controls
        self.sup_lbl_b = QLabel("State B: 2p (2,1,0)")
        self.sup_lbl_b.setWordWrap(True)
        self.sup_slider_nb = self._make_slider(1, 4, 2)
        self.sup_slider_lb = self._make_slider(0, 1, 1)
        self.sup_slider_mb = self._make_slider(-1, 1, 0)

        for s in [self.sup_slider_na, self.sup_slider_la, self.sup_slider_ma,
                  self.sup_slider_nb, self.sup_slider_lb, self.sup_slider_mb]:
            s.valueChanged.connect(self._on_sup_state_changed)

        layout.addWidget(self.sup_lbl_a)
        layout.addWidget(self.sup_slider_na)
        layout.addWidget(self.sup_slider_la)
        layout.addWidget(self.sup_slider_ma)

        layout.addWidget(self.sup_lbl_b)
        layout.addWidget(self.sup_slider_nb)
        layout.addWidget(self.sup_slider_lb)
        layout.addWidget(self.sup_slider_mb)

        # Mixing Angle Slider
        self.sup_mix_label = QLabel("Mixing: 50% A + 50% B")
        self.sup_mix_label.setWordWrap(True)
        self.sup_mix_slider = self._make_slider(0, 90, 45)
        self.sup_mix_slider.valueChanged.connect(self._on_mix_changed)
        layout.addWidget(self.sup_mix_label)
        layout.addWidget(self.sup_mix_slider)

        # Speed Slider
        self.sup_speed_label = QLabel("Beat period: 3.0 s")
        self.sup_speed_label.setWordWrap(True)
        self.sup_speed_slider = self._make_slider(1, 10, 3)
        self.sup_speed_slider.valueChanged.connect(self._on_speed_changed)
        layout.addWidget(self.sup_speed_label)
        layout.addWidget(self.sup_speed_slider)

        # Play/Pause button and timescale label
        self.play_button = QPushButton("PAUSE")
        self.play_button.setFont(self.app_font)
        self.play_button.clicked.connect(self._toggle_playback)
        layout.addWidget(self.play_button)

        self.sup_scale_label = QLabel("Slowed: 1 s = 0.14 fs")
        self.sup_scale_label.setFont(QFont(FONT_FAMILY, 9.5))
        self.sup_scale_label.setFont(QFont(FONT_FAMILY, 10))
        self.sup_scale_label.setFont(QFont(FONT_FAMILY, 10))
        self.sup_scale_label.setStyleSheet(f"color: {ACCENT_COLOR};")
        self.sup_scale_label.setWordWrap(True)
        layout.addWidget(self.sup_scale_label)

        self.sup_widget.setVisible(False)
        self.qn_layout.addWidget(self.sup_widget)

    def _build_stark_controls(self):
        self.stark_widget = QWidget()
        layout = QVBoxLayout(self.stark_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.stark_f_label = QLabel("Electric Field F: 1.0e+09 V/m")
        self.stark_f_label.setWordWrap(True)
        self.stark_f_slider = self._make_slider(0, 50, 10)
        self.stark_f_slider.valueChanged.connect(self._on_stark_f_changed)
        layout.addWidget(self.stark_f_label)
        layout.addWidget(self.stark_f_slider)

        stark_choice_lbl = QLabel("Eigenstate:")
        stark_choice_lbl.setWordWrap(True)
        layout.addWidget(stark_choice_lbl)

        self.stark_state_combo = QComboBox()
        self.stark_state_combo.setFont(self.app_font)
        self.stark_state_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.stark_state_combo.setMinimumContentsLength(18)
        self.stark_state_combo.addItems([
            "Upper state (+) : E0 + |W|",
            "Lower state (-) : E0 - |W|",
        ])
        self.stark_state_combo.currentIndexChanged.connect(self._on_stark_state_changed)
        layout.addWidget(self.stark_state_combo)

        self.stark_warning_label = QLabel("")
        self.stark_warning_label.setFont(QFont(FONT_FAMILY, 9.5))
        self.stark_warning_label.setFont(QFont(FONT_FAMILY, 10))
        self.stark_warning_label.setFont(QFont(FONT_FAMILY, 10))
        self.stark_warning_label.setStyleSheet(f"color: {WARNING_COLOR};")
        self.stark_warning_label.setWordWrap(True)
        layout.addWidget(self.stark_warning_label)

        self.stark_widget.setVisible(False)
        self.qn_layout.addWidget(self.stark_widget)

    def _make_slider(self, min_val: int, max_val: int, start: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(start)
        return slider

    def _populate_elements_dropdown(self):
        self.element_dropdown.blockSignals(True)
        self.element_dropdown.clear()
        for Z in AVAILABLE_Z:
            sym = ELEMENT_SYMBOLS.get(Z, f"Z{Z}")
            if self.mode == "Hydrogenic":
                ion_lbl = get_ion_symbol(Z)
                txt = f"{ion_lbl}  (Z={Z})"
            else:
                txt = f"{sym}  (Z={Z})"
            self.element_dropdown.addItem(txt, Z)
        self.element_dropdown.blockSignals(False)

    # ──────────────────────────────────────────────────────────────────────────
    # RIGHT CONTAINER (3D VIEW + 2D ANALYSIS PLOT)
    # ──────────────────────────────────────────────────────────────────────────
    def _build_right_container(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Title Label
        self.title_label = QLabel("H | 1s | n=1, l=0, m=0 | E=-13.61 eV")
        self.title_label.setFont(QFont(FONT_FAMILY, 12, QFont.DemiBold))
        self.title_label.setStyleSheet(f"color: {TITLE_TEXT_COLOR}; padding: 4px 0;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        # Vertical Splitter holding 3D View (top) and 2D Plot (bottom)
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(6)

        # Top: 3D pyqtgraph.opengl GLViewWidget + Unboxed Overlays
        self.gl_container = self._build_gl_viewport()
        self.splitter.addWidget(self.gl_container)

        # Bottom: 2D pyqtgraph PlotWidget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(BG_COLOR)
        self.plot_widget.setMinimumHeight(150)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.getAxis("bottom").setPen(pg.mkPen(color=BORDER_COLOR))
        self.plot_widget.getAxis("left").setPen(pg.mkPen(color=BORDER_COLOR))
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen(color=TICK_COLOR))
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen(color=TICK_COLOR))
        self.plot_widget.getAxis("bottom").setStyle(tickFont=QFont(FONT_FAMILY, 9.5))
        self.plot_widget.getAxis("left").setStyle(tickFont=QFont(FONT_FAMILY, 9.5))
        self.plot_widget.getAxis("bottom").setStyle(tickFont=QFont(FONT_FAMILY, 10))
        self.plot_widget.getAxis("left").setStyle(tickFont=QFont(FONT_FAMILY, 10))
        self.plot_widget.getAxis("bottom").setStyle(tickFont=QFont(FONT_FAMILY, 10))
        self.plot_widget.getAxis("left").setStyle(tickFont=QFont(FONT_FAMILY, 10))
        self.plot_widget.getViewBox().setMouseEnabled(x=True, y=False)
        self.plot_widget.getViewBox().setLimits(xMin=0.0, yMin=0.0)

        # Install double-click filter on 2D plot
        self.plot_widget.viewport().installEventFilter(self)

        self.splitter.addWidget(self.plot_widget)
        self.splitter.setSizes([750, 220])
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, True)

        layout.addWidget(self.splitter, stretch=1)
        return container

    def _build_gl_viewport(self) -> QWidget:
        viewport_widget = QWidget()
        v_layout = QVBoxLayout(viewport_widget)
        v_layout.setContentsMargins(0, 0, 0, 0)

        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor(BG_COLOR)
        self.gl_view.setCameraPosition(distance=25, elevation=20, azimuth=45)
        self.gl_view.installEventFilter(self)

        # Sized Grid in calm #1d222a
        self.gl_grid = gl.GLGridItem()
        self.gl_grid.setColor(QColor(GRID_COLOR))
        self.gl_grid.setSize(30, 30, 0)
        self.gl_grid.setSpacing(5, 5, 0)
        self.gl_view.addItem(self.gl_grid)

        # Origin Axes (X, Y, Z)
        self.gl_axis = gl.GLAxisItem()
        self.gl_axis.setSize(5, 5, 5)
        self.gl_view.addItem(self.gl_axis)

        # Nucleus: single white marker at origin
        self.nucleus_item = gl.GLScatterPlotItem(
            pos=np.array([[0.0, 0.0, 0.0]]),
            color=np.array([[1.0, 1.0, 1.0, 1.0]]),
            size=6.0,
            pxMode=True,
        )
        self.gl_view.addItem(self.nucleus_item)

        # Orbital Point Cloud with Additive Blending
        self.cloud_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=np.array([[1.0, 1.0, 1.0, 0.5]]),
            size=2.0,
            pxMode=True,
        )
        try:
            self.cloud_item.setGLOptions("additive")
        except Exception:
            pass
        self.gl_view.addItem(self.cloud_item)

        v_layout.addWidget(self.gl_view)

        # Unboxed Overlays (T4: plain line scale bar bottom-left, slim strip legend top-right)
        self.hud_scale_widget = ScaleBarOverlay(self.gl_view)
        self.hud_legend_widget = LegendOverlay(self.gl_view)

        return viewport_widget

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "gl_view") and hasattr(self, "hud_scale_widget") and hasattr(self, "hud_legend_widget"):
            h = self.gl_view.height()
            w = self.gl_view.width()
            self.hud_scale_widget.move(14, max(10, h - 35))
            self.hud_legend_widget.move(max(10, w - self.hud_legend_widget.width() - 10), 10)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonDblClick:
            if obj == self.gl_view:
                self._reset_camera_view()
                return True
            elif hasattr(self, "plot_widget") and obj == self.plot_widget.viewport():
                self._reset_plot_view()
                return True
        return super().eventFilter(obj, event)

    def _reset_camera_view(self):
        cur_dist = self.gl_view.opts["distance"]
        self.gl_view.setCameraPosition(distance=cur_dist, elevation=20, azimuth=45)
        self.status_bar.showMessage("3D camera view reset to default", 2000)

    def _reset_plot_view(self):
        self.plot_widget.autoRange()

    # ──────────────────────────────────────────────────────────────────────────
    # EVENT DISPATCHERS & CONTROL HANDLERS
    # ──────────────────────────────────────────────────────────────────────────
    def _on_mode_changed(self, index: int):
        self.mode = self.mode_combo.currentText()
        self.anim_timer.stop()

        self.hydro_widget.setVisible(self.mode == "Hydrogenic")
        self.scf_widget.setVisible(self.mode == "SCF (multi-electron)")
        self.sup_widget.setVisible(self.mode == "Superposition")
        self.stark_widget.setVisible(self.mode == "Stark (n=2)")

        # Update element dropdown label style
        cur_z = self.Z
        self._populate_elements_dropdown()
        idx = self.element_dropdown.findData(cur_z)
        if idx >= 0:
            self.element_dropdown.setCurrentIndex(idx)

        # Force Z=1 for Stark mode
        if self.mode == "Stark (n=2)":
            self.element_group.setEnabled(False)
            self.Z = 1
        else:
            self.element_group.setEnabled(True)

        self.render_orbital()

    def _on_element_changed(self, index: int):
        if index < 0:
            return
        self.Z = self.element_dropdown.itemData(index)
        if self.mode == "SCF (multi-electron)":
            self._update_scf_subshells()
        self.render_orbital()

    def _on_n_changed(self, val: int):
        self.n = val
        self.n_label.setText(f"n = {val}  (principal)")
        self.l_slider.setMaximum(val - 1)
        if self.l >= val:
            self.l_slider.setValue(val - 1)
        self._debounce_timer.start(150)

    def _on_l_changed(self, val: int):
        self.l = val
        letter = ORBITAL_LETTERS[val] if val < len(ORBITAL_LETTERS) else f"l={val}"
        self.l_label.setText(f"l = {val}  ({letter})")
        self.m_slider.setMinimum(-val)
        self.m_slider.setMaximum(val)
        if abs(self.m) > val:
            self.m_slider.setValue(0)
        self._debounce_timer.start(150)

    def _on_m_changed(self, val: int):
        self.m = val
        self.m_label.setText(f"m = {val}  (magnetic)")
        self._debounce_timer.start(150)

    def _on_slider_released(self):
        self._debounce_timer.start(50)

    def _on_scf_orb_changed(self, index: int):
        data = self.scf_orb_combo.currentData()
        if data is not None:
            self.scf_selected_nl = data
            self.render_orbital()

    def _update_scf_subshells(self):
        self.scf_orb_combo.blockSignals(True)
        self.scf_orb_combo.clear()
        config = get_ground_state_config(self.Z)
        for (n, l), occ in sorted(config.items()):
            if occ > 0:
                letter = ORBITAL_LETTERS[l]
                self.scf_orb_combo.addItem(f"{n}{letter} (occ={occ:g})", (n, l))
        self.scf_orb_combo.blockSignals(False)
        if self.scf_orb_combo.count() > 0:
            self.scf_selected_nl = self.scf_orb_combo.itemData(0)

    def _on_preset_changed(self, index: int):
        txt = self.preset_combo.currentText()
        if txt == "1s + 2pz":
            self.sup_state_a = [1, 0, 0]
            self.sup_state_b = [2, 1, 0]
        elif txt == "2s + 2pz":
            self.sup_state_a = [2, 0, 0]
            self.sup_state_b = [2, 1, 0]
        elif txt == "1s + 2s":
            self.sup_state_a = [1, 0, 0]
            self.sup_state_b = [2, 0, 0]
        elif txt == "2pz + 3dz²":
            self.sup_state_a = [2, 1, 0]
            self.sup_state_b = [3, 2, 0]
        self._sync_sup_sliders()
        self.render_orbital()

    def _sync_sup_sliders(self):
        for s, n, l, m in [
            ("a", self.sup_state_a[0], self.sup_state_a[1], self.sup_state_a[2]),
            ("b", self.sup_state_b[0], self.sup_state_b[1], self.sup_state_b[2]),
        ]:
            if s == "a":
                self.sup_slider_na.setValue(n)
                self.sup_slider_la.setMaximum(n - 1)
                self.sup_slider_la.setValue(l)
                self.sup_slider_ma.setRange(-l, l)
                self.sup_slider_ma.setValue(m)
            else:
                self.sup_slider_nb.setValue(n)
                self.sup_slider_lb.setMaximum(n - 1)
                self.sup_slider_lb.setValue(l)
                self.sup_slider_mb.setRange(-l, l)
                self.sup_slider_mb.setValue(m)

    def _on_sup_state_changed(self):
        na = self.sup_slider_na.value()
        self.sup_slider_la.setMaximum(na - 1)
        la = min(self.sup_slider_la.value(), na - 1)
        self.sup_slider_ma.setRange(-la, la)
        ma = max(-la, min(la, self.sup_slider_ma.value()))
        self.sup_state_a = [na, la, ma]
        let_a = ORBITAL_LETTERS[la]
        self.sup_lbl_a.setText(f"State A: {na}{let_a} ({na},{la},{ma})")

        nb = self.sup_slider_nb.value()
        self.sup_slider_lb.setMaximum(nb - 1)
        lb = min(self.sup_slider_lb.value(), nb - 1)
        self.sup_slider_mb.setRange(-lb, lb)
        mb = max(-lb, min(lb, self.sup_slider_mb.value()))
        self.sup_state_b = [nb, lb, mb]
        let_b = ORBITAL_LETTERS[lb]
        self.sup_lbl_b.setText(f"State B: {nb}{let_b} ({nb},{lb},{mb})")

        self.render_orbital()

    def _on_mix_changed(self, val: int):
        self.sup_mix_angle = float(val)
        c_a = math.cos(math.radians(val))
        p_a = int(round(c_a ** 2 * 100))
        p_b = 100 - p_a
        self.sup_mix_label.setText(f"Mixing: {p_a}% A + {p_b}% B")
        self.render_orbital()

    def _on_speed_changed(self, val: int):
        self.sup_period_s = float(val)
        self.sup_speed_label.setText(f"Beat period: {val:.1f} s")
        self._update_sup_scale_label()

    def _toggle_playback(self):
        self.is_playing = not self.is_playing
        self.play_button.setText("PAUSE" if self.is_playing else "PLAY")

    def _on_stark_f_changed(self, val: int):
        self.stark_f_vm = val * 1.0e8
        self.stark_f_label.setText(f"Electric Field F: {self.stark_f_vm:.1e} V/m")
        self.render_orbital()

    def _on_stark_state_changed(self, idx: int):
        self.stark_state_choice = self.stark_state_combo.currentText()
        self.render_orbital()

    def _open_validation_dialog(self):
        dialog = ValidationDialog(self)
        dialog.exec_()

    # ──────────────────────────────────────────────────────────────────────────
    # RENDERING DISPATCHER
    # ──────────────────────────────────────────────────────────────────────────
    def render_orbital(self):
        t_start = time.perf_counter()
        self.render_button.setEnabled(False)
        self.render_button.setText("Calculating...")
        self.status_state_label.setText("Rendering orbital...")

        if self.mode == "Hydrogenic":
            self._render_hydrogenic()
        elif self.mode == "SCF (multi-electron)":
            self._render_scf()
        elif self.mode == "Superposition":
            self._render_superposition()
        elif self.mode == "Stark (n=2)":
            self._render_stark()

        if self.mode != "SCF (multi-electron)" or (self.Z in _SCF_CACHE):
            self.render_button.setEnabled(True)
            self.render_button.setText("Render orbital")
            self.status_state_label.setText("Ready")

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        self.status_perf_label.setText(f"{self.num_points} pts · {dt_ms:.0f} ms")

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 1: HYDROGENIC
    # ──────────────────────────────────────────────────────────────────────────
    def _render_hydrogenic(self):
        self.anim_timer.stop()
        letter = ORBITAL_LETTERS[self.l]
        orb_str = f"{self.n}{letter}"
        ion_str = get_ion_symbol(self.Z)
        energy_ev = get_energy(self.n, self.Z, unit="eV")

        title_str = f"{ion_str} | {orb_str} | n={self.n}, l={self.l}, m={self.m} | E={energy_ev:.2f} eV"
        self.title_label.setText(title_str)

        self._set_info([
            ("Orbital", f"{ion_str} {orb_str}"),
            ("Energy", f"{energy_ev:.2f} eV"),
            ("Shape", f"{ORBITAL_SHAPES.get(self.l, 'Custom')}"),
            ("Nodes", f"{self.n - self.l - 1} radial, {self.l} angular"),
            ("Type", "One-electron ion (exact)"),
        ])

        # Sample 3D point cloud
        points, psi2 = sample_hydrogenic_orbital(self.n, self.l, self.m, Z=self.Z, n_points=self.num_points)

        p_min, p_max = float(np.min(psi2)), float(np.max(psi2))
        p_range = p_max - p_min if (p_max - p_min) > 1e-12 else 1.0
        norm_p = (psi2 - p_min) / p_range
        colors = cm.plasma(norm_p)
        colors[:, 3] = 0.45

        dpr = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else float(self.devicePixelRatio())
        self.cloud_item.setData(pos=points, color=colors, size=2.0 * max(1.0, dpr))

        r999 = get_hydrogenic_r999(self.n, self.l, self.Z)
        self._update_grid_and_camera(r999)

        r, P = radial_plot_data("Hydrogenic", self.n, self.l, self.Z, num_points=500)
        self._plot_radial_distribution(r, P, orb_str)

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 2: SCF (MULTI-ELECTRON)
    # ──────────────────────────────────────────────────────────────────────────
    def _render_scf(self):
        self.anim_timer.stop()
        if self.scf_orb_combo.count() == 0:
            self._update_scf_subshells()

        n, l = self.scf_selected_nl
        letter = ORBITAL_LETTERS[l]
        orb_str = f"{n}{letter}"
        sym = ELEMENT_SYMBOLS.get(self.Z, f"Z{self.Z}")

        if self.Z in _SCF_CACHE:
            res = _SCF_CACHE[self.Z]
            self._apply_scf_result(res, n, l, orb_str, sym)
            return

        self._scf_request_counter += 1
        req_id = self._scf_request_counter

        self.status_state_label.setText(f"Calculating SCF for {sym} (Z={self.Z})...")
        self.status_progress.setVisible(True)
        self.render_button.setEnabled(False)
        self.render_button.setText("Calculating...")

        worker = SCFWorkerThread(self.Z, req_id)
        worker.finished_res.connect(self._on_scf_worker_finished)
        self._current_scf_worker = worker
        worker.start()

    def _on_scf_worker_finished(self, req_id: int, res: Any):
        if req_id != self._scf_request_counter:
            return

        self.status_progress.setVisible(False)
        self.render_button.setEnabled(True)
        self.render_button.setText("Render orbital")
        self.status_state_label.setText("Ready")

        n, l = self.scf_selected_nl
        letter = ORBITAL_LETTERS[l]
        orb_str = f"{n}{letter}"
        sym = ELEMENT_SYMBOLS.get(self.Z, f"Z{self.Z}")
        self._apply_scf_result(res, n, l, orb_str, sym)

    def _apply_scf_result(self, res: Any, n: int, l: int, orb_str: str, sym: str):
        self.scf_res = res
        orb = res.orbitals.get((n, l))
        energy_ev = orb.eps * HARTREE_EV if orb else 0.0

        title_str = f"{sym} | {orb_str} | n={n}, l={l}, m=0 | E={energy_ev:.2f} eV"
        self.title_label.setText(title_str)

        r_mean = float(np.trapezoid(orb.u ** 2 * res.r, res.r)) if orb else 0.0
        self._set_info([
            ("Orbital", f"{sym} {orb_str}"),
            ("Energy", f"{energy_ev:.2f} eV"),
            ("Shape", f"{ORBITAL_SHAPES.get(l, 'Custom')}"),
            ("Nodes", f"{n - l - 1} radial, {l} angular"),
            ("Occ", f"{orb.occ:g} e⁻"),
            ("ε", f"{orb.eps:.4f} Ha"),
            ("⟨r⟩", f"{r_mean:.2f} a₀"),
        ], honesty="Approximate: spherically averaged Xα")

        P_r = orb.u ** 2
        points, psi2 = sample_orbital(res.r, P_r, l, m=0, n_points=self.num_points)

        p_min, p_max = float(np.min(psi2)), float(np.max(psi2))
        p_range = p_max - p_min if (p_max - p_min) > 1e-12 else 1.0
        norm_p = (psi2 - p_min) / p_range
        colors = cm.plasma(norm_p)
        colors[:, 3] = 0.45

        dpr = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else float(self.devicePixelRatio())
        self.cloud_item.setData(pos=points, color=colors, size=2.0 * max(1.0, dpr))

        cum_prob = np.cumsum(P_r * (res.r * (np.log(res.r[1]) - np.log(res.r[0]))))
        idx_cut = int(np.searchsorted(cum_prob, 0.999))
        r_cut = float(res.r[min(idx_cut, len(res.r) - 1)])
        self._update_grid_and_camera(r_cut)

        r, P = radial_plot_data("SCF", n, l, self.Z, scf_orb=orb, scf_res=res, num_points=500)
        self._plot_radial_distribution(r, P, orb_str)

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 3: SUPERPOSITION
    # ──────────────────────────────────────────────────────────────────────────
    def _render_superposition(self):
        c_a = math.cos(math.radians(self.sup_mix_angle))
        c_b = math.sin(math.radians(self.sup_mix_angle))
        na, la, ma = self.sup_state_a
        nb, lb, mb = self.sup_state_b

        states = [(c_a, na, la, ma), (c_b, nb, lb, mb)]
        self.sup_obj = SuperpositionState(states, Z=self.Z, n_points=self.num_points)

        t_info = transition_info((na, la, ma), (nb, lb, mb), Z=self.Z)
        title_str = f"Superposition: {na}{ORBITAL_LETTERS[la]} + {nb}{ORBITAL_LETTERS[lb]} | ΔE={t_info.delta_e_ev:.2f} eV"
        self.title_label.setText(title_str)

        ea_ev = get_energy(na, self.Z, unit="eV")
        eb_ev = get_energy(nb, self.Z, unit="eV")
        dipole_mag = math.sqrt(t_info.dipole_sq)
        status_txt = "Allowed (E1)" if t_info.is_allowed else f"Forbidden ({t_info.forbidden_reason})"
        self._set_info([
            ("States", f"{na}{ORBITAL_LETTERS[la]} + {nb}{ORBITAL_LETTERS[lb]}"),
            ("ΔE", f"{t_info.delta_e_ev:.3f} eV ({t_info.wavelength_nm:.1f} nm)"),
            ("Dipole", f"|d| = {dipole_mag:.4f} a₀"),
            ("Status", f"E1: {status_txt}"),
            ("Rate", f"A = {t_info.einstein_a_s:.2e} s⁻¹" if t_info.is_allowed else "A = 0 (forbidden)"),
            ("Life", f"τ = {t_info.lifetime_s * 1e9:.2f} ns" if t_info.is_allowed else "τ = ∞ (stable)"),
        ])

        self._update_sup_scale_label()

        dpr = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else float(self.devicePixelRatio())
        self.cloud_item.setData(pos=self.sup_obj.points, size=2.0 * max(1.0, dpr))

        r999_a = get_hydrogenic_r999(na, la, self.Z)
        r999_b = get_hydrogenic_r999(nb, lb, self.Z)
        max_r = max(r999_a, r999_b)
        self._update_grid_and_camera(max_r)

        self._plot_expectation_z_curve()

        self.anim_time_s = 0.0
        self.last_anim_tick = time.perf_counter()
        self.anim_timer.start()

    def _update_sup_scale_label(self):
        na, la, ma = self.sup_state_a
        nb, lb, mb = self.sup_state_b
        ea = get_energy(na, self.Z, unit="Hartree")
        eb = get_energy(nb, self.Z, unit="Hartree")
        delta_e = abs(eb - ea)

        if delta_e < 1e-12:
            self.sup_scale_label.setText("Degenerate: density is static")
        else:
            t_real_s = 2.0 * math.pi * AU_TIME_S / delta_e
            t_ratio = t_real_s / max(0.1, self.sup_period_s)
            fs_per_sec = t_ratio * 1.0e15
            self.sup_scale_label.setText(f"Slowed: 1 s on screen = {fs_per_sec:.2f} fs")

    def _plot_expectation_z_curve(self):
        if self.sup_obj is None:
            return
        self.plot_widget.clear()

        ea = get_energy(self.sup_state_a[0], self.Z, unit="Hartree")
        eb = get_energy(self.sup_state_b[0], self.Z, unit="Hartree")
        de = abs(eb - ea)
        t_period_au = 2.0 * math.pi / de if de > 1e-12 else 100.0

        t_samples_au = np.linspace(0.0, t_period_au, 200)
        t_samples_fs = t_samples_au * AU_TIME_S * 1.0e15
        z_vals = [self.sup_obj.expectation_z(t) for t in t_samples_au]

        self.sup_t_au = t_samples_au
        self.sup_t_fs = t_samples_fs
        self.sup_z_vals = np.array(z_vals)

        self.plot_widget.plot(t_samples_fs, z_vals, pen=pg.mkPen(color=ACCENT_COLOR, width=1.5))
        self.sup_marker = self.plot_widget.plot([t_samples_fs[0]], [z_vals[0]],
                                                pen=None, symbol="o", symbolBrush="#ffffff", symbolSize=7)

        self.plot_widget.setLabel("bottom", "Time t (fs)", color=TICK_COLOR)
        self.plot_widget.setLabel("left", "⟨z⟩(t)  [a₀]", color=TICK_COLOR)
        self.plot_widget.setTitle("Exact Dipole Oscillation ⟨z⟩(t)", color=TITLE_TEXT_COLOR, size="11pt")
        self.plot_widget.getViewBox().setLimits(xMin=None, yMin=None)
        self.plot_widget.autoRange()

    def _on_anim_frame(self):
        now = time.perf_counter()
        dt_sec = now - self.last_anim_tick
        self.last_anim_tick = now

        if not self.is_playing or self.sup_obj is None:
            return

        fps = 1.0 / max(1e-4, dt_sec)
        self.status_perf_label.setText(f"{self.num_points} pts · {fps:.0f} fps")

        self.anim_time_s += dt_sec
        ea = get_energy(self.sup_state_a[0], self.Z, unit="Hartree")
        eb = get_energy(self.sup_state_b[0], self.Z, unit="Hartree")
        de = abs(eb - ea)
        omega_sim = (2.0 * math.pi / max(0.1, self.sup_period_s))
        phase = omega_sim * self.anim_time_s
        t_au = phase / de if de > 1e-12 else self.anim_time_s

        w = self.sup_obj.density_weights(t_au)
        norm_w = np.clip(w / 2.5, 0.0, 1.0)
        colors = cm.plasma(norm_w)
        colors[:, 3] = np.clip(w * 0.45, 0.05, 0.95)
        self.cloud_item.setData(color=colors)

        if hasattr(self, "sup_t_fs") and len(self.sup_t_fs) > 0 and de > 1e-12:
            t_period_fs = (2.0 * math.pi / de) * AU_TIME_S * 1.0e15
            cur_fs = (phase % (2.0 * math.pi)) / (2.0 * math.pi) * t_period_fs
            cur_z = self.sup_obj.expectation_z(t_au)
            self.sup_marker.setData([cur_fs], [cur_z])

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 4: STARK EFFECT
    # ──────────────────────────────────────────────────────────────────────────
    def _render_stark(self):
        self.anim_timer.stop()
        res = stark_n2_hydrogen(self.stark_f_vm)
        is_upper = ("+" in self.stark_state_choice)
        e_ev = res.energy_plus_ev if is_upper else res.energy_minus_ev
        mu_z = res.dipole_plus_a0 if is_upper else res.dipole_minus_a0
        state_str = "2s + 2pz (+)" if is_upper else "2s - 2pz (-)"

        title_str = f"H | Stark n=2 | F={self.stark_f_vm:.1e} V/m | E={e_ev:.2f} eV"
        self.title_label.setText(title_str)

        shift_val = (res.delta_e_ev / 2.0) * (1 if is_upper else -1)
        self._set_info([
            ("Orbital", f"H Stark {state_str}"),
            ("Energy", f"{e_ev:.3f} eV"),
            ("Shift", f"ΔE = {shift_val:+.3f} eV"),
            ("Dipole", f"μ_z = {mu_z:+.1f} a₀"),
            ("Coupling", f"W = {res.W_ha * HARTREE_EV:.3f} eV"),
            ("State", "Degenerate: 2px, 2py unshifted"),
        ], honesty=res.warning or "")

        if res.warning is None:
            self.stark_warning_label.setText("")
        else:
            self.stark_warning_label.setText(res.warning)

        # Sample polarized hybrid
        c_s = 1.0 / math.sqrt(2.0)
        c_p = (1.0 / math.sqrt(2.0)) if is_upper else (-1.0 / math.sqrt(2.0))
        states = [(c_s, 2, 0, 0), (c_p, 2, 1, 0)]
        stark_sup = SuperpositionState(states, Z=1, n_points=self.num_points)
        w = stark_sup.density_weights(0.0)
        norm_w = np.clip(w / 2.5, 0.0, 1.0)
        colors = cm.plasma(norm_w)
        colors[:, 3] = 0.5

        dpr = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else float(self.devicePixelRatio())
        self.cloud_item.setData(pos=stark_sup.points, color=colors, size=2.0 * max(1.0, dpr))

        r999 = get_hydrogenic_r999(2, 0, 1)
        self._update_grid_and_camera(r999)

        # Plot Stark energy splitting vs electric field
        self.plot_widget.clear()
        f_vals_vm = np.linspace(0.0, 5.0e9, 100)
        f_vals_plot = f_vals_vm * 1e-9  # in 10^9 V/m
        shift_ev = [stark_n2_hydrogen(f).delta_e_ev / 2.0 for f in f_vals_vm]
        shift_ev = np.array(shift_ev)

        self.plot_widget.plot(f_vals_plot, shift_ev, pen=pg.mkPen(color=ACCENT_COLOR, width=1.5), name="E+ (+3F)")
        self.plot_widget.plot(f_vals_plot, -shift_ev, pen=pg.mkPen(color=WARNING_COLOR, width=1.5), name="E- (-3F)")

        cur_f_plot = self.stark_f_vm * 1e-9
        cur_shift = (res.delta_e_ev / 2.0) * (1 if is_upper else -1)
        self.plot_widget.plot([cur_f_plot], [cur_shift],
                              pen=None, symbol="o", symbolBrush="#ffffff", symbolSize=7)

        self.plot_widget.setLabel("bottom", "Electric Field F (10⁹ V/m)", color=TICK_COLOR)
        self.plot_widget.setLabel("left", "Energy Shift ΔE (eV)", color=TICK_COLOR)
        self.plot_widget.setTitle("Linear Stark Splitting: ΔE = ± 3 F e a₀", color=TITLE_TEXT_COLOR, size="11pt")
        self.plot_widget.getViewBox().setLimits(xMin=0.0, yMin=None)
        self.plot_widget.autoRange()

    # ──────────────────────────────────────────────────────────────────────────
    # HELPER: DYNAMIC GRID & 3D CAMERA FIT
    # ──────────────────────────────────────────────────────────────────────────
    def _update_grid_and_camera(self, r_extent: float):
        if r_extent <= 6.0:
            step = 1.0
        elif r_extent <= 15.0:
            step = 2.0
        elif r_extent <= 35.0:
            step = 5.0
        elif r_extent <= 75.0:
            step = 10.0
        else:
            step = 20.0

        extent = math.ceil(r_extent / step) * step
        self.gl_grid.setSize(extent * 2, extent * 2, 0)
        self.gl_grid.setSpacing(step, step, 0)

        axis_len = max(2.0, step * 1.5)
        self.gl_axis.setSize(axis_len, axis_len, axis_len)

        # Unboxed scale bar text
        self.hud_scale_widget.set_step(f"{step:g} a₀")

        # Camera auto-fit
        cam_dist = max(8.0, r_extent * 2.2)
        cur_elev = self.gl_view.opts.get("elevation", 20)
        cur_azim = self.gl_view.opts.get("azimuth", 45)
        self.gl_view.setCameraPosition(distance=cam_dist, elevation=cur_elev, azimuth=cur_azim)

    # ──────────────────────────────────────────────────────────────────────────
    # HELPER: 2D RADIAL PROBABILITY PLOT (T4)
    # ──────────────────────────────────────────────────────────────────────────
    def _plot_radial_distribution(self, r: np.ndarray, P: np.ndarray, orb_label: str):
        self.plot_widget.clear()

        # Accent curve (1.5px anti-aliased) with very faint fill (<= 8% alpha)
        self.plot_widget.plot(
            r, P,
            pen=pg.mkPen(color=ACCENT_COLOR, width=1.5),
            fillLevel=0.0,
            brush=pg.mkBrush(QColor(212, 137, 58, 16)),
        )

        # Most Probable Radius (Peak) marker + label
        r_mp = find_most_probable_radius(r, P)
        idx_mp = int(np.argmax(P))
        p_mp = P[idx_mp]
        self.plot_widget.plot(
            [r_mp], [p_mp],
            pen=None, symbol="o", symbolBrush=ACCENT_COLOR, symbolPen=BORDER_COLOR, symbolSize=6
        )

        # Position r_mp label with generous headroom so it can never be clipped
        txt_peak = pg.TextItem(f"r_mp = {r_mp:.2f} a₀", color=ACCENT_COLOR, anchor=(0.0, 1.1))
        txt_peak.setFont(QFont(FONT_FAMILY, 9.5))
        txt_peak.setFont(QFont(FONT_FAMILY, 10))
        txt_peak.setFont(QFont(FONT_FAMILY, 10))
        txt_peak.setPos(r_mp + 0.12 * (r[-1] / 10.0), p_mp * 0.98)
        self.plot_widget.addItem(txt_peak)

        # Mean Radius ⟨r⟩ dashed vertical line
        r_mean = find_mean_radius(r, P)
        mean_line = pg.InfiniteLine(
            pos=r_mean, angle=90,
            pen=pg.mkPen(color=TEXT_DIM, width=1.0, style=Qt.DashLine)
        )
        self.plot_widget.addItem(mean_line)
        txt_mean = pg.TextItem(f"⟨r⟩ = {r_mean:.2f} a₀", color=TEXT_DIM, anchor=(0.0, 0.5))
        txt_mean.setFont(QFont(FONT_FAMILY, 9))
        txt_mean.setPos(r_mean + 0.12 * (r[-1] / 10.0), p_mp * 0.38)
        self.plot_widget.addItem(txt_mean)

        # Radial node ticks at local minima
        nodes = find_radial_nodes(r, P, threshold=0.02)
        for r_node in nodes:
            tick_line = pg.InfiniteLine(
                pos=r_node, angle=90,
                pen=pg.mkPen(color=TEXT_DIM, width=1.0, style=Qt.DotLine)
            )
            self.plot_widget.addItem(tick_line)

        # Strict axis bounds and labels
        self.plot_widget.setLabel("bottom", "r (a₀)", color=TICK_COLOR)
        self.plot_widget.setLabel("left", "P(r)  [1/a₀]", color=TICK_COLOR)
        self.plot_widget.setTitle(f"Radial Probability Distribution: {orb_label}", color=TITLE_TEXT_COLOR, size="11pt")

        self.plot_widget.getViewBox().setLimits(xMin=0.0, yMin=0.0)
        self.plot_widget.setXRange(0.0, float(r[-1]), padding=0.01)
        # 25% headroom so peak label is never clipped
        self.plot_widget.setYRange(0.0, float(p_mp) * 1.25, padding=0.01)

    def _set_info(self, entries: list[tuple[str, str]], honesty: str = ""):
        for idx in range(len(self.info_rows)):
            lbl_k, lbl_v = self.info_rows[idx]
            if idx < len(entries):
                k, v = entries[idx]
                lbl_k.setText(f"{k}")
                lbl_v.setText(v)
                lbl_k.setVisible(True)
                lbl_v.setVisible(True)
            else:
                lbl_k.setVisible(False)
                lbl_v.setVisible(False)
        self.info_honesty_label.setText(honesty)
        self.info_honesty_label.setVisible(bool(honesty.strip()))


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = QuantumSimulatorApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
