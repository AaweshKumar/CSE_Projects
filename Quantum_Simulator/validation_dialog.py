"""Validation Dialog for Quantum Atom Simulator (Phase P3 / U2 / T5).

Displays dark-styled plots and tables comparing simulator predictions
with NIST ionization energies and experimental Cu Kα emission.
"""

from __future__ import annotations
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

import os
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QDialog, QHeaderView, QLabel, QProgressBar, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget
)

from theme import (
    ACCENT_COLOR, ACCENT_HOVER, APP_STYLESHEET, BG_COLOR, BORDER_COLOR,
    FONT_FAMILY, FONT_SIZE, GRID_COLOR, PANEL_COLOR, RAISED_COLOR,
    TEXT_COLOR, TEXT_DIM, TICK_COLOR, TITLE_TEXT_COLOR, WARNING_COLOR,
    apply_dark_title_bar, get_application_font
)
from validation import (
    IESeriesResult, MoseleyResult, compute_ionization_energies, compute_k_alpha_series
)

# Matplotlib global font configuration
mpl.rcParams["font.family"] = ["Calibri", "Carlito", "DejaVu Sans", "sans-serif"]
mpl.rcParams["axes.unicode_minus"] = False


def apply_dark_mpl_style(ax, fig) -> None:
    """Apply the calm desktop-science dark theme to matplotlib axes and figure."""
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1.0)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TITLE_TEXT_COLOR)
    ax.tick_params(axis="x", colors=TICK_COLOR, labelsize=9.5)
    ax.tick_params(axis="y", colors=TICK_COLOR, labelsize=9.5)
    ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.8, alpha=0.8)


class ValidationWorker(QThread):
    """Background worker thread to compute validation data without freezing the GUI."""
    progress = pyqtSignal(int, int, str)
    finished_data = pyqtSignal(object, object)

    def run(self) -> None:
        def ie_cb(cur, tot):
            self.progress.emit(cur, tot * 2, f"Computing IE: Z={cur}/{tot}...")

        ie_result = compute_ionization_energies(Z_max=36, progress_cb=ie_cb)

        def ka_cb(cur, tot):
            self.progress.emit(36 + cur, 36 + tot, f"Computing Kα: Z={10 + cur}...")

        moseley_result = compute_k_alpha_series(Z_min=11, Z_max=36, progress_cb=ka_cb)

        self.finished_data.emit(ie_result, moseley_result)


class ValidationDialog(QDialog):
    """Modal validation dialog presenting NIST benchmark comparisons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Validation: Ionization Energies & Moseley Kα")
        self.setMinimumSize(950, 650)
        self.resize(1000, 680)
        self.setSizeGripEnabled(True)
        self.setStyleSheet(APP_STYLESHEET)

        icon_path = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))), "app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        app_font = get_application_font()
        self.setFont(app_font)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Status / progress header
        self.status_label = QLabel("Initializing validation calculations...", self)
        self.status_label.setFont(app_font)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # Tab widget (flat tabs with accent underline on selection)
        self.tabs = QTabWidget(self)
        self.tabs.setFont(app_font)

        # Tab 1: IE Trend
        self.ie_canvas = FigureCanvas(Figure(figsize=(8, 5)))
        self.tabs.addTab(self.ie_canvas, "1. Ionization Energies (Z=1..36)")

        # Tab 2: Moseley Plot
        self.moseley_canvas = FigureCanvas(Figure(figsize=(8, 5)))
        self.tabs.addTab(self.moseley_canvas, "2. Moseley Plot (Kα)")

        # Tab 3: Error Table
        self.table_widget = QWidget()
        table_layout = QVBoxLayout(self.table_widget)
        table_layout.setContentsMargins(8, 8, 8, 8)
        table_layout.setSpacing(8)

        self.summary_label = QLabel("", self.table_widget)
        self.summary_label.setFont(app_font)
        self.summary_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 9.5pt;")
        table_layout.addWidget(self.summary_label)

        self.error_table = QTableWidget(self.table_widget)
        self.error_table.setFont(app_font)
        self.error_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {PANEL_COLOR};
                color: {TEXT_COLOR};
                gridline-color: {BORDER_COLOR};
                border: 1px solid {BORDER_COLOR};
                selection-background-color: {RAISED_COLOR};
                selection-color: {ACCENT_COLOR};
            }}
            QHeaderView::section {{
                background-color: {RAISED_COLOR};
                color: {TEXT_DIM};
                border: 1px solid {BORDER_COLOR};
                padding: 6px;
                font-family: "{FONT_FAMILY}", "Carlito", "Segoe UI", sans-serif;
                font-weight: 600;
                font-size: 10pt;
            }}
        """)
        table_layout.addWidget(self.error_table)
        self.tabs.addTab(self.table_widget, "3. Error Analysis")

        layout.addWidget(self.tabs)

        # Start worker
        self.worker = ValidationWorker(self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_data.connect(self._on_finished)
        self.worker.start()

    def showEvent(self, event):
        super().showEvent(event)
        apply_dark_title_bar(self)

    def _on_progress(self, cur: int, tot: int, msg: str) -> None:
        pct = int(100.0 * cur / max(1, tot))
        self.progress_bar.setValue(pct)
        self.status_label.setText(msg)

    def _on_finished(self, ie_res: IESeriesResult, mos_res: MoseleyResult) -> None:
        self.progress_bar.setVisible(False)
        self.status_label.setText(
            f"Validation complete: IE TS MAE = {ie_res.mae_ts_ev:.2f} eV | Cu Kα error = {mos_res.cu_err_pct:.2f}%"
        )
        self._plot_ie(ie_res)
        self._plot_moseley(mos_res)
        self._populate_table(ie_res, mos_res)

    def _plot_ie(self, res: IESeriesResult) -> None:
        fig = self.ie_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        apply_dark_mpl_style(ax, fig)

        zs = [p.Z for p in res.points]
        nist = [p.ie_nist_ev for p in res.points]
        koop = [p.ie_koopmans_ev for p in res.points]
        ts = [p.ie_ts_ev for p in res.points]

        ax.scatter(zs, nist, color=TEXT_COLOR, s=36, label="NIST Experiment", zorder=4)
        ax.plot(zs, koop, color=ACCENT_COLOR, linestyle="--", linewidth=1.5,
                label=f"Koopmans (MAE={res.mae_koopmans_ev:.2f} eV)", zorder=3)
        ax.plot(zs, ts, color=WARNING_COLOR, linewidth=1.8,
                label=f"Slater TS (MAE={res.mae_ts_ev:.2f} eV)", zorder=3)

        # Label noble gases and alkali metals
        for p in res.points:
            if p.Z in [2, 10, 18, 36]:
                ax.annotate(p.symbol, (p.Z, p.ie_nist_ev), textcoords="offset points", xytext=(0, 6),
                            ha="center", color=TEXT_DIM, fontsize=8.5, fontfamily=FONT_FAMILY)
            elif p.Z in [3, 11, 19]:
                ax.annotate(p.symbol, (p.Z, p.ie_nist_ev), textcoords="offset points", xytext=(0, -12),
                            ha="center", color=TEXT_DIM, fontsize=8.5, fontfamily=FONT_FAMILY)

        ax.set_title("First Ionization Energies vs NIST (Z = 1 to 36)", fontfamily=FONT_FAMILY, fontsize=11, fontweight="600")
        ax.set_xlabel("Atomic Number Z", fontfamily=FONT_FAMILY, fontsize=10)
        ax.set_ylabel("Ionization Energy (eV)", fontfamily=FONT_FAMILY, fontsize=10)
        ax.legend(facecolor=RAISED_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR,
                  prop={"family": FONT_FAMILY, "size": 9})
        fig.tight_layout()
        self.ie_canvas.draw()

    def _plot_moseley(self, res: MoseleyResult) -> None:
        fig = self.moseley_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        apply_dark_mpl_style(ax, fig)

        zs = [p.Z for p in res.points]
        y_data = [p.sqrt_e_ry for p in res.points]

        ax.scatter(zs, y_data, color=ACCENT_COLOR, s=36, label="Xα Model Kα (2p→1s)", zorder=3)

        z_line = np.linspace(min(zs), max(zs), 100)
        y_line = res.slope * z_line + res.intercept
        fit_lbl = f"Fit: y = {res.slope:.4f}·Z + ({res.intercept:.4f})  [R²={res.r_squared:.5f}]"
        ax.plot(z_line, y_line, color=TEXT_DIM, linestyle="--", linewidth=1.4, label=fit_lbl, zorder=2)

        # Copper experimental marker
        cu_ry = (res.cu_exp_kev * 1000.0) / 13.605693009
        sqrt_cu_ry = float(np.sqrt(cu_ry))
        ax.scatter([29], [sqrt_cu_ry], color=WARNING_COLOR, marker="*", s=120,
                   label=f"Cu Kα Exp (8.048 keV, err={res.cu_err_pct:.1f}%)", zorder=5)

        ax.set_title("Moseley's Law: √(E_Kα / Ry) vs Z (Z = 11 to 36)", fontfamily=FONT_FAMILY, fontsize=11, fontweight="600")
        ax.set_xlabel("Atomic Number Z", fontfamily=FONT_FAMILY, fontsize=10)
        ax.set_ylabel("√(E_Kα / Ry)", fontfamily=FONT_FAMILY, fontsize=10)
        ax.legend(facecolor=RAISED_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR,
                  prop={"family": FONT_FAMILY, "size": 9})
        fig.tight_layout()
        self.moseley_canvas.draw()

    def _populate_table(self, ie_res: IESeriesResult, mos_res: MoseleyResult) -> None:
        summary_text = (
            f"SUMMARY STATISTICS (Z = 1 to 36):\n"
            f"Koopmans Theorem:        MAE = {ie_res.mae_koopmans_ev:6.2f} eV | Mean Relative Error = {ie_res.mre_koopmans * 100:5.1f}%\n"
            f"Slater Transition State: MAE = {ie_res.mae_ts_ev:6.2f} eV | Mean Relative Error = {ie_res.mre_ts * 100:5.1f}%\n"
            f"Moseley Law Regression:   Slope = {mos_res.slope:6.4f} (theo: 0.8660) | Intercept = {mos_res.intercept:6.4f} | R² = {mos_res.r_squared:.6f}\n"
            f"Cu Kα Validation:         Model = {mos_res.cu_kalpha_kev:6.3f} keV vs Exp = {mos_res.cu_exp_kev:6.3f} keV (Error = {mos_res.cu_err_pct:4.2f}%)"
        )
        self.summary_label.setText(summary_text)

        headers = ["Z", "Symbol", "NIST (eV)", "Koopmans (eV)", "TS (eV)", "Δ_Koop (eV)", "Δ_TS (eV)", "Rel Err TS (%)"]
        self.error_table.setColumnCount(len(headers))
        self.error_table.setRowCount(len(ie_res.points))
        self.error_table.setHorizontalHeaderLabels(headers)
        self.error_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for row, p in enumerate(ie_res.points):
            vals = [
                str(p.Z),
                p.symbol,
                f"{p.ie_nist_ev:.2f}",
                f"{p.ie_koopmans_ev:.2f}",
                f"{p.ie_ts_ev:.2f}",
                f"{p.error_koopmans_ev:+.2f}",
                f"{p.error_ts_ev:+.2f}",
                f"{p.rel_error_ts * 100:.1f}%",
            ]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                self.error_table.setItem(row, col, item)
