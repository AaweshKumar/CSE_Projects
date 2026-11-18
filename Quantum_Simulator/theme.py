"""Theme and styling configuration for Quantum Atom Simulator (Revision v3).

Calm, restrained desktop-science-tool theme:
- Window / 3D / plot background: #0e1014
- Panel surface: #14171c
- Raised (inputs, buttons): #1c2027
- Border / divider: #2a2f38
- Text: #d9dce1
- Dim text: #8b929d
- Accent (muted amber): #d4893a (hover: #e09d57)
- Warning / error: #c75d4a
- 3D grid: #1d222a (ticks: #59616d)
- Font: Calibri 11pt (fallbacks: Carlito, Segoe UI, Arial)
"""

from __future__ import annotations
import ctypes
from ctypes import byref, c_int
import os
import sys

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPalette, QPen
from PyQt5.QtWidgets import QApplication, QWidget

# Palette Constants (T3)
BG_COLOR = "#0e1014"          # Window / 3D / plot background
PANEL_COLOR = "#14171c"       # Left panel & dialog surface
RAISED_COLOR = "#1c2027"      # Raised controls (inputs, button backgrounds)
BORDER_COLOR = "#2a2f38"      # Border / divider lines
TEXT_COLOR = "#d9dce1"        # Primary readable text
TEXT_DIM = "#8b929d"          # Muted labels / secondary text
ACCENT_COLOR = "#d4893a"      # Muted amber accent
ACCENT_HOVER = "#e09d57"      # Accent hover state
WARNING_COLOR = "#c75d4a"     # Warning / error state
GRID_COLOR = "#1d222a"        # 3D grid and plot grid lines
TICK_COLOR = "#59616d"        # Plot axis tick labels
TITLE_TEXT_COLOR = "#e6e8ec"  # Window and plot titles

FONT_FAMILY = "Calibri"
FONT_SIZE = 11


def ensure_chevron_icon() -> str:
    """Ensure a crisp 12x12 chevron icon exists on disk for QComboBox arrows."""
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    icon_path = os.path.join(base_dir, "chevron_down.png")
    if not os.path.exists(icon_path):
        try:
            img = QImage(12, 12, QImage.Format_RGBA8888)
            img.fill(0)
            p = QPainter(img)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor(TEXT_DIM), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(QPointF(2.5, 4.5), QPointF(6.0, 8.0))
            p.drawLine(QPointF(6.0, 8.0), QPointF(9.5, 4.5))
            p.end()
            img.save(icon_path, "PNG")
        except Exception:
            pass
    return icon_path.replace("\\", "/")


CHEVRON_ICON_PATH = ensure_chevron_icon()


def get_application_font() -> QFont:
    """Return configured Calibri font with graceful fallbacks."""
    font = QFont(FONT_FAMILY, FONT_SIZE)
    font.setFamilies(["Calibri", "Carlito", "Segoe UI", "Arial"])
    # Check if Calibri is present in system
    families = QFontDatabase().families()
    if "Calibri" not in families:
        import logging
        logging.warning("Calibri font not found on system; falling back to Carlito/Segoe UI/Arial.")
    return font


def apply_dark_theme(app: QApplication) -> None:
    """Apply the Fusion style and a full dark QPalette to QApplication (T1)."""
    app.setStyle("Fusion")

    palette = QPalette()
    # Active / Inactive color roles
    palette.setColor(QPalette.Window, QColor(BG_COLOR))
    palette.setColor(QPalette.WindowText, QColor(TEXT_COLOR))
    palette.setColor(QPalette.Base, QColor(PANEL_COLOR))
    palette.setColor(QPalette.AlternateBase, QColor(RAISED_COLOR))
    palette.setColor(QPalette.ToolTipBase, QColor(RAISED_COLOR))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT_COLOR))
    palette.setColor(QPalette.Text, QColor(TEXT_COLOR))
    palette.setColor(QPalette.Button, QColor(RAISED_COLOR))
    palette.setColor(QPalette.ButtonText, QColor(TEXT_COLOR))
    palette.setColor(QPalette.BrightText, QColor("#ffffff"))
    palette.setColor(QPalette.Link, QColor(ACCENT_COLOR))
    palette.setColor(QPalette.Highlight, QColor(ACCENT_COLOR))
    palette.setColor(QPalette.HighlightedText, QColor(PANEL_COLOR))
    palette.setColor(QPalette.PlaceholderText, QColor(TEXT_DIM))

    # Disabled color roles
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(TEXT_DIM))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_DIM))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TEXT_DIM))
    palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(BORDER_COLOR))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(TEXT_DIM))

    app.setPalette(palette)
    app.setFont(get_application_font())


def apply_dark_title_bar(window: QWidget) -> None:
    """Enable dark native title bar on Windows 10/11 using DwmSetWindowAttribute (T1)."""
    if sys.platform != "win32":
        return
    try:
        hwnd = int(window.winId())
        dwm = ctypes.windll.dwmapi.DwmSetWindowAttribute
        # DWMWA_USE_IMMERSIVE_DARK_MODE (20 on Win10 20H1+ and Win11; 19 on older builds)
        dark_mode = c_int(1)
        res = dwm(hwnd, 20, byref(dark_mode), ctypes.sizeof(dark_mode))
        if res != 0:
            dwm(hwnd, 19, byref(dark_mode), ctypes.sizeof(dark_mode))
        # DWMWA_CAPTION_COLOR = 35 on Windows 11 (0x0014100e in 0x00BBGGRR format for #0e1014)
        caption_color = c_int(0x0014100E)
        dwm(hwnd, 35, byref(caption_color), ctypes.sizeof(caption_color))
    except Exception:
        pass


APP_STYLESHEET = f"""
/* Global Window and Base Widget Style */
QMainWindow, QDialog {{
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
    font-family: "{FONT_FAMILY}", "Carlito", "Segoe UI", sans-serif;
    font-size: {FONT_SIZE}pt;
}}

QWidget {{
    font-family: "{FONT_FAMILY}", "Carlito", "Segoe UI", sans-serif;
    font-size: {FONT_SIZE}pt;
    color: {TEXT_COLOR};
}}

QWidget#centralWidget {{
    background-color: {BG_COLOR};
}}

/* Left Panel Container & Scroll Area (T1: No white or light-grey surfaces) */
QWidget#leftPanelContainer {{
    background-color: {PANEL_COLOR};
    border-right: 1px solid {BORDER_COLOR};
}}

QScrollArea {{
    background-color: {PANEL_COLOR};
    border: none;
}}

QScrollArea > QWidget > QWidget {{
    background-color: {PANEL_COLOR};
}}

/* Section Headers: flat, no bordered boxes, 1px top border (T4) */
QGroupBox {{
    background-color: transparent;
    border: none;
    border-top: 1px solid {BORDER_COLOR};
    margin-top: 20px;
    padding-top: 8px;
    padding-left: 0px;
    padding-right: 0px;
    padding-bottom: 2px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 0px;
    padding: 0 4px 2px 0;
    color: {TEXT_DIM};
    font-size: 10pt;
    font-weight: 600;
    background-color: transparent;
}}

/* Push Buttons (T4: flat, 3px radius, 1px border) */
QPushButton {{
    background-color: {RAISED_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    color: {TEXT_COLOR};
    padding: 5px 12px;
    min-height: 24px;
}}

QPushButton:hover {{
    border: 1px solid {ACCENT_COLOR};
    background-color: #222730;
    color: #ffffff;
}}

QPushButton:pressed {{
    background-color: #161a20;
    border: 1px solid {ACCENT_COLOR};
}}

QPushButton:disabled {{
    background-color: {PANEL_COLOR};
    border: 1px solid {BORDER_COLOR};
    color: {TEXT_DIM};
}}

/* Primary Action Button (Render orbital): solid muted accent, normal case, 38px */
QPushButton#primaryButton {{
    background-color: {ACCENT_COLOR};
    border: 1px solid {ACCENT_COLOR};
    border-radius: 3px;
    color: {PANEL_COLOR};
    font-weight: 600;
    font-size: 11pt;
    padding: 6px 14px;
}}

QPushButton#primaryButton:hover {{
    background-color: {ACCENT_HOVER};
    border: 1px solid {ACCENT_HOVER};
    color: {PANEL_COLOR};
}}

QPushButton#primaryButton:pressed {{
    background-color: #be772d;
    border: 1px solid #be772d;
    color: {PANEL_COLOR};
}}

QPushButton#primaryButton:disabled {{
    background-color: {BORDER_COLOR};
    border: 1px solid {BORDER_COLOR};
    color: {TEXT_DIM};
}}

/* Sliders (T4: 3px groove, 12px round handle, no glow) */
QSlider::groove:horizontal {{
    height: 3px;
    background: {BORDER_COLOR};
    border-radius: 1px;
}}

QSlider::sub-page:horizontal {{
    background: {ACCENT_COLOR};
    border-radius: 1px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT_COLOR};
    border: none;
    width: 12px;
    height: 12px;
    margin: -4.5px 0;
    border-radius: 6px;
}}

QSlider::handle:horizontal:hover {{
    background: {ACCENT_HOVER};
}}

/* Combo Boxes (T4: 1px border, 3px radius, crisp chevron indicator, no grey square) */
QComboBox {{
    background-color: {RAISED_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    padding: 4px 8px;
    color: {TEXT_COLOR};
    min-height: 24px;
}}

QComboBox:hover, QComboBox:focus {{
    border: 1px solid {ACCENT_COLOR};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid {BORDER_COLOR};
    background: transparent;
}}

QComboBox::down-arrow {{
    image: url({CHEVRON_ICON_PATH});
    width: 10px;
    height: 10px;
}}

QComboBox QAbstractItemView {{
    background-color: {RAISED_COLOR};
    border: 1px solid {BORDER_COLOR};
    selection-background-color: #252b35;
    selection-color: {ACCENT_COLOR};
    color: {TEXT_COLOR};
    outline: none;
    padding: 4px;
}}

/* Minimalist Dark Scrollbars (8px wide, handle #2a2f38, no track/arrows) */
QScrollBar:vertical {{
    border: none;
    background: transparent;
    width: 8px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: {BORDER_COLOR};
    min-height: 20px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: {ACCENT_COLOR};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    border: none;
    background: none;
    height: 0px;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

/* Menu Bar & Menus (T1 / T4) */
QMenuBar {{
    background-color: {PANEL_COLOR};
    color: {TEXT_COLOR};
    border-bottom: 1px solid {BORDER_COLOR};
    padding: 2px 6px;
}}

QMenuBar::item {{
    background: transparent;
    padding: 4px 10px;
    border-radius: 2px;
}}

QMenuBar::item:selected {{
    background-color: {RAISED_COLOR};
    color: {ACCENT_COLOR};
}}

QMenu {{
    background-color: {RAISED_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    padding: 4px;
}}

QMenu::item {{
    padding: 6px 22px;
    border-radius: 2px;
}}

QMenu::item:selected {{
    background-color: #252b35;
    color: {ACCENT_COLOR};
}}

/* Status Bar (T4: plain text, no segment borders) */
QStatusBar {{
    background-color: {PANEL_COLOR};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER_COLOR};
    font-size: 10pt;
}}

QStatusBar QLabel {{
    color: {TEXT_DIM};
}}

QStatusBar::item {{
    border: none;
}}

/* Progress Bar */
QProgressBar {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 2px;
    background-color: {BG_COLOR};
    text-align: center;
    color: {TEXT_COLOR};
    max-height: 12px;
    font-size: 8pt;
}}

QProgressBar::chunk {{
    background-color: {ACCENT_COLOR};
}}

/* Tooltips */
QToolTip {{
    background-color: {RAISED_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
    padding: 5px 8px;
    font-family: "{FONT_FAMILY}", "Carlito", "Segoe UI", sans-serif;
    font-size: 10pt;
}}

/* Tab Widget (Validation Dialog: flat tabs with accent underline on selection) */
QTabWidget::pane {{
    border: 1px solid {BORDER_COLOR};
    background-color: {PANEL_COLOR};
}}

QTabBar::tab {{
    background-color: {RAISED_COLOR};
    color: {TEXT_DIM};
    padding: 7px 16px;
    border: 1px solid {BORDER_COLOR};
    border-bottom: none;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
    margin-right: 2px;
}}

QTabBar::tab:selected {{
    background-color: {PANEL_COLOR};
    color: {ACCENT_COLOR};
    border-top: 2px solid {ACCENT_COLOR};
    border-left: 1px solid {BORDER_COLOR};
    border-right: 1px solid {BORDER_COLOR};
    border-bottom: 1px solid {PANEL_COLOR};
}}

QTabBar::tab:hover:!selected {{
    color: {ACCENT_HOVER};
}}

/* Splitters (T4: 1px line, 6px hit area, accent hover) */
QSplitter::handle {{
    background-color: {BORDER_COLOR};
}}

QSplitter::handle:hover {{
    background-color: {ACCENT_COLOR};
}}
"""
