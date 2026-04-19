"""Catppuccin Mocha palette + shared PyQt5 stylesheet.

Every window in the app imports `STYLESHEET` and `PALETTE` so colours
stay consistent. To reskin the app, replace this one file.
"""

from __future__ import annotations

import os

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_ARROW_DOWN = os.path.join(_ASSETS, "arrow_down.svg").replace("\\", "/")
_ARROW_UP = os.path.join(_ASSETS, "arrow_up.svg").replace("\\", "/")


PALETTE = {
    "bg":       "#1e1e2e",
    "surface":  "#313244",
    "surface2": "#45475a",
    "text":     "#cdd6f4",
    "text_dim": "#a6adc8",
    "accent":   "#89b4fa",
    "success":  "#a6e3a1",
    "warn":     "#f9e2af",
    "err":      "#f38ba8",
    "muted":    "#6c7086",
}


STYLESHEET = f"""
QWidget {{
    background-color: {PALETTE["bg"]};
    color: {PALETTE["text"]};
    font-family: "Inter", "DejaVu Sans", sans-serif;
    font-size: 13px;
}}
QFrame#Card, QGroupBox {{
    background-color: {PALETTE["surface"]};
    border: 1px solid {PALETTE["surface2"]};
    border-radius: 10px;
}}
QPushButton {{
    background-color: {PALETTE["surface"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["surface2"]};
    padding: 8px 14px;
    border-radius: 8px;
}}
QPushButton:hover {{
    background-color: {PALETTE["surface2"]};
    border-color: {PALETTE["accent"]};
}}
QPushButton:pressed {{
    background-color: {PALETTE["accent"]};
    color: {PALETTE["bg"]};
}}
QPushButton:disabled {{
    color: {PALETTE["muted"]};
}}
QComboBox, QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {PALETTE["surface"]};
    color: {PALETTE["text"]};
    border: 1px solid {PALETTE["surface2"]};
    padding: 6px 8px;
    border-radius: 6px;
}}
/* Spinboxes use PlusMinus button symbols (set per-widget via Python helper)
   so the +/− labels stay clearly visible on the dark theme. Style the
   button slots so they blend with the card. */
QSpinBox, QDoubleSpinBox {{
    padding-right: 4px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 22px;
    background: {PALETTE["surface2"]};
    border: none;
}}
QSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    border-top-right-radius: 5px;
    border-left: 1px solid {PALETTE["bg"]};
    border-bottom: 1px solid {PALETTE["bg"]};
}}
QSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    border-bottom-right-radius: 5px;
    border-left: 1px solid {PALETTE["bg"]};
}}
QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    border-top-right-radius: 5px;
    border-left: 1px solid {PALETTE["bg"]};
    border-bottom: 1px solid {PALETTE["bg"]};
}}
QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    border-bottom-right-radius: 5px;
    border-left: 1px solid {PALETTE["bg"]};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {PALETTE["accent"]};
    color: {PALETTE["bg"]};
}}
/* Combo-box drop-down: tinted button with an SVG down-arrow. Qt needs an
   explicit image for ::down-arrow — otherwise the subcontrol renders
   blank once any other combo sub-control is styled. */
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid {PALETTE["bg"]};
    border-top-right-radius: 5px;
    border-bottom-right-radius: 5px;
    background: {PALETTE["surface2"]};
}}
QComboBox::drop-down:hover {{
    background: {PALETTE["accent"]};
}}
QComboBox::down-arrow {{
    image: url("{_ARROW_DOWN}");
    width: 10px;
    height: 6px;
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {PALETTE["surface2"]};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {PALETTE["accent"]};
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}}
QTabWidget::pane {{
    border: none;
    background: {PALETTE["bg"]};
}}
QTabBar::tab {{
    background: transparent;
    color: {PALETTE["text_dim"]};
    padding: 8px 14px;
    border: none;
}}
QTabBar::tab:selected {{
    color: {PALETTE["accent"]};
    border-bottom: 2px solid {PALETTE["accent"]};
}}
QLabel {{
    background: transparent;
}}
QLabel#Heading {{
    font-size: 18px;
    font-weight: 600;
    color: {PALETTE["text"]};
}}
QLabel#Subtle {{
    color: {PALETTE["text_dim"]};
}}
QListWidget, QScrollArea {{
    background: {PALETTE["bg"]};
    border: none;
}}
"""


def apply_theme(app) -> None:
    """Apply the global stylesheet + fusion style. Call once in main()."""
    try:
        from PyQt5.QtWidgets import QStyleFactory

        app.setStyle(QStyleFactory.create("Fusion"))
    except Exception:
        pass
    app.setStyleSheet(STYLESHEET)
