"""Catppuccin Mocha palette + shared PyQt5 stylesheet.

Every window in the app imports `STYLESHEET` and `PALETTE` so colours
stay consistent. To reskin the app, replace this one file.
"""

from __future__ import annotations


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
