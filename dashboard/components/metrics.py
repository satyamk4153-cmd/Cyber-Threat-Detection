"""
Reusable UI metric card components with SOC Dark Theme CSS styling.
"""

from typing import Optional


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",  # 'normal', 'danger', 'success'
    border_color: str = "#3A86FF",
    icon: str = "",
) -> str:
    """Returns HTML for a cybersecurity SOC styled KPI card."""
    delta_html = ""
    if delta:
        d_color = "#38B000" if delta_color == "success" else ("#FF0054" if delta_color == "danger" else "#00F5D4")
        delta_html = f"<div style='font-size: 0.8rem; color: {d_color}; margin-top: 4px;'>{delta}</div>"

    return f"""
    <div style="
        background: linear-gradient(135deg, #1C2541 0%, #111827 100%);
        border: 1px solid {border_color}44;
        border-left: 4px solid {border_color};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    ">
        <div style="color: #94A3B8; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">
            {icon} {label}
        </div>
        <div style="color: #F8F9FA; font-size: 1.7rem; font-weight: 700; margin-top: 4px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
            {value}
        </div>
        {delta_html}
    </div>
    """


def render_status_pill(is_active: bool, active_text: str, inactive_text: str) -> str:
    """Renders a pulsing status indicator pill."""
    bg = "#38B00022" if is_active else "#FFBE0B22"
    border = "#38B000" if is_active else "#FFBE0B"
    color = "#38B000" if is_active else "#FFBE0B"
    text = active_text if is_active else inactive_text
    dot = "●"

    return f"""
    <span style="
        background: {bg};
        border: 1px solid {border};
        color: {color};
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        display: inline-block;
    ">
        {dot} {text}
    </span>
    """

