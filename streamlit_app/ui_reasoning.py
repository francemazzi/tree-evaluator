from __future__ import annotations

import streamlit as st


def inject_custom_css() -> None:
    """Inject CSS for reasoning section styling."""
    st.markdown("""
    <style>
    .reasoning-container {
        background: linear-gradient(135deg, rgba(45, 55, 72, 0.95) 0%, rgba(26, 32, 44, 0.95) 100%);
        border: 1px solid rgba(99, 179, 237, 0.3);
        border-left: 4px solid #63b3ed;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 0.9em;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    .reasoning-header {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #90cdf4;
        font-weight: 600;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(99, 179, 237, 0.2);
    }

    .reasoning-content {
        color: #e2e8f0;
        line-height: 1.6;
    }

    .reasoning-step {
        padding: 8px 12px;
        margin: 6px 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        border-left: 2px solid rgba(99, 179, 237, 0.5);
    }

    .reasoning-step-emoji {
        margin-right: 8px;
    }

    .reasoning-loading {
        display: inline-block;
        animation: pulse 1.5s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }

    .stExpander[data-testid="stExpander"] > div:first-child {
        background: linear-gradient(135deg, rgba(45, 55, 72, 0.8) 0%, rgba(26, 32, 44, 0.8) 100%);
        border: 1px solid rgba(99, 179, 237, 0.3);
        border-radius: 8px;
    }

    .stExpander[data-testid="stExpander"] summary {
        color: #90cdf4 !important;
    }

    .tool-result {
        background: rgba(72, 187, 120, 0.1);
        border-left: 3px solid #48bb78;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.88em;
    }

    .validation-result {
        background: rgba(159, 122, 234, 0.1);
        border-left: 3px solid #9f7aea;
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.88em;
    }
    </style>
    """, unsafe_allow_html=True)


def format_reasoning_step(step: str) -> str:
    """Format a reasoning step with appropriate icon and styling."""
    step_lower = step.lower()

    if "✅" in step or "risultati" in step_lower:
        icon = "✅"
        css_class = "tool-result"
    elif "✓" in step or "validazione" in step_lower:
        icon = "✓"
        css_class = "validation-result"
    elif "query sql" in step_lower or "select" in step_lower:
        icon = "🔍"
        css_class = "reasoning-step"
    else:
        icon = "💭"
        css_class = "reasoning-step"

    return f'<div class="{css_class}"><span class="reasoning-step-emoji">{icon}</span>{step}</div>'


def render_reasoning_box(steps: list, is_loading: bool = False) -> str:
    """Render reasoning steps in a styled container."""
    header_text = "🧠 Processo di ragionamento"
    if is_loading:
        header_text += ' <span class="reasoning-loading">●●●</span>'

    formatted_steps = "".join([format_reasoning_step(step) for step in steps])
    return f"""
    <div class="reasoning-container">
        <div class="reasoning-header">{header_text}</div>
        <div class="reasoning-content">{formatted_steps}</div>
    </div>
    """
