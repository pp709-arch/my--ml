import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="EnergySense AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("<div class='sidebar-title'>⚡ EnergySense AI Platform</div>", unsafe_allow_html=True)

    nav_items = [
        "Dashboard",
        "Analytics",
        "Demand Forecaster",
        "Prediction",
        "Model Comparison",
        "Active Model",
        "Best Results",
        "About Project"
    ]

    for item in nav_items:
        if item == "Dashboard":
            st.markdown(f"<div class='nav-item nav-active'>{item}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='nav-item'>{item}</div>", unsafe_allow_html=True)

# HEADER
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("""
    <div style='margin-top:10px;'>
        <span style='background:#dcfce7;color:#15803d;padding:6px 12px;border-radius:999px;font-size:12px;font-weight:600;'>Live System</span>
        <h1 style='margin-top:10px;'>Dashboard Overview</h1>
        <p style='color:#6b7280;'>Hotel energy monitoring powered by XGBoost ML</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background:white;padding:14px;border-radius:14px;margin-top:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
        <span style='color:#6b7280;'>Active Model:</span><br>
        <strong style='color:#7c3aed;'>XGBoost Classifier</strong>
    </div>
    """, unsafe_allow_html=True)

# METRIC CARDS
cards = st.columns(4)

metrics = [
    ("Total Records", "300", "+12.5% from last month", "#3b82f6"),
    ("Avg. Demand (KW)", "45.67", "+8.3% from last month", "#22c55e"),
    ("Max Demand (KW)", "198.6", "+15.2% from last month", "#f59e0b"),
    ("Best Model Accuracy", "96.72%", "XGBoost", "#8b5cf6")
]

for col, metric in zip(cards, metrics):
    title, value, change, color = metric

    with col:
        st.markdown(f"""
        <div class='card' style='border-top-color:{color}'>
            <div class='metric-title'>{title}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-change' style='color:{color}'>{change}</div>
        </div>
        """, unsafe_allow_html=True)

# DATA
pie_df = pd.DataFrame({
    "Category": ["Normal", "High", "Critical"],
    "Value": [73.3, 20, 6.7]
})

trend_x = np.arange(1, 51)

trend_df = pd.DataFrame({
    "Sample": trend_x,
    "AC Usage": np.sin(trend_x/2.5)*20 + 60,
    "Lighting": np.cos(trend_x/4)*3 + 6
})

hour_df = pd.DataFrame({
    "Hour": [f"{i:02d}:00" for i in range(0, 22, 2)],
    "Demand": [42, 38, 35, 50, 72, 88, 95, 102, 98, 116, 118]
})

bar_df = pd.DataFrame({
    "Hotel": ["Luxury", "Business", "Resort"],
    "Normal": [78, 86, 58],
    "High": [22, 20, 18],
    "Critical": [8, 6, 7]
})

# ROW 1
left, right = st.columns([1, 2])

with left:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Demand Distribution</div>", unsafe_allow_html=True)

    fig = px.pie(
        pie_df,
        names="Category",
        values="Value",
        hole=0.65,
        color="Category",
        color_discrete_map={
            "Normal": "#22c55e",
            "High": "#f59e0b",
            "Critical": "#ef4444"
        }
    )

    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>AC Usage Trend (50 Samples)</div>", unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend_df["Sample"],
        y=trend_df["AC Usage"],
        mode='lines',
        fill='tozeroy',
        name='AC Usage',
        line=dict(color='#8b5cf6', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=trend_df["Sample"],
        y=trend_df["Lighting"],
        mode='lines',
        fill='tozeroy',
        name='Lighting',
        line=dict(color='#22c55e', width=2)
    ))

    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ROW 2
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Energy Demand by Hour</div>", unsafe_allow_html=True)

    fig = px.area(hour_df, x="Hour", y="Demand")
    fig.update_traces(line_color="#3b82f6")
    fig.update_layout(height=320)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Energy Status by Hotel Type</div>", unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(name='Normal', x=bar_df['Hotel'], y=bar_df['Normal']))
    fig.add_trace(go.Bar(name='High', x=bar_df['Hotel'], y=bar_df['High']))
    fig.add_trace(go.Bar(name='Critical', x=bar_df['Hotel'], y=bar_df['Critical']))

    fig.update_layout(barmode='group', height=320)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# MODEL COMPARISON
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Model Quick Comparison</div>", unsafe_allow_html=True)

comparison_cols = st.columns(3)

models = [
    ("XGBoost", "96.72%"),
    ("Random Forest", "93.12%"),
    ("Decision Tree", "89.45%")
]

for col, model in zip(comparison_cols, models):
    name, acc = model

    with col:
        st.markdown(f"""
        <div style='background:#faf5ff;padding:20px;border-radius:16px;border:1px solid #e9d5ff;'>
            <h3 style='color:#7c3aed'>{name}</h3>
            <h1>{acc}</h1>
            <p style='color:#6b7280'>Accuracy Score</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
