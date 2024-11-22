# Toggle this flag to show or hide the login functionality
SHOW_LOGIN = False

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import firebase_admin
from firebase_admin import credentials

# Set page configuration
st.set_page_config(
    page_title="FoxEdge - Predictive Analytics",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Firebase Admin Initialization
base_dir = os.path.dirname(os.path.abspath(__file__))
service_account_path = os.path.join(base_dir, "utils", "serviceAccountKey.json")

try:
    cred = credentials.Certificate(service_account_path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
except Exception as e:
    st.error(f"Error initializing Firebase: {e}")

# Add CSS for branding and animation
st.markdown('''
    <style>
    .hero {
        position: relative;
        text-align: center;
        padding: 4em 1em;
        overflow: hidden;
    }

    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at center, rgba(255, 255, 255, 0.1), transparent);
        animation: rotate 30s linear infinite;
    }

    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .hero h1 {
        font-size: 3.5em;
        margin-bottom: 0.2em;
    }

    .hero p {
        font-size: 1.5em;
        margin-bottom: 1em;
        color: #CCCCCC;
    }

    .footer {
        text-align: center;
        margin-top: 3em;
        color: #999999;
    }

    .footer a {
        color: #2CFFAA;
        text-decoration: none;
    }
    </style>
''', unsafe_allow_html=True)

# App Homepage
st.markdown('''
    <div class="hero">
        <h1>FoxEdge</h1>
        <p>Your Ultimate Toolkit for Predictive Betting Insights</p>
    </div>
''', unsafe_allow_html=True)

# Chart Section
st.subheader("Betting Line Movement Analysis")
categories = ['3+ Toward Favorite', '2.0-2.5 Toward Favorite', '0.5-1.5 Toward Favorite', 
              'No Movement', '0.5-1.5 Toward Underdog', '2.0-2.5 Toward Underdog', '3+ Toward Underdog']
ats_cover = [42.1, 49.2, 50.3, 48.7, 48.3, 48.8, 48.0]
over_under = [49.7, 51.0, 51.6, 53.3, 50.7, 52.1, 53.7]

np.random.seed(42)
ats_prediction_upper = [x + np.random.uniform(1, 3) for x in ats_cover]
ats_prediction_lower = [x - np.random.uniform(1, 3) for x in ats_cover]
over_under_upper = [x + np.random.uniform(1, 3) for x in over_under]
over_under_lower = [x - np.random.uniform(1, 3) for x in over_under]

fig = go.Figure()

# ATS Cover Line
fig.add_trace(go.Scatter(
    x=categories,
    y=ats_cover,
    mode='lines+markers',
    name='ATS Cover %',
    line=dict(color='#2CFFAA', width=4),
    marker=dict(size=10, color='#2CFFAA'),
    hovertemplate='<b>%{x}</b><br>ATS Cover: %{y:.1f}%<extra></extra>'
))

# ATS Predictive Range
fig.add_trace(go.Scatter(
    x=categories + categories[::-1],
    y=ats_prediction_upper + ats_prediction_lower[::-1],
    fill='toself',
    fillcolor='rgba(44, 255, 170, 0.1)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

# Over/Under Line
fig.add_trace(go.Scatter(
    x=categories,
    y=over_under,
    mode='lines+markers',
    name='Over/Under %',
    line=dict(color='#A56BFF', width=4, dash='dot'),
    marker=dict(size=10, color='#A56BFF'),
    hovertemplate='<b>%{x}</b><br>Over/Under: %{y:.1f}%<extra></extra>'
))

# Over/Under Predictive Range
fig.add_trace(go.Scatter(
    x=categories + categories[::-1],
    y=over_under_upper + over_under_lower[::-1],
    fill='toself',
    fillcolor='rgba(165, 107, 255, 0.1)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False
))

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    xaxis=dict(title='Line Movement Category', titlefont=dict(size=18, color='#F5F5F5')),
    yaxis=dict(title='Percentage (%)', titlefont=dict(size=18, color='#F5F5F5')),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=12)),
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# Main Dashboard Layout
st.subheader("Explore Our Tools")
col1, col2, col3 = st.columns(3)

# Column 1: Key Stats Analysis
with col1:
    st.markdown("### Key Stats Analysis")
    st.markdown("Uncover key statistics to refine your betting strategies.")
    if st.button("Explore Key Stats"):
        st.session_state.page = "Key Stats Analysis"

# Column 2: Predictive Analytics
with col2:
    st.markdown("### Predictive Analytics")
    st.markdown("Run advanced simulations and predictive models for upcoming games.")
    if st.button("Run Simulations"):
        st.session_state.page = "Predictive Analytics"

# Column 3: Quantum-Inspired Tools
with col3:
    st.markdown("### NCAAB Quantum Tools")
    st.markdown("Leverage quantum-inspired simulations for college basketball predictions.")
    if st.button("Explore NCAAB Simulations"):
        st.session_state.page = "NCAAB Quantum-Inspired Game Simulations"

# Conditional Navigation Logic
page = st.session_state.get("page", "Home")

if page == "Key Stats Analysis":
    st.subheader("Key Stats Analysis")
    st.markdown("**Feature coming soon...**")

elif page == "Predictive Analytics":
    st.subheader("Predictive Analytics")
    st.markdown("**Feature coming soon...**")

elif page == "NCAAB Quantum-Inspired Game Simulations":
    # Import and display the content from the NCAAB script
    import pages.NCAAB_Quantum_Inspired_Game_Simulations as ncaab_simulations
    ncaab_simulations.display()

# Add Footer
st.markdown('''
    <div class="footer">
        &copy; 2024 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
