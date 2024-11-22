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

# Add CSS for branding, animation, and styling
st.markdown('''
    <style>
    .hero {
        position: relative;
        text-align: center;
        padding: 4em 1em;
        overflow: hidden;
        background: linear-gradient(135deg, #2CFFAA, #A56BFF);
        color: #FFFFFF;
        border-radius: 10px;
        margin-bottom: 2em;
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
        color: #FFFFFF;
    }

    .cta-button {
        display: inline-block;
        padding: 0.8em 1.5em;
        font-size: 1.2em;
        color: #FFFFFF;
        background-color: #FF5733;
        border-radius: 5px;
        text-decoration: none;
        margin-top: 1em;
        transition: background-color 0.3s ease;
    }

    .cta-button:hover {
        background-color: #C70039;
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

    .tool-card {
        background-color: rgba(44, 255, 170, 0.1);
        border: 1px solid #2CFFAA;
        border-radius: 8px;
        padding: 1em;
        margin-bottom: 1em;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .tool-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 10px rgba(44, 255, 170, 0.3);
    }

    .tool-card h3 {
        font-size: 1.5em;
        color: #2CFFAA;
        margin-bottom: 0.5em;
    }

    .tool-card p {
        font-size: 1em;
        color: #FFFFFF;
    }
    </style>
''', unsafe_allow_html=True)

# Navigation Sidebar
tools = {
    "Home": "Welcome to FoxEdge! Your toolkit for predictive betting insights.",
    "Key Stats Analysis": "Analyze impactful stats for game outcomes.",
    "Predictive Analytics": "Advanced tools for smarter betting decisions.",
    "NCAAB Quantum Simulations": "Quantum-inspired predictions for NCAA basketball.",
    "Upcoming Games": "Analyze and predict outcomes for upcoming matchups.",
    "Betting Trends": "Explore betting patterns and trends.",
    "Line Movement Insights": "Understand how line movements impact predictions.",
    "Odds Comparisons": "Compare odds across sportsbooks for the best value.",
    "Simulation Settings": "Customize simulation parameters for better accuracy.",
    "Team Statistics": "Dive deep into team performance stats.",
}

# Default to Home if no page is selected
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar navigation
st.sidebar.title("FoxEdge Navigation")
selected_page = st.sidebar.radio("Go to", list(tools.keys()), index=list(tools.keys()).index(st.session_state.page))

# Update page when a new one is selected
st.session_state.page = selected_page

# Render the selected page
if st.session_state.page == "Home":
    # Hero Section
    st.markdown('''
        <div class="hero">
            <h1>FoxEdge</h1>
            <p>Your Ultimate Toolkit for Predictive Betting Insights</p>
            <a href="#tools" class="cta-button">Get Started</a>
        </div>
    ''', unsafe_allow_html=True)

    # Explore Tools Section
    st.markdown('<div id="tools"></div>', unsafe_allow_html=True)
    st.subheader("Explore Our Tools and Features")
    cols = st.columns(3)
    tool_names = list(tools.keys())[1:]  # Exclude "Home" from tools

    for idx, tool in enumerate(tool_names):
        with cols[idx % 3]:
            st.markdown(f'''
                <div class="tool-card">
                    <h3>{tool}</h3>
                    <p>{tools[tool]}</p>
                </div>
            ''', unsafe_allow_html=True)
            if st.button(f"Explore {tool}", key=f"btn_{tool.replace(' ', '_')}"):
                st.session_state.page = tool  # Navigate to the corresponding page

else:
    # Render the selected tool's content dynamically
    st.subheader(st.session_state.page)
    st.markdown(f"**{tools[st.session_state.page]}**")

# Footer
st.markdown('''
    <div class="footer">
        &copy; 2024 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
