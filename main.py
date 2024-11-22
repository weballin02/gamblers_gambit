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

    .hero .button {
        display: inline-block;
        margin-top: 1em;
        padding: 10px 20px;
        font-size: 1em;
        color: #FFFFFF;
        background: linear-gradient(45deg, #2CFFAA, #A56BFF);
        text-decoration: none;
        border-radius: 20px;
        transition: transform 0.3s ease;
    }

    .hero .button:hover {
        transform: translateY(-5px);
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

# Main Dashboard Layout
st.subheader("Welcome to FoxEdge - Explore Our Tools")

# Create a multi-column layout for tools and features
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
