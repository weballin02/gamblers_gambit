import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import firebase_admin
from firebase_admin import credentials

# Set page configuration (must be the first Streamlit command)
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

# Sidebar Navigation
if "user" not in st.session_state:
    pages = ["Home", "Key Stats Analysis"]
    page = st.sidebar.radio("Go to:", pages, key="public_navigation")
else:
    st.sidebar.success(f"Welcome, {st.session_state['user']['email']}!")
    if st.sidebar.button("Logout", key="logout_button"):
        del st.session_state['user']
        st.experimental_rerun()
    pages = ["Home", "Key Stats Analysis", "Predictive Analytics"]
    page = st.sidebar.radio("Go to:", pages, key="private_navigation")

# Authentication Tabs
if "user" not in st.session_state:
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email_tab1")
        password = st.text_input("Password", type="password", key="login_password_tab1")
        if st.button("Login", key="login_button_tab1"):
            try:
                # Firebase Authentication (placeholder logic)
                st.session_state['user'] = {"email": email}
                st.success(f"Logged in as: {email}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
    with tab2:
        st.subheader("Register")
        email = st.text_input("Register Email", key="register_email_tab2")
        password = st.text_input("Register Password", type="password", key="register_password_tab2")
        if st.button("Register", key="register_button_tab2"):
            try:
                # Firebase Registration (placeholder logic)
                st.success(f"User registered successfully with email: {email}")
            except Exception as e:
                st.error(f"Error registering user: {e}")

# Page Logic
if page == "Home":
    # Hero Section
    st.markdown('''
        <div class="hero">
            <h1>FoxEdge Predictive Analytics</h1>
            <p>The Future of Sports Betting Confidence</p>
            <a href="#data-section" class="button">Discover More</a>
        </div>
    ''', unsafe_allow_html=True)

    # Data for the Chart
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

    st.markdown('''
        <div class="footer">
            &copy; 2024 <a href="#">FoxEdge</a>. All rights reserved.
        </div>
    ''', unsafe_allow_html=True)
