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
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&family=Open+Sans:wght@400;600&display=swap');

        /* Root Variables with Updated Colors */
        :root {
            --background-gradient-start: #2CFFAA;  /* Teal start */
            --background-gradient-end: #A56BFF;   /* Purple end */
            --primary-text-color: #ECECEC;
            --heading-text-color: #F5F5F5;
            --accent-color-teal: #FF5733;         /* Accent */
            --accent-color-purple: #C70039;      /* Accent */
            --highlight-color: #FFC300;          /* Highlight */
            --font-heading: 'Raleway', sans-serif;
            --font-body: 'Open Sans', sans-serif;
        }

        /* Global Styles */
        body, html {
            background: linear-gradient(135deg, var(--background-gradient-start), var(--background-gradient-end));
            color: var(--primary-text-color);
            font-family: var(--font-body);
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }

        h1, h2, h3 {
            font-family: var(--font-heading);
            color: var(--heading-text-color);
        }

        /* Hero Section */
        .hero {
            position: relative;
            text-align: center;
            padding: 6em 1em;
            overflow: hidden;
            background: linear-gradient(135deg, var(--background-gradient-start), var(--background-gradient-end));
            animation: gradient-flip 10s infinite ease-in-out, rotate-background 30s infinite linear;
            border-radius: 10px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        }

        /* Gradient Flip Animation */
        @keyframes gradient-flip {
            0% {
                background: linear-gradient(135deg, var(--background-gradient-start), var(--background-gradient-end));
            }
            25% {
                background: linear-gradient(135deg, var(--background-gradient-end), black);
            }
            50% {
                background: linear-gradient(135deg, black, var(--background-gradient-start));
            }
            75% {
                background: linear-gradient(135deg, black, var(--background-gradient-end));
            }
            100% {
                background: linear-gradient(135deg, var(--background-gradient-start), var(--background-gradient-end));
            }
        }

        /* Radial Gradient Rotation */
        .hero::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at center, rgba(255, 255, 255, 0.2), transparent);
            animation: rotate-background 30s linear infinite;
        }

        @keyframes rotate-background {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }

        /* Text and Buttons */
        .hero h1 {
            font-size: 4em;
            margin-bottom: 0.2em;
            color: var(--heading-text-color);
        }

        .hero p {
            font-size: 1.5em;
            margin-bottom: 1em;
            color: #CCCCCC;
        }

        .button {
            background: linear-gradient(45deg, var(--accent-color-teal), var(--accent-color-purple));
            border: none;
            padding: 1em 2em;
            color: #FFFFFF;
            font-size: 1.2em;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.3s ease;
            text-decoration: none;
        }

        .button:hover {
            transform: translateY(-5px);
        }
    </style>
''', unsafe_allow_html=True)

# Hero Section with Animation
st.markdown('''
    <div class="hero">
        <h1>FoxEdge</h1>
        <p>Your Ultimate Toolkit for Predictive Betting Insights</p>
        <a href="#tools" class="button">Get Started</a>
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

# Explore Tools Section
st.markdown('<div id="tools"></div>', unsafe_allow_html=True)
st.subheader("Explore Our Tools and Features")
tools = [
    {"name": "Key Stats Analysis", "description": "Uncover the most impactful stats driving game outcomes."},
    {"name": "Predictive Analytics", "description": "Advanced tools for smarter betting decisions."},
    {"name": "NCAAB Quantum Simulations", "description": "Quantum-inspired NCAA basketball predictions."},
    {"name": "Upcoming Games", "description": "Analyze and predict outcomes for upcoming matchups."},
    {"name": "Betting Trends", "description": "Explore betting patterns and trends."},
    {"name": "Line Movement Insights", "description": "See how line movements impact predictions."},
    {"name": "Odds Comparisons", "description": "Compare odds across sportsbooks."},
    {"name": "Simulation Settings", "description": "Customize simulation parameters for better accuracy."},
    {"name": "Team Statistics", "description": "Dive deep into team performance stats."}
]

cols = st.columns(3)
for idx, tool in enumerate(tools):
    with cols[idx % 3]:
        st.markdown(f'''
            <div class="tool-card">
                <h3>{tool["name"]}</h3>
                <p>{tool["description"]}</p>
                <a href="#!" class="cta-button">Explore</a>
            </div>
        ''', unsafe_allow_html=True)

# Placeholder Feature Pages
page = st.session_state.get("page", "Home")

if page == "Key Stats Analysis":
    st.subheader("Key Stats Analysis")
    st.markdown("**Feature coming soon...**")

elif page == "Predictive Analytics":
    st.subheader("Predictive Analytics")
    st.markdown("**Feature coming soon...**")

elif page == "NCAAB Quantum Simulations":
    st.subheader("NCAAB Quantum Simulations")
    st.markdown("**Feature coming soon...**")

# Add Footer
st.markdown('''
    <div class="footer">
        &copy; 2024 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
