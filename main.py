# Toggle this flag to show or hide the login functionality
SHOW_LOGIN = False

# Import Statements
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import firebase_admin
from firebase_admin import credentials

# Define icons directory path
ICONS_DIR = "icons"  # Ensure this is set correctly

# Helper function to get image path
def get_image_path(icon_name):
    return os.path.join(ICONS_DIR, icon_name)

# Set page configuration with theming
st.set_page_config(
    page_title="FoxEdge - Predictive Analytics",
    page_icon=get_image_path("logo.png"),  # Replace "fox_icon.png" with the actual file name
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Session State for Dark Mode and Page Navigation
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

if 'page' not in st.session_state:
    st.session_state['page'] = "Dashboard"

# Function to Toggle Dark Mode
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Dark Mode Toggle Button
st.sidebar.button("Toggle Dark Mode", on_click=toggle_dark_mode)

# Apply Theme Based on Dark Mode
if st.session_state.dark_mode:
    st.markdown("""
        <style>
        body {
            background-color: #2C3E50; /* Charcoal Dark Gray */
            color: #FFFFFF; /* Crisp White */
        }
        .cta-button {
            background-color: #FF4500; /* Fiery Red */
            color: #FFFFFF;
        }
        .footer a {
            color: #1E90FF; /* Electric Blue */
        }
        .tool-card {
            background-color: rgba(44, 62, 80, 0.8); /* Charcoal Dark Gray */
            border: 1px solid #32CD32; /* Lime Green border */
            color: #FFFFFF;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {
            background-color: #FFFFFF; /* Crisp White */
            color: #333333; /* Dark text */
        }
        .cta-button {
            background-color: #FF4500; /* Fiery Red */
            color: #FFFFFF;
        }
        .footer a {
            color: #1E90FF; /* Electric Blue */
        }
        .tool-card {
            background-color: rgba(245, 245, 245, 0.8); /* Light Gray */
            border: 1px solid #32CD32; /* Lime Green border */
            color: #333333;
        }
        </style>
    """, unsafe_allow_html=True)

# CSS for Branding, Hero Section, and Tool Cards
st.markdown('''
    <style>
    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, #1E90FF, #FF4500); /* Electric Blue to Fiery Red */
        padding: 3em;
        border-radius: 15px;
        text-align: center;
        color: #FFFFFF;
        margin-bottom: 2em;
    }
    .hero h1 {
        font-size: 3.5em;
        margin: 0;
        font-family: 'Montserrat', sans-serif;
    }
    .hero p {
        font-size: 1.5em;
        margin-top: 0.5em;
        font-family: 'Open Sans', sans-serif;
    }
    .cta-button {
        background-color: #FF4500; /* Fiery Red */
        color: #FFFFFF;
        padding: 10px 25px;
        border-radius: 50px;
        text-decoration: none;
        font-size: 1.2em;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
        transition: background-color 0.3s, transform 0.2s;
    }
    .cta-button:hover {
        background-color: #FF8C00; /* Deep Orange */
        transform: scale(1.05);
    }
    .tool-card {
        background-color: rgba(255, 255, 255, 0.9); /* White background for better contrast */
        border: 1px solid #32CD32; /* Lime Green border */
        border-radius: 10px;
        padding: 20px; /* Increased padding for better spacing */
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */
    }
    .tool-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(50, 205, 50, 0.3); /* Enhanced shadow on hover */
    }
    .footer {
        text-align: center;
        margin-top: 3em;
        font-size: 0.9em;
        color: #999999;
    }
    .footer a {
        color: #1E90FF; /* Electric Blue */
        text-decoration: none;
    }
    </style>
''', unsafe_allow_html=True)

# Firebase Admin Initialization
@st.cache_resource
def initialize_firebase():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    service_account_path = os.path.join(base_dir, "utils", "serviceAccountKey.json")
    cred = credentials.Certificate(service_account_path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return True

try:
    if initialize_firebase():
        st.sidebar.success("Firebase initialized successfully!")
except FileNotFoundError:
    st.sidebar.error("Firebase service account key not found. Please check the path.")
except ValueError as ve:
    st.sidebar.error(f"Firebase initialization error: {ve}")
except Exception as e:
    st.sidebar.error(f"An unexpected error occurred: {e}")

# Function to Load Data
@st.cache_data(ttl=3600)
def load_data_func():
    return pd.DataFrame({
        'Team': ['Team A', 'Team B', 'Team C'],
        'Wins': [20, 15, 10],
        'Losses': [5, 10, 15]
    })

data = load_data_func()

# Function to Render Hero Section
def render_hero_section():
    st.markdown('''
        <div class="hero">
            <h1>FoxEdge</h1>
            <p>Your Ultimate Toolkit for Predictive Betting Insights</p>
            <a href="#tools" class="cta-button">Get Started</a>
        </div>
    ''', unsafe_allow_html=True)

# Function to Render Chart
def render_chart():
    st.subheader("Betting Line Movement Analysis")
    categories = ['3+ Toward Favorite', '2-2.5 Toward Favorite', '0.5-1.5 Toward Favorite', 'No Movement']
    ats_cover = [42, 50, 52, 49]
    over_under = [48, 52, 54, 51]

    # Simulated confidence intervals
    ats_upper = [x + 2 for x in ats_cover]
    ats_lower = [x - 2 for x in ats_cover]
    ou_upper = [x + 2 for x in over_under]
    ou_lower = [x - 2 for x in over_under]

    fig = go.Figure()

    # ATS Cover Line
    fig.add_trace(go.Scatter(
        x=categories,
        y=ats_cover,
        mode='lines+markers',
        name='ATS Cover %',
        line=dict(color='#1E90FF', width=3),  # Electric Blue
        marker=dict(size=8, color='#1E90FF'),
        hovertemplate='%{y:.1f}% ATS Cover<extra></extra>'
    ))

    # ATS Confidence Band
    fig.add_trace(go.Scatter(
        x=categories + categories[::-1],
        y=ats_upper + ats_lower[::-1],
        fill='toself',
        fillcolor='rgba(50, 205, 50, 0.2)',  # Lime Green transparency
        line=dict(color='rgba(255,255,255,0)'),
        name='ATS Confidence Range',
        hoverinfo='skip'
    ))

    # Over/Under Line
    fig.add_trace(go.Scatter(
        x=categories,
        y=over_under,
        mode='lines+markers',
        name='Over/Under %',
        line=dict(color='#FF4500', width=3),  # Fiery Red
        marker=dict(size=8, color='#FF4500'),
        hovertemplate='%{y:.1f}% Over/Under<extra></extra>'
    ))

    # Over/Under Confidence Band
    fig.add_trace(go.Scatter(
        x=categories + categories[::-1],
        y=ou_upper + ou_lower[::-1],
        fill='toself',
        fillcolor='rgba(255, 140, 0, 0.2)',  # Deep Orange transparency
        line=dict(color='rgba(255,255,255,0)'),
        name='Over/Under Confidence Range',
        hoverinfo='skip'
    ))

    fig.update_layout(
        template='plotly_dark',  # Dark theme for the chart
        xaxis=dict(
            title='Line Movement Category',
            titlefont=dict(size=14, color='#FFFFFF'),  # White axis title
            tickfont=dict(color='#FFFFFF')  # White tick labels
        ),
        yaxis=dict(
            title='Percentage (%)',
            titlefont=dict(size=14, color='#FFFFFF'),  # White axis title
            tickfont=dict(color='#FFFFFF')  # White tick labels
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color='#FFFFFF')  # White legend text
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode='x unified',
        plot_bgcolor='#000000',  # Black plot background
        paper_bgcolor='#000000',  # Black chart background
    )
    st.plotly_chart(fig, use_container_width=True)

# Function to Render Tool Cards
def render_tool_cards():
    st.subheader("Explore Our Tools and Features")
    tools = [
        {"name": "Key Stats Analysis", "description": "Uncover impactful stats driving game outcomes.", "url": "https://foxedge.streamlit.app/Key_Stat_Analysis", "icon": "🎱"},
        {"name": "NFL FoxEdge", "description": "Explore NFL betting insights.", "url": "https://foxedge.streamlit.app/NFL_FoxEdge", "icon": "nfl_icon.png"},
        {"name": "NBA FoxEdge", "description": "Get NBA predictive tools.", "url": "https://foxedge.streamlit.app/NBA_FoxEdge", "icon": "nba_icon.png"},
        {"name": "NCAAB Quantum Simulations", "description": "Quantum-inspired NCAA basketball predictions.", "url": "https://foxedge.streamlit.app/NCAAB_Quantum-Inspired_Game_Predictions", "icon": "ncaab_icon.png"},
        {"name": "NFL Betting Insights", "description": "Advanced NFL trends.", "url": "https://foxedge.streamlit.app/Enhanced_NFL_Betting_Insights", "icon": "nfl_insights_icon.png"},
        {"name": "NFL Game Simulations", "description": "Simulate NFL games for insights.", "url": "https://foxedge.streamlit.app/NFL_Game_Simulations", "icon": "nfl_gmae_sim.png"},
        {"name": "NFL Team Scoring Predictions", "description": "Predict NFL team scoring outcomes.", "url": "https://foxedge.streamlit.app/NFL_Team_Scoring_Predictions", "icon": "nfl_score.png"},
        {"name": "NBA Betting Insights", "description": "Explore NBA betting trends.", "url": "https://foxedge.streamlit.app/Enhanced_NBA_Betting_Insights", "icon": "nba_insight.png"},
        {"name": "NBA Team Scoring Predictions", "description": "Predict NBA team scoring outcomes.", "url": "https://foxedge.streamlit.app/NBA_Team_Scoring_Predictions", "icon": "nba_icon.png"},
        {"name": "NBA Quantum Simulations", "description": "Quantum-inspired simulations for NBA games.", "url": "https://foxedge.streamlit.app/NBA_Quantum-Inspired_Game_Predictions", "icon": "nba_quant.png"},
        {"name": "NFL Quantum Simulations", "description": "Quantum-inspired simulations for NFL games.", "url": "https://foxedge.streamlit.app/NFL_Quantum-Inspired_Game_Predictions", "icon": "nfl_quant.png"},
        {"name": "Gambler's Gambit", "description": "Read articles and insights.", "url": "https://foxedge.streamlit.app/Blog", "icon": "blog.png"}
    ]
    
    for tool in tools:
        image_path = get_image_path(tool['icon'])
        print(f"Attempting to load image from: {image_path}")  # Debugging line
        st.markdown(f'''
            <div class="tool-card">
                <div class="tool-card-icon">
                    <img src="{image_path}" alt="{tool['name']}" style="width: 40px; height: 40px;">
                </div>
                <h3>{tool["name"]}</h3>
                <p>{tool["description"]}</p>
                <a href="{tool["url"]}" target="_blank" class="cta-button">Explore</a>
            </div>
        ''', unsafe_allow_html=True)

# Render Sections
render_hero_section()
render_chart()
render_tool_cards()

# Footer
st.markdown('''
    <div class="footer">
        &copy; 2024 FoxEdge. All rights reserved.
    </div>
''', unsafe_allow_html=True)
