# Import Statements
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import firebase_admin
from firebase_admin import credentials
import datetime

# Define icons directory path
ICONS_DIR = "icons"  # Ensure this is set correctly

# Helper function to get image path
def get_image_path(icon_name):
    return os.path.join(ICONS_DIR, icon_name)

# Set page configuration with theming
st.set_page_config(
    page_title="FoxEdge - Predictive Analytics",
    page_icon=get_image_path("logo.png"),  # Replace with your actual logo file
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
st.sidebar.button("🌗 Toggle Dark Mode", on_click=toggle_dark_mode)

# Apply Theme Based on Dark Mode
if st.session_state.dark_mode:
    primary_bg = "#121212"
    primary_text = "#FFFFFF"
    secondary_bg = "#1E1E1E"
    accent_color = "#BB86FC"
    highlight_color = "#03DAC6"
else:
    primary_bg = "#FFFFFF"
    primary_text = "#000000"
    secondary_bg = "#F5F5F5"
    accent_color = "#6200EE"
    highlight_color = "#03DAC6"

# Custom CSS for Novel Design
st.markdown(f'''
    <style>
    /* Global Styles */
    body {{
        background-color: {primary_bg};
        color: {primary_text};
        font-family: 'Roboto', sans-serif;
    }}
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    /* Hero Section */
    .hero {{
        background: linear-gradient(135deg, {accent_color}, {highlight_color});
        padding: 3em;
        border-radius: 20px;
        text-align: center;
        color: #FFFFFF;
        margin-bottom: 2em;
        position: relative;
        overflow: hidden;
    }}
    .hero::before {{
        content: '';
        background-image: radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.1), transparent);
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        animation: rotation 30s infinite linear;
    }}
    @keyframes rotation {{
        from {{transform: rotate(0deg);}}
        to {{transform: rotate(360deg);}}
    }}
    .hero h1 {{
        font-size: 3.5em;
        margin: 0;
        font-weight: bold;
        letter-spacing: -1px;
    }}
    .hero p {{
        font-size: 1.5em;
        margin-top: 0.5em;
    }}
    .cta-button {{
        background-color: {highlight_color};
        color: {primary_text};
        padding: 10px 25px;
        border-radius: 50px;
        text-decoration: none;
        font-size: 1.2em;
        margin-top: 1em;
        display: inline-block;
        transition: background-color 0.3s, transform 0.2s;
    }}
    .cta-button:hover {{
        background-color: #018786;
        transform: translateY(-5px);
    }}
    /* Tool Card Styles */
    .tool-card {{
        background-color: #000000;  /* Set background to black */
        border: 1px solid {highlight_color};  /* Keep the border color */
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5);  /* Darker shadow for better visibility */
        color: #FFFFFF;  /* Set text color to white */
    }}
    .tool-card h3 {{
        margin-top: 0.5em;
        color: #FFFFFF;  /* Ensure heading is white */
    }}
    .tool-card p {{
        color: #FFFFFF;  /* Ensure paragraph text is white */
    }}
    .footer {{
        text-align: center;
        margin-top: 3em;
        font-size: 0.9em;
        color: #999999;
    }}
    .footer a {{
        color: {accent_color};
        text-decoration: none;
    }}
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
    st.markdown(f'''
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
        line=dict(color=accent_color, width=3),
        marker=dict(size=8, color=accent_color),
        hovertemplate='%{y:.1f}% ATS Cover<extra></extra>'
    ))

    # ATS Confidence Band
    fig.add_trace(go.Scatter(
        x=categories + categories[::-1],
        y=ats_upper + ats_lower[::-1],
        fill='toself',
        fillcolor='rgba(50, 205, 50, 0.2)',
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
        line=dict(color=highlight_color, width=3),
        marker=dict(size=8, color=highlight_color),
        hovertemplate='%{y:.1f}% Over/Under<extra></extra>'
    ))

    # Over/Under Confidence Band
    fig.add_trace(go.Scatter(
        x=categories + categories[::-1],
        y=ou_upper + ou_lower[::-1],
        fill='toself',
        fillcolor='rgba(255, 140, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Over/Under Confidence Range',
        hoverinfo='skip'
    ))

    fig.update_layout(
        template='plotly_white',
        xaxis=dict(
            title='Line Movement Category',
            titlefont=dict(size=14, color=primary_text),
            tickfont=dict(color=primary_text)
        ),
        yaxis=dict(
            title='Percentage (%)',
            titlefont=dict(size=14, color=primary_text),
            tickfont=dict(color=primary_text)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color=primary_text)
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode='x unified',
        plot_bgcolor=secondary_bg,
        paper_bgcolor=secondary_bg,
    )
    st.plotly_chart(fig, use_container_width=True)

# Function to Render Tool Cards
def render_tool_cards():
    st.subheader("Explore Our Tools and Features")
    tools = [
        {"name": "Key Stats Analysis", "description": "Uncover impactful stats driving game outcomes.", "url": "https://foxedge.streamlit.app/Key_Stat_Analysis", "icon": "🎱"},
        {"name": "NFL FoxEdge", "description": "Explore NFL betting insights.", "url": "https://foxedge.streamlit.app/NFL_FoxEdge", "icon": "🏈"},
        {"name": "NBA FoxEdge", "description": "Get NBA predictive tools.", "url": "https://foxedge.streamlit.app/NBA_FoxEdge", "icon": "🏀"},
        {"name": "NCAAB Quantum Simulations", "description": "Quantum-inspired NCAA basketball predictions.", "url": "https://foxedge.streamlit.app/NCAAB_Quantum-Inspired_Game_Predictions", "icon": "🎓"},
        {"name": "NFL Betting Insights", "description": "Advanced NFL trends.", "url": "https://foxedge.streamlit.app/Enhanced_NFL_Betting_Insights", "icon": "📊"},
        {"name": "NFL Game Simulations", "description": "Simulate NFL games for insights.", "url": "https://foxedge.streamlit.app/NFL_Game_Simulations", "icon": "🎮"},
        {"name": "NFL Team Scoring Predictions", "description": "Predict NFL team scoring outcomes.", "url": "https://foxedge.streamlit.app/NFL_Team_Scoring_Predictions", "icon": "📈"},
        {"name": "NBA Betting Insights", "description": "Explore NBA betting trends.", "url": "https://foxedge.streamlit.app/Enhanced_NBA_Betting_Insights", "icon": "📊"},
        {"name": "NBA Team Scoring Predictions", "description": "Predict NBA team scoring outcomes.", "url": "https://foxedge.streamlit.app/NBA_Team_Scoring_Predictions", "icon": "📈"},
        {"name": "NBA Quantum Simulations", "description": "Quantum-inspired simulations for NBA games.", "url": "https://foxedge.streamlit.app/NBA_Quantum-Inspired_Game_Predictions", "icon": "🔮"},
        {"name": "NFL Quantum Simulations", "description": "Quantum-inspired simulations for NFL games.", "url": "https://foxedge.streamlit.app/NFL_Quantum-Inspired_Game_Predictions", "icon": "🔮"},
        {"name": "Gambler's Gambit", "description": "Read articles and insights.", "url": "https://foxedge.streamlit.app/Blog", "icon": "📝"}
    ]
    
    cols = st.columns(3)
    for idx, tool in enumerate(tools):
        with cols[idx % 3]:
            st.markdown(f'''
                <div class="tool-card">
                    <div style="font-size: 2em;">{tool["icon"]}</div>
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
st.markdown(f'''
    <div class="footer">
        &copy; {datetime.datetime.now().year} FoxEdge. All rights reserved.
    </div>
''', unsafe_allow_html=True)
