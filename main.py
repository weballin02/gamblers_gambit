import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="FoxEdge - Synesthetic Interface",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Synesthetic Interface CSS
st.markdown('''
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&family=Open+Sans:wght@400;600&display=swap');

        /* Root Variables */
        :root {
            --background-gradient-start: #0F2027;
            --background-gradient-end: #203A43;
            --primary-text-color: #ECECEC;
            --heading-text-color: #F5F5F5;
            --accent-color-teal: #2CFFAA;
            --accent-color-purple: #A56BFF;
            --highlight-color: #FF6B6B;
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
            font-size: 4em;
            margin-bottom: 0.2em;
        }

        .hero p {
            font-size: 1.5em;
            margin-bottom: 1em;
            color: #CCCCCC;
        }

        /* Buttons */
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

        /* Data Section */
        .data-section {
            padding: 4em 1em;
            text-align: center;
        }

        .data-section h2 {
            font-size: 2.5em;
            margin-bottom: 0.5em;
        }

        .data-section p {
            font-size: 1.2em;
            color: #CCCCCC;
            margin-bottom: 2em;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2em 1em;
            color: #999999;
            font-size: 0.9em;
        }

        .footer a {
            color: var(--accent-color-teal);
            text-decoration: none;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .hero h1 {
                font-size: 2.5em;
            }

            .hero p {
                font-size: 1.2em;
            }
        }
    </style>
''', unsafe_allow_html=True)

# Main Content

# Hero Section
st.markdown('''
    <div class="hero">
        <h1>FoxEdge</h1>
        <p>Experience Data Beyond Sight</p>
        <a href="#data-section" class="button">Discover More</a>
    </div>
''', unsafe_allow_html=True)

# Data Section
st.markdown('''
    <div class="data-section" id="data-section">
        <h2>Immersive Analytics</h2>
        <p>Interact with data like never before through our synesthetic visualizations.</p>
    </div>
''', unsafe_allow_html=True)

# Sample Data Visualization
# Generate random data for demonstration
np.random.seed(42)
x_data = np.linspace(0, 10, 500)
y_data = np.sin(x_data) + np.random.normal(scale=0.5, size=500)

# Create a Plotly figure
fig = go.Figure(data=go.Scatter(
    x=x_data,
    y=y_data,
    mode='lines',
    line=dict(color='#A56BFF', width=3),
    hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
))

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    hovermode='x unified'
)

# Display the figure
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown('''
    <div class="footer">
        &copy; 2023 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
