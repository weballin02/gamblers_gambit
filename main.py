import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="FoxEdge - Predictive Analytics",
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
        <h1>FoxEdge Quantum Analytics</h1>
        <p>The Future of Sports Betting Confidence</p>
        <a href="#data-section" class="button">Discover More</a>
    </div>
''', unsafe_allow_html=True)

# Data Section
st.markdown('''
    <div class="data-section" id="data-section">
        <!-- Removed the top label as per your request -->
        <p>Unlock insights with our advanced analytics.</p>
    </div>
''', unsafe_allow_html=True)

# Data for the Chart
categories = ['3+ Toward Favorite', '2.0-2.5 Toward Favorite', '0.5-1.5 Toward Favorite', 
              'No Movement', '0.5-1.5 Toward Underdog', '2.0-2.5 Toward Underdog', '3+ Toward Underdog']
ats_cover = [42.1, 49.2, 50.3, 48.7, 48.3, 48.8, 48.0]
over_under = [49.7, 51.0, 51.6, 53.3, 50.7, 52.1, 53.7]

# Simulate Predictive Ranges (Replace with actual predictive data if available)
np.random.seed(42)
ats_prediction_upper = [x + np.random.uniform(1, 3) for x in ats_cover]
ats_prediction_lower = [x - np.random.uniform(1, 3) for x in ats_cover]
over_under_upper = [x + np.random.uniform(1, 3) for x in over_under]
over_under_lower = [x - np.random.uniform(1, 3) for x in over_under]

# Build the Chart
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

# Update Layout
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    # Removed the title from the chart
    xaxis=dict(
        title='Line Movement Category',
        titlefont=dict(size=18, color='#F5F5F5'),
        tickfont=dict(size=12),
        showgrid=False,
    ),
    yaxis=dict(
        title='Percentage (%)',
        titlefont=dict(size=18, color='#F5F5F5'),
        tickfont=dict(size=12),
        showgrid=False,
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5,
        font=dict(size=12)
    ),
    margin=dict(l=20, r=20, t=20, b=20),
    hovermode='x unified'
)

# Display the Chart
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown('''
    <div class="footer">
        &copy; 2023 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
