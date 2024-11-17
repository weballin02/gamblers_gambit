# Import Libraries
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import streamlit as st
from pmdarima import auto_arima
import joblib
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="FoxEdge - NFL Scoring Predictions",
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
            padding: 4em 1em;
            overflow: hidden;
        }

        /* Buttons */
        .button {
            background: linear-gradient(45deg, var(--accent-color-teal), var(--accent-color-purple));
            border: none;
            padding: 0.8em 2em;
            color: #FFFFFF;
            font-size: 1.1em;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.3s ease;
            text-decoration: none;
            display: inline-block;
            margin-top: 1em;
        }

        .button:hover {
            transform: translateY(-5px);
        }

        /* Data Section */
        .data-section {
            padding: 2em 1em;
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

        /* Enhanced Summary Styling */
        .summary-section {
            padding: 2em 1em;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            margin-bottom: 2em;
        }

        .summary-section h3 {
            font-size: 2em;
            margin-bottom: 0.5em;
            color: var(--accent-color-teal);
        }

        .summary-section p {
            font-size: 1.1em;
            color: #E0E0E0;
            line-height: 1.6;
        }

        /* Team Trends Styling Update */
        .team-trends {
            display: flex;
            flex-wrap: wrap;
            gap: 2em;
            justify-content: space-around;  /* Aligns cards neatly side-by-side */
            margin-top: 2em;
        }

        .team-card {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1.5em;
            width: calc(33% - 2em);  /* Each card takes up approximately 1/3rd of the row, with gaps */
            min-width: 300px;         /* Ensure cards maintain a minimum width */
            max-width: 400px;         /* Optionally limit the maximum width */
            text-align: center;
        }

        /* Streamlit Elements */
        .stButton > button {
            background: linear-gradient(45deg, var(--accent-color-teal), var(--accent-color-purple));
            border: none;
            padding: 0.8em 2em;
            color: #FFFFFF;
            font-size: 1.1em;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.3s ease;
            margin-top: 1em;
        }

        .stButton > button:hover {
            transform: translateY(-5px);
        }

        .stCheckbox > div {
            padding: 0.5em 0;
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
            .team-trends {
                flex-direction: column;
                align-items: center;
            }

            .team-card {
                width: 90%;
            }
        }
    </style>
''', unsafe_allow_html=True)

# Streamlit App
st.markdown('''
    <div class="hero">
        <h1>FoxEdge - NFL Scoring Predictions</h1>
        <p>View projected NFL team scores for upcoming games based on recent stats.</p>
    </div>
''', unsafe_allow_html=True)

st.markdown('''
    <div class="data-section">
        <h2>Select Team to View Scoring Predictions</h2>
        <p>Choose a team to see historical scores and predictions for the upcoming games.</p>
    </div>
''', unsafe_allow_html=True)

# Load Data
file_path = 'data/nfl_data.csv'  # Update this path
team_data = load_and_preprocess_data(file_path)

# Dropdown menu for selecting a team
teams_list = sorted(team_data['team'].unique())
team = st.selectbox('Select a team for prediction:', teams_list)

if team:
    team_scores = team_data[team_data['team'] == team]['score']
    team_scores.index = pd.to_datetime(team_scores.index)

    st.markdown(f'''
        <div class="data-section">
            <h2>Historical Scores for {team}</h2>
        </div>
    ''', unsafe_allow_html=True)
    st.line_chart(team_scores)

    # Display future predictions
    team_forecast = all_forecasts[all_forecasts['Team'] == team]

    st.markdown(f'''
        <div class="data-section">
            <h2>Predicted Scores for Next 5 Games ({team})</h2>
        </div>
    ''', unsafe_allow_html=True)
    st.write(team_forecast[['Date', 'Predicted_Score']])

    # Plot the historical and predicted scores
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert dates to Matplotlib's numeric format
    forecast_dates = mdates.date2num(team_forecast['Date'])
    historical_dates = mdates.date2num(team_scores.index)

    # Plot historical scores
    ax.plot(
        historical_dates,
        team_scores.values,
        label=f'Historical Scores for {team}',
        color='blue'
    )
    # Plot predicted scores
    ax.plot(
        forecast_dates,
        team_forecast['Predicted_Score'],
        label='Predicted Scores',
        color='red'
    )
    # Plot confidence interval (using +/- 5 as placeholder)
    lower_bound = team_forecast['Predicted_Score'] - 5
    upper_bound = team_forecast['Predicted_Score'] + 5

    # Ensure no non-finite values
    finite_indices = np.isfinite(forecast_dates) & np.isfinite(lower_bound) & np.isfinite(upper_bound)

    ax.fill_between(
        forecast_dates[finite_indices],
        lower_bound.values[finite_indices],
        upper_bound.values[finite_indices],
        color='gray',
        alpha=0.2,
        label='Confidence Interval'
    )

    ax.xaxis_date()
    fig.autofmt_xdate()

    ax.set_title(f'Score Prediction for {team}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# NFL Game Predictions Section
st.write('---')
st.markdown('''
    <div class="data-section">
        <h2>NFL Game Predictions for Upcoming Games</h2>
        <p>Select an upcoming game to view predicted scores and the likely winner.</p>
    </div>
''', unsafe_allow_html=True)

# Fetch upcoming games
upcoming_games = fetch_upcoming_games()

# Create game labels
upcoming_games['game_label'] = [
    f"{row['away_team_full']} at {row['home_team_full']} ({row['game_datetime'].strftime('%Y-%m-%d %H:%M %Z')})"
    for _, row in upcoming_games.iterrows()
]

# Let the user select a game
if not upcoming_games.empty:
    game_selection = st.selectbox('Select an upcoming game:', upcoming_games['game_label'])
    selected_game = upcoming_games[upcoming_games['game_label'] == game_selection].iloc[0]

    home_team = selected_game['home_team_full']
    away_team = selected_game['away_team_full']

    # Predict scores
    home_team_score = predict_team_score(home_team)
    away_team_score = predict_team_score(away_team)

    if home_team_score is not None and away_team_score is not None:
        st.markdown(f'''
            <div class="summary-section">
                <h3>Predicted Scores</h3>
                <p><strong>{home_team}: {home_team_score:.2f}</strong></p>
                <p><strong>{away_team}: {away_team_score:.2f}</strong></p>
            </div>
        ''', unsafe_allow_html=True)

        if home_team_score > away_team_score:
            st.success(f"**Predicted Winner:** {home_team}")
        elif away_team_score > home_team_score:
            st.success(f"**Predicted Winner:** {away_team}")
        else:
            st.info("**Predicted Outcome:** Tie")
    else:
        st.error("Prediction models for one or both teams are not available.")

# Footer
st.markdown('''
    <div class="footer">
        &copy; 2023 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
