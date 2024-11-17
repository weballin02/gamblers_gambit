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

# Team Abbreviation to Full Name Mapping
team_abbrev_mapping = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs',
    'LAC': 'Los Angeles Chargers',
    'LAR': 'Los Angeles Rams',
    'LV': 'Las Vegas Raiders',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots',
    'NO': 'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SEA': 'Seattle Seahawks',
    'SF': 'San Francisco 49ers',
    'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders',
}

# Invert the mapping for reverse lookup
full_name_to_abbrev = {v: k for k, v in team_abbrev_mapping.items()}

# Load and Preprocess Data Using nfl_data_py
@st.cache_data
def load_and_preprocess_data():
    current_year = datetime.now().year
    previous_years = [current_year - 1, current_year - 2]
    
    # Importing schedules for the current and previous seasons
    schedule = nfl.import_schedules([current_year] + previous_years)
    
    # Converting dates to datetime and splitting data for home and away teams
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    
    # Prepare home and away data
    home_df = schedule[['gameday', 'home_team', 'home_score']].copy()
    home_df.rename(columns={'home_team': 'team', 'home_score': 'score'}, inplace=True)

    away_df = schedule[['gameday', 'away_team', 'away_score']].copy()
    away_df.rename(columns={'away_team': 'team', 'away_score': 'score'}, inplace=True)

    # Combine both DataFrames
    team_data = pd.concat([home_df, away_df], ignore_index=True)
    team_data.dropna(subset=['score'], inplace=True)
    team_data['score'] = pd.to_numeric(team_data['score'], errors='coerce')
    team_data.set_index('gameday', inplace=True)
    team_data.sort_index(inplace=True)

    return team_data

# Load Data
team_data = load_and_preprocess_data()

# Get list of teams
teams_list = team_data['team'].unique()

# Train or Load Models
@st.cache_resource
def get_team_models(team_data):
    model_dir = 'models/nfl/'
    os.makedirs(model_dir, exist_ok=True)

    team_models = {}
    teams_list = team_data['team'].unique()

    for team in teams_list:
        model_path = os.path.join(model_dir, f'{team}_arima_model.pkl')

        team_scores = team_data[team_data['team'] == team]['score']
        team_scores.reset_index(drop=True, inplace=True)

        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                st.write(f"Model loaded successfully for team: {team}")
            except Exception as e:
                st.write(f"Error loading model for team {team}: {e}")
                continue
        else:
            if len(team_scores) < 5:
                st.write(f"Not enough data points to train a model for team: {team}. (Data points: {len(team_scores)})")
                continue

            try:
                model = auto_arima(
                    team_scores,
                    seasonal=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True
                )
                model.fit(team_scores)
                joblib.dump(model, model_path)
                st.write(f"Model trained and saved successfully for team: {team}")
            except Exception as e:
                st.write(f"Error training model for team {team}: {e}")
                continue

        team_models[team] = model

    if not team_models:
        st.write("No models were successfully created. Please check data availability and model configurations.")

    return team_models

# Get Team Models
team_models = get_team_models(team_data)

# Function to Predict Team Score
def predict_team_score(team_abbrev, periods=1):
    model = team_models.get(team_abbrev)
    if model:
        forecast = model.predict(n_periods=periods)
        # Ensure forecast is a numpy array
        if isinstance(forecast, pd.Series):
            forecast = forecast.values
        return forecast[0]
    else:
        st.write(f"Prediction model not found for team: {team_abbrev}")
        return None

# Streamlit Interface for Selecting a Team and Viewing Predictions
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

    # Predict future scores
    if team in team_models:
        forecast = team_models[team].predict(n_periods=5)
        future_dates = pd.date_range(start=team_scores.index[-1] + pd.Timedelta(days=7), periods=5, freq='7D')
        prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted_Score': forecast})
        st.write(f"### Predicted Scores for {team} (Next 5 Games)")
        st.write(prediction_df)

# Fetch Upcoming Games and Predict Scores
st.write('---')
st.markdown('''
    <div class="data-section">
        <h2>NFL Game Predictions for Upcoming Games</h2>
        <p>Select an upcoming game to view predicted scores and the likely winner.</p>
    </div>
''', unsafe_allow_html=True)

# Fetch upcoming games
@st.cache_data(ttl=3600)
def fetch_upcoming_games():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])

    # Combine 'gameday' and 'gametime' to create 'game_datetime'
    schedule['game_datetime'] = pd.to_datetime(
        schedule['gameday'].astype(str) + ' ' + schedule['gametime'].astype(str),
        errors='coerce',
        utc=True
    )

    # Drop rows where 'game_datetime' could not be parsed
    schedule.dropna(subset=['game_datetime'], inplace=True)

    # Get current time in UTC
    now = datetime.now(pytz.UTC)

    # Filter for upcoming regular-season games
    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') &
        (schedule['game_datetime'] >= now)
    ]

    # Select necessary columns
    upcoming_games = upcoming_games[['game_id', 'game_datetime', 'home_team', 'away_team']]

    # Apply mapping
    upcoming_games['home_team_full'] = upcoming_games['home_team'].map(team_abbrev_mapping)
    upcoming_games['away_team_full'] = upcoming_games['away_team'].map(team_abbrev_mapping)

    # Remove games where team names couldn't be mapped
    upcoming_games.dropna(subset=['home_team_full', 'away_team_full'], inplace=True)

    # Reset index
    upcoming_games.reset_index(drop=True, inplace=True)

    return upcoming_games

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

    # Convert full team names to abbreviations to use the saved models
    home_team_abbrev = full_name_to_abbrev.get(home_team)
    away_team_abbrev = full_name_to_abbrev.get(away_team)

    if home_team_abbrev and away_team_abbrev:
        # Predict scores
        home_team_score = predict_team_score(home_team_abbrev)
        away_team_score = predict_team_score(away_team_abbrev)

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
    else:
        st.error("Could not find abbreviation for one or both teams.")

# Footer
st.markdown('''
    <div class="footer">
        &copy; 2023 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
