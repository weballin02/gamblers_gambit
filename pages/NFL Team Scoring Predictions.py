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
    page_title="FoxEdge - NFL Team Scoring Predictions",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load and preprocess data
@st.cache_data
def fetch_and_preprocess_data():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])
    home_df = schedule[['gameday', 'home_team', 'home_score']].rename(columns={'home_team': 'team', 'home_score': 'score'})
    away_df = schedule[['gameday', 'away_team', 'away_score']].rename(columns={'away_team': 'team', 'away_score': 'score'})
    team_data = pd.concat([home_df, away_df], ignore_index=True)
    team_data['gameday'] = pd.to_datetime(team_data['gameday'], errors='coerce')
    team_data.dropna(subset=['score'], inplace=True)
    team_data.set_index('gameday', inplace=True)
    team_data.sort_index(inplace=True)
    return team_data

team_data = fetch_and_preprocess_data()

# Train or load ARIMA models
@st.cache_resource
def get_team_models(team_data):
    model_dir = 'models/nfl/'
    os.makedirs(model_dir, exist_ok=True)
    team_models = {}
    for team in team_data['team'].unique():
        model_path = os.path.join(model_dir, f'{team}_arima_model.pkl')
        team_scores = team_data[team_data['team'] == team]['score'].reset_index(drop=True)
        if os.path.exists(model_path):
            team_models[team] = joblib.load(model_path)
        elif len(team_scores) >= 10:
            model = auto_arima(team_scores, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
            model.fit(team_scores)
            joblib.dump(model, model_path)
            team_models[team] = model
    return team_models

team_models = get_team_models(team_data)

# Function to predict scores
def predict_team_score(team, periods=1):
    model = team_models.get(team)
    if model:
        forecast = model.predict(n_periods=periods)
        return forecast[0] if isinstance(forecast, (np.ndarray, list, pd.Series)) else None
    return None

# Compute forecasts
@st.cache_data
def compute_team_forecasts(_models, data):
    forecasts = {}
    for team, model in _models.items():
        scores = data[data['team'] == team]['score']
        if not scores.empty:
            forecast_dates = pd.date_range(start=scores.index[-1] + timedelta(days=7), periods=5, freq='7D')
            predictions = model.predict(n_periods=5)
            forecasts[team] = pd.DataFrame({'Date': forecast_dates, 'Predicted_Score': predictions, 'Team': team})
    return pd.concat(forecasts.values(), ignore_index=True) if forecasts else pd.DataFrame(columns=['Date', 'Predicted_Score', 'Team'])

# Interactive UI for team selection
st.sidebar.markdown("### Select a Team for Analysis")
team_selection = st.sidebar.selectbox("Choose a team:", sorted(team_data['team'].unique()))

if team_selection:
    team_scores = team_data[team_data['team'] == team_selection]['score']
    st.line_chart(team_scores)
    team_forecast = all_forecasts[all_forecasts['Team'] == team_selection]
    if not team_forecast.empty:
        st.dataframe(team_forecast[['Date', 'Predicted_Score']])
        fig, ax = plt.subplots()
        ax.plot(team_scores.index, team_scores, label="Historical Scores")
        ax.plot(team_forecast['Date'], team_forecast['Predicted_Score'], label="Forecasted Scores", linestyle='--')
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("No forecasts available for this team.")

# Fetch upcoming games
@st.cache_data(ttl=3600)
def fetch_upcoming_games():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])
    schedule['game_datetime'] = pd.to_datetime(
        schedule['gameday'].astype(str) + ' ' + schedule['gametime'].astype(str),
        errors='coerce',
        utc=True
    )
    schedule.dropna(subset=['game_datetime'], inplace=True)
    now = datetime.now(pytz.UTC)
    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') &
        (schedule['game_datetime'] > now)
    ][['game_id', 'game_datetime', 'home_team', 'away_team']]
    return upcoming_games

upcoming_games = fetch_upcoming_games()

# Predictions for upcoming games
st.markdown("### Upcoming Game Predictions")
if not upcoming_games.empty:
    for _, game in upcoming_games.iterrows():
        home_team = game['home_team']
        away_team = game['away_team']
        home_score = predict_team_score(home_team)
        away_score = predict_team_score(away_team)
        if home_score is not None and away_score is not None:
            st.markdown(f"**{home_team} vs {away_team}:**")
            st.markdown(f"- **{home_team} Predicted Score:** {home_score:.2f}")
            st.markdown(f"- **{away_team} Predicted Score:** {away_score:.2f}")
else:
    st.write("No upcoming games found.")
