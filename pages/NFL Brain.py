# Unified NFL Prediction System with All Features
# Import Libraries
import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import requests
import pytz
import plotly.express as px
import joblib

# Set Page Configuration
st.set_page_config(
    page_title="🏈 FoxEdge - NFL Predictions",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Sidebar Injury Toggle
include_injuries = st.checkbox("Include Injury Impact")


# Utility Functions
@st.cache_data(ttl=3600)
def load_nfl_data():
    current_year = datetime.now().year
    previous_years = [current_year - 1, current_year - 2]
    schedule = nfl.import_schedules([current_year] + previous_years)
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    home_df = schedule[['gameday', 'home_team', 'home_score']].rename(columns={'home_team': 'team', 'home_score': 'score'})
    away_df = schedule[['gameday', 'away_team', 'away_score']].rename(columns={'away_team': 'team', 'away_score': 'score'})
    team_data = pd.concat([home_df, away_df]).dropna(subset=['score'])
    team_data.set_index('gameday', inplace=True)
    return team_data

@st.cache_resource
def train_ml_models(team_data):
    models = {}
    scaler = StandardScaler()
    for team in team_data['team'].unique():
        team_scores = team_data[team_data['team'] == team]['score']
        if len(team_scores) < 5:
            continue
        X = np.arange(len(team_scores)).reshape(-1, 1)
        y = team_scores.values
        X_scaled = scaler.fit_transform(X)
        model = GradientBoostingRegressor().fit(X_scaled, y)
        models[team] = model
    return models

@st.cache_resource
def train_arima_models(team_data):
    arima_models = {}
    for team in team_data['team'].unique():
        team_scores = team_data[team_data['team'] == team]['score']
        if len(team_scores) < 5:
            continue
        model = auto_arima(team_scores, seasonal=False, trace=False, suppress_warnings=True)
        arima_models[team] = model
    return arima_models

@st.cache_data(ttl=3600)
def fetch_upcoming_games():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])
    schedule['game_datetime'] = pd.to_datetime(schedule['gameday'].astype(str) + ' ' + schedule['gametime'].astype(str), errors='coerce', utc=True)
    now = datetime.now(pytz.UTC)
    weekday = now.weekday()
    target_days = {3: [3, 6, 0], 6: [6, 0, 3], 0: [0, 3, 6]}.get(weekday, [3, 6, 0])
    upcoming_game_dates = [now + timedelta(days=(d - weekday + 7) % 7) for d in target_days]
    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') &
        (schedule['game_datetime'].dt.date.isin([date.date() for date in upcoming_game_dates]))
    ].sort_values('game_datetime')
    return upcoming_games[['game_datetime', 'home_team', 'away_team']]

# Fetch Injury Data
@st.cache_data(ttl=3600)
def fetch_injury_data():
    current_year = datetime.now().year
    injuries = nfl.import_injuries([current_year])
    key_positions = ['QB', 'RB', 'WR', 'OL']
    relevant_injuries = injuries[injuries['position'].isin(key_positions) & (injuries['report_status'] == 'Out')]
    return relevant_injuries

# Adjust Scores Based on Injuries
def adjust_scores_for_injuries(team, base_scores, injury_data):
    """
    Adjust scores based on injury data for the given team.
    
    Args:
        team (str): The team abbreviation (e.g., "NE").
        base_scores (pd.Series): Base predicted scores for the team.
        injury_data (pd.DataFrame): Injury data fetched from nfl_data_py.
        
    Returns:
        pd.Series: Adjusted scores accounting for injuries.
    """
    team_injuries = injury_data[injury_data['team'] == team]
    injury_impact = {
        'QB': 0.15,  # Quarterback has the highest impact
        'RB': 0.07,  # Running back
        'WR': 0.08,  # Wide receiver
        'OL': 0.05   # Offensive lineman
    }
    
    # Calculate total impact for the team
    total_impact = team_injuries['position'].map(injury_impact).sum()
    adjusted_scores = base_scores * (1 - total_impact)
    return adjusted_scores

def fetch_weather_data(location, game_datetime):
    API_KEY = "YOUR_API_KEY"  # Replace with a valid API key
    date_str = game_datetime.strftime('%Y-%m-%d')
    url = f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={location}&dt={date_str}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        forecast = data.get('forecast', {}).get('forecastday', [{}])[0]
        return {
            'temperature': forecast.get('day', {}).get('avgtemp_f', 70),
            'humidity': forecast.get('day', {}).get('avghumidity', 50),
            'precipitation': forecast.get('day', {}).get('totalprecip_in', 0),
            'conditions': forecast.get('day', {}).get('condition', {}).get('text', 'Clear')
        }
    except Exception:
        return {'temperature': 70, 'humidity': 50, 'precipitation': 0, 'conditions': 'Clear'}
def monte_carlo_simulation(home_team, away_team, num_simulations=1000, include_injuries=False):
    home_scores = pd.Series(predict_scores(home_team, periods=num_simulations)).reset_index(drop=True)
    away_scores = pd.Series(predict_scores(away_team, periods=num_simulations)).reset_index(drop=True)

    if include_injuries:
        injury_data = fetch_injury_data()
        home_scores = adjust_scores_for_injuries(home_team, home_scores, injury_data)
        away_scores = adjust_scores_for_injuries(away_team, away_scores, injury_data)

    if home_scores is None or away_scores is None:
        return None

    # Introduce variability with normal distribution
    home_variation = np.random.normal(0, home_scores.std(), num_simulations)
    away_variation = np.random.normal(0, away_scores.std(), num_simulations)

    adjusted_home_scores = home_scores + home_variation
    adjusted_away_scores = away_scores + away_variation

    # Perform simulations
    results = {
        "Home Win %": (adjusted_home_scores - adjusted_away_scores) * 100,
        "Away Win %": (adjusted_away_scores - adjusted_home_scores) * 100,
        "Average Total": (adjusted_home_scores + adjusted_away_scores).mean(),
        "Average Differential": (adjusted_home_scores - adjusted_away_scores).mean()
    }
    return results


def predict_scores(team, periods=1):
    ml_model = ml_models.get(team)
    arima_model = arima_models.get(team)
    if not ml_model or not arima_model:
        return None
    X_future = np.arange(len(team_data[team_data['team'] == team]['score']), len(team_data[team_data['team'] == team]['score']) + periods).reshape(-1, 1)
    ml_predictions = ml_model.predict(X_future)
    arima_predictions = arima_model.predict(n_periods=periods)
    combined_predictions = (ml_predictions + arima_predictions) / 2
    return combined_predictions

# Load Data and Train Models
team_data = load_nfl_data()
ml_models = train_ml_models(team_data)
arima_models = train_arima_models(team_data)
upcoming_games = fetch_upcoming_games()

# User Interface
st.title("🏈 FoxEdge - NFL Predictions")
st.sidebar.markdown("## Settings")


if not upcoming_games.empty:
    upcoming_games['game_label'] = [
        f"{row['away_team']} at {row['home_team']} ({row['game_datetime'].strftime('%Y-%m-%d %H:%M %Z')})"
        for _, row in upcoming_games.iterrows()
    ]
    game = st.selectbox('Select an upcoming game:', upcoming_games['game_label'])
    selected_game = upcoming_games[upcoming_games['game_label'] == game].iloc[0]
    home_team, away_team = selected_game['home_team'], selected_game['away_team']

    include_injuries = st.checkbox("Include Injury Adjustments")
    include_weather = st.checkbox("Include Weather Adjustments")
    num_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000, step=100)

    with st.spinner("Running Monte Carlo simulations..."):
        results = monte_carlo_simulation(home_team, away_team, num_simulations)

# Enhanced Output for Bettors
if results:
    # Section 1: Key Predictions
    st.markdown("## 🏈 Key Game Predictions")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Total Points", f"{results['Average Total']:.2f}", help="Predicted total points scored in the game.")
    col2.metric("Average Score Differential", f"{results['Average Differential']:.2f}", help="Predicted margin of victory (positive: home team, negative: away team).")
    
    # Predicted Winner with Confidence
    predicted_winner = selected_game['home_team'] if results['Average Differential'] > 0 else selected_game['away_team']
    st.success(f"**Predicted Winner**: {predicted_winner}")

    # Section 2: Score Differential Distribution
    st.markdown("## 📊 Score Differential Distribution")
    home_scores = predict_scores(home_team, periods=num_simulations)
    away_scores = predict_scores(away_team, periods=num_simulations)
    score_differentials = home_scores - away_scores

    # Histogram
    fig = px.histogram(
        score_differentials,
        nbins=30,
        title="Score Differential Distribution (Home Team - Away Team)",
        labels={'value': 'Score Differential', 'count': 'Frequency'},
        color_discrete_sequence=['#1E90FF']
    )
    fig.update_layout(
        xaxis_title="Score Differential",
        yaxis_title="Frequency",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Section 3: Suggested Bets
    st.markdown("## 💡 Suggested Bets")
    spread_line = st.slider("Set Betting Line (Spread)", min_value=-20.0, max_value=20.0, step=0.5, value=0.0, help="Adjust the spread line to evaluate betting scenarios.")
    total_line = st.slider("Set Total Line", min_value=20.0, max_value=100.0, step=0.5, value=results['Average Total'], help="Adjust the total points line for over/under evaluation.")

    # Betting Scenarios
    home_covers = (score_differentials > spread_line).mean() * 100
    away_covers = (score_differentials < spread_line).mean() * 100
    over_hits = (home_scores + away_scores > total_line).mean() * 100
    under_hits = (home_scores + away_scores < total_line).mean() * 100

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Spread Bet Analysis")
        st.metric(selected_game['home_team'], f"{home_covers:.2f}%", help="Probability that the home team covers the spread.")
        st.metric(selected_game['away_team'], f"{away_covers:.2f}%", help="Probability that the away team covers the spread.")
    with col2:
        st.markdown("### Total Bet Analysis")
        st.metric("Over Hits %", f"{over_hits:.2f}%", help="Probability that the total points exceed the set line.")
        st.metric("Under Hits %", f"{under_hits:.2f}%", help="Probability that the total points are below the set line.")
# Display Advanced Insights for Injuries
st.markdown("### 🔍 Advanced Insights")

if include_injuries:
    injury_data = fetch_injury_data()

    st.markdown("#### Injury Adjustments")
    
    # Check available columns and adjust dynamically
    available_columns = injury_data.columns.tolist()
    columns_to_display = [col for col in ['position', 'report_status', 'full_name', 'player'] if col in available_columns]
    
    # Display Home Team Injuries
    st.markdown(f"**{home_team} Key Injuries**:")
    home_injuries = injury_data[injury_data['team'] == home_team]
    if not home_injuries.empty:
        st.dataframe(home_injuries[columns_to_display], use_container_width=True)
    else:
        st.write("No significant injuries for the home team.")

    # Display Away Team Injuries
    st.markdown(f"**{away_team} Key Injuries**:")
    away_injuries = injury_data[injury_data['team'] == away_team]
    if not away_injuries.empty:
        st.dataframe(away_injuries[columns_to_display], use_container_width=True)
    else:
        st.write("No significant injuries for the away team.")
else:
    st.write("Injury adjustments are not included. Enable them in the sidebar.")



