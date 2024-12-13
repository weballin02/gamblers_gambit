import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from pmdarima import auto_arima

warnings.filterwarnings('ignore')

# =======================
# 1. Custom CSS Styling
# =======================
st.markdown('''
    <style>
        body, html {
            background: #2C3E50;
            color: #FFFFFF;
            font-family: 'Open Sans', sans-serif;
        }

        .hero {
            text-align: center;
            padding: 3em 1em;
            background: linear-gradient(120deg, #1E90FF, #FF8C00);
            color: #FFFFFF;
        }

        .data-section {
            padding: 2em 1em;
            background-color: #1E1E1E;
            border-radius: 10px;
            margin-bottom: 1.5em;
        }

        .prediction-card {
            background-color: #1E1E1E;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        .footer {
            text-align: center;
            padding: 2em 1em;
            color: #999999;
        }
    </style>
''', unsafe_allow_html=True)

# =======================
# 2. Data Loading
# =======================
nba_team_list = nba_teams.get_teams()
team_name_mapping = {team['abbreviation']: team['full_name'] for team in nba_team_list}
id_to_abbrev = {team['id']: team['abbreviation'] for team in nba_team_list}

@st.cache_data
def load_nba_game_logs(season='2023-24'):
    game_logs = LeagueGameLog(season=season, season_type_all_star='Regular Season', player_or_team_abbreviation='T')
    games = game_logs.get_data_frames()[0]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    return games

game_logs = load_nba_game_logs()

# =======================
# 3. Model Training
# =======================
def train_team_models(game_logs):
    models = {}
    for team in game_logs['TEAM_ABBREVIATION'].unique():
        team_df = game_logs[game_logs['TEAM_ABBREVIATION'] == team]
        if len(team_df) < 20:
            continue
        X = team_df[['GAME_DATE']].apply(lambda x: x.map(lambda d: d.timestamp()))
        y = team_df['PTS']
        models[team] = RandomForestRegressor().fit(X, y)
    return models

models = train_team_models(game_logs)

# =======================
# 4. ARIMA Models
# =======================
def train_arima_models(game_logs):
    arima_models = {}
    for team in game_logs['TEAM_ABBREVIATION'].unique():
        team_points = game_logs[game_logs['TEAM_ABBREVIATION'] == team]['PTS']
        if len(team_points) < 20:
            continue
        try:
            arima_model = auto_arima(team_points, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
            arima_models[team] = arima_model
        except Exception as e:
            st.error(f"Failed to train ARIMA for {team}: {str(e)}")
    return arima_models

arima_models = train_arima_models(game_logs)

# =======================
# 5. Prediction Function
# =======================
def predict_team_score(team, models, arima_models):
    try:
        arima_model = arima_models.get(team, None)
        arima_prediction = 0
        
        if arima_model:
            arima_result = arima_model.predict(n_periods=1)
            if arima_result is not None and len(arima_result) > 0:
                arima_prediction = arima_result[0]
    except Exception as e:
        st.error(f"ARIMA prediction failed for {team}: {str(e)}")
        arima_prediction = 0
    
    if arima_prediction == 0 and team in models:
        try:
            X_new = pd.DataFrame({'GAME_DATE': [datetime.now().timestamp()]})
            rf_prediction = models[team].predict(X_new)[0]
            arima_prediction = rf_prediction
        except Exception as e:
            st.error(f"RandomForest fallback failed for {team}: {str(e)}")
            arima_prediction = 0
    
    return round(arima_prediction, 1)

# =======================
# 6. Main App Logic
# =======================
st.markdown('''
    <div class="hero">
        <h1>NBA Prediction Center</h1>
        <p>Smart Predictions for Smarter Bets</p>
    </div>
''', unsafe_allow_html=True)

upcoming_games = ScoreboardV2(game_date=datetime.now().strftime('%Y-%m-%d')).get_data_frames()[0]
upcoming_games['game_label'] = upcoming_games.apply(lambda x: f"{team_name_mapping[id_to_abbrev[x['VISITOR_TEAM_ID']]]} @ {team_name_mapping[id_to_abbrev[x['HOME_TEAM_ID']]]}", axis=1)

if not upcoming_games.empty:
    selected_game = st.selectbox('Select a game to display predictions:', upcoming_games['game_label'])

    if selected_game:
        selected_row = upcoming_games[upcoming_games['game_label'] == selected_game].iloc[0]
        home_team = id_to_abbrev[selected_row['HOME_TEAM_ID']]
        away_team = id_to_abbrev[selected_row['VISITOR_TEAM_ID']]

        home_score = predict_team_score(home_team, models, arima_models)
        away_score = predict_team_score(away_team, models, arima_models)

        st.markdown(f'''
            <div class="data-section">
                <h2>🏠 {team_name_mapping[home_team]} vs 🚗 {team_name_mapping[away_team]}</h2>
                <div class="prediction-card">
                    <p><strong>{team_name_mapping[home_team]}:</strong> {home_score}</p>
                    <p><strong>{team_name_mapping[away_team]}:</strong> {away_score}</p>
                </div>
            </div>
        ''', unsafe_allow_html=True)
else:
    st.warning("No upcoming games available.")
