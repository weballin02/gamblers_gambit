# NBA Enhanced Prediction Script with Optimized Performance

# =======================
# 1. Import Libraries
# =======================
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap

warnings.filterwarnings('ignore')

# =======================
# 2. Streamlit App Configuration
# =======================
st.set_page_config(
    page_title="NBA Advanced Betting Insights",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Session State for Theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Function to Toggle Theme
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Theme Toggle Button
st.sidebar.button("🌗 Toggle Theme", on_click=toggle_theme)

# Apply Theme Based on Dark Mode
if st.session_state.dark_mode:
    primary_bg = "#121212"
    primary_text = "#FFFFFF"
    secondary_bg = "#1E1E1E"
    accent_color = "#BB86FC"
    highlight_color = "#03DAC6"
    chart_template = "plotly_dark"
else:
    primary_bg = "#FFFFFF"
    primary_text = "#000000"
    secondary_bg = "#F5F5F5"
    accent_color = "#6200EE"
    highlight_color = "#03DAC6"
    chart_template = "plotly_white"

# Custom CSS for Novel Design
st.markdown(f"""
    <style>
    body {{
        background-color: {primary_bg};
        color: {primary_text};
        font-family: 'Roboto', sans-serif;
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .header {{
        background-color: {accent_color};
        padding: 3em;
        border-radius: 20px;
        text-align: center;
        color: #FFFFFF;
        margin-bottom: 2em;
    }}
    .header h1 {{ font-size: 3.5em; }}
    .header p {{ font-size: 1.5em; }}
    </style>
""", unsafe_allow_html=True)

# =======================
# 3. Header Section
# =======================
def render_header():
    st.markdown(f'''
        <div class="header">
            <h1>NBA Advanced Betting Insights</h1>
            <p>Predictive Analytics with Enhanced Features</p>
        </div>
    ''', unsafe_allow_html=True)

# =======================
# 4. Data Fetching and Preprocessing
# =======================
nba_team_list = nba_teams.get_teams()
team_name_mapping = {team['abbreviation']: team['full_name'] for team in nba_team_list}
id_to_abbrev = {team['id']: team['abbreviation'] for team in nba_team_list}

current_season = '2023-24'
previous_seasons = ['2022-23', '2021-22']
season_weights = {current_season: 1.0, '2022-23': 0.7, '2021-22': 0.5}

@st.cache_data
def load_and_preprocess_logs(seasons):
    """Load game logs and preprocess data."""
    all_games = []
    for season in seasons:
        try:
            game_logs = LeagueGameLog(
                season=season, 
                season_type_all_star='Regular Season', 
                player_or_team_abbreviation='T'
            ).get_data_frames()[0]
            game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'])
            game_logs['SEASON'] = season
            game_logs['WEIGHT'] = season_weights[season]
            all_games.append(game_logs)
        except Exception as e:
            st.error(f"Error loading game logs for {season}: {e}")
    if not all_games:
        return None
    combined_logs = pd.concat(all_games, ignore_index=True)
    combined_logs.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'], inplace=True)
    return combined_logs

@st.cache_data
def preprocess_team_data(game_logs):
    """Preprocess team-level data."""
    team_data = game_logs[['GAME_DATE', 'TEAM_ABBREVIATION', 'PTS', 'MATCHUP']].copy()
    team_data.rename(columns={'TEAM_ABBREVIATION': 'team', 'PTS': 'score'}, inplace=True)
    team_data['is_home'] = team_data['MATCHUP'].str.contains('vs.').astype(int)
    team_data['rest_days'] = team_data.groupby('team')['GAME_DATE'].diff().dt.days.fillna(7)
    team_data['game_number'] = team_data.groupby('team').cumcount() + 1
    team_data['score_rolling_mean'] = team_data.groupby('team')['score'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    return team_data

game_logs = load_and_preprocess_logs([current_season] + previous_seasons)
team_data = preprocess_team_data(game_logs)

# =======================
# 5. Model Training
# =======================
def train_team_models(team_data):
    models, team_stats = {}, {}
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }

    for team, team_df in team_data.groupby('team'):
        if len(team_df) < 15:
            continue

        features = team_df[['game_number', 'is_home', 'rest_days', 'score_rolling_mean', 'opponent_avg_score']]
        target = team_df['score']
        features.fillna(method='ffill', inplace=True)

        tscv = TimeSeriesSplit(n_splits=3)
        model_candidates = {
            'gbr': GradientBoostingRegressor(),
            'rf': RandomForestRegressor(),
            'xgb': XGBRegressor(eval_metric='rmse', use_label_encoder=False)
        }

        team_models = {}
        for model_name, model in model_candidates.items():
            search = RandomizedSearchCV(
                model, param_grid, cv=tscv, n_jobs=-1, n_iter=10, random_state=42
            )
            search.fit(features, target)
            team_models[model_name] = search.best_estimator_

        models[team] = {
            'models': team_models,
            'features': features.columns
        }
        team_stats[team] = {
            'avg_score': target.mean(),
            'std_dev': target.std(),
            'min_score': target.min(),
            'max_score': target.max()
        }

    return models, team_stats

models, team_stats = train_team_models(team_data)

# =======================
# 6. Prediction Logic
# =======================
def predict_team_score(team, models, team_stats, team_data):
    if team not in models or team not in team_stats:
        return None, None

    model_dict = models[team]['models']
    team_features = team_data[team_data['team'] == team].iloc[-1]
    next_features = pd.DataFrame({
        'game_number': [team_features['game_number'] + 1],
        'is_home': [team_features['is_home']],
        'rest_days': [7],
        'score_rolling_mean': [team_features['score_rolling_mean']],
        'opponent_avg_score': [team_features['opponent_avg_score']]
    })

    predictions = [model.predict(next_features)[0] for model in model_dict.values()]
    ensemble_prediction = np.mean(predictions)

    stats = team_stats[team]
    confidence_interval = (stats['avg_score'] - 1.96 * stats['std_dev'], 
                           stats['avg_score'] + 1.96 * stats['std_dev'])

    return ensemble_prediction, confidence_interval

# =======================
# 7. Fetch Upcoming Games
# =======================
@st.cache_data(ttl=3600)
def fetch_nba_games():
    today = datetime.now().date()
    next_day = today + timedelta(days=1)
    today_scoreboard = ScoreboardV2(game_date=today.strftime('%Y-%m-%d'))
    tomorrow_scoreboard = ScoreboardV2(game_date=next_day.strftime('%Y-%m-%d'))

    try:
        today_games = today_scoreboard.get_data_frames()[0]
        tomorrow_games = tomorrow_scoreboard.get_data_frames()[0]
        combined_games = pd.concat([today_games, tomorrow_games], ignore_index=True)
    except Exception as e:
        st.error(f"Error fetching games: {e}")
        return pd.DataFrame()

    combined_games['HOME_TEAM_ABBREV'] = combined_games['HOME_TEAM_ID'].map(id_to_abbrev)
    combined_games['VISITOR_TEAM_ABBREV'] = combined_games['VISITOR_TEAM_ID'].map(id_to_abbrev)
    combined_games.dropna(subset=['HOME_TEAM_ABBREV', 'VISITOR_TEAM_ABBREV'], inplace=True)

    combined_games['GAME_LABEL'] = combined_games.apply(
        lambda row: f"{team_name_mapping.get(row['VISITOR_TEAM_ABBREV'])} at {team_name_mapping.get(row['HOME_TEAM_ABBREV'])}",
        axis=1
    )
    return combined_games[['GAME_ID', 'HOME_TEAM_ABBREV', 'VISITOR_TEAM_ABBREV', 'GAME_LABEL']]

upcoming_games = fetch_nba_games()

# =======================
# 8. Main App Logic
# =======================
def main():
    render_header()

    st.markdown("### NBA Game Predictions")
    if not upcoming_games.empty:
        game_selection = st.selectbox('Select an upcoming game:', upcoming_games['GAME_LABEL'])
        selected_game = upcoming_games[upcoming_games['GAME_LABEL'] == game_selection].iloc[0]

        home_team = selected_game['HOME_TEAM_ABBREV']
        away_team = selected_game['VISITOR_TEAM_ABBREV']

        home_pred, home_ci = predict_team_score(home_team, models, team_stats, team_data)
        away_pred, away_ci = predict_team_score(away_team, models, team_stats, team_data)

        # Predictions
        st.write(f"**Home Team:** {home_team} | Predicted Score: {home_pred}")
        st.write(f"**Away Team:** {away_team} | Predicted Score: {away_pred}")
    else:
        st.warning("No upcoming games found.")

if __name__ == "__main__":
    main()
