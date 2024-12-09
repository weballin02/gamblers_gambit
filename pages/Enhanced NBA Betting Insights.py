# NBA Enhanced Prediction Script with Advanced Features and Models

# =======================
# 1. Import Libraries
# =======================
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from nba_api.stats.endpoints import (
    LeagueGameLog,
    ScoreboardV2,
    teamgamelog,
)
from nba_api.stats.static import teams as nba_teams
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
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
    /* Global Styles */
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
        position: relative;
        overflow: hidden;
    }}
    .header::before {{
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
    .header h1 {{
        font-size: 3.5em;
        margin: 0;
        font-weight: bold;
        letter-spacing: -1px;
    }}
    .header p {{
        font-size: 1.5em;
        margin-top: 0.5em;
    }}
    .prediction-card {{
        background-color: {accent_color};
        padding: 2em;
        border-radius: 15px;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2em;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .prediction-card:hover {{
        transform: translateY(-10px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }}
    .prediction-card h2 {{
        font-size: 2em;
        margin-bottom: 0.5em;
    }}
    .prediction-card p {{
        font-size: 1.2em;
        margin-bottom: 0.5em;
    }}
    .data-section {{
        padding: 2em 1em;
        background-color: {accent_color};
        border-radius: 15px;
        margin-bottom: 2em;
    }}
    .data-section h2 {{
        font-size: 2em;
        margin-bottom: 1em;
    }}
    .footer {{
        text-align: center;
        padding: 2em 1em;
        color: {primary_text};
        opacity: 0.6;
        font-size: 0.9em;
    }}
    .footer a {{
        color: {highlight_color};
        text-decoration: none;
    }}
    @media (max-width: 768px) {{
        .header h1 {{
            font-size: 2.5em;
        }}
        .header p {{
            font-size: 1em;
        }}
    }}
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
def load_nba_game_logs(seasons):
    """Fetch and preprocess game logs for the specified NBA seasons."""
    all_games = []
    for season in seasons:
        try:
            game_logs = LeagueGameLog(season=season, season_type_all_star='Regular Season', player_or_team_abbreviation='T')
            games = game_logs.get_data_frames()[0]
            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
            games['SEASON'] = season
            games['WEIGHT'] = season_weights[season]
            all_games.append(games)
        except Exception as e:
            st.error(f"Error loading NBA game logs for {season}: {str(e)}")
    return pd.concat(all_games, ignore_index=True) if all_games else None

def preprocess_data(game_logs):
    # Prepare team data
    team_data = game_logs[['GAME_DATE', 'TEAM_ABBREVIATION', 'PTS', 'MATCHUP']]
    team_data = team_data.rename(columns={'TEAM_ABBREVIATION': 'team', 'PTS': 'score'})
    team_data['GAME_DATE'] = pd.to_datetime(team_data['GAME_DATE'])
    team_data.sort_values(['team', 'GAME_DATE'], inplace=True)

    team_data['is_home'] = team_data['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    team_data['rest_days'] = team_data.groupby('team')['GAME_DATE'].diff().dt.days.fillna(7)
    team_data['game_number'] = team_data.groupby('team').cumcount() + 1

    team_data['score_rolling_mean'] = team_data.groupby('team')['score'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

    team_data['opponent'] = team_data['MATCHUP'].apply(
        lambda x: x.split('vs. ')[1] if 'vs.' in x else x.split('@ ')[1]
    )

    opponent_avg_score = team_data.groupby('team')['score'].mean().reset_index()
    opponent_avg_score.columns = ['opponent', 'opponent_avg_score']
    team_data = team_data.merge(opponent_avg_score, on='opponent', how='left')

    decay = 0.9
    team_data['decay_weight'] = team_data.groupby('team').cumcount().apply(lambda x: decay ** x)
    team_data.dropna(inplace=True)
    return team_data

game_logs = load_nba_game_logs([current_season] + previous_seasons)
team_data = preprocess_data(game_logs)

# =======================
# 5. Model Training
# =======================
@st.cache_data
def train_team_models(team_data):
    models = {}
    team_stats = {}
    for team in team_data['team'].unique():
        team_df = team_data[team_data['team'] == team]
        if len(team_df) < 15:
            continue  # Skip teams with insufficient data

        features = team_df[['game_number', 'is_home', 'rest_days', 'score_rolling_mean', 'opponent_avg_score']]
        target = team_df['score']
        features.fillna(method='ffill', inplace=True)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }

        tscv = TimeSeriesSplit(n_splits=3)

        gbr = GradientBoostingRegressor()
        rf = RandomForestRegressor()
        xgb = XGBRegressor(eval_metric='rmse', use_label_encoder=False)

        gbr_grid = GridSearchCV(gbr, param_grid, cv=tscv)
        gbr_grid.fit(features, target)

        rf_grid = GridSearchCV(rf, {'n_estimators': [100, 200], 'max_depth': [5, 10]}, cv=tscv)
        rf_grid.fit(features, target)

        xgb_grid = GridSearchCV(xgb, param_grid, cv=tscv)
        xgb_grid.fit(features, target)

        models[team] = {
            'gbr': gbr_grid.best_estimator_,
            'rf': rf_grid.best_estimator_,
            'xgb': xgb_grid.best_estimator_,
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
    ensemble_prediction = None
    confidence_interval = None

    if team in models:
        team_df = team_data[team_data['team'] == team].iloc[-1]
        next_features = pd.DataFrame({
            'game_number': [team_df['game_number'] + 1],
            'is_home': [team_df['is_home']],
            'rest_days': [7],
            'score_rolling_mean': [team_df['score_rolling_mean']],
            'opponent_avg_score': [team_df['opponent_avg_score']]
        })

        model_dict = models[team]
        predictions = []
        for model_name in ['gbr', 'rf', 'xgb']:
            model = model_dict[model_name]
            pred = model.predict(next_features[model_dict['features']])[0]
            predictions.append(pred)

        ensemble_prediction = np.mean(predictions)

    if team in team_stats:
        avg = team_stats[team]['avg_score']
        std = team_stats[team]['std_dev']
        confidence_interval = (avg - 1.96 * std, avg + 1.96 * std)

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
    st.markdown('<div id="prediction"></div>', unsafe_allow_html=True)

    st.markdown(f'''
        <div class="data-section">
            <h2>NBA Game Predictions with Enhanced Models</h2>
        </div>
    ''', unsafe_allow_html=True)

    if not upcoming_games.empty:
        game_selection = st.selectbox('Select an upcoming game:', upcoming_games['GAME_LABEL'])
        selected_game = upcoming_games[upcoming_games['GAME_LABEL'] == game_selection].iloc[0]

        home_team = selected_game['HOME_TEAM_ABBREV']
        away_team = selected_game['VISITOR_TEAM_ABBREV']

        home_pred, home_ci = predict_team_score(home_team, models, team_stats, team_data)
        away_pred, away_ci = predict_team_score(away_team, models, team_stats, team_data)

        st.markdown("### 📊 Predictions")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"#### 🏠 **{team_name_mapping.get(home_team, home_team)}**")
            if home_pred is not None and home_ci is not None:
                st.markdown(f"**Ensemble Prediction:** {home_pred:.2f}")
                st.markdown(f"**95% Confidence Interval:** {home_ci[0]:.2f} - {home_ci[1]:.2f}")
            else:
                st.warning("Insufficient data for predictions.")

        with col2:
            st.markdown(f"#### 🚗 **{team_name_mapping.get(away_team, away_team)}**")
            if away_pred is not None and away_ci is not None:
                st.markdown(f"**Ensemble Prediction:** {away_pred:.2f}")
                st.markdown(f"**95% Confidence Interval:** {away_ci[0]:.2f} - {away_ci[1]:.2f}")
            else:
                st.warning("Insufficient data for predictions.")

        st.markdown("### 💡 Betting Insights")
        with st.expander("View Betting Suggestions"):
            if home_pred is not None and away_pred is not None:
                suggested_winner = home_team if home_pred > away_pred else away_team
                margin_of_victory = abs(home_pred - away_pred)
                total_points = home_pred + away_pred
                over_under_threshold = team_data['score'].mean() * 2  
                over_under = "Over" if total_points > over_under_threshold else "Under"

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**🏆 Suggested Winner:** **{team_name_mapping.get(suggested_winner, suggested_winner)}**")

                with col2:
                    st.markdown(f"**📈 Margin of Victory:** {margin_of_victory:.2f} pts")

                with col3:
                    st.markdown(f"**🔢 Total Points:** {total_points:.2f}")

                st.markdown(f"**💰 Over/Under Suggestion:** **Take the {over_under} ({over_under_threshold:.2f})**")
            else:
                st.warning("Insufficient data to provide betting insights for this game.")

        st.markdown("### 📊 Predicted Scores Comparison")
        scores_df = pd.DataFrame({
            "Team": [team_name_mapping.get(home_team, home_team), team_name_mapping.get(away_team, away_team)],
            "Predicted Score": [
                home_pred if home_pred is not None else 0,
                away_pred if away_pred is not None else 0
            ]
        })

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.set_style("whitegrid")
        sns.barplot(x="Team", y="Predicted Score", data=scores_df, palette="viridis", ax=ax)
        ax.set_title("Predicted Scores")
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(scores_df["Predicted Score"].max(), 150) + 10 if scores_df["Predicted Score"].max() > 0 else 150)

        for index, row in scores_df.iterrows():
            ax.text(index, row["Predicted Score"] + 0.5, f'{row["Predicted Score"]:.2f}', color='black', ha="center")

        st.pyplot(fig)

        st.markdown("### 🔍 Feature Importance")
        with st.expander("View Feature Importance for Home Team"):
            if home_team in models:
                model_dict = models[home_team]
                model = model_dict['xgb']  
                explainer = shap.TreeExplainer(model)
                team_df = team_data[team_data['team'] == home_team]
                team_features = team_df[model_dict['features']].iloc[-1:]
                shap_values = explainer.shap_values(team_features)
                shap.initjs()
                plt.title(f"Feature Importance for {team_name_mapping.get(home_team, home_team)}")
                shap.summary_plot(shap_values, team_features, plot_type="bar")
                st.pyplot(bbox_inches='tight')
            else:
                st.warning("Insufficient data for feature importance visualization.")

        with st.expander("View Feature Importance for Away Team"):
            if away_team in models:
                model_dict = models[away_team]
                model = model_dict['xgb']
                explainer = shap.TreeExplainer(model)
                team_df = team_data[team_data['team'] == away_team]
                team_features = team_df[model_dict['features']].iloc[-1:]
                shap_values = explainer.shap_values(team_features)
                shap.initjs()
                plt.title(f"Feature Importance for {team_name_mapping.get(away_team, away_team)}")
                shap.summary_plot(shap_values, team_features, plot_type="bar")
                st.pyplot(bbox_inches='tight')
            else:
                st.warning("Insufficient data for feature importance visualization.")

        st.markdown("### 📋 Team Statistics")
        with st.expander("View Team Performance Stats"):
            stats_df = pd.DataFrame(team_stats).T
            stats_df = stats_df.rename_axis("Team").reset_index()
            st.dataframe(stats_df.style.highlight_max(axis=0), use_container_width=True)

    else:
        st.warning("No upcoming games found.")

    st.markdown(f'''
        <div class="footer">
            &copy; {datetime.now().year} NBA Advanced Betting Insights. All rights reserved.
        </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
