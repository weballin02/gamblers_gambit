# Integrated_NBA_Predictions.py

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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from pmdarima import auto_arima
from sklearn.cluster import KMeans
import shap

# =======================
# 2. Configuration and Setup
# =======================
warnings.filterwarnings('ignore')

# Streamlit App Configuration
st.set_page_config(
    page_title="Integrated NBA Predictions",
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

# Custom CSS for Styling
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
    </style>
""", unsafe_allow_html=True)

# =======================
# 3. Header Section
# =======================
def render_header():
    st.markdown(f'''
        <div class="header">
            <h1>Integrated NBA Predictions</h1>
            <p>Combining Multiple Prediction Logics for Enhanced Accuracy</p>
        </div>
    ''', unsafe_allow_html=True)

render_header()

# =======================
# 4. Data Fetching and Preprocessing
# =======================
# Fetch NBA Teams
def get_nba_teams():
    nba_team_list = nba_teams.get_teams()
    team_name_mapping = {team['abbreviation']: team['full_name'] for team in nba_team_list}
    id_to_abbrev = {team['id']: team['abbreviation'] for team in nba_team_list}
    return nba_team_list, team_name_mapping, id_to_abbrev

nba_team_list, team_name_mapping, id_to_abbrev = get_nba_teams()

# Load NBA Game Logs
@st.cache_data(ttl=3600)
def load_nba_game_logs(_seasons):
    """Fetch and preprocess game logs for the specified NBA seasons."""
    all_games = []
    for season in _seasons:
        try:
            game_logs = LeagueGameLog(season=season, season_type_all_star='Regular Season', player_or_team_abbreviation='T')
            games = game_logs.get_data_frames()[0]
            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
            games['SEASON'] = season
            games['WEIGHT'] = season_weights.get(season, 1.0)
            all_games.append(games)
        except Exception as e:
            st.error(f"Error loading NBA game logs for {season}: {str(e)}")
    return pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()

# Preprocess Game Logs
def preprocess_game_logs(game_logs):
    if game_logs.empty:
        return pd.DataFrame()
    
    # Rename columns for consistency
    game_logs.rename(columns={
        'TEAM_ID': 'TEAM_ID',
        'TEAM_ABBREVIATION': 'TEAM_ABBREVIATION',
        'PTS': 'PTS',
        'MATCHUP': 'MATCHUP'
    }, inplace=True)
    
    # Feature Engineering
    game_logs['is_home'] = game_logs['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    game_logs['opponent'] = game_logs['MATCHUP'].apply(
        lambda x: x.split('vs. ')[1] if 'vs.' in x else x.split('@ ')[1]
    )
    game_logs['rest_days'] = game_logs.groupby('TEAM_ABBREVIATION')['GAME_DATE'].diff().dt.days.fillna(7)
    game_logs['game_number'] = game_logs.groupby('TEAM_ABBREVIATION').cumcount() + 1
    game_logs['score_rolling_mean'] = game_logs.groupby('TEAM_ABBREVIATION')['PTS'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    game_logs['opponent_avg_score'] = game_logs.groupby('opponent')['PTS'].transform('mean')
    
    # Drop rows with missing values
    game_logs.dropna(inplace=True)
    
    return game_logs

# Define Seasons and Weights
current_season = '2023-24'
previous_seasons = ['2022-23', '2021-22']
season_weights = {current_season: 1.0, '2022-23': 0.7, '2021-22': 0.5}

# Load and Preprocess Data
game_logs = load_nba_game_logs([current_season] + previous_seasons)
processed_game_logs = preprocess_game_logs(game_logs)

# =======================
# 5. Model Training Modules
# =======================

# 5.1. Regression Models (Enhanced NBA.py)
@st.cache_resource
def train_regression_models(_processed_game_logs, _season_weights):
    models = {}
    team_stats = {}
    for team in _processed_game_logs['TEAM_ABBREVIATION'].unique():
        team_df = _processed_game_logs[_processed_game_logs['TEAM_ABBREVIATION'] == team]
        if len(team_df) < 15:
            continue  # Skip teams with insufficient data

        # Features and Target
        features = team_df[['game_number', 'is_home', 'rest_days', 'score_rolling_mean', 'opponent_avg_score']]
        target = team_df['PTS']
        features.fillna(method='ffill', inplace=True)

        # Hyperparameter Tuning Grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }

        tscv = TimeSeriesSplit(n_splits=3)

        # Gradient Boosting Regressor
        gbr = GradientBoostingRegressor()
        gbr_grid = GridSearchCV(gbr, param_grid, cv=tscv)
        gbr_grid.fit(features, target)

        # Random Forest Regressor
        rf = RandomForestRegressor()
        rf_grid = GridSearchCV(rf, {'n_estimators': [100, 200], 'max_depth': [5, 10]}, cv=tscv)
        rf_grid.fit(features, target)

        # XGBoost Regressor
        xgb = XGBRegressor(eval_metric='rmse', use_label_encoder=False)
        xgb_grid = GridSearchCV(xgb, param_grid, cv=tscv)
        xgb_grid.fit(features, target)

        # Store Best Estimators
        models[team] = {
            'gbr': gbr_grid.best_estimator_,
            'rf': rf_grid.best_estimator_,
            'xgb': xgb_grid.best_estimator_,
            'features': features.columns
        }

        # Store Team Statistics
        team_stats[team] = {
            'avg_score': target.mean(),
            'std_dev': target.std(),
            'min_score': target.min(),
            'max_score': target.max()
        }

    return models, team_stats

regression_models, regression_team_stats = train_regression_models(processed_game_logs, season_weights)

# 5.2. ARIMA Models (NBA FoxEdge, NBA Quantum.py, NBA Team Scoring.py)
@st.cache_resource
def train_arima_models(_processed_game_logs):
    model_dir = 'models/arima'
    os.makedirs(model_dir, exist_ok=True)
    team_models_arima = {}
    for team_abbrev in _processed_game_logs['TEAM_ABBREVIATION'].unique():
        team_points = _processed_game_logs[_processed_game_logs['TEAM_ABBREVIATION'] == team_abbrev]['PTS']
        team_points.reset_index(drop=True, inplace=True)
        if len(team_points) < 10:
            continue  # Skip teams with insufficient data
        try:
            model = auto_arima(
                team_points,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True
            )
            model.fit(team_points)
            model_path = os.path.join(model_dir, f"{team_abbrev}_arima_model.pkl")
            joblib.dump(model, model_path)
            team_models_arima[team_abbrev] = model
        except Exception as e:
            st.warning(f"ARIMA model training failed for {team_abbrev}: {e}")
            continue
    return team_models_arima

@st.cache_data
def compute_team_forecasts_arima(_team_models_arima, _processed_game_logs, forecast_periods=5):
    team_forecasts_arima = {}
    for team_abbrev, model in _team_models_arima.items():
        team_points = _processed_game_logs[_processed_game_logs['TEAM_ABBREVIATION'] == team_abbrev]['PTS']
        if team_points.empty:
            continue
        try:
            forecast = model.predict(n_periods=forecast_periods)
            predictions = pd.DataFrame({
                'Date': pd.date_range(start=team_points.index.max() + 1, periods=forecast_periods, freq='D'),
                'Predicted_PTS': forecast,
                'Team': team_abbrev
            })
            team_forecasts_arima[team_abbrev] = predictions
        except Exception as e:
            st.warning(f"ARIMA forecast failed for {team_abbrev}: {e}")
            continue
    if team_forecasts_arima:
        return pd.concat(team_forecasts_arima.values(), ignore_index=True)
    else:
        return pd.DataFrame()

team_models_arima = train_arima_models(processed_game_logs)
team_forecasts_arima = compute_team_forecasts_arima(team_models_arima, processed_game_logs)

# 5.3. Clustering (NBA Quantum.py, NBA Team Scoring.py)
@st.cache_data
def apply_clustering(_processed_game_logs, n_clusters=3):
    cluster_data = {}
    for team_abbrev in _processed_game_logs['TEAM_ABBREVIATION'].unique():
        team_points = _processed_game_logs[_processed_game_logs['TEAM_ABBREVIATION'] == team_abbrev]['PTS'].values.reshape(-1, 1)
        if len(team_points) < n_clusters:
            cluster_data[team_abbrev] = [np.mean(team_points)]
            continue
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(team_points)
            cluster_centers = kmeans.cluster_centers_.flatten()
            cluster_data[team_abbrev] = cluster_centers
        except Exception as e:
            st.warning(f"KMeans clustering failed for {team_abbrev}: {e}")
            cluster_data[team_abbrev] = [np.mean(team_points)]
    return cluster_data

team_clusters = apply_clustering(processed_game_logs)

# =======================
# 6. Prediction Logic
# =======================

# 6.1. Regression Prediction
def predict_team_score_regression(team, models, team_stats, processed_game_logs):
    ensemble_prediction = None
    confidence_interval = None

    if team in models:
        team_df = processed_game_logs[processed_game_logs['TEAM_ABBREVIATION'] == team].iloc[-1]
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
            try:
                pred = model.predict(next_features[model_dict['features']])[0]
                predictions.append(pred)
            except Exception as e:
                st.warning(f"Prediction failed for {team} using {model_name}: {e}")
                continue

        if predictions:
            ensemble_prediction = np.mean(predictions)

    if team in team_stats:
        avg = team_stats[team]['avg_score']
        std = team_stats[team]['std_dev']
        confidence_interval = (avg - 1.96 * std, avg + 1.96 * std)

    return ensemble_prediction, confidence_interval

# 6.2. Simulation Prediction
def quantum_monte_carlo_simulation(home_team_abbrev, away_team_abbrev, spread_adjustment, num_simulations, team_stats, team_forecasts_arima):
    if home_team_abbrev not in team_stats or away_team_abbrev not in team_stats:
        st.error("Team stats not available for selected teams")
        return None

    home_forecast_arima = team_forecasts_arima[team_forecasts_arima['Team'] == home_team_abbrev]['Predicted_PTS'].mean()
    away_forecast_arima = team_forecasts_arima[team_forecasts_arima['Team'] == away_team_abbrev]['Predicted_PTS'].mean()

    home_forecast = home_forecast_arima + spread_adjustment
    away_forecast = away_forecast_arima

    home_stats = team_stats[home_team_abbrev]
    away_stats = team_stats[away_team_abbrev]

    # Simulate scores based on normal distribution
    home_scores = np.random.normal(loc=home_forecast, scale=home_stats['std_dev'], size=num_simulations)
    away_scores = np.random.normal(loc=away_forecast, scale=away_stats['std_dev'], size=num_simulations)

    # Ensure scores are non-negative
    home_scores = np.maximum(home_scores, 0)
    away_scores = np.maximum(away_scores, 0)

    score_diff = home_scores - away_scores
    home_wins = np.sum(score_diff > 0)
    away_wins = np.sum(score_diff < 0)
    ties = np.sum(score_diff == 0)

    results = {
        "Home_Win_Percentage": round(home_wins / num_simulations * 100, 2),
        "Away_Win_Percentage": round(away_wins / num_simulations * 100, 2),
        "Tie_Percentage": round(ties / num_simulations * 100, 2),
        "Average_Home_Score": round(np.mean(home_scores), 2),
        "Average_Away_Score": round(np.mean(away_scores), 2),
        "Average_Total_Score": round(np.mean(home_scores + away_scores), 2),
        "Average_Score_Difference": round(np.mean(score_diff), 2)
    }

    return results

# 6.3. Aggregated Prediction
def aggregate_predictions(regression_pred, simulation_pred, arima_pred):
    weights = {'regression': 0.5, 'simulation': 0.3, 'arima': 0.2}
    final_pred = 0
    if regression_pred is not None:
        final_pred += regression_pred * weights['regression']
    if simulation_pred is not None:
        final_pred += simulation_pred['Average_Home_Score'] * weights['simulation']
    if arima_pred is not None:
        final_pred += arima_pred * weights['arima']
    return final_pred

# =======================
# 7. Fetch Upcoming Games
# =======================
@st.cache_data(ttl=3600)
def fetch_upcoming_games():
    today = datetime.now().date()
    next_day = today + timedelta(days=1)
    try:
        today_scoreboard = ScoreboardV2(game_date=today.strftime('%Y-%m-%d'))
        tomorrow_scoreboard = ScoreboardV2(game_date=next_day.strftime('%Y-%m-%d'))
        combined_games = pd.concat([today_scoreboard.get_data_frames()[0], tomorrow_scoreboard.get_data_frames()[0]], ignore_index=True)
        if combined_games.empty:
            st.info(f"No upcoming games scheduled for today ({today}) and tomorrow ({next_day}).")
            return pd.DataFrame()
        combined_games['HOME_TEAM_ABBREV'] = combined_games['HOME_TEAM_ID'].map(id_to_abbrev)
        combined_games['VISITOR_TEAM_ABBREV'] = combined_games['VISITOR_TEAM_ID'].map(id_to_abbrev)
        combined_games.dropna(subset=['HOME_TEAM_ABBREV', 'VISITOR_TEAM_ABBREV'], inplace=True)
        combined_games['GAME_LABEL'] = combined_games.apply(
            lambda row: f"{team_name_mapping.get(row['VISITOR_TEAM_ABBREV'])} at {team_name_mapping.get(row['HOME_TEAM_ABBREV'])}",
            axis=1
        )
        return combined_games[['GAME_ID', 'HOME_TEAM_ABBREV', 'VISITOR_TEAM_ABBREV', 'GAME_LABEL']]
    except Exception as e:
        st.error(f"Error fetching upcoming games: {e}")
        return pd.DataFrame()

upcoming_games = fetch_upcoming_games()

# =======================
# 8. Main App Logic
# =======================
def main():
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

        # Regression Predictions
        home_pred_reg, home_ci_reg = predict_team_score_regression(home_team, regression_models, regression_team_stats, processed_game_logs)
        away_pred_reg, away_ci_reg = predict_team_score_regression(away_team, regression_models, regression_team_stats, processed_game_logs)

        # ARIMA Predictions
        home_pred_arima = team_forecasts_arima[team_forecasts_arima['Team'] == home_team]['Predicted_PTS'].mean()
        away_pred_arima = team_forecasts_arima[team_forecasts_arima['Team'] == away_team]['Predicted_PTS'].mean()

        # User Inputs for Simulation
        st.markdown("### 🛠️ Simulation Controls")
        spread_adjustment = st.slider(
            "Home Team Spread Adjustment",
            -10.0, 10.0, 0.0, step=0.5
        )
        num_simulations = st.selectbox(
            "Select Number of Simulations",
            [1000, 10000, 100000]
        )
        simulate_button = st.button("Run Simulation")

        # Run Simulation
        simulation_results = None
        if simulate_button:
            with st.spinner("Running Monte Carlo simulations..."):
                simulation_results = quantum_monte_carlo_simulation(
                    home_team, away_team, spread_adjustment, num_simulations, regression_team_stats, team_forecasts_arima
                )
            st.success("Simulation completed!")

        # Aggregate Predictions
        final_home_pred = aggregate_predictions(home_pred_reg, simulation_results, home_pred_arima) if simulation_results else home_pred_reg
        final_away_pred = aggregate_predictions(away_pred_reg, simulation_results, away_pred_arima) if simulation_results else away_pred_reg

        # Display Predictions
        st.markdown("### 📊 Final Predictions")
        prediction_df = pd.DataFrame({
            "Team": [team_name_mapping.get(home_team, home_team), team_name_mapping.get(away_team, away_team)],
            "Predicted Score": [final_home_pred, final_away_pred]
        })
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.set_style("whitegrid")
        sns.barplot(x="Team", y="Predicted Score", data=prediction_df, palette="viridis", ax=ax)
        ax.set_title("Predicted Scores")
        ax.set_ylabel("Score")
        ax.set_ylim(0, max(prediction_df["Predicted Score"].max(), 150) + 10 if prediction_df["Predicted Score"].max() > 0 else 150)

        for index, row in prediction_df.iterrows():
            ax.text(index, row["Predicted Score"] + 0.5, f'{row["Predicted Score"]:.2f}', color='black', ha="center")

        st.pyplot(fig)

        # Betting Insights
        st.markdown("### 💡 Betting Insights")
        if simulation_results:
            suggested_winner = home_team if simulation_results['Average_Score_Difference'] > 0 else away_team
            margin_of_victory = abs(simulation_results['Average_Score_Difference'])
            total_points = simulation_results['Average_Total_Score']
            over_under_threshold = processed_game_logs['PTS'].mean() * 2  # Example threshold
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
            st.warning("Run simulations to get betting insights.")

        # Feature Importance (Enhanced NBA.py)
        st.markdown("### 🔍 Feature Importance")
        with st.expander("View Feature Importance for Home Team"):
            if home_team in regression_models:
                model_dict = regression_models[home_team]
                model = model_dict['xgb']  # Using XGBoost for SHAP
                try:
                    explainer = shap.TreeExplainer(model)
                    team_df = processed_game_logs[processed_game_logs['TEAM_ABBREVIATION'] == home_team]
                    team_features = team_df[model_dict['features']].iloc[-1:]
                    shap_values = explainer.shap_values(team_features)
                    shap.initjs()
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, team_features, plot_type="bar", show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"SHAP analysis failed for {home_team}: {e}")
            else:
                st.warning("Insufficient data for feature importance visualization.")

        with st.expander("View Feature Importance for Away Team"):
            if away_team in regression_models:
                model_dict = regression_models[away_team]
                model = model_dict['xgb']  # Using XGBoost for SHAP
                try:
                    explainer = shap.TreeExplainer(model)
                    team_df = processed_game_logs[processed_game_logs['TEAM_ABBREVIATION'] == away_team]
                    team_features = team_df[model_dict['features']].iloc[-1:]
                    shap_values = explainer.shap_values(team_features)
                    shap.initjs()
                    fig, ax = plt.subplots()
                    shap.summary_plot(shap_values, team_features, plot_type="bar", show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"SHAP analysis failed for {away_team}: {e}")
            else:
                st.warning("Insufficient data for feature importance visualization.")

        # Team Statistics
        st.markdown("### 📋 Team Statistics")
        with st.expander("View Team Performance Stats"):
            stats_df = pd.DataFrame(regression_team_stats).T
            stats_df = stats_df.rename_axis("Team").reset_index()
            st.dataframe(stats_df.style.highlight_max(axis=0), use_container_width=True)

    else:
        st.warning("No upcoming games found.")

    # =======================
    # 9. Footer Section
    # =======================
    st.markdown(f'''
        <div class="footer">
            &copy; {datetime.now().year} Integrated NBA Predictions. All rights reserved.
        </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
