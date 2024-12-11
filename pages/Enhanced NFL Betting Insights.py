# foxedge_nfl_insights.py

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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nfl_data_py as nfl
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
import shap
from pmdarima import auto_arima
import matplotlib.dates as mdates
from scipy.stats import truncnorm, t, logistic

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =======================
# 2. Streamlit App Configuration
# =======================
st.set_page_config(
    page_title="🏈 FoxEdge NFL Insights",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =======================
# 3. Theme and CSS Styling
# =======================

# Initialize Session State for Theme and High Contrast
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'high_contrast' not in st.session_state:
    st.session_state.high_contrast = False

# Function to Toggle Theme
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.experimental_rerun()

# Function to Toggle High Contrast
def toggle_high_contrast():
    st.session_state.high_contrast = not st.session_state.high_contrast
    st.experimental_rerun()

# Theme Toggle Buttons in Sidebar
st.sidebar.button("🌗 Toggle Theme", on_click=toggle_theme)
st.sidebar.button("🎨 Toggle High Contrast", on_click=toggle_high_contrast)

# Apply Theme Based on Dark Mode and High Contrast
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

if st.session_state.high_contrast:
    primary_text = "#FFFF00"  # Yellow text for high contrast
    accent_color = "#FF0000"  # Red accent for high contrast

# Custom CSS for Novel Design
st.markdown(f"""
    <style>
    /* Overall Page Styling */
    body {{
        background-color: {primary_bg};
        color: {primary_text};
        font-family: 'Open Sans', sans-serif;
    }}

    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Header Styling */
    .header-title {{
        font-family: 'Montserrat', sans-serif;
        background: linear-gradient(120deg, {highlight_color}, {accent_color});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5em;
    }}

    /* Subheader Styling */
    .subheader-text {{
        color: {primary_text};
        opacity: 0.7;
        font-size: 1.5em;
        text-align: center;
        margin-bottom: 1.5em;
    }}

    /* Prediction Card Styling */
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

    /* Data Section Styling */
    .data-section {{
        padding: 2em 1em;
        background-color: {secondary_bg};
        border-radius: 15px;
        margin-bottom: 2em;
    }}

    .data-section h2 {{
        font-size: 2em;
        margin-bottom: 1em;
        color: {highlight_color};
    }}

    .data-section p {{
        font-size: 1em;
        color: {primary_text};
        opacity: 0.8;
        margin-bottom: 1em;
    }}

    /* Summary Section Styling */
    .summary-section {{
        padding: 2em 1em;
        background-color: {secondary_bg};
        border-radius: 15px;
        margin-bottom: 2em;
    }}

    .summary-section h3 {{
        font-size: 2em;
        margin-bottom: 0.5em;
        color: {highlight_color};
    }}

    .summary-section p {{
        font-size: 1.1em;
        color: {primary_text};
        opacity: 0.9;
        line-height: 1.6;
    }}

    /* Footer Styling */
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

    /* Responsive Design */
    @media (max-width: 768px) {{
        .header-title {{
            font-size: 2em;
        }}

        .subheader-text {{
            font-size: 1em;
        }}

        .prediction-card, .data-section, .summary-section {{
            padding: 1em;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

# =======================
# 4. Header Section
# =======================
def render_header():
    st.markdown(f'''
        <div>
            <h1 class="header-title">FoxEdge NFL Insights</h1>
            <p class="subheader-text">Comprehensive NFL Betting Predictions and Analytics</p>
        </div>
    ''', unsafe_allow_html=True)

# =======================
# 5. Utility Functions
# =======================

# --- Helper Function to Round to Nearest 0.5 ---
def round_to_nearest_half(x):
    return round(x * 2) / 2

# --- Team Abbreviation Mapping ---
def get_team_mappings():
    # Assuming you have a mapping of NFL team abbreviations to full names
    # This should be replaced with actual NFL team data
    nfl_teams = {
        'DAL': 'Dallas Cowboys',
        'NE': 'New England Patriots',
        'KC': 'Kansas City Chiefs',
        'GB': 'Green Bay Packers',
        'SF': 'San Francisco 49ers',
        # Add all NFL teams here
    }
    full_to_abbrev = {v: k for k, v in nfl_teams.items()}
    return nfl_teams, full_to_abbrev

abbrev_to_full, full_to_abbrev = get_team_mappings()

# --- Generate Truncated Normal Distribution ---
def generate_truncated_normal(mean, std, lower, upper, size):
    a = (lower - mean) / std
    b = (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

# --- Dynamic Blended Prediction ---
def dynamic_blend_weight(team_stats, team):
    # Weight shifts towards GBR as win streak increases
    win_streak = team_stats.get(team, {}).get('win_streak', 0)
    weight = 0.5 + 0.05 * win_streak  # Each win in the streak increases GBR's weight by 5%
    weight = min(0.9, max(0.1, weight))  # Ensure the weight is bounded between 0.1 and 0.9
    return weight

def blended_prediction(arima_pred, gbr_pred, weight):
    return weight * gbr_pred + (1 - weight) * arima_pred

# =======================
# 6. Data Fetching and Preprocessing
# =======================

# --- Fetch and Preprocess NFL Game Logs ---
@st.cache_data(ttl=3600)
def load_nfl_game_logs(seasons):
    """Fetch and preprocess game logs for the specified NFL seasons."""
    all_games = []
    for season in seasons:
        try:
            game_logs = fetch_nfl_game_logs(season=season)
            game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'], errors='coerce')
            game_logs['SEASON'] = season
            all_games.append(game_logs)
        except Exception as e:
            st.error(f"Error loading NFL game logs for {season}: {str(e)}")
    return pd.concat(all_games, ignore_index=True) if all_games else None

def preprocess_nfl_data(game_logs):
    """Prepare team data from game logs."""
    if game_logs is None:
        st.error("No game logs to preprocess.")
        return None
    
    team_data = game_logs[['GAME_DATE', 'TEAM_ABBREVIATION', 'PTS', 'MATCHUP']]
    team_data = team_data.rename(columns={'TEAM_ABBREVIATION': 'team', 'PTS': 'score'})
    team_data['GAME_DATE'] = pd.to_datetime(team_data['GAME_DATE'], errors='coerce')
    team_data.sort_values('GAME_DATE', inplace=True)  # Ensure chronological order
    team_data.set_index('GAME_DATE', inplace=True)
    
    # Determine Home/Away status
    team_data['is_home'] = team_data['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)
    
    # Calculate Rest Days
    team_data['rest_days'] = team_data.groupby('team')['GAME_DATE'].diff().dt.days.fillna(7)
    
    # Rolling Mean of Scores (Last 5 Games)
    team_data['score_rolling_mean'] = team_data.groupby('team')['score'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Extract Opponent
    team_data['opponent'] = team_data['MATCHUP'].apply(
        lambda x: x.split('vs. ')[1] if 'vs.' in x else x.split('@ ')[1]
    )
    
    # Calculate Opponent Average Score
    opponent_avg_score = team_data.groupby('team')['score'].mean().reset_index()
    opponent_avg_score.columns = ['opponent', 'opponent_avg_score']
    team_data = team_data.merge(opponent_avg_score, on='opponent', how='left')
    
    # Assign Decay Weights (for weighting past games)
    decay = 0.9
    team_data['decay_weight'] = team_data.groupby('team').cumcount().apply(lambda x: decay ** x)
    
    # Compute Momentum (Win Streak and Form)
    team_data = compute_momentum(team_data)
    
    team_data.dropna(inplace=True)
    return team_data

# Load Game Logs
current_season = "2024"  # Update as needed
seasons = [current_season, "2023", "2022"]
game_logs = load_nfl_game_logs(seasons)
team_data = preprocess_nfl_data(game_logs)

# =======================
# 7. Feature Engineering and Clustering
# =======================

# --- Aggregate Team Statistics ---
def aggregate_team_stats(team_data):
    """Aggregate statistics for each team."""
    team_stats = {}
    for team in team_data['team'].unique():
        team_df = team_data[team_data['team'] == team]
        if len(team_df) < 5:
            continue  # Skip teams with insufficient data
        scores = team_df['score'].values
        team_stats[team] = {
            'avg_score': round(np.mean(scores), 2),
            'std_dev': round(np.std(scores), 2),
            'min_score': round(np.min(scores), 2),
            'max_score': round(np.max(scores), 2),
            'recent_form': round(team_df['score_rolling_mean'].iloc[-1], 2),
            'win_streak': int(team_df['Win_Streak'].iloc[-1]),
            'games_played': len(scores)
        }
    return team_stats

team_stats = aggregate_team_stats(team_data)

# --- Dynamic Clustering of Teams ---
@st.cache_data
def perform_clustering(team_stats, n_clusters=5):
    """Cluster teams based on average score and standard deviation."""
    clustering_data = pd.DataFrame(team_stats).T[['avg_score', 'std_dev']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clustering_data['cluster'] = kmeans.fit_predict(clustering_data)
    cluster_labels = clustering_data['cluster'].to_dict()
    return cluster_labels, kmeans

cluster_labels, kmeans_model = perform_clustering(team_stats, n_clusters=5)

# Calculate Opponent Strength based on Clustering
def calculate_opponent_strength(team_data, cluster_labels):
    """Calculate opponent strength based on clustering."""
    team_data = team_data.copy()
    team_data['opponent_cluster'] = team_data['opponent'].map(cluster_labels)
    opponent_strength = team_data.groupby('team')['opponent_cluster'].mean().to_dict()
    return opponent_strength

opponent_strength = calculate_opponent_strength(team_data, cluster_labels)

# =======================
# 8. Model Training
# =======================

# --- Train Blended ARIMA + GBR Models ---
@st.cache_resource
def train_blended_models(team_data, team_stats, cluster_labels):
    """Train blended ARIMA and Gradient Boosting models for each team."""
    model_dir = 'models/nfl/blended'
    os.makedirs(model_dir, exist_ok=True)
    blended_models = {}
    
    for team, stats in team_stats.items():
        team_df = team_data[team_data['team'] == team]
        if len(team_df) < 10:
            continue  # Skip teams with insufficient data
        
        # Feature Engineering
        features = team_df[['is_home', 'rest_days', 'score_rolling_mean', 'opponent_avg_score']]
        # Enhanced Features
        features['win_streak'] = team_df['win_streak']
        features['opponent_strength'] = team_df['opponent'].map(opponent_strength).fillna(0)
        features['back_to_back'] = team_df['rest_days'] < 2  # Binary feature
        features['home_avg_score'] = team_df.apply(lambda x: team_stats[x['team']]['avg_score'] if x['is_home'] else 0, axis=1)
        features['away_avg_score'] = team_df.apply(lambda x: team_stats[x['team']]['avg_score'] if not x['is_home'] else 0, axis=1)
        
        # Include ELO Ratings (Placeholder calculation)
        # In a real scenario, use actual ELO ratings
        team_data['elo_rating'] = team_data.groupby('team')['score'].cumsum() / team_data['games_played']
        features['elo_rating'] = team_df['elo_rating']
        
        # Drop any remaining NaNs
        features.fillna(method='ffill', inplace=True)
        
        target = team_df['score']
        
        # SHAP-Based Feature Selection
        # Train initial GBR model to compute SHAP values
        gbr_initial = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gbr_initial.fit(features, target)
        explainer = shap.Explainer(gbr_initial)
        shap_values = explainer(features)
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': np.abs(shap_values.values).mean(axis=0)
        })
        # Select features with importance above the median
        median_importance = feature_importance['importance'].median()
        selected_features = feature_importance[feature_importance['importance'] >= median_importance]['feature'].tolist()
        features_selected = features[selected_features]
        
        # Retrain GBR with selected features
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gbr.fit(features_selected, target)
        
        # Save Feature Selection
        feature_selection_path = os.path.join(model_dir, f"{team}_features.pkl")
        joblib.dump(selected_features, feature_selection_path)
        
        # ARIMA Model
        arima_model_path = os.path.join(model_dir, f"{team}_arima.pkl")
        arima_model = auto_arima(
            team_df['score'],
            seasonal=True,
            m=16,  # Number of weeks in NFL regular season
            trace=False,
            error_action='ignore',
            suppress_warnings=True
        )
        joblib.dump(arima_model, arima_model_path)
        
        # Gradient Boosting Regressor Model
        gbr_model_path = os.path.join(model_dir, f"{team}_gbr.pkl")
        joblib.dump(gbr, gbr_model_path)
        
        # Store Blended Models
        blended_models[team] = {
            'arima': arima_model,
            'gbr': gbr,
            'selected_features': selected_features
        }
    
    return blended_models

blended_models = train_blended_models(team_data, team_stats, cluster_labels)

# =======================
# 9. Prediction Logic
# =======================

# --- Predict Team Score with Blended Model ---
def predict_team_score_blended(team, blended_models, team_stats, team_data):
    """Predict team score using blended ARIMA and GBR models."""
    if team not in blended_models:
        return None, None
    
    models = blended_models[team]
    arima_model = models['arima']
    gbr_model = joblib.load(os.path.join('models/nfl/blended', f"{team}_gbr.pkl"))
    selected_features = joblib.load(os.path.join('models/nfl/blended', f"{team}_features.pkl"))
    
    team_df = team_data[team_data['team'] == team]
    if team_df.empty:
        return None, None
    
    latest_game = team_df.iloc[-1]
    next_features = pd.DataFrame({
        'is_home': [1],  # Assuming the next game is at home; adjust as needed
        'rest_days': [7],  # Assuming a week of rest
        'score_rolling_mean': [latest_game['score_rolling_mean']],
        'opponent_avg_score': [latest_game['opponent_avg_score']],
        'win_streak': [latest_game['win_streak']],
        'opponent_strength': [opponent_strength.get(latest_game['opponent'], 0)],
        'back_to_back': [latest_game['rest_days'] < 2],
        'home_avg_score': [team_stats[team]['avg_score'] if latest_game['is_home'] else 0],
        'away_avg_score': [team_stats[team]['avg_score'] if not latest_game['is_home'] else 0],
        'elo_rating': [team_data.loc[latest_game.name, 'elo_rating']]
    })
    
    # Select features
    features_selected = next_features[selected_features]
    
    # ARIMA Prediction
    arima_pred = arima_model.predict(n_periods=1)[0]
    
    # GBR Prediction
    gbr_pred = gbr_model.predict(features_selected)[0]
    
    # Dynamic Blended Prediction
    weight = dynamic_blend_weight(team_stats, team)
    blended_pred = blended_prediction(arima_pred, gbr_pred, weight)
    
    # Confidence via Logistic Transformation
    confidence = logistic.cdf(blended_pred + team_stats[team]['std_dev']) - logistic.cdf(blended_pred - team_stats[team]['std_dev'])
    confidence = round(confidence * 100, 2)
    
    return blended_pred, confidence

# =======================
# 10. Monte Carlo Simulation Enhancements
# =======================

# --- Monte Carlo Simulation with Truncated Normal and Student's T Distribution ---
def monte_carlo_simulation_enhanced(home_team, away_team, blended_models, team_stats, team_data, num_simulations=1000, spread_adjustment=0, total_line=50.0, betting_percentage_home=50.0):
    """Enhanced Monte Carlo simulation using truncated normal and Student's T distributions."""
    # Predict scores
    home_pred, home_confidence = predict_team_score_blended(home_team, blended_models, team_stats, team_data)
    away_pred, away_confidence = predict_team_score_blended(away_team, blended_models, team_stats, team_data)
    
    if home_pred is None or away_pred is None:
        return None, None, None, None, None
    
    # Adjust ELO Ratings
    # Placeholder: In a real scenario, use actual ELO ratings
    home_elo = team_data.loc[team_data['team'] == home_team, 'elo_rating'].iloc[-1]
    away_elo = team_data.loc[team_data['team'] == away_team, 'elo_rating'].iloc[-1]
    
    # Adjust based on cluster
    if cluster_labels[home_team] == cluster_labels[away_team]:
        home_std = team_stats[home_team]['std_dev'] * 0.8  # Reduce spread
        away_std = team_stats[away_team]['std_dev'] * 0.8
    else:
        home_std = team_stats[home_team]['std_dev']
        away_std = team_stats[away_team]['std_dev']
    
    # Truncated Normal Parameters
    home_mean = home_pred
    away_mean = away_pred
    
    # Use Student's T distribution for heavy tails
    df = 5  # Degrees of freedom; adjust as needed for heavier tails
    home_scores = t.rvs(df, loc=home_mean, scale=home_std, size=num_simulations)
    away_scores = t.rvs(df, loc=away_mean, scale=away_std, size=num_simulations)
    
    # Apply Truncated Normal to ensure realistic scores
    home_scores = generate_truncated_normal(home_mean, home_std, 0, 100, num_simulations)
    away_scores = generate_truncated_normal(away_mean, away_std, 0, 100, num_simulations)
    
    # Apply Spread Adjustment
    home_scores += spread_adjustment
    
    # Ensure scores are non-negative
    home_scores = np.maximum(home_scores, 0)
    away_scores = np.maximum(away_scores, 0)
    
    # Adjust for Travel Fatigue (Back-to-Back)
    home_rest_days = team_data.loc[team_data['team'] == home_team, 'rest_days'].iloc[-1]
    away_rest_days = team_data.loc[team_data['team'] == away_team, 'rest_days'].iloc[-1]
    
    if home_rest_days < 2:
        home_scores -= 2  # Subtract 2 points for fatigue
    if away_rest_days < 2:
        away_scores -= 2  # Subtract 2 points for fatigue
    
    # Ensure scores are non-negative after adjustment
    home_scores = np.maximum(home_scores, 0)
    away_scores = np.maximum(away_scores, 0)
    
    # Calculate score differences and totals
    score_diff = home_scores - away_scores
    total_scores = home_scores + away_scores
    
    # Calculate win counts
    home_wins = np.sum(score_diff > 0)
    away_wins = np.sum(score_diff < 0)
    ties = np.sum(score_diff == 0)
    
    # Calculate metrics
    results = {
        "Home Win %": round((home_wins / num_simulations) * 100, 2),
        "Away Win %": round((away_wins / num_simulations) * 100, 2),
        "Ties %": round((ties / num_simulations) * 100, 2),
        "Average Home Score": round(np.mean(home_scores), 2),
        "Average Away Score": round(np.mean(away_scores), 2),
        "Average Total Score": round(np.mean(total_scores), 2),
        "Average Score Differential": round(np.mean(score_diff), 2)
    }
    
    # Confidence Intervals using Bayesian Bootstrapping
    bootstrap_samples = 1000
    boot_diff_means = []
    boot_total_means = []
    for _ in range(bootstrap_samples):
        # Bayesian Bootstrapping
        weights = np.random.dirichlet(np.ones(num_simulations))
        sample_diff = np.random.choice(score_diff, size=num_simulations, p=weights)
        sample_total = np.random.choice(total_scores, size=num_simulations, p=weights)
        boot_diff_means.append(np.mean(sample_diff))
        boot_total_means.append(np.mean(sample_total))
    
    lower_diff, upper_diff = round_to_nearest_half(np.percentile(boot_diff_means, 5)), round_to_nearest_half(np.percentile(boot_diff_means, 95))
    lower_total, upper_total = round_to_nearest_half(np.percentile(boot_total_means, 5)), round_to_nearest_half(np.percentile(boot_total_means, 95))
    
    # Fade Public Betting Confidence
    confidence = results["Home Win %"] if betting_percentage_home > 70 else results["Away Win %"]
    confidence = fade_public(betting_percentage_home, confidence)
    
    return results, score_diff, (lower_diff, upper_diff), (lower_total, upper_total), confidence

# --- Fade Public Betting ---
def fade_public(bet_percentage, confidence):
    if bet_percentage > 70:
        confidence *= 0.9  # Reduce confidence by 10%
    return confidence

# =======================
# 11. Logic Improvements
# =======================

# --- Smarter Injury Handling ---
def adjust_for_injuries(team, injury_data, team_stats):
    """Adjust team stats based on injuries."""
    if team not in injury_data:
        return team_stats[team]['avg_score']
    
    # Define impact scores based on player positions or usage
    key_positions = ['QB', 'RB', 'WR', 'OL']
    position_impact = {'QB': 0.15, 'RB': 0.07, 'WR': 0.05, 'OL': 0.03}
    
    total_impact = 0
    for _, row in injury_data.iterrows():
        if row['team'] == team and row['position'] in key_positions:
            impact = position_impact.get(row['position'], 0.02)
            total_impact += impact
    
    adjusted_avg = team_stats[team]['avg_score'] * (1 - total_impact)
    return adjusted_avg

# Example Injury Data (Placeholder)
# In a real scenario, fetch injury data from a reliable source
# Example DataFrame structure:
# injury_data = pd.DataFrame({
#     'team': ['DAL', 'NE'],
#     'player': ['Dak Prescott', 'Cam Newton'],
#     'position': ['QB', 'QB']
# })
injury_data = pd.DataFrame({
    'team': ['DAL', 'NE'],
    'player': ['Dak Prescott', 'Cam Newton'],
    'position': ['QB', 'QB']
})

# =======================
# 12. Data Presentation Improvements
# =======================

# --- Display Prediction Results ---
def display_prediction_results(results, home_team_full, away_team_full, lower_diff, upper_diff, lower_total, upper_total, total_line, parlay_suggestion):
    """Display simulation results with confidence intervals."""
    if results:
        st.markdown(f'''
            <div class="prediction-card">
                <h2>🏈 Simulation Results</h2>
                <p><strong>{home_team_full} Win Percentage:</strong> {results["Home Win %"]}%</p>
                <p><strong>{away_team_full} Win Percentage:</strong> {results["Away Win %"]}%</p>
                <p><strong>Ties Percentage:</strong> {results["Ties %"]}%</p>
                <p><strong>Average Total Score:</strong> {results["Average Total Score"]}</p>
                <p><strong>Average Score Differential:</strong> {results["Average Score Differential"]}</p>
                <p><strong>90% Confidence Interval for Score Differential:</strong> {lower_diff} to {upper_diff}</p>
                <p><strong>90% Confidence Interval for Total Points:</strong> {lower_total} to {upper_total}</p>
            </div>
        ''', unsafe_allow_html=True)
        
        # Betting Insights
        st.markdown("### 💡 Betting Insights")
        # Over/Under Recommendation
        if results["Average Total Score"] > total_line:
            over_under = "Take the Over"
        else:
            over_under = "Take the Under"
        
        # Spread Bet Recommendation
        suggested_winner = "Home Team" if results["Home Win %"] > results["Away Win %"] else "Away Team"
        spread_recommendation = f"**Spread Bet:** {suggested_winner} to cover the spread."
        
        # Over/Under Recommendation
        over_under_recommendation = f"**Over/Under Bet:** {over_under} ({total_line}) based on predicted total points."
        
        # Parlay Suggestion
        parlay_suggestion_text = f"**Parlay Suggestion:** {parlay_suggestion}"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**🏆 Suggested Winner:** **{suggested_winner}**")
        with col2:
            st.markdown(f"**💰 Over/Under Recommendation:** {over_under}")
        with col3:
            st.markdown(f"**🎯 Spread Bet Recommendation:** {'Home' if suggested_winner == 'Home Team' else 'Away'} Team to cover the spread.")
        
        st.markdown(over_under_recommendation)
        st.markdown(parlay_suggestion_text)
        
        # Visualizations
        st.markdown("### 📊 Simulation Results Visualization")
        
        # Score Differential Distribution
        fig = px.histogram(
            score_diff,
            nbins=50,
            title="Score Differential Distribution (Home Team - Away Team)",
            labels={'value': 'Score Differential', 'count': 'Frequency'},
            opacity=0.75,
            color_discrete_sequence=[highlight_color],
            template=chart_template
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red",
                     annotation_text="Break-Even", annotation_position="top left")
        fig.update_layout(
            xaxis_title="Score Differential",
            yaxis_title="Frequency",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key='score_diff_distribution')
        
        # Betting Line vs. Prediction Overlay
        fig_bet = go.Figure(data=[
            go.Bar(name='Predicted Total', x=["Predicted Total"], y=[results["Average Total Score"]], marker_color=highlight_color),
            go.Bar(name='Betting Total', x=["Betting Total"], y=[total_line], marker_color="orange")
        ])
        fig_bet.update_layout(
            barmode='group',
            title="Predicted Total vs. Betting Total",
            yaxis_title="Points",
            template=chart_template
        )
        st.plotly_chart(fig_bet, use_container_width=True, key='betting_total_overlay')
        
        # Total Points Confidence Interval
        fig_total_ci = go.Figure()
        fig_total_ci.add_trace(go.Box(y=total_scores, name='Total Points', boxpoints='all', jitter=0.3, pointpos=-1.8))
        fig_total_ci.update_layout(
            title="Total Points Distribution",
            yaxis_title="Total Points",
            template=chart_template
        )
        st.plotly_chart(fig_total_ci, use_container_width=True, key='total_points_distribution')
        
    else:
        st.warning("No simulation results to display.")

# =======================
# 13. Main Application Logic
# =======================

def main():
    # Render Header
    render_header()
    
    # Fetch and preprocess data
    game_logs = load_nfl_game_logs(seasons)
    team_data = preprocess_nfl_data(game_logs)
    team_stats = aggregate_team_stats(team_data)
    
    # Perform Clustering
    cluster_labels, kmeans_model = perform_clustering(team_stats, n_clusters=5)
    
    # Calculate Opponent Strength
    opponent_strength = calculate_opponent_strength(team_data, cluster_labels)
    
    # Train Blended Models
    blended_models = train_blended_models(team_data, team_stats, cluster_labels)
    
    # Compute Team Forecasts (Placeholder: Implement as needed)
    # team_forecasts = compute_team_forecasts(blended_models, team_data)
    
    # Fetch upcoming games
    upcoming_games = fetch_upcoming_games()  # Placeholder function
    
    # Create Tabs for Organization
    tabs = st.tabs(["🏈 Predictions", "📈 Team Scoring", "🔮 Quantum Insights", "📊 Data & Analytics", "⚙️ Settings"])
    
    with tabs[0]:
        # Predictions Tab
        st.markdown('<div class="data-section"><h2>Select a Game for Prediction</h2></div>', unsafe_allow_html=True)
        
        if upcoming_games.empty:
            st.warning("No upcoming games available for prediction.")
        else:
            game_selection = st.selectbox('Select an upcoming game:', upcoming_games['GAME_LABEL'])
            selected_game = upcoming_games[upcoming_games['GAME_LABEL'] == game_selection].iloc[0]
            home_team, away_team = selected_game['HOME_TEAM_ABBREV'], selected_game['VISITOR_TEAM_ABBREV']
            home_team_full, away_team_full = abbrev_to_full.get(home_team, home_team), abbrev_to_full.get(away_team, away_team)
            
            # Injury Adjustments
            home_avg = adjust_for_injuries(home_team, injury_data, team_stats)
            away_avg = adjust_for_injuries(away_team, injury_data, team_stats)
            team_stats[home_team]['avg_score'] = home_avg
            team_stats[away_team]['avg_score'] = away_avg
            
            # Spread Adjustment Slider
            st.markdown("### Adjust Spread")
            spread_adjustment = st.slider(
                "Home Team Spread Adjustment",
                -10.0, 10.0, 0.0, step=0.5,
                help="Adjust the spread line to evaluate betting scenarios."
            )
            
            # Total Line Input
            st.markdown("### Set Total Points Line")
            total_line = st.number_input(
                "Total Points Line",
                min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                help="Set the over/under total points line for the game."
            )
            
            # Number of Simulations
            st.markdown("### Number of Simulations")
            num_simulations = st.selectbox(
                "Select Number of Simulations",
                [1000, 5000, 10000],
                index=0,
                help="Choose the number of Monte Carlo simulations to run for prediction accuracy."
            )
            
            # Public Betting Percentage Input
            st.markdown("### Public Betting Percentage")
            betting_percentage_home = st.slider(
                "Public Betting Percentage on Home Team",
                0.0, 100.0, 50.0, step=1.0,
                help="Enter the percentage of public bets on the home team."
            )
            
            # Parlay Suggestions
            parlay_suggestion = "Combine your top 3 confident bets for a parlay."
            
            # Run Simulation Button
            if st.button("Run Simulation"):
                with st.spinner("Running simulation..."):
                    results, score_diff, (lower_diff, upper_diff), (lower_total, upper_total), confidence = monte_carlo_simulation_enhanced(
                        home_team, away_team, blended_models, team_stats, team_data,
                        num_simulations=num_simulations,
                        spread_adjustment=spread_adjustment,
                        total_line=total_line,
                        betting_percentage_home=betting_percentage_home
                    )
                    if results:
                        display_prediction_results(results, home_team_full, away_team_full, lower_diff, upper_diff, lower_total, upper_total, total_line, parlay_suggestion)
                    else:
                        st.warning("Unable to perform simulation due to insufficient data.")

    with tabs[1]:
        # Team Scoring Tab
        st.markdown('<div class="data-section"><h2>📈 NFL Team Scoring Predictions</h2></div>', unsafe_allow_html=True)
        
        # Dropdown menu for selecting a team
        selected_team = st.selectbox('Select a team for scoring prediction:', sorted(team_stats.keys()), format_func=lambda x: abbrev_to_full.get(x, x))
        
        if selected_team:
            team_full_name = abbrev_to_full.get(selected_team, "Unknown Team")
            st.markdown(f"### 📊 Historical Points for {team_full_name}")
            
            # Plot Historical Points
            team_points = team_data[team_data['team'] == selected_team]['score']
            st.line_chart(team_points)
            
            # Display Forecasted Points
            st.markdown(f"### 🔮 Predicted Points for Next 5 Games ({team_full_name})")
            # Placeholder: Implement team_forecasts as needed
            # team_forecast = team_forecasts[team_forecasts['Team'] == selected_team]
            # st.write(team_forecast[['Date', 'Predicted_PTS']])
            st.write("Forecasted points functionality is under development.")
            
            # Plot Forecast with Confidence Interval
            st.markdown("### 📉 Forecast Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            historical_dates = mdates.date2num(team_points.index)
            # forecast_dates = mdates.date2num(team_forecast['Date'])
            # ax.plot_date(historical_dates, team_points.values, '-', label='Historical Points', color='blue')
            # ax.plot_date(forecast_dates, team_forecast['Predicted_PTS'], '-', label='Predicted Points', color='orange')
            # lower_bound = team_forecast['Predicted_PTS'] - 5
            # upper_bound = team_forecast['Predicted_PTS'] + 5
            # ax.fill_between(forecast_dates, lower_bound, upper_bound, color='gray', alpha=0.2, label='Confidence Interval')
            ax.plot_date(historical_dates, team_points.values, '-', label='Historical Points', color='blue')
            ax.set_title(f"Points Prediction for {team_full_name}", color=primary_text)
            ax.set_xlabel("Date", color=primary_text)
            ax.set_ylabel("Points", color=primary_text)
            ax.legend()
            ax.grid(True)
            ax.set_facecolor(secondary_bg)
            fig.patch.set_facecolor(primary_bg)
            st.pyplot(fig)
    
    with tabs[2]:
        # Quantum Insights Tab
        st.markdown('<div class="data-section"><h2>🔮 Quantum-Inspired Predictions</h2></div>', unsafe_allow_html=True)
        
        # Description or additional quantum-inspired features can be added here
        st.write("""
            Explore advanced quantum-inspired simulations and predictive analytics to enhance your NFL betting strategies. Leverage Monte Carlo simulations, cluster analyses, and more to gain deeper insights into game outcomes.
        """)
        
        # Example: Quantum Monte Carlo Simulation Button
        if st.button("Run Quantum Monte Carlo Simulation for All Games"):
            with st.spinner("Running quantum-inspired simulations..."):
                all_results = []
                if not upcoming_games.empty:
                    for _, game in upcoming_games.iterrows():
                        home_team_abbrev = game['HOME_TEAM_ABBREV']
                        away_team_abbrev = game['VISITOR_TEAM_ABBREV']
                        home_team_full = abbrev_to_full.get(home_team_abbrev, home_team_abbrev)
                        away_team_full = abbrev_to_full.get(away_team_abbrev, away_team_abbrev)
                        
                        # Perform simulation
                        results, score_diff, (lower_diff, upper_diff), (lower_total, upper_total), confidence = monte_carlo_simulation_enhanced(
                            home_team_abbrev, away_team_abbrev,
                            blended_models, team_stats, team_data,
                            num_simulations=1000,
                            spread_adjustment=0,
                            total_line=50.0,  # Default; adjust as needed
                            betting_percentage_home=50.0  # Placeholder; replace with actual data
                        )
                        if results:
                            all_results.append({
                                'Home Team': home_team_full,
                                'Away Team': away_team_full,
                                'Home Win %': results['Home Win %'],
                                'Away Win %': results['Away Win %'],
                                'Ties %': results['Ties %'],
                                'Avg Home Score': results['Average Home Score'],
                                'Avg Away Score': results['Average Away Score'],
                                'Avg Total Score': results['Average Total Score'],
                                'Score Differential': results['Average Score Differential'],
                                'CI Lower Diff': lower_diff,
                                'CI Upper Diff': upper_diff,
                                'CI Lower Total': lower_total,
                                'CI Upper Total': upper_total,
                                'Confidence Level (%)': confidence
                            })
                
                if all_results:
                    results_df = pd.DataFrame(all_results)
                    st.write("### Simulation Summary for All Games")
                    st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
                    
                    # Visualize Win Percentages
                    fig = go.Figure(data=[
                        go.Bar(name='Home Win %', x=results_df['Home Team'], y=results_df['Home Win %'], marker_color='green'),
                        go.Bar(name='Away Win %', x=results_df['Away Team'], y=results_df['Away Win %'], marker_color='red')
                    ])
                    fig.update_layout(
                        barmode='group',
                        title="Win Percentage Comparison",
                        xaxis_title="Teams",
                        yaxis_title="Win Percentage (%)",
                        template=chart_template
                    )
                    st.plotly_chart(fig, use_container_width=True, key='quantum_win_percentage')
                    
                    # Detailed Results with Confidence Intervals
                    st.markdown("### Detailed Simulation Results")
                    for index, row in results_df.iterrows():
                        st.markdown(f'''
                            <div class="prediction-card">
                                <h2>{row['Home Team']} vs {row['Away Team']}</h2>
                                <p><strong>Home Win %:</strong> {row['Home Win %']}%</p>
                                <p><strong>Away Win %:</strong> {row['Away Win %']}%</p>
                                <p><strong>Ties %:</strong> {row['Ties %']}%</p>
                                <p><strong>Average Total Score:</strong> {row['Avg Total Score']}</p>
                                <p><strong>Score Differential:</strong> {row['Score Differential']}</p>
                                <p><strong>90% Confidence Interval for Score Differential:</strong> {row['CI Lower Diff']} to {row['CI Upper Diff']}</p>
                                <p><strong>90% Confidence Interval for Total Points:</strong> {row['CI Lower Total']} to {row['CI Upper Total']}</p>
                                <p><strong>Confidence Level:</strong> {row['Confidence Level (%)']}%</p>
                            </div>
                        ''', unsafe_allow_html=True)
                    
                else:
                    st.warning("No simulation results to display.")

    with tabs[3]:
        # Data & Analytics Tab
        st.markdown('<div class="data-section"><h2>📊 Data & Analytics</h2></div>', unsafe_allow_html=True)
        
        # Display Team Statistics
        st.markdown("### 📋 Team Performance Statistics")
        stats_df = pd.DataFrame(team_stats).T
        stats_df = stats_df.rename_axis("Team").reset_index()
        stats_df.rename(columns={'team': 'Team'}, inplace=True)
        st.dataframe(stats_df.style.highlight_max(axis=0, color='lightgreen').format({
            'avg_score': "{:.2f}",
            'std_dev': "{:.2f}",
            'min_score': "{:.2f}",
            'max_score': "{:.2f}",
            'recent_form': "{:.2f}",
            'win_streak': "{:d}",
            'games_played': "{:d}"
        }), use_container_width=True)
        
        # SHAP Feature Importance
        st.markdown("### 🔍 Feature Importance")
        selected_team_for_shap = st.selectbox('Select a team to view feature importance:', sorted(team_stats.keys()), format_func=lambda x: abbrev_to_full.get(x, x))
        if selected_team_for_shap:
            if selected_team_for_shap in blended_models:
                model_info = blended_models[selected_team_for_shap]
                gbr_model = joblib.load(os.path.join('models/nfl/blended', f"{selected_team_for_shap}_gbr.pkl"))
                selected_features = joblib.load(os.path.join('models/nfl/blended', f"{selected_team_for_shap}_features.pkl"))
                team_features = team_data[team_data['team'] == selected_team_for_shap].iloc[-1:]
                if not team_features.empty:
                    explainer = shap.Explainer(gbr_model)
                    shap_values = explainer(team_features[selected_features])
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.markdown(f"#### Feature Importance for {abbrev_to_full.get(selected_team_for_shap, selected_team_for_shap)}")
                    shap.summary_plot(shap_values, team_features[selected_features], plot_type="bar", show=False)
                    st.pyplot(bbox_inches='tight')
                else:
                    st.warning("Insufficient data for SHAP analysis.")
            else:
                st.warning("Model not available for the selected team.")

    with tabs[4]:
        # Settings Tab
        st.markdown('<div class="data-section"><h2>⚙️ Settings</h2></div>', unsafe_allow_html=True)
        
        # Data Management
        with st.expander("💾 Data Management"):
            if st.button("Download Team Statistics CSV"):
                csv = pd.DataFrame(team_stats).T.to_csv().encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='team_statistics.csv',
                    mime='text/csv',
                    key='download_team_stats_csv'
                )
            if st.button("Download Simulation Results CSV"):
                # Placeholder: Implement simulation results download as needed
                st.warning("Simulation results download functionality is under development.")
        
        # Advanced Settings
        with st.expander("🔧 Advanced Settings"):
            st.markdown("Adjust simulation parameters to customize predictions.")
            new_num_simulations = st.slider(
                "Default Number of Simulations",
                1000, 10000, 1000, step=500,
                help="Set the default number of Monte Carlo simulations for predictions."
            )
            st.session_state['default_simulations'] = new_num_simulations
        
        # Accessibility Settings
        with st.expander("🎨 Accessibility"):
            high_contrast = st.checkbox("Enable High Contrast Mode", value=st.session_state.high_contrast)
            if high_contrast != st.session_state.high_contrast:
                toggle_high_contrast()
    
    # =======================
    # 14. Footer Section
    # =======================
    st.markdown(f'''
        <div class="footer">
            &copy; {datetime.now().year} **FoxEdge**. All rights reserved.
        </div>
    ''', unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    main()
