# FoxEdge - Comprehensive NFL Predictions Streamlit App

# =======================
# 1. Import Libraries
# =======================
import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from sklearn.ensemble import GradientBoostingRegressor
from pmdarima import auto_arima
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import requests
import pytz
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import truncnorm
from sklearn.linear_model import LogisticRegression
from collections import deque
import joblib

warnings.filterwarnings('ignore')

# =======================
# 2. Streamlit App Configuration
# =======================
st.set_page_config(
    page_title="🏈 FoxEdge - NFL Predictions",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =======================
# 3. Custom CSS Styling
# =======================
st.markdown('''
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&family=Open+Sans:wght@400;600&display=swap');

        /* Root Variables */
        :root {
            --background-color: #2C3E50; /* Charcoal Dark Gray */
            --primary-color: #1E90FF; /* Electric Blue */
            --secondary-color: #FF8C00; /* Deep Orange */
            --accent-color: #FF4500; /* Fiery Red */
            --success-color: #32CD32; /* Lime Green */
            --alert-color: #FFFF33; /* Neon Yellow */
            --text-color: #FFFFFF; /* Crisp White */
            --heading-text-color: #F5F5F5; /* Light Gray */
            --font-heading: 'Raleway', sans-serif;
            --font-body: 'Open Sans', sans-serif;
        }

        /* Global Styles */
        body, html {
            background: var(--background-color);
            color: var(--text-color);
            font-family: var(--font-body);
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }

        h1, h2, h3, h4 {
            font-family: var(--font-heading);
            color: var(--primary-color);
        }

        /* Hero Section */
        .hero {
            position: relative;
            text-align: center;
            padding: 4em 1em;
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
            font-size: 3.5em;
            margin-bottom: 0.2em;
            background: linear-gradient(120deg, var(--success-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            font-size: 1.5em;
            margin-bottom: 1em;
            color: #CCCCCC; /* Light Gray */
        }

        /* Buttons */
        .button {
            background: var(--primary-color);
            border: none;
            padding: 0.8em 2em;
            color: var(--text-color);
            font-size: 1.1em;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.3s ease, background-color 0.3s ease;
            text-decoration: none;
            display: inline-block;
            margin-top: 1em;
        }

        .button:hover {
            background-color: var(--accent-color); /* Fiery Red */
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
            color: var(--success-color); /* Lime Green */
        }

        .data-section p {
            font-size: 1.2em;
            color: #CCCCCC; /* Light Gray */
            margin-bottom: 2em;
        }

        /* Prediction and Summary Cards */
        .prediction-card, .summary-card, .team-card {
            background-color: #1E1E1E; /* Dark background for cards */
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2em 1em;
            color: #999999; /* Light Gray */
            font-size: 0.9em;
        }

        .footer a {
            color: var(--primary-color); /* Electric Blue */
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

            .team-trends {
                flex-direction: column;
                align-items: center;
            }

            .team-card {
                width: 90%;
            }
        }

        /* Tooltip Styles */
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted black;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 220px;
            background-color: #555;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position above the text */
            left: 50%;
            margin-left: -110px;
            opacity: 0;
            transition: opacity 0.3s;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }

        /* Data Table Styles */
        table.dataframe {
            background-color: #1E1E1E;
            color: #FFFFFF;
            border-collapse: collapse;
            width: 100%;
        }

        table.dataframe th, table.dataframe td {
            border: 1px solid #444444;
            padding: 8px;
            text-align: center;
        }

        table.dataframe th {
            background-color: var(--primary-color);
        }

        table.dataframe tr:nth-child(even) {
            background-color: #2C3E50;
        }

        table.dataframe tr:hover {
            background-color: #34495E;
        }
    </style>
''', unsafe_allow_html=True)

# =======================
# 4. Header Section
# =======================
st.markdown('''
    <div class="hero">
        <h1>FoxEdge</h1>
        <p>Advanced NFL Betting Insights and Predictions</p>
    </div>
''', unsafe_allow_html=True)

# =======================
# 5. Sidebar Configuration
# =======================
st.sidebar.header("⚙️ Settings")

# Theme Toggle (Simplified as Light/Dark Mode is handled via CSS)
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.experimental_rerun()

st.sidebar.button("🌗 Toggle Theme", on_click=toggle_theme)

# Apply Theme Based on Dark Mode
if st.session_state.dark_mode:
    primary_bg = "#121212"
    secondary_bg = "#1E1E1E"
    primary_text = "#FFFFFF"
    secondary_text = "#B0B0B0"
    accent_color = "#BB86FC"  # Purple
    highlight_color = "#03DAC6"  # Teal
    chart_color = "#BB86FC"
    chart_template = "plotly_dark"
else:
    primary_bg = "#FFFFFF"
    secondary_bg = "#F0F0F0"
    primary_text = "#000000"
    secondary_text = "#4F4F4F"
    accent_color = "#6200EE"
    highlight_color = "#03DAC6"
    chart_color = "#6200EE"
    chart_template = "plotly_white"

# =======================
# 6. Utility Functions
# =======================

# --- Helper Function to Round to Nearest 0.5 ---
def round_to_nearest_half(x):
    return round(x * 2) / 2

# --- Data Loading and Preprocessing ---
@st.cache_data(ttl=3600)
def load_nfl_data():
    try:
        current_year = datetime.now().year
        previous_years = [current_year - 1, current_year - 2]
        schedule = nfl.import_schedules([current_year] + previous_years)
        if schedule.empty:
            st.error(f"No data available for the years: {current_year}, {current_year -1}, {current_year -2}")
            return None
        schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
        return schedule
    except Exception as e:
        st.error(f"Error loading NFL data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def preprocess_data(schedule):
    home_df = schedule[['gameday', 'home_team', 'home_score', 'away_score', 'gametime', 'game_type']].rename(
        columns={'home_team': 'team', 'home_score': 'score', 'away_score': 'opp_score'}
    )
    away_df = schedule[['gameday', 'away_team', 'away_score', 'home_score', 'gametime', 'game_type']].rename(
        columns={'away_team': 'team', 'away_score': 'score', 'home_score': 'opp_score'}
    )
    team_data = pd.concat([home_df, away_df], ignore_index=True)
    team_data.dropna(subset=['score'], inplace=True)
    team_data.sort_values('gameday', inplace=True)  # Ensure chronological order
    team_data.set_index('gameday', inplace=True)
    return team_data

# --- Team Abbreviation Mapping ---
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

# --- Fetch Upcoming Games ---
@st.cache_data(ttl=3600)
def fetch_upcoming_games(schedule):
    if schedule is None:
        return pd.DataFrame()
    schedule['game_datetime'] = pd.to_datetime(schedule['gameday'].astype(str) + ' ' + schedule['gametime'].astype(str), errors='coerce', utc=True)
    now = datetime.now(pytz.UTC)
    weekday = now.weekday()

    # Define target days based on current weekday
    target_days = {3: [3, 6, 0], 6: [6, 0, 3], 0: [0, 3, 6]}.get(weekday, [3, 6, 0])
    upcoming_game_dates = [now + timedelta(days=(d - weekday + 7) % 7) for d in target_days]

    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') &
        (schedule['game_datetime'].dt.date.isin([date.date() for date in upcoming_game_dates]))
    ].sort_values('game_datetime')

    return upcoming_games[['game_datetime', 'home_team', 'away_team']]

# --- Fetch Injury Data ---
@st.cache_data(ttl=3600)
def fetch_injury_data(current_year):
    try:
        injuries = nfl.import_injuries([current_year])
        key_positions = ['QB', 'RB', 'WR', 'OL']
        
        # Debugging: Display available columns in injury_data
        # Comment out the next line in production
        # st.write("Available Columns in Injury Data:", injuries.columns.tolist())
        
        # Define possible column names for player names, including 'full_name'
        possible_player_columns = [
            'player_name', 'name', 'player_full_name', 'playerName', 
            'player_fullName', 'full_name'  # Added 'full_name'
        ]
        
        # Identify and rename the player name column to 'player'
        player_col = None
        for col in possible_player_columns:
            if col in injuries.columns:
                player_col = col
                injuries = injuries.rename(columns={col: 'player'})
                break
        
        if not player_col:
            st.error(f"Injury data does not contain any of the expected player name columns: {possible_player_columns}")
            return pd.DataFrame()
        
        # Check for 'impact_score' column; if missing, assign a default value
        if 'impact_score' not in injuries.columns:
            st.warning("Injury data does not contain 'impact_score'. Assigning default impact score of 10%.")
            injuries['impact_score'] = 10  # Default impact score; adjust as needed
        
        # Filter relevant injuries
        relevant_injuries = injuries[
            injuries['position'].isin(key_positions) & 
            (injuries['report_status'] == 'Out')
        ]
        
        # Parse and filter recent injuries (last 7 days)
        today = datetime.now(pytz.UTC)
        one_week_ago = today - timedelta(days=7)
        relevant_injuries['date_modified'] = pd.to_datetime(relevant_injuries['date_modified'], errors='coerce')
        recent_injuries = relevant_injuries[
            relevant_injuries['date_modified'] >= one_week_ago
        ]
        
        if recent_injuries.empty:
            st.info("No recent injuries found in the last week.")
        
        return recent_injuries
    except Exception as e:
        st.error(f"Error fetching injury data: {str(e)}")
        return pd.DataFrame()

# --- Fetch Weather Data ---
def get_weather_data(location, game_datetime):
    API_KEY = st.session_state.get('weather_api_key', None)
    if not API_KEY:
        # Use default weather conditions if API key is not provided
        return {'temperature': 70, 'humidity': 50, 'precipitation': 0, 'conditions': 'Clear', 'wind': 0}
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
            'conditions': forecast.get('day', {}).get('condition', {}).get('text', 'Clear'),
            'wind': forecast.get('day', {}).get('maxwind_mph', 0)
        }
    except Exception:
        # Return default weather conditions in case of failure
        return {'temperature': 70, 'humidity': 50, 'precipitation': 0, 'conditions': 'Clear', 'wind': 0}

# --- Adjust Team Ratings Based on Injuries ---
def adjust_rating_for_injuries(team, base_rating, injury_data):
    team_injuries = injury_data[injury_data['team'] == team]
    impact_score = 0
    key_players = ['Patrick Mahomes', 'Josh Allen', 'Joe Burrow']  # Add more key players as needed

    for _, row in team_injuries.iterrows():
        position = row['position']
        player = row['player']
        # Base impact based on position
        position_impact = {'QB': 0.15, 'RB': 0.07, 'WR': 0.08, 'OL': 0.05}.get(position, 0.02)
        # Increase impact if key player
        if player in key_players:
            position_impact *= 1.25
        impact_score += position_impact

    adjusted_rating = base_rating * (1 - impact_score)
    return adjusted_rating, round(impact_score * 100, 2)  # Percentage impact

# --- Adjust Team Ratings Based on Weather ---
def adjust_rating_for_weather(base_rating, weather_data):
    # Nonlinear adjustment for wind speed
    wind = weather_data.get('wind', 0)
    if wind > 15:
        wind_penalty = -0.02 * ((wind - 15) ** 2)
    else:
        wind_penalty = 0

    # Other adjustments
    conditions = weather_data.get('conditions', 'Clear')
    precipitation = weather_data.get('precipitation', 0)

    if conditions in ['Rain', 'Snow', 'Blustery', 'Windy']:
        precipitation_penalty = -0.03 if precipitation > 0.5 else -0.02
    else:
        precipitation_penalty = 0

    adjustment = wind_penalty + precipitation_penalty
    adjusted_rating = base_rating * (1 + adjustment)
    return adjusted_rating, round(adjustment * 100, 2)  # Percentage adjustment

# --- Aggregate Team Statistics ---
def aggregate_team_stats(team_data):
    team_stats = team_data.groupby('team').apply(
        lambda x: pd.Series({
            'avg_score': np.average(x['score']),
            'min_score': x['score'].min(),
            'max_score': x['score'].max(),
            'std_dev': x['score'].std(),
            'games_played': x['score'].count(),
            'recent_form': round(x['score'].rolling(window=5).mean().iloc[-1], 2) if len(x) >=5 else round(x['score'].mean(), 2),
            'avg_points_allowed': round(np.average(x['opp_score']), 2),
            'home_avg_score': 0,  # Placeholder for home/away splits
            'away_avg_score': 0,  # Placeholder for home/away splits
            'win_streak': 0,  # Placeholder, to be calculated
            'recent_opponent_avg_score': 0  # Placeholder, to be calculated
        })
    ).to_dict(orient='index')

    # Advanced Feature Engineering
    for team in team_stats.keys():
        team_games = team_data[team_data['team'] == team].sort_index()
        # Calculate Win Streak
        wins = team_games['score'] > team_games['opp_score']
        win_streak = 0
        max_streak = 0
        for win in wins:
            if win:
                win_streak += 1
                if win_streak > max_streak:
                    max_streak = win_streak
            else:
                win_streak = 0
        team_stats[team]['win_streak'] = max_streak

        # Calculate Home/Away Scoring Splits
        home_games = schedule[(schedule['home_team'] == team) & (schedule['game_type'] == 'REG')]
        away_games = schedule[(schedule['away_team'] == team) & (schedule['game_type'] == 'REG')]
        team_stats[team]['home_avg_score'] = round(home_games['home_score'].mean(), 2) if not home_games.empty else team_stats[team]['avg_score']
        team_stats[team]['away_avg_score'] = round(away_games['away_score'].mean(), 2) if not away_games.empty else team_stats[team]['avg_score']

        # Calculate Recent Opponent Average Score
        recent_games = team_games.tail(5)
        recent_opponents = recent_games['opp_score']
        team_stats[team]['recent_opponent_avg_score'] = round(recent_opponents.mean(), 2) if not recent_opponents.empty else team_stats[team]['avg_points_allowed']

        # **Refinement: Cap the Standard Deviation to Reduce Spread**
        max_std_dev = 10  # Define maximum standard deviation
        if team_stats[team]['std_dev'] > max_std_dev:
            team_stats[team]['std_dev'] = max_std_dev

    return team_stats

# --- Train Machine Learning Models ---
@st.cache_resource
def train_ml_models(team_data):
    models = {}
    scalers = {}
    for team in team_data['team'].unique():
        team_scores = team_data[team_data['team'] == team]['score']
        if len(team_scores) < 5:
            continue
        X = team_data[team_data['team'] == team].index.astype(int).values.reshape(-1, 1)
        y = team_scores.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        gbr = GradientBoostingRegressor()
        gbr.fit(X_scaled, y)
        models[team] = gbr
        scalers[team] = scaler
    return models, scalers

# --- Train ARIMA Models ---
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

# --- Prepare Data for Clustering ---
def prepare_clustering_data(team_stats):
    # Convert team_stats dictionary to DataFrame
    stats_df = pd.DataFrame.from_dict(team_stats, orient='index')

    # Select relevant features for clustering
    features = ['avg_score', 'avg_points_allowed', 'std_dev', 'recent_form', 'win_streak']
    clustering_data = stats_df[features]

    # Handle missing values if any
    clustering_data.fillna(clustering_data.mean(), inplace=True)

    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    return scaled_data, stats_df.index.tolist()

# --- Determine Optimal Clusters ---
def determine_optimal_clusters(scaled_data, max_k=10):
    sse = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        sse.append(kmeans.inertia_)

    # Elbow Method: Identify the elbow point where SSE starts to decrease more slowly
    # For automation, one might use the 'kneed' library, but we'll default to K=4 for simplicity
    optimal_k = 4  # Adjust based on your data and domain knowledge
    return optimal_k

# --- Monte Carlo Simulation with Clustering ---
def monte_carlo_simulation_with_clustering(home_team, away_team, clusters, team_cluster_map, spread_adjustment=0, num_simulations=1000, team_stats=None, team_mae_dict=None):
    if home_team not in team_stats or away_team not in team_stats:
        st.error("Team stats not available for selected teams")
        return None, None  # Return None for score_diff_sim

    home_cluster = team_cluster_map.get(home_team)
    away_cluster = team_cluster_map.get(away_team)
    
    # Adjust simulation parameters based on clusters
    cluster_adjustment = 0
    if home_cluster == away_cluster:
        cluster_adjustment = 0.05  # Slight increase in competition
    
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]

    # Base Ratings
    home_rating = home_stats['avg_score'] * 0.5 + home_stats['max_score'] * 0.2 + home_stats['recent_form'] * 0.3
    away_rating = away_stats['avg_score'] * 0.5 + away_stats['max_score'] * 0.2 + away_stats['recent_form'] * 0.3

    # Weighted Averaging based on MAE
    home_weight = 1 / team_mae_dict.get(home_team, 1)
    away_weight = 1 / team_mae_dict.get(away_team, 1)
    combined_home_rating = (home_rating * home_weight) / (home_weight + away_weight)
    combined_away_rating = (away_rating * away_weight) / (home_weight + away_weight)

    # Apply cluster adjustment
    combined_home_rating *= (1 + cluster_adjustment)
    combined_away_rating *= (1 - cluster_adjustment)

    # **Refinement: Introduce a Minimum Standard Deviation to Prevent Zero Variance**
    min_std_dev = 3  # Define minimum standard deviation
    home_std_dev = max(home_stats['std_dev'], min_std_dev)
    away_std_dev = max(away_stats['std_dev'], min_std_dev)

    # Simulations using Truncated Normal Distribution
    home_scores = generate_truncated_normal(combined_home_rating, home_std_dev, 0, 70, num_simulations)
    away_scores = generate_truncated_normal(combined_away_rating, away_std_dev, 0, 70, num_simulations)

    score_diff = home_scores - away_scores
    home_wins = np.sum(score_diff > spread_adjustment)

    results = {
        "Home Win %": round((home_wins / num_simulations) * 100, 2),
        "Away Win %": round(((num_simulations - home_wins) / num_simulations) * 100, 2),
        "Average Total": round_to_nearest_half(np.mean(home_scores + away_scores)),
        "Average Differential": round_to_nearest_half(np.mean(score_diff))  # Predicted Scoring Margin
    }

    return results, score_diff  # Return score_diff along with results

# --- Generate Truncated Normal Distribution ---
def generate_truncated_normal(mean, std, lower, upper, size):
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

# --- Predict Game Outcome ---
def predict_game_outcome(home_team, away_team, game_datetime, use_injury_impact=False, use_weather_impact=False, team_stats=None, injury_data=None, team_stadium_locations=None):
    home_stats = team_stats.get(home_team, {})
    away_stats = team_stats.get(away_team, {})

    if home_stats and away_stats:
        # Base Ratings
        home_rating = home_stats['avg_score'] * 0.5 + home_stats['max_score'] * 0.2 + home_stats['recent_form'] * 0.3
        away_rating = away_stats['avg_score'] * 0.5 + away_stats['max_score'] * 0.2 + away_stats['recent_form'] * 0.3

        # Adjust for Injuries
        if use_injury_impact and injury_data is not None and not injury_data.empty:
            home_rating, home_injury_impact = adjust_rating_for_injuries(home_team, home_rating, injury_data)
            away_rating, away_injury_impact = adjust_rating_for_injuries(away_team, away_rating, injury_data)
        else:
            home_injury_impact = away_injury_impact = 0

        # Adjust for Weather
        if use_weather_impact and team_stadium_locations is not None:
            location = team_stadium_locations.get(home_team, 'Unknown Location')
            weather_data = get_weather_data(location, game_datetime)
            home_rating, home_weather_adjustment = adjust_rating_for_weather(home_rating, weather_data)
            away_rating, away_weather_adjustment = adjust_rating_for_weather(away_rating, weather_data)
        else:
            weather_data = {}
            home_weather_adjustment = away_weather_adjustment = 0

        # Confidence Calculation using Logistic Transformation
        rating_diff = home_rating - away_rating
        log_odds = 0.1 * rating_diff  # Adjustable parameter
        win_probability = 1 / (1 + np.exp(-log_odds))
        confidence = round(win_probability * 100, 2)  # Convert to percentage

        predicted_winner = home_team if home_rating > away_rating else away_team

        return (predicted_winner, rating_diff, confidence, home_rating, away_rating,
                home_injury_impact, away_injury_impact, weather_data)
    else:
        return "Unavailable", "N/A", "N/A", None, None, 0, 0, {}

# --- Bayesian Updating ---
def bayesian_update(team, actual_score, expected_score, team_stats):
    # Simple Bayesian updating: adjust the mean slightly based on actual performance
    alpha = 0.1  # Learning rate
    if team in team_stats:
        team_stats[team]['avg_score'] = (1 - alpha) * team_stats[team]['avg_score'] + alpha * actual_score
    return team_stats

# =======================
# 7. Main Application Logic
# =======================

# Load and preprocess data
schedule = load_nfl_data()
if schedule is None:
    st.stop()

team_data = preprocess_data(schedule)

# Feature Engineering
team_stats = aggregate_team_stats(team_data)

# Prepare Data for Clustering
clustering_data, team_list = prepare_clustering_data(team_stats)
optimal_k = determine_optimal_clusters(clustering_data)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(clustering_data)
clusters = kmeans.labels_

# Map each team to its cluster
team_cluster_map = dict(zip(team_list, clusters))

# Model Training
ml_models, scalers = train_ml_models(team_data)
arima_models = train_arima_models(team_data)

# Fetch Upcoming Games
upcoming_games = fetch_upcoming_games(schedule)

# Fetch Injury Data
current_year = datetime.now().year
injury_data = fetch_injury_data(current_year)

# Team Stadium Locations (Ensure accuracy)
team_stadium_locations = {
    'ARI': 'Glendale, Arizona', 'ATL': 'Atlanta, Georgia', 'BAL': 'Baltimore, Maryland', 'BUF': 'Orchard Park, New York',
    'CAR': 'Charlotte, North Carolina', 'CHI': 'Chicago, Illinois', 'CIN': 'Cincinnati, Ohio', 'CLE': 'Cleveland, Ohio',
    'DAL': 'Arlington, Texas', 'DEN': 'Denver, Colorado', 'DET': 'Detroit, Michigan', 'GB': 'Green Bay, Wisconsin',
    'HOU': 'Houston, Texas', 'IND': 'Indianapolis, Indiana', 'JAX': 'Jacksonville, Florida', 'KC': 'Kansas City, Missouri',
    'LAC': 'Inglewood, California', 'LAR': 'Inglewood, California', 'LV': 'Las Vegas, Nevada', 'MIA': 'Miami Gardens, Florida',
    'MIN': 'Minneapolis, Minnesota', 'NE': 'Foxborough, Massachusetts', 'NO': 'New Orleans, Louisiana',
    'NYG': 'East Rutherford, New Jersey', 'NYJ': 'East Rutherford, New Jersey', 'PHI': 'Philadelphia, Pennsylvania',
    'PIT': 'Pittsburgh, Pennsylvania', 'SEA': 'Seattle, Washington', 'SF': 'Santa Clara, California',
    'TB': 'Tampa, Florida', 'TEN': 'Nashville, Tennessee', 'WAS': 'Landover, Maryland'
}

# Calculate MAE for models using cross-validation on recent data (last 2-3 seasons)
team_mae_dict = {}
tscv = TimeSeriesSplit(n_splits=3)
for team, model in ml_models.items():
    team_scores = team_data[team_data['team'] == team]['score'].values
    if len(team_scores) < 5:
        continue
    # Use only the last 2 seasons for training to prevent overfitting
    recent_games = team_data[team_data['team'] == team].tail(16)  # Approx. 16 games per season
    X = recent_games.index.astype(int).values.reshape(-1, 1)
    y = recent_games['score'].values
    scaler = scalers.get(team)
    if scaler is None:
        continue
    X_scaled = scaler.transform(X)
    errors = []
    for train_index, test_index in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model_fold = GradientBoostingRegressor().fit(X_train, y_train)
        predictions = model_fold.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        errors.append(mae)
    avg_mae = np.mean(errors)
    team_mae_dict[team] = avg_mae

# User Interface for Game Selection
st.markdown('''
    <div class="data-section">
        <h2>Select a Game for Prediction</h2>
    </div>
''', unsafe_allow_html=True)

# Weather API Key Input
with st.sidebar.expander("🔑 Enter WeatherAPI Key (Optional)"):
    st.session_state.weather_api_key = st.text_input("WeatherAPI Key:", type="password", help="Enter your WeatherAPI key to enable weather-based adjustments. Leave blank to use default weather conditions.")

use_injury_impact = st.sidebar.checkbox("Include Injury Impact in Prediction")
use_weather_impact = st.sidebar.checkbox("Include Weather Impact in Prediction")

# Move Betting Line and Total Line Sliders to Sidebar (Global Scope)
with st.sidebar.expander("📉 Betting Lines"):
    spread_adjustment = st.slider("Set Betting Line (Spread)", -20.0, 20.0, 0.0, step=0.5, help="Adjust the spread line to evaluate betting scenarios.")
    total_line = st.slider("Set Total Line", 20.0, 100.0, 50.0, step=0.5, help="Adjust the total points line for over/under evaluation.")

# Define Divisions for Rivalry Games
divisions = {
    'AFC East': ['NE', 'MIA', 'BUF', 'NYJ'],
    'NFC East': ['PHI', 'DAL', 'NYG', 'WAS'],
    'AFC West': ['KC', 'LV', 'DEN', 'LAC'],
    'NFC West': ['SEA', 'SF', 'LAR', 'ARI'],
    'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
    'NFC North': ['GB', 'CHI', 'MIN', 'DET'],
    'AFC South': ['IND', 'JAX', 'HOU', 'TEN'],
    'NFC South': ['TB', 'NO', 'CAR', 'ATL'],
}

# Identify if the game is a rivalry game
def is_rivalry_game(home_team, away_team):
    for division, teams in divisions.items():
        if home_team in teams and away_team in teams:
            return True
    return False

# Creating Tabs
tabs = st.tabs(["🏈 Predictions", "💡 Insights", "📊 Historical Data", "📋 Team Statistics", "⚙️ Settings"])

with tabs[0]:
    # Predictions Content
    if upcoming_games.empty:
        st.warning("No upcoming games available for prediction.")
    else:
        upcoming_games['game_label'] = [
            f"{team_abbrev_mapping.get(row['away_team'], row['away_team'])} at {team_abbrev_mapping.get(row['home_team'], row['home_team'])} ({row['game_datetime'].strftime('%Y-%m-%d %H:%M %Z')})"
            for _, row in upcoming_games.iterrows()
        ]
        game_selection = st.selectbox('Select an upcoming game:', upcoming_games['game_label'])
        selected_game = upcoming_games[upcoming_games['game_label'] == game_selection].iloc[0]
        home_team, away_team, game_datetime = selected_game['home_team'], selected_game['away_team'], selected_game['game_datetime']

        # Run Simulation Button
        if st.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                # Predict Game Outcome
                (predicted_winner, rating_diff, confidence, home_rating, away_rating,
                 home_injury_impact, away_injury_impact, weather_data) = predict_game_outcome(
                    home_team, away_team, game_datetime, use_injury_impact, use_weather_impact,
                    team_stats, injury_data, team_stadium_locations
                )

                if predicted_winner != "Unavailable":
                    # Retrieve team statistics
                    home_stats = team_stats.get(home_team, {})
                    away_stats = team_stats.get(away_team, {})

                    # Display Prediction Results
                    st.markdown(f'''
                        <div class="data-section">
                            <h2>🏈 Predicted Outcome</h2>
                            <div class="prediction-card">
                                <p><strong>{team_abbrev_mapping.get(home_team, home_team)}</strong> vs. <strong>{team_abbrev_mapping.get(away_team, away_team)}</strong></p>
                                <p>Predicted Winner: {team_abbrev_mapping.get(predicted_winner, predicted_winner)}</p>
                                <p>Confidence Level: {confidence}%</p>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

                    # Betting Insights
                    with st.container():
                        st.markdown('''
                            <div class="data-section">
                                <h2>💡 Betting Insights</h2>
                                <div class="summary-card">
                        ''', unsafe_allow_html=True)

                        # Recalculate betting insights based on new lines using Monte Carlo Simulation with Clustering
                        simulation_results, score_diff_sim = monte_carlo_simulation_with_clustering(
                            home_team, away_team, clusters, team_cluster_map,
                            spread_adjustment=spread_adjustment,
                            num_simulations=1000,
                            team_stats=team_stats,
                            team_mae_dict=team_mae_dict
                        )

                        if simulation_results and score_diff_sim is not None:
                            # **Refinement: Use Actual Simulation Data for Confidence Interval**
                            lower_bound = round_to_nearest_half(np.percentile(score_diff_sim, 5))
                            upper_bound = round_to_nearest_half(np.percentile(score_diff_sim, 95))

                            # Determine if it's a rivalry game
                            rivalry = is_rivalry_game(home_team, away_team)
                            if rivalry:
                                confidence_adjusted = round(confidence * 0.85, 2)  # Lower the confidence in divisional games
                            else:
                                confidence_adjusted = confidence

                            # **Additional Refinement: Cap the Confidence Interval Spread**
                            # Optionally, impose a maximum spread to prevent excessively wide intervals
                            max_spread = 30  # Define maximum spread
                            current_spread = upper_bound - lower_bound
                            if current_spread > max_spread:
                                adjustment = (current_spread - max_spread) / 2
                                lower_bound += adjustment
                                upper_bound -= adjustment

                            # Prepare Betting Insights Content
                            betting_insights_md = f'''
                                <p><strong>Spread Bet Analysis:</strong></p>
                                <p>{team_abbrev_mapping.get(home_team, home_team)} Win Percentage: {simulation_results["Home Win %"]}%</p>
                                <p>{team_abbrev_mapping.get(away_team, away_team)} Win Percentage: {simulation_results["Away Win %"]}%</p>
                                
                                <p><strong>Total Bet Analysis:</strong></p>
                                <p>Over {total_line}: {simulation_results["Average Total"]} Points</p>
                                <p>Under {total_line}: {round_to_nearest_half(100 - simulation_results["Home Win %"])}%</p>
                                
                                <p><strong>Predicted Scoring Margin:</strong> {simulation_results["Average Differential"]} points</p>
                                
                                <p><strong>Score Differential 90% Confidence Interval:</strong> {lower_bound} to {upper_bound} points</p>
                            '''

                            st.markdown(betting_insights_md, unsafe_allow_html=True)
                        else:
                            st.warning("Unable to perform betting analysis due to insufficient data.")

                        st.markdown('''
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)

                    # Visualizations
                    with st.container():
                        st.markdown('''
                            <div class="data-section">
                                <h2>📊 Visual Insights</h2>
                        ''', unsafe_allow_html=True)

                        # Probability Distribution of Score Differential
                        fig = px.histogram(
                            score_diff_sim,
                            nbins=30,
                            title="Score Differential Distribution (Home Team - Away Team)",
                            labels={'value': 'Score Differential', 'count': 'Frequency'},
                            opacity=0.75,
                            color_discrete_sequence=[chart_color]
                        )
                        fig.add_vline(x=spread_adjustment, line_dash="dash", line_color="red", annotation_text="Spread Adjustment", annotation_position="top left")
                        fig.update_layout(
                            xaxis_title="Score Differential",
                            yaxis_title="Frequency",
                            template=chart_template
                        )
                        st.plotly_chart(fig, use_container_width=True, key='fig_score_diff_distribution')

                        # Betting Line vs. Prediction Overlay
                        predicted_total = simulation_results["Average Total"]
                        betting_total = total_line  # From user input

                        fig_bet = go.Figure()
                        fig_bet.add_trace(go.Bar(
                            x=["Predicted Total", "Betting Total"],
                            y=[predicted_total, betting_total],
                            name="Totals",
                            marker_color=['#32CD32', '#1E90FF']
                        ))
                        fig_bet.update_layout(title="Predicted Total vs. Betting Total", yaxis_title="Points", template=chart_template)
                        st.plotly_chart(fig_bet, use_container_width=True, key='fig_bet_prediction')

                        # Probability Distribution of Score Differential
                        fig_prob = px.histogram(
                            score_diff_sim,
                            nbins=30,
                            title="Probability Distribution of Score Differential",
                            labels={'value': 'Score Differential', 'count': 'Frequency'},
                            opacity=0.75,
                            color_discrete_sequence=[chart_color]
                        )
                        fig_prob.update_layout(
                            xaxis_title="Score Differential",
                            yaxis_title="Frequency",
                            template=chart_template
                        )
                        st.plotly_chart(fig_prob, use_container_width=True, key='fig_prob_score_diff')

                        st.markdown('''
                            </div>
                        ''', unsafe_allow_html=True)

                    # Team Performance Trends
                    with st.container():
                        st.markdown('''
                            <div class="data-section">
                                <h2>📈 Team Performance Trends</h2>
                                <div style="display: flex; justify-content: center; gap: 20px;">
                        ''', unsafe_allow_html=True)

                        # Home Team Trends
                        st.markdown(f'''
                            <div class="team-card">
                                <h4>{team_abbrev_mapping.get(home_team, home_team)} Trends</h4>
                                <p>Recent Form (Last 5 Games): <strong>{home_stats.get('recent_form', 'N/A')}</strong>
                                    <span class="tooltip">ℹ️
                                        <span class="tooltiptext">Average score over the last 5 games.</span>
                                    </span>
                                </p>
                                <p>Consistency (Std Dev): <strong>{round(home_stats.get('std_dev', 0), 2)}</strong>
                                    <span class="tooltip">ℹ️
                                        <span class="tooltiptext">Standard deviation of scores; lower means more consistent.</span>
                                    </span>
                                </p>
                                <p>Win Streak: <strong>{home_stats.get('win_streak', 0)}</strong>
                                    <span class="tooltip">ℹ️
                                        <span class="tooltiptext">Number of consecutive wins.</span>
                                    </span>
                                </p>
                                <p>Home/Away Scoring Difference: <strong>Home: {home_stats.get('home_avg_score', 'N/A')}, Away: {home_stats.get('away_avg_score', 'N/A')}</strong>
                                    <span class="tooltip">ℹ️
                                        <span class="tooltiptext">Average scores at home vs. away.</span>
                                    </span>
                                </p>
                                <p>Recent Opponent Avg Score: <strong>{home_stats.get('recent_opponent_avg_score', 'N/A')}</strong>
                                    <span class="tooltip">ℹ️
                                        <span class="tooltiptext">Average score of recent opponents.</span>
                                    </span>
                                </p>
                                <p>Tip: A higher recent form score suggests the team is on a good streak, indicating better performance in the upcoming game. Consistency is also key—lower std dev values mean more reliable scoring.</p>
                            </div>
                        ''', unsafe_allow_html=True)

                        # Away Team Trends
                        st.markdown(f'''
                            <div class="team-card">
                                <h4>{team_abbrev_mapping.get(away_team, away_team)} Trends</h4>
                                <p>Recent Form (Last 5 Games): <strong>{away_stats.get('recent_form', 'N/A')}</strong>
                                    <span class="tooltip">ℹ️
                                        <span class="tooltiptext">Average score over the last 5 games.</span>
                                    </span>
                                </p>
                                <p>Consistency (Std Dev): <strong>{round(away_stats.get('std_dev', 0), 2)}</strong>
                                    <span class="tooltip">ℹ️
                                        <span class="tooltiptext">Standard deviation of scores; lower means more consistent.</span>
                                    </span>
                                </p>
                                <p>Win Streak: <strong>{away_stats.get('win_streak', 0)}</strong>
                                    <span class="tooltip">ℹ️
                                        <span class="tooltiptext">Number of consecutive wins.</span>
                                    </span>
                                </p>
                                <p>Home/Away Scoring Difference: <strong>Home: {away_stats.get('home_avg_score', 'N/A')}, Away: {away_stats.get('away_avg_score', 'N/A')}</strong>
                                    <span class="tooltip">ℹ️
                                        <span class="tooltiptext">Average scores at home vs. away.</span>
                                    </span>
                                </p>
                                <p>Recent Opponent Avg Score: <strong>{away_stats.get('recent_opponent_avg_score', 'N/A')}</strong>
                                    <span class="tooltip">ℹ️
                                        <span class="tooltiptext">Average score of recent opponents.</span>
                                    </span>
                                </p>
                                <p>Tip: For betting totals (over/under), look at consistency. Highly consistent teams can make predicting total points easier, while erratic scores suggest less predictable outcomes.</p>
                            </div>
                        ''', unsafe_allow_html=True)

                        st.markdown('''
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)

                    # Player Impact Section
                    with st.container():
                        st.markdown('''
                            <div class="data-section">
                                <h2>🛡️ Player Impact</h2>
                                <div class="prediction-card">
                        ''', unsafe_allow_html=True)

                        key_players = ['Patrick Mahomes', 'Josh Allen', 'Joe Burrow']
                        injured_key_players = injury_data[injury_data['player'].isin(key_players)]

                        if not injured_key_players.empty:
                            for _, row in injured_key_players.iterrows():
                                player = row['player']
                                position = row['position']
                                # Calculate impact score if not already present
                                impact_score = row.get('impact_score', 10)  # Default to 10% if not present
                                st.markdown(f'''
                                    <p><strong>{player} ({position})</strong> is currently <span style="color:red;">Out</span> with an impact score of {impact_score}%.</p>
                                ''', unsafe_allow_html=True)
                        else:
                            st.markdown("<p>No key players are currently injured.</p>", unsafe_allow_html=True)

                        st.markdown('''
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)

                    # Performance Metrics Dashboard
                    with st.container():
                        st.markdown('''
                            <div class="data-section">
                                <h2>📊 Performance Metrics</h2>
                                <div style="display: flex; justify-content: space-around; gap: 20px;">
                        ''', unsafe_allow_html=True)

                        avg_mae = round(np.mean(list(team_mae_dict.values())), 2)
                        simulations = 1000

                        st.markdown(f'''
                            <div class="summary-card">
                                <h4>Prediction Accuracy</h4>
                                <p>{avg_mae} MAE</p>
                            </div>
                        ''', unsafe_allow_html=True)

                        st.markdown(f'''
                            <div class="summary-card">
                                <h4>Number of Simulations</h4>
                                <p>{simulations}</p>
                            </div>
                        ''', unsafe_allow_html=True)

                        st.markdown('''
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)

                    # Downloadable Reports
                    with st.container():
                        st.markdown('''
                            <div class="data-section">
                                <h2>📥 Downloadable Reports</h2>
                                <div class="summary-card">
                        ''', unsafe_allow_html=True)

                        # Prepare Prediction Data
                        prediction_data = {
                            "Predicted Winner": [team_abbrev_mapping.get(predicted_winner, predicted_winner)],
                            "Confidence Level (%)": [confidence],
                            "Average Total Points": [simulation_results["Average Total"]],
                            "Predicted Scoring Margin": [simulation_results["Average Differential"]],  # Added Field
                            "Average Differential": [simulation_results["Average Differential"]],
                            "Score Differential 90% CI Lower": [lower_bound],
                            "Score Differential 90% CI Upper": [upper_bound]
                        }
                        prediction_df = pd.DataFrame(prediction_data)

                        csv = prediction_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Prediction CSV",
                            data=csv,
                            file_name='prediction.csv',
                            mime='text/csv',
                            key='download_prediction_csv'
                        )

                        st.markdown('''
                                </div>
                            </div>
                        ''', unsafe_allow_html=True)

                    # Footer
                    st.markdown(f'''
                        <div class="footer">
                            &copy; {datetime.now().year} FoxEdge. All rights reserved.
                        </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.error("Prediction data unavailable.")

with tabs[1]:
    # Insights Content
    st.markdown('''
        <div class="data-section">
            <h2>💡 Insights</h2>
            <p>Explore deeper insights into team performance, trends, and factors influencing game outcomes.</p>
        </div>
    ''', unsafe_allow_html=True)

    # Display Betting Line vs Prediction Overlay
    st.markdown('''
        <div class="data-section">
            <h3>📊 Betting Line vs. Prediction</h3>
        </div>
    ''', unsafe_allow_html=True)

    # Betting Line vs Prediction Example (Assuming variables are set)
    if 'simulation_results' in locals() and 'spread_adjustment' in locals() and 'total_line' in locals():
        # Predicted vs Betting Total
        fig_bet = go.Figure()
        fig_bet.add_trace(go.Bar(
            x=["Predicted Total", "Betting Total"],
            y=[simulation_results["Average Total"], total_line],
            name="Totals",
            marker_color=['#32CD32', '#1E90FF']
        ))
        fig_bet.update_layout(title="Predicted Total vs. Betting Total", yaxis_title="Points", template=chart_template)
        st.plotly_chart(fig_bet, use_container_width=True, key='insights_fig_bet_prediction')

        # Predicted Winner vs Betting Line
        fig_winner = go.Figure()
        fig_winner.add_trace(go.Bar(
            x=[team_abbrev_mapping.get(home_team, home_team), team_abbrev_mapping.get(away_team, away_team)],
            y=[simulation_results["Home Win %"], simulation_results["Away Win %"]],
            name="Win Probability",
            marker_color=['#32CD32', '#FF4500']
        ))
        fig_winner.update_layout(title="Win Probability", yaxis_title="Percentage", template=chart_template)
        st.plotly_chart(fig_winner, use_container_width=True, key='insights_fig_win_probability')

with tabs[2]:
    # Historical Data Content
    st.markdown('''
        <div class="data-section">
            <h2>📊 Historical Performance</h2>
            <p>View interactive graphs showcasing historical performance metrics for each team.</p>
        </div>
    ''', unsafe_allow_html=True)

    selected_team_history = st.selectbox('Select Team for Historical Performance:', list(team_stats.keys()))

    if selected_team_history:
        team_history = team_data[team_data['team'] == selected_team_history].sort_index()
        if not team_history.empty:
            fig = px.line(
                team_history.reset_index(),
                x='gameday',
                y='score',
                title=f"{team_abbrev_mapping.get(selected_team_history, selected_team_history)} Score Over Time",
                labels={'gameday': 'Date', 'score': 'Score'},
                hover_data={'gameday': '|%B %d, %Y'}
            )
            st.plotly_chart(fig, use_container_width=True, key=f'historical_fig_{selected_team_history}')
        else:
            st.warning("No historical data available for the selected team.")

with tabs[3]:
    # Team Statistics Content
    st.markdown('''
        <div class="data-section">
            <h2>📋 Team Statistics</h2>
            <p>Detailed statistics for all teams, including averages, consistency, and recent form.</p>
        </div>
    ''', unsafe_allow_html=True)

    stats_df = pd.DataFrame.from_dict(team_stats, orient='index')
    stats_df.reset_index(inplace=True)
    stats_df.rename(columns={'index': 'Team'}, inplace=True)

    st.dataframe(stats_df.style.highlight_max(axis=0, color='lightgreen').set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#1E90FF'), ('color', 'white'), ('font-size', '12px')]}
    ]), use_container_width=True)

with tabs[4]:
    # Settings Content
    st.markdown('''
        <div class="data-section">
            <h2>⚙️ Settings</h2>
            <p>Adjust various settings to customize your predictions and insights.</p>
        </div>
    ''', unsafe_allow_html=True)

    with st.expander("🔧 Filters"):
        selected_division = st.selectbox("Select Division:", ["All"] + list(divisions.keys()))
        if selected_division != "All":
            teams_in_division = divisions[selected_division]
            selected_team_filter = st.selectbox("Select Team:", teams_in_division)
        else:
            selected_team_filter = st.selectbox("Select Team:", list(team_stats.keys()))

    with st.expander("📊 Visualization Settings"):
        show_confidence_interval = st.checkbox("Show Confidence Interval", value=True)
        show_rivalry_adjustment = st.checkbox("Highlight Rivalry Games", value=True)

    with st.expander("💾 Data Management"):
        if st.button("Download All Team Statistics"):
            csv_all = stats_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Team Statistics CSV",
                data=csv_all,
                file_name='team_statistics.csv',
                mime='text/csv',
                key='download_team_statistics_csv'
            )

# =======================
# 8. Footer Section
# =======================
st.markdown(f'''
    <div class="footer">
        &copy; {datetime.now().year} FoxEdge. All rights reserved.
    </div>
''', unsafe_allow_html=True)
