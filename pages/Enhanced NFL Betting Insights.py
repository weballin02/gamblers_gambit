# Import Libraries
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz
import requests
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="FoxEdge - Enhanced NFL Betting Insights",
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
            --background-color: #2C3E50; /* Charcoal Dark Gray */
            --primary-color: #1E90FF; /* Electric Blue */
            --secondary-color: #FF8C00; /* Deep Orange */
            --accent-color: #FF4500; /* Fiery Red */
            --success-color: #32CD32; /* Lime Green */
            --alert-color: #FFFF33; /* Neon Yellow */
            --text-color: #FFFFFF; /* Crisp White */
            --heading-text-color: #F5F5F5; /* Adjusted for contrast */
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

        h1, h2, h3 {
            font-family: var(--font-heading);
            color: var(--primary-color);
        }

        /* Hero Section */
        .hero {
            position: relative;
            text-align: center;
            padding: 6em 1em;
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
            font-size: 4em;
            margin-bottom: 0.2em;
        }

        .hero p {
            font-size: 1.5em;
            margin-bottom: 1em;
            color: #CCCCCC; /* Keep this for contrast */
        }

        /* Buttons */
        .button {
            background: var(--primary-color);
            border: none;
            padding: 1em 2em;
            color: var(--text-color);
            font-size: 1.2em;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.3s ease, background-color 0.3s ease;
            text-decoration: none;
        }

        .button:hover {
            background-color: var(--accent-color); /* Fiery Red */
            transform: translateY(-5px);
        }

        /* Data Section */
        .data-section {
            padding: 4em 1em;
            text-align: center;
        }

        .data-section h2 {
            font-size: 2.5em;
            margin-bottom: 0.5em;
            color: var(--success-color); /* Lime Green for success indicators */
        }

        .data-section p {
            font-size: 1.2em;
            color: #CCCCCC; /* Keep this for contrast */
            margin-bottom: 2em;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2em 1em;
            color: #999999; /* Keep this for contrast */
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
        }
    </style>
''', unsafe_allow_html=True)

# Main Content

# Hero Section
st.markdown('''
    <div class="hero">
        <h1>FoxEdge</h1>
        <p>Enhanced NFL Betting Insights</p>
    </div>
''', unsafe_allow_html=True)

# Functionality

# Data Visualizations and Insights Section
st.markdown('''
    <div class="data-section">
        <h2>NFL Game Predictions with Detailed Analysis</h2>
        <p>Analyze team trends, weather, and injuries for smarter predictions.</p>
    </div>
''', unsafe_allow_html=True)

# Define Seasons and Weights
current_year = datetime.now().year
previous_years = [current_year - 1, current_year - 2]
season_weights = {current_year: 1.0, current_year - 1: 0.7, current_year - 2: 0.5}

# Load and Preprocess Data from Multiple Seasons
@st.cache_data
def load_and_preprocess_data():
    all_team_data = []
    for year in [current_year] + previous_years:
        schedule = nfl.import_schedules([year])
        schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
        schedule['season_weight'] = season_weights[year]
        
        # Split into home and away data and standardize columns
        home_df = schedule[['gameday', 'home_team', 'home_score', 'season_weight']].copy().rename(columns={'home_team': 'team', 'home_score': 'score'})
        away_df = schedule[['gameday', 'away_team', 'away_score', 'season_weight']].copy().rename(columns={'away_team': 'team', 'away_score': 'score'})
        
        season_data = pd.concat([home_df, away_df], ignore_index=True)
        season_data.dropna(subset=['score'], inplace=True)
        season_data.set_index('gameday', inplace=True)
        
        all_team_data.append(season_data)

    return pd.concat(all_team_data, ignore_index=False)

team_data = load_and_preprocess_data()

# Aggregate Team Stats with Weights
def aggregate_team_stats(team_data):
    team_stats = team_data.groupby('team').apply(
        lambda x: pd.Series({
            'avg_score': np.average(x['score'], weights=x['season_weight']),
            'min_score': x['score'].min(),
            'max_score': x['score'].max(),
            'std_dev': x['score'].std(),
            'games_played': x['score'].count()
        })
    ).to_dict(orient='index')
    return team_stats

team_stats = aggregate_team_stats(team_data)

# Filter Current Season Data Only for Predictions
def get_current_season_stats():
    schedule = nfl.import_schedules([current_year])
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    home_df = schedule[['gameday', 'home_team', 'home_score']].copy().rename(columns={'home_team': 'team', 'home_score': 'score'})
    away_df = schedule[['gameday', 'away_team', 'away_score']].copy().rename(columns={'away_team': 'team', 'away_score': 'score'})
    current_season_data = pd.concat([home_df, away_df], ignore_index=True)
    current_season_data.dropna(subset=['score'], inplace=True)
    current_season_data.set_index('gameday', inplace=True)
    current_season_data.sort_index(inplace=True)
    return current_season_data

current_season_data = get_current_season_stats()

# Calculate current season stats for all teams
def calculate_current_season_stats():
    current_season_stats = current_season_data.groupby('team').apply(
        lambda x: pd.Series({
            'avg_score': round(x['score'].mean(), 2),
            'min_score': round(x['score'].min(), 2),
            'max_score': round(x['score'].max(), 2),
            'std_dev': round(x['score'].std(), 2),
            'games_played': x['score'].count(),
            'recent_form': round(x['score'].tail(5).mean(), 2) if len(x) >= 5 else round(x['score'].mean(), 2)
        })
    ).to_dict(orient='index')
    return current_season_stats

current_season_stats = calculate_current_season_stats()

# Injury Data Retrieval Function
@st.cache_data(ttl=3600)
def fetch_injury_data():
    injury_data = nfl.import_injuries([current_year])
    key_positions = ['QB', 'RB', 'WR', 'OL']
    key_injuries = injury_data[(injury_data['position'].isin(key_positions)) & (injury_data['report_status'] == 'Out')]
    today = datetime.now(pytz.UTC)
    one_week_ago = today - timedelta(days=7)
    key_injuries['date_modified'] = pd.to_datetime(key_injuries['date_modified'], errors='coerce')
    recent_injuries = key_injuries[key_injuries['date_modified'] >= one_week_ago]
    return recent_injuries

# Adjust Team Rating Based on Injury Impact
def adjust_rating_for_injuries(team, base_rating, injury_data):
    team_injuries = injury_data[injury_data['team'] == team]
    impact_score = sum({
        'QB': 0.15, 'RB': 0.07, 'WR': 0.08, 'OL': 0.05
    }.get(row['position'], 0.02) for _, row in team_injuries.iterrows())
    point_decrease = round(impact_score * 3, 2)
    adjusted_rating = base_rating * (1 - impact_score)
    return adjusted_rating, point_decrease

# Weather Impact Code Integration
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

# Get weather data function
def get_weather_data(location, game_datetime):
    API_KEY = 'YOUR_API_KEY_HERE'  # Replace with your API key
    date_str = game_datetime.strftime('%Y-%m-%d')
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{date_str}?unitGroup=us&key={API_KEY}&include=current"

    try:
        response = requests.get(url)
        if response.status_code != 200:
            return {'temperature': 70, 'humidity': 50, 'wind_speed': 0, 'precipitation': 0, 'conditions': 'Clear'}
        
        data = response.json().get('currentConditions', {})
        return {
            'temperature': data.get('temp', 70),
            'humidity': data.get('humidity', 50),
            'wind_speed': data.get('windspeed', 0),
            'precipitation': data.get('precipprob', 0),
            'conditions': data.get('conditions', 'Clear')
        }
    except Exception:
        return {'temperature': 70, 'humidity': 50, 'wind_speed': 0, 'precipitation': 0, 'conditions': 'Clear'}

# Function to adjust rating based on weather
def adjust_rating_for_weather(base_rating, weather_data):
    adjustment = (-0.05 if weather_data['wind_speed'] > 15 else 0) + (-0.03 if weather_data['precipitation'] > 50 else 0)
    adjusted_rating = base_rating * (1 + adjustment)
    return adjusted_rating, round(adjustment * 3, 2)

# Function to predict game outcome with optional injury and weather impact
def predict_game_outcome(home_team, away_team, game_datetime, use_injury_impact=False, use_weather_impact=False):
    home_stats = current_season_stats.get(home_team, {})
    away_stats = current_season_stats.get(away_team, {})

    if home_stats and away_stats:
        home_team_rating = (
            home_stats['avg_score'] * 0.5 +
            home_stats['max_score'] * 0.2 +
            home_stats['recent_form'] * 0.3
        )
        away_team_rating = (
            away_stats['avg_score'] * 0.5 +
            away_stats['max_score'] * 0.2 +
            away_stats['recent_form'] * 0.3
        )

        if use_injury_impact:
            injury_data = fetch_injury_data()
            home_team_rating, home_point_decrease = adjust_rating_for_injuries(home_team, home_team_rating, injury_data)
            away_team_rating, away_point_decrease = adjust_rating_for_injuries(away_team, away_team_rating, injury_data)
        else:
            home_point_decrease = away_point_decrease = 0

        if use_weather_impact:
            location = team_stadium_locations.get(home_team, 'Unknown Location')
            weather_data = get_weather_data(location, game_datetime)
            home_team_rating, home_weather_point_decrease = adjust_rating_for_weather(home_team_rating, weather_data)
            away_team_rating, away_weather_point_decrease = adjust_rating_for_weather(away_team_rating, weather_data)
        else:
            weather_data = {}
            home_weather_point_decrease = away_weather_point_decrease = 0

        home_total_point_decrease = home_point_decrease + home_weather_point_decrease
        away_total_point_decrease = away_point_decrease + away_weather_point_decrease

        confidence = min(100, max(0, 50 + abs(home_team_rating - away_team_rating) * 5))
        predicted_winner = home_team if home_team_rating > away_team_rating else away_team

        return (predicted_winner, home_team_rating - away_team_rating, confidence, home_team_rating, away_team_rating,
                home_total_point_decrease, away_total_point_decrease, weather_data)
    else:
        return "Unavailable", "N/A", "N/A", None, None, 0, 0, {}

# Enhanced Summary for Betting Insights
def enhanced_summary(home_team, away_team, home_stats, away_stats, home_team_rating, away_team_rating,
                     home_point_decrease, away_point_decrease, weather_data, use_injury_impact):
    st.markdown('''
        <div class="summary-section">
            <h3>Enhanced Betting Insights Summary</h3>
    ''', unsafe_allow_html=True)

    # Display injury impact if selected
    if use_injury_impact:
        st.markdown(f'''
            <p><strong>Injury Impact on Team Strength:</strong></p>
            <p>{home_team} Injury Impact Score: <strong>{home_point_decrease}</strong></p>
            <p>{away_team} Injury Impact Score: <strong>{away_point_decrease}</strong></p>
        ''', unsafe_allow_html=True)

    # Display weather impact if available
    if weather_data:
        st.markdown(f'''
            <p><strong>Weather Conditions at {team_stadium_locations.get(home_team, 'Unknown Location')}:</strong></p>
            <p>Temperature: <strong>{weather_data['temperature']}°F</strong></p>
            <p>Wind Speed: <strong>{weather_data['wind_speed']} mph</strong></p>
            <p>Conditions: <strong>{weather_data['conditions']}</strong></p>
        ''', unsafe_allow_html=True)

    # Prediction and confidence
    likely_advantage = home_team if home_team_rating > away_team_rating else away_team
    st.markdown(f'''
            <p><strong>Overall Prediction and Confidence:</strong></p>
            <p>Predicted Advantage: <strong>{likely_advantage}</strong> is expected to have an edge.</p>
            <p>Confidence Level: <strong>{round(abs(home_team_rating - away_team_rating) * 5, 2)}%</strong></p>
        </div>
    ''', unsafe_allow_html=True)

    # Display team performance trends
    st.markdown('''
        <div class="data-section">
            <h2>Team Performance Trends</h2>
            <div class="team-trends">
    ''', unsafe_allow_html=True)

    # Home team trends
    st.markdown(f'''
        <div class="team-card">
            <h4>{home_team} Trends</h4>
            <p>Recent Form (Last 5 Games): <strong>{home_stats.get('recent_form', 'N/A')}</strong></p>
            <p>Consistency (Std Dev): <strong>{home_stats.get('std_dev', 'N/A')}</strong></p>
            <p>Tip: A higher recent form score suggests the team is on a good streak, which may indicate better performance in the upcoming game. Consistency is also key—lower values mean more reliable scoring.</p>
        </div>
    ''', unsafe_allow_html=True)

    # Away team trends
    st.markdown(f'''
        <div class="team-card">
            <h4>{away_team} Trends</h4>
            <p>Recent Form (Last 5 Games): <strong>{away_stats.get('recent_form', 'N/A')}</strong></p>
            <p>Consistency (Std Dev): <strong>{away_stats.get('std_dev', 'N/A')}</strong></p>
            <p>Tip: For betting totals (over/under), look at consistency. Highly consistent teams can make predicting total points easier, while erratic scores suggest less predictable outcomes.</p>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
            </div>
        </div>
    ''', unsafe_allow_html=True)

# Fetch upcoming games based on the current day of the week
@st.cache_data(ttl=3600)
def fetch_upcoming_games():
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

upcoming_games = fetch_upcoming_games()

# Streamlit UI for Team Prediction
st.markdown('''
    <div class="data-section">
        <h2>Select a Game for Prediction</h2>
    </div>
''', unsafe_allow_html=True)

use_injury_impact = st.checkbox("Include Injury Impact in Prediction")
use_weather_impact = st.checkbox("Include Weather Impact in Prediction")

upcoming_games['game_label'] = [
    f"{row['away_team']} at {row['home_team']} ({row['game_datetime'].strftime('%Y-%m-%d %H:%M %Z')})"
    for _, row in upcoming_games.iterrows()
]
game_selection = st.selectbox('Select an upcoming game:', upcoming_games['game_label'])
selected_game = upcoming_games[upcoming_games['game_label'] == game_selection].iloc[0]

home_team, away_team, game_datetime = selected_game['home_team'], selected_game['away_team'], selected_game['game_datetime']
(predicted_winner, predicted_score_diff, confidence, home_team_rating, away_team_rating,
 home_total_point_decrease, away_total_point_decrease, weather_data) = predict_game_outcome(
    home_team, away_team, game_datetime, use_injury_impact, use_weather_impact
)

if predicted_winner != "Unavailable":
    st.markdown(f'''
        <div class="data-section">
            <h2>Predicted Outcome</h2>
            <p><strong>{home_team}</strong> vs. <strong>{away_team}</strong></p>
            <p>Predicted Winner: <strong>{predicted_winner}</strong></p>
            <p>Confidence Level: <strong>{round(confidence, 2)}%</strong></p>
            <p>Expected Score Difference: <strong>{round(predicted_score_diff, 2)}</strong></p>
        </div>
    ''', unsafe_allow_html=True)

    enhanced_summary(
        home_team, away_team,
        current_season_stats.get(home_team, {}), current_season_stats.get(away_team, {}),
        home_team_rating, away_team_rating,
        home_total_point_decrease, away_total_point_decrease,
        weather_data, use_injury_impact
    )
else:
    st.error("Prediction data unavailable.")

# Footer
st.markdown('''
    <div class="footer">
        &copy; 2023 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
