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
            --background-gradient-start: #0F2027;
            --background-gradient-end: #203A43;
            --primary-text-color: #ECECEC;
            --heading-text-color: #F5F5F5;
            --accent-color-teal: #2CFFAA;
            --accent-color-purple: #A56BFF;
            --highlight-color: #FF6B6B;
            --font-heading: 'Raleway', sans-serif;
            --font-body: 'Open Sans', sans-serif;
        }

        /* Global Styles */
        body, html {
            background: linear-gradient(135deg, var(--background-gradient-start), var(--background-gradient-end));
            color: var(--primary-text-color);
            font-family: var(--font-body);
            margin: 0;
            padding: 0;
            overflow-x: hidden;
        }

        h1, h2, h3 {
            font-family: var(--font-heading);
            color: var(--heading-text-color);
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
        }

        .hero p {
            font-size: 1.5em;
            margin-bottom: 1em;
            color: #CCCCCC;
        }

        /* Buttons */
        .button {
            background: linear-gradient(45deg, var(--accent-color-teal), var(--accent-color-purple));
            border: none;
            padding: 0.8em 2em;
            color: #FFFFFF;
            font-size: 1.1em;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.3s ease;
            text-decoration: none;
            display: inline-block;
            margin-top: 1em;
        }

        .button:hover {
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
        }

        .data-section p {
            font-size: 1.2em;
            color: #CCCCCC;
            margin-bottom: 2em;
        }

        /* Enhanced Summary Styling */
        .summary-section {
            padding: 2em 1em;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            margin-bottom: 2em;
        }

        .summary-section h3 {
            font-size: 2em;
            margin-bottom: 0.5em;
            color: var(--accent-color-teal);
        }

        .summary-section p {
            font-size: 1.1em;
            color: #E0E0E0;
            line-height: 1.6;
        }

              /* Team Trends Styling Update */
        .team-trends {
            display: flex;
            flex-wrap: wrap;
            gap: 2em;
            justify-content: space-around;  /* Aligns cards neatly side-by-side */
            margin-top: 2em;
        }

        .team-card {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1.5em;
            width: calc(33% - 2em);  /* Each card takes up approximately 1/3rd of the row, with gaps */
            min-width: 300px;         /* Ensure cards maintain a minimum width */
            max-width: 400px;         /* Optionally limit the maximum width */
            text-align: center;
        }

        /* Streamlit Elements */
        .stButton > button {
            background: linear-gradient(45deg, var(--accent-color-teal), var(--accent-color-purple));
            border: none;
            padding: 0.8em 2em;
            color: #FFFFFF;
            font-size: 1.1em;
            border-radius: 30px;
            cursor: pointer;
            transition: transform 0.3s ease;
            margin-top: 1em;
        }

        .stButton > button:hover {
            transform: translateY(-5px);
        }

        .stCheckbox > div {
            padding: 0.5em 0;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 2em 1em;
            color: #999999;
            font-size: 0.9em;
        }

        .footer a {
            color: var(--accent-color-teal);
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
            'games_played': x['score'].count(),
            'avg_points_allowed': np.average(x['points_allowed'], weights=x['season_weight']) if 'points_allowed' in x.columns else 0,  # Calculate average points allowed
            'winning_percentage': (x['score'] > 0).sum() / x['score'].count() * 100 if x['score'].count() > 0 else 0,  # Calculate winning percentage
            'last_5_results': x['score'].tail(5).tolist()  # Get last 5 results
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
            'avg_score': x['score'].mean(),
            'min_score': x['score'].min(),
            'max_score': x['score'].max(),
            'std_dev': x['score'].std(),
            'games_played': x['score'].count(),
            'recent_form': x['score'].tail(5).mean() if len(x) >= 5 else x['score'].mean()
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
    API_KEY = '88H6RKM5HJT8NMDGBFA8ZBM7S'  # Replace with your API key
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

def enhanced_summary(home_team, away_team, home_stats, away_stats, home_team_rating, away_team_rating,
                     home_point_decrease, away_point_decrease, weather_data, use_injury_impact):
    st.markdown('''
        <div class="summary-section">
            <h3>Enhanced Betting Insights Summary</h3>
    ''', unsafe_allow_html=True)

    # Display injury impact if selected
    if use_injury_impact:
        st.markdown(f'''
            <div class="flex-container">
                <div class="impact-card">
                    <p><strong>Injury Impact on Team Strength:</strong></p>
                    <div class="injury-impacts">
                        <div class="injury-impact-item">
                            <span>{home_team} Impact:</span>
                            <strong>{home_point_decrease}</strong>
                        </div>
                        <div class="injury-impact-item">
                            <span>{away_team} Impact:</span>
                            <strong>{away_point_decrease}</strong>
                        </div>
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

    # Display weather impact if available
    if weather_data:
        st.markdown(f'''
            <div class="weather-card">
                <p><strong>Weather Conditions at {team_stadium_locations.get(home_team, 'Unknown Location')}:</strong></p>
                <div class="weather-grid">
                    <div class="weather-item">
                        <span>Temperature:</span>
                        <strong>{weather_data['temperature']}°F</strong>
                    </div>
                    <div class="weather-item">
                        <span>Wind Speed:</span>
                        <strong>{weather_data['wind_speed']} mph</strong>
                    </div>
                    <div class="weather-item">
                        <span>Conditions:</span>
                        <strong>{weather_data['conditions']}</strong>
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

    # Prediction and confidence
    likely_advantage = home_team if home_team_rating > away_team_rating else away_team
    st.markdown(f'''
            <div class="prediction-card">
                <p><strong>Overall Prediction and Confidence:</strong></p>
                <div class="prediction-details">
                    <div class="prediction-item">
                        <span>Predicted Advantage:</span>
                        <strong>{likely_advantage}</strong>
                    </div>
                    <div class="prediction-item">
                        <span>Confidence Level:</span>
                        <strong>{round(abs(home_team_rating - away_team_rating) * 5, 2)}%</strong>
                    </div>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # Display team performance trends side by side
    st.markdown(f'''
        <div class="data-section">
            <h2>Team Performance Comparison</h2>
            <div class="team-comparison">
                <div class="team-card">
                    <h4>{home_team} Trends</h4>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span>Recent Form:</span>
                            <strong>{round(home_stats.get('recent_form', 0), 2)}</strong>
                        </div>
                        <div class="stat-item">
                            <span>Consistency:</span>
                            <strong>{round(home_stats.get('std_dev', 0), 2)}</strong>
                        </div>
                        <div class="stat-item">
                            <span>Avg Points Scored:</span>
                            <strong>{round(home_stats.get('avg_score', 0), 2)}</strong>
                        </div>
                        <div class="stat-item">
                            <span>Avg Points Allowed:</span>
                            <strong>{round(home_stats.get('avg_points_allowed', 0), 2)}</strong>
                        </div>
                        <div class="stat-item">
                            <span>Winning Percentage:</span>
                            <strong>{round(home_stats.get('winning_percentage', 0), 2)}%</strong>
                        </div>
                        <div class="stat-item">
                            <span>Last 5 Results:</span>
                            <strong>{', '.join(home_stats.get('last_5_results', []))}</strong>
                        </div>
                    </div>
                    <p class="trend-tip">Higher recent form suggests better performance potential.</p>
                    <div class="betting-impact">
                        <h5>Betting Impact:</h5>
                        <p>With a predicted score differential of {round(predicted_score_diff, 2)} in favor of {predicted_winner}, consider betting on the spread. If {predicted_winner} is favored by less than {round(predicted_score_diff, 2)}, it may be a good idea to take the points.</p>
                        <p>Based on the predicted total score of {round(total_predicted_score, 2)}, consider betting on the <strong>{'over' if total_predicted_score > 45 else 'under'}</strong> for totals bets.</p>
                    </div>
                </div>
                <div class="team-card">
                    <h4>{away_team} Trends</h4>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span>Recent Form:</span>
                            <strong>{round(away_stats.get('recent_form', 0), 2)}</strong>
                        </div>
                        <div class="stat-item">
                            <span>Consistency:</span>
                            <strong>{round(away_stats.get('std_dev', 0), 2)}</strong>
                        </div>
                        <div class="stat-item">
                            <span>Avg Points Scored:</span>
                            <strong>{round(away_stats.get('avg_score', 0), 2)}</strong>
                        </div>
                        <div class="stat-item">
                            <span>Avg Points Allowed:</span>
                            <strong>{round(away_stats.get('avg_points_allowed', 0), 2)}</strong>
                        </div>
                        <div class="stat-item">
                            <span>Winning Percentage:</span>
                            <strong>{round(away_stats.get('winning_percentage', 0), 2)}%</strong>
                        </div>
                        <div class="stat-item">
                            <span>Last 5 Results:</span>
                            <strong>{', '.join(away_stats.get('last_5_results', []))}</strong>
                        </div>
                    </div>
                    <p class="trend-tip">Lower consistency values indicate more reliable scoring.</p>
                    <div class="betting-impact">
                        <h5>Betting Impact:</h5>
                        <p>With a predicted score differential of {round(predicted_score_diff, 2)} in favor of {predicted_winner}, consider betting on the spread. If {predicted_winner} is favored by less than {round(predicted_score_diff, 2)}, it may be a good idea to take the points.</p>
                        <p>Based on the predicted total score of {round(total_predicted_score, 2)}, consider betting on the <strong>{'over' if total_predicted_score > 45 else 'under'}</strong> for totals bets.</p>
                    </div>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

# Update the CSS section to include new styles for the side-by-side layout
st.markdown('''
    <style>
        /* Previous CSS styles remain the same until team-trends section */
        
        .team-comparison {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 2rem auto;
            max-width: 1200px;
        }
        
        .team-card {
            flex: 1;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1.5rem;
            max-width: 500px;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
        }
        
        .team-card:hover {
            transform: translateY(-5px);
        }
        
        .stats-grid {
            display: grid;
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .stat-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }
        
        .trend-tip {
            font-size: 0.9rem;
            color: var(--accent-color-teal);
            margin-top: 1rem;
            font-style: italic;
        }
        
        .flex-container {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .impact-card, .weather-card, .prediction-card {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 1.5rem;
            width: 100%;
            max-width: 500px;
        }
        
        .injury-impacts, .weather-grid, .prediction-details {
            display: grid;
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .injury-impact-item, .weather-item, .prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }
        
        @media (max-width: 768px) {
            .team-comparison {
                flex-direction: column;
                align-items: center;
            }
            
            .team-card {
                width: 100%;
                margin: 1rem 0;
            }
        }
    </style>
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

# Calculate predicted scores for both teams
home_predicted_score = home_team_rating  # Assuming home_team_rating is the predicted score for the home team
away_predicted_score = away_team_rating  # Assuming away_team_rating is the predicted score for the away team
total_predicted_score = home_predicted_score + away_predicted_score

# Update the predicted outcome section
if predicted_winner != "Unavailable":
    st.markdown(f'''
        <div class="data-section">
            <h2>Predicted Outcome</h2>
            <p><strong>{home_team}</strong> vs. <strong>{away_team}</strong></p>
            <p>Predicted Winner: <strong>{predicted_winner}</strong></p>
            <p>Confidence Level: <strong>{round(confidence, 2)}%</strong></p>
            <p>Expected Score Difference: <strong>{round(predicted_score_diff, 2)}</strong></p>
            <p>Total Predicted Score: <strong>{round(total_predicted_score, 2)}</strong></p>
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
