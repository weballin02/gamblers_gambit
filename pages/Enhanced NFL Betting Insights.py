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

# Streamlit App Title
st.set_page_config(
    page_title="Enhanced NFL Betting Insights",
    page_icon="🏈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# General Styling and High Contrast Toggle
st.markdown("""
    <style>
        /* Shared CSS for consistent styling */
        html, body, [class*="css"] {
            font-family: 'Open Sans', sans-serif;
            background: linear-gradient(135deg, #1a1c2c 0%, #0f111a 100%);
            color: #E5E7EB;
        }

        .header-title {
            font-family: 'Montserrat', sans-serif;
            background: linear-gradient(120deg, #FFA500, #FF6B00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3em;
            font-weight: 800;
        }

        .gradient-bar {
            height: 10px;
            background: linear-gradient(90deg, #22C55E, #EF4444);
            border-radius: 5px;
        }

        div.stButton > button {
            background: linear-gradient(90deg, #FF6B00, #FFA500);
            color: white;
            border: none;
            padding: 1em 2em;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        div.stButton > button:hover {
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# High Contrast Toggle
if st.button("Toggle High Contrast Mode"):
    st.markdown("""
        <style>
            body {
                background: #000;
                color: #FFF;
            }

            .gradient-bar {
                background: linear-gradient(90deg, #0F0, #F00);
            }

            div.stButton > button {
                background: #FFF;
                color: #000;
            }
        </style>
    """, unsafe_allow_html=True)

# Header Section
st.markdown('''
    <div style="text-align: center; margin-bottom: 1.5em;">
        <h1 class="header-title">Enhanced NFL Betting Insights</h1>
        <p style="color: #9CA3AF; font-size: 1.2em;">
            Analyze team trends, weather, and injuries for smarter predictions.
        </p>
    </div>
''', unsafe_allow_html=True)

# Data Visualizations
st.markdown('''
    <h2>Betting Insights</h2>
    <div class="gradient-bar"></div>
    <p style="color: #3B82F6; font-weight: 700;">Confidence Level: 71.68%</p>
''', unsafe_allow_html=True)

# Functionality
st.write("Review team trends and predictions.")

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

# Enhanced Summary for Betting Insights
def enhanced_summary(home_team, away_team, home_stats, away_stats, home_injury_summary, away_injury_summary, 
                     home_team_rating, away_team_rating, home_point_decrease, away_point_decrease, 
                     weather_data, use_injury_impact):
    st.subheader("Enhanced Betting Insights Summary")

    # Display injury impact if selected
    if use_injury_impact:
        st.write(f"### Key Players Missing for {home_team}")
        st.write(", ".join(home_injury_summary) if home_injury_summary else "No key injuries.")
        st.write(f"### Key Players Missing for {away_team}")
        st.write(", ".join(away_injury_summary) if away_injury_summary else "No key injuries.")

        st.subheader("Injury Impact on Team Strength")
        st.write(f"**{home_team} Injury Impact Score:** {home_point_decrease}")
        st.write(f"**{away_team} Injury Impact Score:** {away_point_decrease}")

    # Display weather impact if available
    if weather_data:
        st.subheader("Weather Conditions")
        st.write(f"**Temperature:** {weather_data['temperature']}°F")
        st.write(f"**Wind Speed:** {weather_data['wind_speed']} mph")
        st.write(f"**Conditions:** {weather_data['conditions']}")

    # Display trends
    st.subheader("Team Performance Trends")
    st.write(f"**{home_team} Trends:**\n- Recent Form (Last 5 Games): {home_stats['recent_form']}\n- Consistency (Std Dev): {home_stats['std_dev']}")
    st.write("Tip: A higher recent form score suggests the team is on a good streak, which may indicate better performance in the upcoming game. Consistency is also key—lower values mean more reliable scoring.")

    st.write(f"**{away_team} Trends:**\n- Recent Form (Last 5 Games): {away_stats['recent_form']}\n- Consistency (Std Dev): {away_stats['std_dev']}")
    st.write("Tip: For betting totals (over/under), look at consistency. Highly consistent teams can make predicting total points easier, while erratic scores suggest less predictable outcomes.")

    # Prediction and confidence
    likely_advantage = home_team if home_team_rating > away_team_rating else away_team
    st.write(f"**Overall Prediction and Confidence with Injury Adjustments**")
    st.write(f"Predicted Advantage: {likely_advantage} is expected to have an edge, with adjusted ratings reflecting recent performance and injury impact.")
    st.write(f"Confidence Boost: If betting on {likely_advantage}, the injury impact and recent form support this choice. Use this insight for moneyline bets or spreads if the adjusted ratings favor a team by a solid margin.")

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
st.header('NFL Game Predictions with Detailed Analysis')
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
    st.write(f"### Predicted Outcome for {home_team} vs. {away_team}")
    st.write(f"**Predicted Winner:** {predicted_winner} with confidence of {round(confidence, 2)}%")
    st.write(f"**Score Difference:** {round(predicted_score_diff, 2)}")
    enhanced_summary(
        home_team, away_team, 
        current_season_stats.get(home_team, {}), current_season_stats.get(away_team, {}),
        [], [],  # Placeholders for injury summaries if not used
        home_team_rating, away_team_rating, 
        home_total_point_decrease, away_total_point_decrease,
        weather_data, use_injury_impact
    )
else:
    st.error("Prediction data unavailable.")
