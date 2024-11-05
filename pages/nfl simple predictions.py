# Import Libraries
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

# Load and Preprocess Data
@st.cache_data
def load_and_preprocess_data():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])

    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    home_df = schedule[['gameday', 'home_team', 'home_score']].copy().rename(columns={'home_team': 'team', 'home_score': 'score'})
    away_df = schedule[['gameday', 'away_team', 'away_score']].copy().rename(columns={'away_team': 'team', 'away_score': 'score'})

    team_data = pd.concat([home_df, away_df], ignore_index=True)
    team_data.dropna(subset=['score'], inplace=True)
    team_data.set_index('gameday', inplace=True)
    team_data.sort_index(inplace=True)
    
    return team_data

team_data = load_and_preprocess_data()

# Aggregate Team Stats
def aggregate_team_stats(team_data):
    team_stats = team_data.groupby('team').agg(
        avg_score=('score', 'mean'),
        min_score=('score', 'min'),
        max_score=('score', 'max'),
        std_dev=('score', 'std'),
        games_played=('score', 'count'),
        recent_form=('score', lambda x: x.tail(5).mean() if len(x) >= 5 else x.mean())
    ).to_dict(orient='index')
    
    return team_stats

team_stats = aggregate_team_stats(team_data)

# Calculate Team Rating
def calculate_team_rating(team_stats):
    team_ratings = {}
    team_rating_values = []

    for team, stats in team_stats.items():
        team_rating = (stats['avg_score'] + stats['max_score'] + stats['recent_form']) - (stats['games_played'] * 0.5)
        team_rating_values.append(team_rating)
        team_ratings[team] = {
            'avg_score': round(stats['avg_score'], 2),
            'min_score': round(stats['min_score'], 2),
            'max_score': round(stats['max_score'], 2),
            'std_dev': round(stats['std_dev'], 2),
            'recent_form': round(stats['recent_form'], 2),
            'games_played': stats['games_played'],
            'team_rating': round(team_rating, 2)
        }
    
    return team_ratings

team_ratings = calculate_team_rating(team_stats)

# Predict Outcome Based on Team Ratings
def predict_game_outcome(home_team, away_team):
    home_team_rating = team_ratings.get(home_team, {}).get('team_rating', None)
    away_team_rating = team_ratings.get(away_team, {}).get('team_rating', None)

    if home_team_rating is not None and away_team_rating is not None:
        if home_team_rating > away_team_rating:
            predicted_winner = home_team
            predicted_score_diff = home_team_rating - away_team_rating
        elif away_team_rating > home_team_rating:
            predicted_winner = away_team
            predicted_score_diff = away_team_rating - home_team_rating
        else:
            predicted_winner = "Tie"
            predicted_score_diff = 0
    else:
        predicted_winner = "Unavailable"
        predicted_score_diff = "N/A"
    
    return predicted_winner, predicted_score_diff

# Fetch Upcoming Games
@st.cache_data(ttl=3600)
def fetch_upcoming_games():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])
    schedule['game_datetime'] = pd.to_datetime(schedule['gameday'].astype(str) + ' ' + schedule['gametime'].astype(str), errors='coerce', utc=True)
    upcoming_games = schedule[(schedule['game_type'] == 'REG') & (schedule['game_datetime'] >= datetime.now(pytz.UTC))]
    return upcoming_games[['game_id', 'game_datetime', 'home_team', 'away_team']]

upcoming_games = fetch_upcoming_games()

# Streamlit UI for Team Prediction
st.title('Enhanced NFL Team Points Prediction for Betting Insights')

# Display Game Predictions
st.header('NFL Game Predictions with Detailed Analysis')

# Create game labels for selection
upcoming_games['game_label'] = [
    f"{row['away_team']} at {row['home_team']} ({row['game_datetime'].strftime('%Y-%m-%d %H:%M %Z')})"
    for _, row in upcoming_games.iterrows()
]

game_selection = st.selectbox('Select an upcoming game:', upcoming_games['game_label'])
selected_game = upcoming_games[upcoming_games['game_label'] == game_selection].iloc[0]

home_team = selected_game['home_team']
away_team = selected_game['away_team']

# Predict Outcome for Selected Game
predicted_winner, predicted_score_diff = predict_game_outcome(home_team, away_team)

# Display Prediction Results with Betting Insights
if predicted_winner != "Unavailable":
    st.write(f"### Predicted Outcome for {home_team} vs. {away_team}")
    st.write(f"**Predicted Winner:** {predicted_winner}")
    st.write(f"**Expected Score Difference:** {predicted_score_diff:.2f}")
    
    # Detailed Team Stats for Betting Insights
    home_stats = team_ratings.get(home_team, {})
    away_stats = team_ratings.get(away_team, {})

    # Display home team stats
    st.subheader(f"{home_team} Performance Summary")
    st.write(f"- **Average Score:** {home_stats['avg_score']}")
    st.write(f"- **Recent Form (Last 5 Games):** {home_stats['recent_form']}")
    st.write(f"- **Score Variability (Std Dev):** {home_stats['std_dev']} (Lower indicates more consistency)")
    st.write(f"- **Games Played:** {home_stats['games_played']}")
    st.write(f"- **Overall Rating:** {home_stats['team_rating']}")

    # Display away team stats
    st.subheader(f"{away_team} Performance Summary")
    st.write(f"- **Average Score:** {away_stats['avg_score']}")
    st.write(f"- **Recent Form (Last 5 Games):** {away_stats['recent_form']}")
    st.write(f"- **Score Variability (Std Dev):** {away_stats['std_dev']} (Lower indicates more consistency)")
    st.write(f"- **Games Played:** {away_stats['games_played']}")
    st.write(f"- **Overall Rating:** {away_stats['team_rating']}")

    # Betting Insights Summary
    st.subheader("Betting Insights")
    st.write(f"**{home_team if home_stats['team_rating'] > away_stats['team_rating'] else away_team}** has a higher team rating, indicating a likely advantage.")
    
    if home_stats['std_dev'] < away_stats['std_dev']:
        st.write(f"**{home_team}** has a more consistent scoring pattern, which may reduce risk in betting.")
    elif away_stats['std_dev'] < home_stats['std_dev']:
        st.write(f"**{away_team}** has a more consistent scoring pattern, which may reduce risk in betting.")
    
    st.write("Consider betting on the team with higher rating and consistency, but monitor recent form as it can reflect current team momentum.")
else:
    st.error("Prediction data for one or both teams is unavailable.")
