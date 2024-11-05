# Import Libraries
import pandas as pd
import numpy as np
import os
import streamlit as st
from datetime import datetime
from CBBpy import CBB
import warnings
warnings.filterwarnings('ignore')

# Initialize CBBpy API
cbb = CBB()

# Load and Preprocess Data
@st.cache_data
def load_and_preprocess_data(start_date: str, end_date: str):
    """
    Load NCAA basketball data within a specified date range.
    """
    games = cbb.get_games_range(start_date, end_date)
    
    # Extract game scores and metadata
    data = []
    for game in games:
        game_info = cbb.get_game_info(game)
        if 'home_team' in game_info and 'home_score' in game_info:
            data.append({
                'date': game_info['date'],
                'team': game_info['home_team'],
                'PTS': game_info['home_score']
            })
        if 'away_team' in game_info and 'away_score' in game_info:
            data.append({
                'date': game_info['date'],
                'team': game_info['away_team'],
                'PTS': game_info['away_score']
            })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# Set date range for historical data
start_date = "2023-01-01"  # Adjust as needed
end_date = "2023-12-31"
data = load_and_preprocess_data(start_date, end_date)

# Aggregate Points by Team and Date
team_data = data.groupby(['date', 'team'])['PTS'].sum().reset_index()
team_data.set_index('date', inplace=True)

# Aggregate Team Stats
def aggregate_team_stats(team_data):
    """
    Calculate average score, recent form, and consistency for each team.
    """
    team_stats = team_data.groupby('team').agg(
        avg_score=('PTS', 'mean'),
        min_score=('PTS', 'min'),
        max_score=('PTS', 'max'),
        std_dev=('PTS', 'std'),
        games_played=('PTS', 'count'),
        recent_form=('PTS', lambda x: x.tail(5).mean() if len(x) >= 5 else x.mean())
    ).to_dict(orient='index')
    return team_stats

team_stats = aggregate_team_stats(team_data)

# Calculate Team Rating
def calculate_team_rating(team_stats):
    """
    Calculate a rating based on stats for each team.
    """
    team_ratings = {}
    for team, stats in team_stats.items():
        team_rating = (stats['avg_score'] + stats['max_score'] + stats['recent_form']) - (stats['games_played'] * 0.5)
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
    """
    Predict the game outcome based on team ratings.
    """
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

# Fetch Upcoming NCAA Games
@st.cache_data(ttl=3600)
def fetch_upcoming_ncaa_games(date: str):
    """
    Fetch NCAA basketball games scheduled for a specific date.
    """
    game_ids = cbb.get_game_ids(date)
    games = []
    for game_id in game_ids:
        game_info = cbb.get_game_info(game_id)
        if 'home_team' in game_info and 'away_team' in game_info:
            games.append({
                'game_id': game_id,
                'game_date': game_info['date'],
                'home_team': game_info['home_team'],
                'away_team': game_info['away_team']
            })
    return pd.DataFrame(games)

# Set the date for upcoming games
game_date = datetime.now().strftime('%Y-%m-%d')
upcoming_games = fetch_upcoming_ncaa_games(game_date)

# Streamlit UI for Team Prediction
st.title('NCAA Basketball Team Points Prediction')

# Display Game Predictions
st.header('NCAA Game Predictions with Detailed Analysis')

# Create game labels for selection
upcoming_games['game_label'] = [
    f"{row['away_team']} at {row['home_team']} ({row['game_date']})"
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

    st.subheader(f"{home_team} Performance Summary")
    st.write(f"- **Average Score:** {home_stats['avg_score']}")
    st.write(f"- **Recent Form (Last 5 Games):** {home_stats['recent_form']}")
    st.write(f"- **Score Variability (Std Dev):** {home_stats['std_dev']} (Lower indicates more consistency)")
    st.write(f"- **Games Played:** {home_stats['games_played']}")
    st.write(f"- **Overall Rating:** {home_stats['team_rating']}")

    st.subheader(f"{away_team} Performance Summary")
    st.write(f"- **Average Score:** {away_stats['avg_score']}")
    st.write(f"- **Recent Form (Last 5 Games):** {away_stats['recent_form']}")
    st.write(f"- **Score Variability (Std Dev):** {away_stats['std_dev']} (Lower indicates more consistency)")
    st.write(f"- **Games Played:** {away_stats['games_played']}")
    st.write(f"- **Overall Rating:** {away_stats['team_rating']}")

    st.subheader("Betting Insights")
    st.write(f"**{home_team if home_stats['team_rating'] > away_stats['team_rating'] else away_team}** has a higher team rating, indicating a likely advantage.")
    
    if home_stats['std_dev'] < away_stats['std_dev']:
        st.write(f"**{home_team}** has a more consistent scoring pattern, which may reduce risk in betting.")
    elif away_stats['std_dev'] < home_stats['std_dev']:
        st.write(f"**{away_team}** has a more consistent scoring pattern, which may reduce risk in betting.")
    
    st.write("Consider betting on the team with higher rating and consistency, but monitor recent form as it can reflect current team momentum.")
else:
    st.error("Prediction data for one or both teams is unavailable.")
