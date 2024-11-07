# Import Libraries
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

# Streamlit App Title
st.title("Enhanced NFL Betting Insights")
st.markdown("Get detailed insights on NFL team trends and betting opportunities. This page combines multi-season stats, recent form, and consistency to highlight spread, moneyline, and over/under suggestions. Just select a game to dive into specific betting angles.")

# Define Seasons and Weights
current_year = datetime.now().year
previous_years = [current_year - 1, current_year - 2]
season_weights = {current_year: 1.0, current_year - 1: 0.7, current_year - 2: 0.5}  # Higher weight for recent data

# Load and Preprocess Data from Multiple Seasons
@st.cache_data
def load_and_preprocess_data():
    all_team_data = []
    for year in [current_year] + previous_years:
        schedule = nfl.import_schedules([year])
        
        # Add date and weights for seasons
        schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
        schedule['season_weight'] = season_weights[year]
        
        # Split into home and away data and standardize columns
        home_df = schedule[['gameday', 'home_team', 'home_score', 'season_weight']].copy().rename(columns={'home_team': 'team', 'home_score': 'score'})
        away_df = schedule[['gameday', 'away_team', 'away_score', 'season_weight']].copy().rename(columns={'away_team': 'team', 'away_score': 'score'})
        
        # Combine home and away data, filter out null scores
        season_data = pd.concat([home_df, away_df], ignore_index=True)
        season_data.dropna(subset=['score'], inplace=True)
        season_data.set_index('gameday', inplace=True)
        
        all_team_data.append(season_data)

    # Concatenate data for all seasons
    return pd.concat(all_team_data, ignore_index=False)

# Load the team data for training purposes
team_data = load_and_preprocess_data()

# Aggregate Team Stats with Weights (Training on All Seasons)
def aggregate_team_stats(team_data):
    # Calculate team stats across multiple seasons with weights
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

# Train using multi-season aggregated data
team_stats = aggregate_team_stats(team_data)

# Filter Current Season Data Only for Predictions
def get_current_season_stats():
    schedule = nfl.import_schedules([current_year])
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    
    # Separate current season's home and away games
    home_df = schedule[['gameday', 'home_team', 'home_score']].copy().rename(columns={'home_team': 'team', 'home_score': 'score'})
    away_df = schedule[['gameday', 'away_team', 'away_score']].copy().rename(columns={'away_team': 'team', 'away_score': 'score'})
    
    # Combine, filter, and sort by date
    current_season_data = pd.concat([home_df, away_df], ignore_index=True)
    current_season_data.dropna(subset=['score'], inplace=True)
    current_season_data.set_index('gameday', inplace=True)
    current_season_data.sort_index(inplace=True)
    
    return current_season_data

# Load current season data for prediction
current_season_data = get_current_season_stats()

# Calculate current season stats only
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

# Calculate current season stats for all teams
current_season_stats = calculate_current_season_stats()

# Predict Outcome Based on Current Season Data
def predict_game_outcome(home_team, away_team):
    home_stats = current_season_stats.get(home_team, {})
    away_stats = current_season_stats.get(away_team, {})
    
    if home_stats and away_stats:
        # Calculate "overall rating" using only current season stats for consistency
        home_team_rating = home_stats['avg_score'] * 0.5 + home_stats['max_score'] * 0.2 + home_stats['recent_form'] * 0.3
        away_team_rating = away_stats['avg_score'] * 0.5 + away_stats['max_score'] * 0.2 + away_stats['recent_form'] * 0.3
        rating_diff = abs(home_team_rating - away_team_rating)
        
        confidence = min(100, max(0, 50 + rating_diff * 5))  # Confidence based on rating difference
        
        if home_team_rating > away_team_rating:
            predicted_winner = home_team
            predicted_score_diff = home_team_rating - away_team_rating
        elif away_team_rating > home_team_rating:
            predicted_winner = away_team
            predicted_score_diff = away_team_rating - home_team_rating
        else:
            predicted_winner = "Tie"
            predicted_score_diff = 0
        return predicted_winner, predicted_score_diff, confidence, home_team_rating, away_team_rating
    else:
        return "Unavailable", "N/A", "N/A", None, None

# Fetch Upcoming Games
@st.cache_data(ttl=3600)
def fetch_upcoming_games():
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
predicted_winner, predicted_score_diff, confidence, home_team_rating, away_team_rating = predict_game_outcome(home_team, away_team)

# Display Prediction Results with Betting Insights
if predicted_winner != "Unavailable":
    st.write(f"### Predicted Outcome for {home_team} vs. {away_team}")
    st.write(f"**Predicted Winner:** {predicted_winner} with a confidence of {round(confidence, 2)}%")
    st.write(f"**Expected Score Difference:** {round(predicted_score_diff, 2)}")
    
    # Display home team stats based on current season data
    home_stats = current_season_stats.get(home_team, {})
    away_stats = current_season_stats.get(away_team, {})

    st.subheader(f"{home_team} Performance Summary (Current Season)")
    st.write(f"- **Average Score:** {round(home_stats['avg_score'], 2)}")
    st.write(f"- **Recent Form (Last 5 Games):** {round(home_stats['recent_form'], 2)}")
    st.write(f"- **Games Played:** {home_stats['games_played']}")
    st.write(f"- **Consistency (Std Dev):** {round(home_stats['std_dev'], 2)}")
    st.write(f"- **Overall Rating:** {round(home_team_rating, 2)}")

    st.subheader(f"{away_team} Performance Summary (Current Season)")
    st.write(f"- **Average Score:** {round(away_stats['avg_score'], 2)}")
    st.write(f"- **Recent Form (Last 5 Games):** {round(away_stats['recent_form'], 2)}")
    st.write(f"- **Games Played:** {away_stats['games_played']}")
    st.write(f"- **Consistency (Std Dev):** {round(away_stats['std_dev'], 2)}")
    st.write(f"- **Overall Rating:** {round(away_team_rating, 2)}")

    # Enhanced Betting Insights Summary
    st.subheader("Betting Insights")

    # Identify which team has a higher rating and consistency
    likely_advantage = home_team if home_team_rating > away_team_rating else away_team
    st.write(f"**Advantage:** {likely_advantage} has a higher overall rating, suggesting a potential advantage.")

    # Consistency Analysis
    if home_stats['std_dev'] < away_stats['std_dev']:
        st.write(f"**Consistency:** {home_team} has a more consistent scoring pattern (lower standard deviation), which may indicate reliability in expected performance. This can be useful for spread or moneyline bets.")
    elif away_stats['std_dev'] < home_stats['std_dev']:
        st.write(f"**Consistency:** {away_team} has a more consistent scoring pattern (lower standard deviation), making them potentially more reliable. Consider this for spread or moneyline bets.")
    else:
        st.write("**Consistency:** Both teams have similar consistency in scoring, making this matchup less predictable for reliability.")

    # Recent Form and Momentum
    if home_stats['recent_form'] > home_stats['avg_score']:
        st.write(f"**Momentum:** {home_team} is currently exceeding their season average in recent games, suggesting they are in good form. This may indicate positive momentum.")
    else:
        st.write(f"**Momentum:** {home_team} is scoring below their season average recently, suggesting a potential decline in form.")

    if away_stats['recent_form'] > away_stats['avg_score']:
        st.write(f"**Momentum:** {away_team} is also performing above their season average recently, indicating positive momentum.")
    else:
        st.write(f"**Momentum:** {away_team} is performing below their season average in recent games, which may signal a downward trend.")

    # Scoring Trends for Over/Under Bets
    avg_total_score = home_stats['avg_score'] + away_stats['avg_score']
    recent_total_score = home_stats['recent_form'] + away_stats['recent_form']
    st.write(f"**Scoring Trends:** The combined average score of both teams is approximately {round(avg_total_score, 2)} points per game.")
    st.write(f"- **Over/Under Insight:** Based on recent form, the teams are scoring a combined {round(recent_total_score, 2)} points. If this exceeds the set over/under line, an over bet might be worth considering, while a lower score could indicate potential for an under bet.")

    # Spread and Moneyline Suggestions
    if predicted_score_diff > 3:
        st.write(f"**Spread Suggestion:** With an expected score difference of {round(predicted_score_diff, 2)}, the spread bet might favor **{predicted_winner}** if the line is less than this expected margin.")
    else:
        st.write("**Spread Suggestion:** The expected score difference is small, suggesting a close game where a spread bet could be risky.")

    st.write("**Moneyline Insight:** Based on overall ratings and recent form, **{likely_advantage}** may be a good choice for a moneyline bet if they’re performing consistently.")

else:
    st.error("Prediction data for one or both teams is unavailable.")


