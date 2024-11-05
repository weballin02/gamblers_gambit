# Import Libraries
import pandas as pd
import numpy as np
import os
import streamlit as st
from datetime import datetime, timedelta
from nba_api.stats.endpoints import ScoreboardV2
from nba_api.stats.static import teams as nba_teams
import warnings
warnings.filterwarnings('ignore')

# Define the start of the current 2024 NBA season
current_season_start = datetime(2024, 10, 1)

# Load and Preprocess Data
@st.cache_data
def load_and_preprocess_data(file_path):
    """Load and preprocess NBA data from CSV."""
    usecols = ['date', 'team', 'PTS']
    data = pd.read_csv(file_path, usecols=usecols)
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    return data

# Apply Team Name Mapping
def apply_team_name_mapping(data):
    """Map team abbreviations to full names."""
    team_name_mapping = {
        'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets', 'CHA': 'Charlotte Hornets',
        'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers', 'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets',
        'DET': 'Detroit Pistons', 'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
        'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat',
        'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks',
        'OKC': 'Oklahoma City Thunder', 'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs', 'TOR': 'Toronto Raptors',
        'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
    }
    data['team_abbrev'] = data['team']
    data['team'] = data['team'].map(team_name_mapping)
    data.dropna(subset=['team'], inplace=True)
    return data, team_name_mapping

# Load Data
file_path = 'data/traditional.csv'  # Update to the NBA data CSV file path
data = load_and_preprocess_data(file_path)

# Apply Team Name Mapping
data, team_name_mapping = apply_team_name_mapping(data)

# Aggregate Points by Team and Date
team_data = data.groupby(['date', 'team_abbrev', 'team'])['PTS'].sum().reset_index()
team_data.set_index('date', inplace=True)

# Adjusted function to filter for the current season's data only
def aggregate_team_stats(team_data):
    """
    Calculate average score, recent form, and consistency for each team, 
    restricted to the current 2024 NBA season.
    """
    # Filter team data to include only games from the current season
    current_season_data = team_data[team_data.index >= current_season_start]

    # Aggregate statistics for each team based on current season data
    team_stats = current_season_data.groupby('team_abbrev').agg(
        avg_score=('PTS', 'mean'),
        min_score=('PTS', 'min'),
        max_score=('PTS', 'max'),
        std_dev=('PTS', 'std'),
        games_played=('PTS', 'count'),
        recent_form=('PTS', lambda x: x.tail(5).mean() if len(x) >= 5 else x.mean())
    ).to_dict(orient='index')
    
    return team_stats

# Calculate team ratings with current season's stats
def calculate_team_rating(team_stats):
    """
    Calculate a rating based on stats for each team, with adjusted weightings.
    """
    team_ratings = {}
    for team, stats in team_stats.items():
        # Calculate team rating with minimal impact from games_played
        team_rating = (
            stats['avg_score'] + stats['max_score'] + stats['recent_form']
            - (stats['games_played'] * 0.1)
        )
        
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

# Re-run the aggregate and rating calculations using the filtered data
team_stats = aggregate_team_stats(team_data)
team_ratings = calculate_team_rating(team_stats)

# Predict Outcome Based on Team Ratings
def predict_game_outcome(home_team, away_team):
    """Predict the game outcome based on team ratings."""
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

# Fetch Upcoming NBA Games for Today and Tomorrow
@st.cache_data(ttl=3600)
def fetch_nba_games():
    """Fetch NBA games scheduled for today and tomorrow using nba_api."""
    today = datetime.now().date()
    next_day = today + timedelta(days=1)
    
    # Fetch games for today
    today_scoreboard = ScoreboardV2(game_date=today.strftime('%Y-%m-%d'))
    tomorrow_scoreboard = ScoreboardV2(game_date=next_day.strftime('%Y-%m-%d'))
    
    # Combine today's and tomorrow's games
    try:
        today_games = today_scoreboard.get_data_frames()[0]
        tomorrow_games = tomorrow_scoreboard.get_data_frames()[0]
        combined_games = pd.concat([today_games, tomorrow_games], ignore_index=True)
    except Exception as e:
        st.error(f"Error fetching games: {e}")
        return pd.DataFrame()  # Return empty DataFrame if an error occurs
    
    # If no games are found, notify user
    if combined_games.empty:
        st.info("No games scheduled for today or tomorrow.")
        return pd.DataFrame()
    
    # Process team abbreviations and full names
    nba_team_list = nba_teams.get_teams()
    id_to_abbrev = {team['id']: team['abbreviation'] for team in nba_team_list}

    combined_games['HOME_TEAM_ABBREV'] = combined_games['HOME_TEAM_ID'].map(id_to_abbrev)
    combined_games['VISITOR_TEAM_ABBREV'] = combined_games['VISITOR_TEAM_ID'].map(id_to_abbrev)
    combined_games.dropna(subset=['HOME_TEAM_ABBREV', 'VISITOR_TEAM_ABBREV'], inplace=True)

    return combined_games[['GAME_ID', 'HOME_TEAM_ABBREV', 'VISITOR_TEAM_ABBREV']]

upcoming_games = fetch_nba_games()

# Streamlit UI for Team Prediction
st.title('NBA Team Points Prediction')

# Display Game Predictions
st.header('NBA Game Predictions with Detailed Analysis')

# Create game labels for selection
upcoming_games['game_label'] = [
    f"{team_name_mapping.get(row['VISITOR_TEAM_ABBREV'])} at {team_name_mapping.get(row['HOME_TEAM_ABBREV'])}"
    for _, row in upcoming_games.iterrows()
]

# Let the user select a game and handle possible empty selections
if not upcoming_games.empty:
    game_selection = st.selectbox('Select an upcoming game:', upcoming_games['game_label'])
    selected_game = upcoming_games[upcoming_games['game_label'] == game_selection]

    # Ensure that the selection is valid before accessing the row
    if not selected_game.empty:
        selected_game = selected_game.iloc[0]
        
        home_team = selected_game['HOME_TEAM_ABBREV']
        away_team = selected_game['VISITOR_TEAM_ABBREV']
        
        # Predict Outcome for Selected Game
        predicted_winner, predicted_score_diff = predict_game_outcome(home_team, away_team)

        # Display Prediction Results with Betting Insights
        if predicted_winner != "Unavailable":
            st.write(f"### Predicted Outcome for {team_name_mapping[home_team]} vs. {team_name_mapping[away_team]}")
            st.write(f"**Predicted Winner:** {team_name_mapping[predicted_winner]}")
            st.write(f"**Expected Score Difference:** {predicted_score_diff:.2f}")

            # Detailed Team Stats for Betting Insights
            home_stats = team_ratings.get(home_team, {})
            away_stats = team_ratings.get(away_team, {})

            st.subheader(f"{team_name_mapping[home_team]} Performance Summary")
            st.write(f"- **Average Score:** {home_stats['avg_score']}")
            st.write(f"- **Recent Form (Last 5 Games):** {home_stats['recent_form']}")
            st.write(f"- **Score Variability (Std Dev):** {home_stats['std_dev']} (Lower indicates more consistency)")
            st.write(f"- **Games Played:** {home_stats['games_played']}")
            st.write(f"- **Overall Rating:** {home_stats['team_rating']}")

            st.subheader(f"{team_name_mapping[away_team]} Performance Summary")
            st.write(f"- **Average Score:** {away_stats['avg_score']}")
            st.write(f"- **Recent Form (Last 5 Games):** {away_stats['recent_form']}")
            st.write(f"- **Score Variability (Std Dev):** {away_stats['std_dev']} (Lower indicates more consistency)")
            st.write(f"- **Games Played:** {away_stats['games_played']}")
            st.write(f"- **Overall Rating:** {away_stats['team_rating']}")
        else:
            st.warning("Prediction data for one or both teams is unavailable.")
else:
    st.warning("No games scheduled for today or tomorrow.")
