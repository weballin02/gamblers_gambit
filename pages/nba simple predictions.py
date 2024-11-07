# Import Libraries
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
import warnings
warnings.filterwarnings('ignore')

# Define season start dates
current_season = '2024-25'
previous_seasons = ['2023-24', '2022-23']
season_weights = {current_season: 1.0, '2023-24': 0.7, '2022-23': 0.5}  # Weighted preference on recent data

# Fetch and Preprocess Game Logs for Multiple Seasons
@st.cache_data
def load_nba_game_logs(seasons):
    """Fetch and preprocess game logs for the specified NBA seasons."""
    all_games = []
    for season in seasons:
        try:
            game_logs = LeagueGameLog(season=season, season_type_all_star='Regular Season', player_or_team_abbreviation='T')
            games = game_logs.get_data_frames()[0]
            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
            games['SEASON'] = season  # Add season label for weighting
            games['WEIGHT'] = season_weights[season]
            all_games.append(games)
        except Exception as e:
            st.error(f"Error loading NBA game logs for {season}: {str(e)}")
    return pd.concat(all_games, ignore_index=True) if all_games else None

# Load game logs for the specified seasons
game_logs = load_nba_game_logs([current_season] + previous_seasons)

# Aggregate Points by Team and Date with Season Weights
def aggregate_team_stats(game_logs):
    """
    Calculate average score, recent form, and consistency for each team using multi-year data,
    with a weighting factor for recent seasons and an emphasis on recent games.
    """
    if game_logs is None:
        return {}

    # Filter data for current season's recent games to calculate "recent form"
    recent_games = game_logs[(game_logs['SEASON'] == current_season) & (game_logs['GAME_DATE'] >= datetime.now() - timedelta(days=30))]

    # Calculate weighted stats for each team
    team_stats = game_logs.groupby('TEAM_ABBREVIATION').apply(
        lambda x: pd.Series({
            'avg_score': np.average(x['PTS'], weights=x['WEIGHT']),
            'min_score': x['PTS'].min(),
            'max_score': x['PTS'].max(),
            'std_dev': np.std(x['PTS']),
            'games_played': x['PTS'].count(),
            'recent_form': recent_games[recent_games['TEAM_ABBREVIATION'] == x.name]['PTS'].mean()
        })
    ).to_dict(orient='index')
    
    return team_stats

# Generate team stats for the season with weights applied
team_stats = aggregate_team_stats(game_logs)

# Calculate Team Ratings
def calculate_team_rating(team_stats):
    """Calculate a rating based on stats for each team, with adjusted weightings."""
    team_ratings = {}
    for team, stats in team_stats.items():
        # Calculate team rating with weighted emphasis on recent form and avg score
        team_rating = (
            stats['avg_score'] * 0.5 + stats['max_score'] * 0.2 + stats['recent_form'] * 0.3
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

# Generate team ratings
team_ratings = calculate_team_rating(team_stats)

# Predict Game Outcome Based on Team Ratings
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
    """Fetch NBA games scheduled for today and tomorrow."""
    today = datetime.now().date()
    next_day = today + timedelta(days=1)
    
    # Fetch games for today and tomorrow
    today_scoreboard = ScoreboardV2(game_date=today.strftime('%Y-%m-%d'))
    tomorrow_scoreboard = ScoreboardV2(game_date=next_day.strftime('%Y-%m-%d'))
    
    try:
        today_games = today_scoreboard.get_data_frames()[0]
        tomorrow_games = tomorrow_scoreboard.get_data_frames()[0]
        combined_games = pd.concat([today_games, tomorrow_games], ignore_index=True)
    except Exception as e:
        st.error(f"Error fetching games: {e}")
        return pd.DataFrame()
    
    # Process games data and add abbreviations for home and away teams
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

# Team Name Mapping for Display Purposes
team_name_mapping = {team['abbreviation']: team['full_name'] for team in nba_teams.get_teams()}

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
