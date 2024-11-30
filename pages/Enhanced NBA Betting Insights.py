# Import Libraries
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
import warnings
warnings.filterwarnings('ignore')

# Streamlit App Title
st.set_page_config(
    page_title="Enhanced NBA Betting Insights",
    page_icon="🏀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# General Styling and High Contrast Toggle
st.markdown("""
    <style>
        /* Shared CSS for consistent styling */
        html, body, [class*="css"] {
            font-family: 'Open Sans', sans-serif;
            background: var(--background-color);
            color: var(--text-color);
        }

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

        .header-title {
            font-family: 'Montserrat', sans-serif;
            background: linear-gradient(120deg, var(--secondary-color), var(--primary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3em;
            font-weight: 800;
        }

        .gradient-bar {
            height: 10px;
            background: linear-gradient(90deg, var(--success-color), var(--accent-color));
            border-radius: 5px;
        }

        div.stButton > button {
            background: linear-gradient(90deg, var(--secondary-color), var(--primary-color));
            color: var(--text-color);
            border: none;
            padding: 1em 2em;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        div.stButton > button:hover {
            background-color: var(--accent-color); /* Fiery Red */
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
        <h1 class="header-title">Enhanced NBA Betting Insights</h1>
        <p style="color: #9CA3AF; font-size: 1.2em;">
            Analyze NBA team performance with detailed insights for smarter betting decisions.
        </p>
    </div>
''', unsafe_allow_html=True)

# Data Visualizations
st.markdown('''
    <h2>Predicted Outcome</h2>
    <div class="gradient-bar"></div>
    <p style="color: var(--primary-color); font-weight: 700;">Milwaukee Bucks Win Probability: 77.9%</p>
''', unsafe_allow_html=True)

# Functionality
st.write("Select an upcoming game to view predictions.")

# Define season start dates and weights for multi-season training
current_season = '2024-25'
previous_seasons = ['2023-24', '2022-23']
season_weights = {current_season: 1.0, '2023-24': 0.7, '2022-23': 0.5}

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
            games['SEASON'] = season
            games['WEIGHT'] = season_weights[season]
            all_games.append(games)
        except Exception as e:
            st.error(f"Error loading NBA game logs for {season}: {str(e)}")
    return pd.concat(all_games, ignore_index=True) if all_games else None

# Aggregate Team Stats with Season Weights for Training
def aggregate_team_stats(game_logs):
    if game_logs is None:
        return {}
    
    team_stats = game_logs.groupby('TEAM_ABBREVIATION').apply(
        lambda x: pd.Series({
            'avg_score': np.average(x['PTS'], weights=x['WEIGHT']),
            'min_score': x['PTS'].min(),
            'max_score': x['PTS'].max(),
            'std_dev': np.std(x['PTS']),
            'games_played': x['PTS'].count()
        })
    ).to_dict(orient='index')
    return team_stats

# Fetch and Calculate Current Season Stats
@st.cache_data
def load_current_season_logs(season):
    """Fetch and calculate current season logs for specific insights."""
    game_logs = LeagueGameLog(season=season, season_type_all_star='Regular Season', player_or_team_abbreviation='T')
    games = game_logs.get_data_frames()[0]
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    return games

def calculate_current_season_stats(current_season_logs):
    """Calculate stats for current season only."""
    recent_games = current_season_logs[current_season_logs['GAME_DATE'] >= datetime.now() - timedelta(days=30)]
    current_season_stats = current_season_logs.groupby('TEAM_ABBREVIATION').apply(
        lambda x: pd.Series({
            'avg_score': x['PTS'].mean(),
            'min_score': x['PTS'].min(),
            'max_score': x['PTS'].max(),
            'std_dev': np.std(x['PTS']),
            'games_played': x['PTS'].count(),
            'recent_form': recent_games[recent_games['TEAM_ABBREVIATION'] == x.name]['PTS'].mean()
        })
    ).to_dict(orient='index')
    return current_season_stats

# Load data when the button is pressed or on first load
if 'game_logs' not in st.session_state:
    st.session_state['game_logs'] = load_nba_game_logs([current_season] + previous_seasons)
    st.session_state['team_stats'] = aggregate_team_stats(st.session_state['game_logs'])
    st.session_state['current_season_logs'] = load_current_season_logs(current_season)
    st.session_state['current_season_stats'] = calculate_current_season_stats(st.session_state['current_season_logs'])

if st.button("Refresh Data & Predict"):
    with st.spinner("Refreshing data and updating predictions..."):
        st.session_state['game_logs'] = load_nba_game_logs([current_season] + previous_seasons)
        st.session_state['team_stats'] = aggregate_team_stats(st.session_state['game_logs'])
        st.session_state['current_season_logs'] = load_current_season_logs(current_season)
        st.session_state['current_season_stats'] = calculate_current_season_stats(st.session_state['current_season_logs'])
        st.success("Data refreshed and predictions updated.")

# Predict Game Outcome Based on Current Season Data
def predict_game_outcome(home_team, away_team):
    home_stats = st.session_state['current_season_stats'].get(home_team, {})
    away_stats = st.session_state['current_season_stats'].get(away_team, {})

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
        rating_diff = abs(home_team_rating - away_team_rating)
        
        confidence = min(100, max(0, 50 + rating_diff * 5))
        
        if home_team_rating > away_team_rating:
            predicted_winner = home_team
            predicted_score_diff = round(home_team_rating - away_team_rating, 2)  # Rounded to 2 decimal places
        elif away_team_rating > home_team_rating:
            predicted_winner = away_team
            predicted_score_diff = round(away_team_rating - home_team_rating, 2)  # Rounded to 2 decimal places
        else:
            predicted_winner = "Tie"
            predicted_score_diff = 0
        return predicted_winner, predicted_score_diff, confidence, round(home_team_rating, 2), round(away_team_rating, 2)  # Rounded to 2 decimal places
    else:
        return "Unavailable", "N/A", "N/A", None, None

# Fetch Upcoming Games
@st.cache_data(ttl=3600)
def fetch_nba_games():
    today = datetime.now().date()
    next_day = today + timedelta(days=1)
    today_scoreboard = ScoreboardV2(game_date=today.strftime('%Y-%m-%d'))
    tomorrow_scoreboard = ScoreboardV2(game_date=next_day.strftime('%Y-%m-%d'))
    
    try:
        today_games = today_scoreboard.get_data_frames()[0]
        tomorrow_games = tomorrow_scoreboard.get_data_frames()[0]
        combined_games = pd.concat([today_games, tomorrow_games], ignore_index=True)
    except Exception as e:
        st.error(f"Error fetching games: {e}")
        return pd.DataFrame()
    
    nba_team_list = nba_teams.get_teams()
    id_to_abbrev = {team['id']: team['abbreviation'] for team in nba_team_list}
    
    combined_games['HOME_TEAM_ABBREV'] = combined_games['HOME_TEAM_ID'].map(id_to_abbrev)
    combined_games['VISITOR_TEAM_ABBREV'] = combined_games['VISITOR_TEAM_ID'].map(id_to_abbrev)
    combined_games.dropna(subset=['HOME_TEAM_ABBREV', 'VISITOR_TEAM_ABBREV'], inplace=True)
    
    return combined_games[['GAME_ID', 'HOME_TEAM_ABBREV', 'VISITOR_TEAM_ABBREV']]

upcoming_games = fetch_nba_games()

# Streamlit UI for Team Prediction
st.title('Enhanced NBA Team Points Prediction for Betting Insights')

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

    if not selected_game.empty:
        selected_game = selected_game.iloc[0]
        
        home_team = selected_game['HOME_TEAM_ABBREV']
        away_team = selected_game['VISITOR_TEAM_ABBREV']
        
        # Predict Outcome for Selected Game
        predicted_winner, predicted_score_diff, confidence, home_team_rating, away_team_rating = predict_game_outcome(home_team, away_team)

        # Display Prediction Results with Enhanced Betting Insights
        if predicted_winner != "Unavailable":
            st.write(f"### Predicted Outcome for {team_name_mapping[home_team]} vs. {team_name_mapping[away_team]}")
            st.write(f"**Predicted Winner:** {team_name_mapping[predicted_winner]} with confidence: {round(confidence, 2)}%")
            st.write(f"**Expected Score Difference:** {round(predicted_score_diff, 2)}")

            # Detailed Team Stats for Betting Insights
            home_stats = st.session_state['current_season_stats'].get(home_team, {})
            away_stats = st.session_state['current_season_stats'].get(away_team, {})

            st.subheader(f"{team_name_mapping[home_team]} Performance Summary (Current Season)")
            st.write(f"- **Average Score:** {round(home_stats['avg_score'], 2)}")
            st.write(f"- **Recent Form (Last 5 Games):** {round(home_stats['recent_form'], 2)}")
            st.write(f"- **Games Played:** {home_stats['games_played']}")
            st.write(f"- **Consistency (Std Dev):** {round(home_stats['std_dev'], 2)}")
            st.write(f"- **Overall Rating:** {round(home_team_rating, 2)}")

            st.subheader(f"{team_name_mapping[away_team]} Performance Summary (Current Season)")
            st.write(f"- **Average Score:** {round(away_stats['avg_score'], 2)}")
            st.write(f"- **Recent Form (Last 5 Games):** {round(away_stats['recent_form'], 2)}")
            st.write(f"- **Games Played:** {away_stats['games_played']}")
            st.write(f"- **Consistency (Std Dev):** {round(away_stats['std_dev'], 2)}")
            st.write(f"- **Overall Rating:** {round(away_team_rating, 2)}")

            # Enhanced Betting Insights Summary
            st.subheader("Betting Insights")
            likely_advantage = home_team if home_team_rating > away_team_rating else away_team
            st.write(f"**Advantage:** {team_name_mapping[likely_advantage]} has a higher team rating, suggesting a potential edge.")
            
            if home_stats['std_dev'] < away_stats['std_dev']:
                st.write(f"**Consistency:** {team_name_mapping[home_team]} is more consistent in scoring, which could reduce risk in betting.")
            elif away_stats['std_dev'] < home_stats['std_dev']:
                st.write(f"**Consistency:** {team_name_mapping[away_team]} is more consistent, potentially lowering bet risk.")
            else:
                st.write("**Consistency:** Both teams have similar scoring consistency.")

            # Momentum and Scoring Trends
            avg_total_score = round(home_stats['avg_score'], 2) + round(away_stats['avg_score'], 2)
            recent_total_score = round(home_stats['recent_form'], 2) + round(away_stats['recent_form'], 2)
            st.write(f"**Scoring Trends:** Combined average score is around {avg_total_score}, recent combined score is {recent_total_score}.")
            
            if predicted_score_diff > 5:
                st.write(f"**Spread Insight:** Consider betting on **{team_name_mapping[predicted_winner]}** if the spread is favorable.")
            else:
                st.write("**Spread Insight:** A close game suggests caution for spread betting.")
                
            st.write(f"**Moneyline Suggestion:** **{team_name_mapping[likely_advantage]}** may be favorable based on rating and consistency.")

