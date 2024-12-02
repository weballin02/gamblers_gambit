# Import Libraries
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="NBA FoxEdge - Enhanced Betting Insights",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Session State for Theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Function to Toggle Theme
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Theme Toggle Button
st.button("🌗 Toggle Theme", on_click=toggle_theme)

# Apply Theme Based on Dark Mode
if st.session_state.dark_mode:
    primary_bg = "#121212"
    primary_text = "#FFFFFF"
    secondary_bg = "#1E1E1E"
    accent_color = "#BB86FC"
    highlight_color = "#03DAC6"
else:
    primary_bg = "#FFFFFF"
    primary_text = "#000000"
    secondary_bg = "#F5F5F5"
    accent_color = "#6200EE"
    highlight_color = "#03DAC6"

# Custom CSS for Novel Design
st.markdown(f"""
    <style>
    /* Global Styles */
    body {{
        background-color: {primary_bg};
        color: {primary_text};
        font-family: 'Roboto', sans-serif;
    }}
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    /* Header Section */
    .header {{
        background-color: {accent_color};
        padding: 3em;
        border-radius: 20px;
        text-align: center;
        color: #FFFFFF;
        margin-bottom: 2em;
        position: relative;
        overflow: hidden;
    }}
    .header::before {{
        content: '';
        background-image: radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.1), transparent);
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        animation: rotation 30s infinite linear;
    }}
    @keyframes rotation {{
        from {{transform: rotate(0deg);}}
        to {{transform: rotate(360deg);}}
    }}
    .header h1 {{
        font-size: 3.5em;
        margin: 0;
        font-weight: bold;
        letter-spacing: -1px;
    }}
    .header p {{
        font-size: 1.5em;
        margin-top: 0.5em;
    }}
    .btn {{
        background-color: {highlight_color};
        color: {primary_text};
        padding: 10px 25px;
        border-radius: 50px;
        text-decoration: none;
        font-size: 1.2em;
        margin-top: 1em;
        display: inline-block;
        transition: background-color 0.3s, transform 0.2s;
    }}
    .btn:hover {{
        background-color: #018786;
        transform: translateY(-5px);
    }}
    /* Prediction Section */
    .prediction-card {{
        background-color: {accent_color};
        padding: 2em;
        border-radius: 15px;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2em;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .prediction-card:hover {{
        transform: translateY(-10px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }}
    .prediction-card h2 {{
        font-size: 2em;
        margin-bottom: 0.5em;
    }}
    .prediction-card p {{
        font-size: 1.2em;
        margin-bottom: 0.5em;
    }}
    /* Data Section */
    .data-section {{
        padding: 2em 1em;
        background-color: {accent_color};
        border-radius: 15px;
        margin-bottom: 2em;
    }}
    .data-section h2 {{
        font-size: 2em;
        margin-bottom: 1em;
    }}
    /* Footer */
    .footer {{
        text-align: center;
        padding: 2em 1em;
        color: {primary_text};
        opacity: 0.6;
        font-size: 0.9em;
    }}
    .footer a {{
        color: {accent_color};
        text-decoration: none;
    }}
    /* Responsive Design */
    @media (max-width: 768px) {{
        .header h1 {{
            font-size: 2.5em;
        }}
        .header p {{
            font-size: 1em;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

# Header Section
def render_header():
    st.markdown(f'''
        <div class="header">
            <h1>NBA FoxEdge</h1>
            <p>Enhanced Betting Insights</p>
        </div>
    ''', unsafe_allow_html=True)

# Functionality

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
            predicted_score_diff = round(home_team_rating - away_team_rating, 2)
        elif away_team_rating > home_team_rating:
            predicted_winner = away_team
            predicted_score_diff = round(away_team_rating - home_team_rating, 2)
        else:
            predicted_winner = "Tie"
            predicted_score_diff = 0
        return predicted_winner, predicted_score_diff, confidence, round(home_team_rating, 2), round(away_team_rating, 2)
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

# Main App Logic
def main():
    render_header()
    st.markdown('<div id="prediction"></div>', unsafe_allow_html=True)

    # Display Game Predictions
    st.markdown(f'''
        <div class="data-section">
            <h2>NBA Game Predictions with Detailed Analysis</h2>
        </div>
    ''', unsafe_allow_html=True)

    # Team Name Mapping for Display Purposes
    team_name_mapping = {team['abbreviation']: team['full_name'] for team in nba_teams.get_teams()}

    # Create game labels for selection
    if not upcoming_games.empty:
        upcoming_games['game_label'] = [
            f"{team_name_mapping.get(row['VISITOR_TEAM_ABBREV'])} at {team_name_mapping.get(row['HOME_TEAM_ABBREV'])}"
            for _, row in upcoming_games.iterrows()
        ]

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
                st.markdown(f'''
                    <div class="prediction-card">
                        <h2>Predicted Outcome</h2>
                        <p><strong>{team_name_mapping[home_team]}</strong> vs. <strong>{team_name_mapping[away_team]}</strong></p>
                        <p><strong>Predicted Winner:</strong> {team_name_mapping[predicted_winner]}</p>
                        <p><strong>Confidence Level:</strong> {round(confidence, 2)}%</p>
                        <p><strong>Expected Score Difference:</strong> {round(predicted_score_diff, 2)}</p>
                        <div class="gradient-bar" style="width: {round(confidence, 2)}%;"></div>
                        <p style="color: {highlight_color}; font-weight: 700;">{team_name_mapping[predicted_winner]} Win Probability: {round(confidence, 2)}%</p>
                    </div>
                ''', unsafe_allow_html=True)

                st.markdown(f'''
                    <div class="data-section">
                        <h2>Team Performance Insights</h2>
                ''', unsafe_allow_html=True)

                # Detailed Team Stats for Betting Insights
                home_stats = st.session_state['current_season_stats'].get(home_team, {})
                away_stats = st.session_state['current_season_stats'].get(away_team, {})

                st.markdown(f'''
                    <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                        <div style="flex: 0 0 45%; margin-bottom: 2em;">
                            <h3>{team_name_mapping[home_team]} Performance Summary</h3>
                            <p><strong>Average Score:</strong> {round(home_stats['avg_score'], 2)}</p>
                            <p><strong>Recent Form (Last 5 Games):</strong> {round(home_stats['recent_form'], 2)}</p>
                            <p><strong>Games Played:</strong> {home_stats['games_played']}</p>
                            <p><strong>Consistency (Std Dev):</strong> {round(home_stats['std_dev'], 2)}</p>
                            <p><strong>Overall Rating:</strong> {round(home_team_rating, 2)}</p>
                        </div>
                        <div style="flex: 0 0 45%; margin-bottom: 2em;">
                            <h3>{team_name_mapping[away_team]} Performance Summary</h3>
                            <p><strong>Average Score:</strong> {round(away_stats['avg_score'], 2)}</p>
                            <p><strong>Recent Form (Last 5 Games):</strong> {round(away_stats['recent_form'], 2)}</p>
                            <p><strong>Games Played:</strong> {away_stats['games_played']}</p>
                            <p><strong>Consistency (Std Dev):</strong> {round(away_stats['std_dev'], 2)}</p>
                            <p><strong>Overall Rating:</strong> {round(away_team_rating, 2)}</p>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)

                # Enhanced Betting Insights Summary
                st.markdown(f'''
                    <div class="prediction-card">
                        <h2>Betting Insights</h2>
                ''', unsafe_allow_html=True)
                likely_advantage = home_team if home_team_rating > away_team_rating else away_team
                st.write(f"**Advantage:** {team_name_mapping[likely_advantage]} has a higher team rating, suggesting a potential edge.")

                if home_stats['std_dev'] < away_stats['std_dev']:
                    st.write(f"**Consistency:** {team_name_mapping[home_team]} is more consistent in scoring, which could reduce risk in betting.")
                elif away_stats['std_dev'] < home_stats['std_dev']:
                    st.write(f"**Consistency:** {team_name_mapping[away_team]} is more consistent, potentially lowering bet risk.")
                else:
                    st.write("**Consistency:** Both teams have similar scoring consistency.")

                avg_total_score = round(home_stats['avg_score'], 2) + round(away_stats['avg_score'], 2)
                recent_total_score = round(home_stats['recent_form'], 2) + round(away_stats['recent_form'], 2)
                st.write(f"**Scoring Trends:** Combined average score is around {avg_total_score}, recent combined score is {recent_total_score}.")

                if predicted_score_diff > 5:
                    st.write(f"**Spread Insight:** Consider betting on **{team_name_mapping[predicted_winner]}** if the spread is favorable.")
                else:
                    st.write("**Spread Insight:** A close game suggests caution for spread betting.")

                st.write(f"**Moneyline Suggestion:** **{team_name_mapping[likely_advantage]}** may be favorable based on rating and consistency.")

                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.warning("Prediction data unavailable for the selected game.")

    else:
        st.warning("No upcoming games found.")

    # Footer
    st.markdown(f'''
        <div class="footer">
            &copy; {datetime.now().year} NBA FoxEdge. All rights reserved.
        </div>
    ''', unsafe_allow_html=True)

# Run the main function
if __name__ == "__main__":
    main()
