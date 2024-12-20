# nfl_quant.py

# ===========================
# 1. Import Libraries
# ===========================

import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz

# ===========================
# 2. Streamlit App Configuration
# ===========================

st.set_page_config(
    page_title="FoxEdge - NFL Quantum Predictions",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===========================
# 3. Custom CSS Styling with FoxEdge Colors
# ===========================

st.markdown('''
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&family=Open+Sans:wght@400;600&display=swap');

        /* Root Variables */
        :root {
            --background-gradient-start: #2C3E50; /* Charcoal Dark Gray */
            --background-gradient-end: #1E90FF;   /* Electric Blue */
            --primary-text-color: #FFFFFF;         /* Crisp White */
            --heading-text-color: #F5F5F5;         /* Light Gray */
            --accent-color-teal: #32CD32;          /* Lime Green */
            --accent-color-purple: #FF8C00;        /* Deep Orange */
            --highlight-color: #FFFF33;            /* Neon Yellow */
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

        h1, h2, h3, h4 {
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
            background: linear-gradient(120deg, var(--accent-color-teal), var(--accent-color-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            font-size: 1.5em;
            margin-bottom: 1em;
            color: #CCCCCC; /* Light Gray */
        }

        /* All other CSS remains the same */
    </style>
    ''', unsafe_allow_html=True)

# ===========================
# 4. Hero Section
# ===========================

st.markdown('''
    <div class="hero">
        <h1>FoxEdge</h1>
        <p>NFL Quantum-Inspired Predictions</p>
    </div>
''', unsafe_allow_html=True)

# ===========================
# 5. Data Loading with Caching
# ===========================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_nfl_data():
    try:
        current_year = datetime.now().year
        games = nfl.import_schedules([current_year])
        if games.empty:
            st.error(f"No data available for {current_year}")
            return None
        return games
    except Exception as e:
        st.error(f"Error loading NFL data: {str(e)}")
        return None

# ===========================
# 6. Team Statistics Calculation
# ===========================

def calculate_team_stats():
    games = load_nfl_data()
    if games is None:
        return {}

    # Convert to Eastern Time for consistency
    eastern = pytz.timezone('US/Eastern')
    games['gameday'] = pd.to_datetime(games['gameday']).dt.tz_localize('UTC').dt.tz_convert(eastern)
    
    past_games = games[games['gameday'] < datetime.now(eastern)]
    past_games = past_games.dropna(subset=['home_score', 'away_score'])
    team_stats = {}

    for team in pd.concat([past_games['home_team'], past_games['away_team']]).unique():
        home_games = past_games[past_games['home_team'] == team]
        away_games = past_games[past_games['away_team'] == team]

        recent_home = home_games.sort_values('gameday').tail(5)
        recent_away = away_games.sort_values('gameday').tail(5)
        
        home_scores = home_games['home_score']
        away_scores = away_games['away_score']
        all_scores = pd.concat([home_scores, away_scores])

        if len(all_scores) > 0:
            try:
                n_clusters = min(3, len(all_scores))
                scores_clustered = KMeans(n_clusters=n_clusters).fit_predict(all_scores.values.reshape(-1, 1))
                avg_scores_by_cluster = all_scores.groupby(scores_clustered).mean()
                
                recent_scores = pd.concat([
                    recent_home['home_score'],
                    recent_away['away_score']
                ]).tail(5)
                recent_form = recent_scores.mean() if not recent_scores.empty else all_scores.mean()
                
                team_stats[team] = {
                    'avg_score': round(all_scores.mean(), 2),
                    'std_dev': round(all_scores.std(), 2),
                    'min_score': round(all_scores.min(), 2),
                    'max_score': round(all_scores.max(), 2),
                    'cluster_avg_scores': [round(score, 2) for score in avg_scores_by_cluster.values.tolist()],
                    'recent_form': round(recent_form, 2),
                    'games_played': len(all_scores)
                }
            except Exception as e:
                st.warning(f"Error calculating stats for {team}: {str(e)}")
                continue

    return team_stats

# ===========================
# 7. Upcoming Games Retrieval
# ===========================

@st.cache_data(ttl=3600)
def get_upcoming_games():
    games = load_nfl_data()
    if games is None:
        return pd.DataFrame()

    # Convert schedule to Eastern Time (ET) since NFL games are scheduled in ET
    schedule = games.copy()
    eastern = pytz.timezone('US/Eastern')
    
    # Convert gameday to datetime with Eastern timezone
    schedule['game_datetime'] = pd.to_datetime(schedule['gameday']).dt.tz_localize('UTC').dt.tz_convert(eastern)
    
    # Get current time in Eastern timezone
    now = datetime.now(eastern)
    today_weekday = now.weekday()

    # Set target game days based on the current weekday
    next_game_days = {
        0: [0, 3, 6],  # Monday -> [Monday, Thursday, Sunday]
        1: [3, 6, 0],  # Tuesday -> [Thursday, Sunday, Monday]
        2: [3, 6, 0],  # Wednesday -> [Thursday, Sunday, Monday]
        3: [3, 6, 0],  # Thursday -> [Thursday, Sunday, Monday]
        4: [6, 0, 3],  # Friday -> [Sunday, Monday, Thursday]
        5: [6, 0, 3],  # Saturday -> [Sunday, Monday, Thursday]
        6: [6, 0, 3],  # Sunday -> [Sunday, Monday, Thursday]
    }

    target_days = next_game_days[today_weekday]

    # Calculate upcoming dates considering time of day
    upcoming_game_dates = []
    for target_day in target_days:
        days_ahead = (target_day - today_weekday) % 7
        if days_ahead == 0 and now.hour >= 20:  # After 8 PM ET on game day
            days_ahead = 7  # Move to next week
        target_date = now.date() + timedelta(days=days_ahead)
        upcoming_game_dates.append(target_date)

    # Filter games
    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') &
        (schedule['game_datetime'].dt.date.isin(upcoming_game_dates))
    ].sort_values('game_datetime')

    # Convert game times to local timezone for display
    local_tz = datetime.now().astimezone().tzinfo
    upcoming_games['game_datetime'] = upcoming_games['game_datetime'].dt.tz_convert(local_tz)
    
    # Add formatted game time column for display
    upcoming_games['formatted_time'] = upcoming_games['game_datetime'].dt.strftime('%Y-%m-%d %I:%M %p %Z')
    
    return upcoming_games[['formatted_time', 'home_team', 'away_team']]

# ===========================
# 8. Quantum Monte Carlo Simulation Function
# ===========================

def quantum_monte_carlo_simulation(home_team, away_team, spread_adjustment, num_simulations, team_stats):
    if home_team not in team_stats or away_team not in team_stats:
        st.error("Team stats not available for selected teams")
        return None

    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    home_cluster_scores = np.array(home_stats['cluster_avg_scores'])
    away_cluster_scores = np.array(away_stats['cluster_avg_scores'])
    
    home_score_avg = np.random.choice(home_cluster_scores, num_simulations) + spread_adjustment
    away_score_avg = np.random.choice(away_cluster_scores, num_simulations)
    
    home_form_factor = (home_stats['recent_form'] / home_stats['avg_score']) if home_stats['avg_score'] != 0 else 1
    away_form_factor = (away_stats['recent_form'] / away_stats['avg_score']) if away_stats['avg_score'] != 0 else 1
    
    home_scores = np.maximum(
        home_stats['min_score'],
        np.random.normal(home_score_avg * home_form_factor, home_stats['std_dev'], num_simulations)
    )
    away_scores = np.maximum(
        away_stats['min_score'],
        np.random.normal(away_score_avg * away_form_factor, away_stats['std_dev'], num_simulations)
    )
    
    score_diff = home_scores - away_scores
    home_wins = np.sum(score_diff > 0)
    
    results = {
        f"{home_team} Win Percentage": round(home_wins / num_simulations * 100, 2),
        f"{away_team} Win Percentage": round((num_simulations - home_wins) / num_simulations * 100, 2),
        f"Average {home_team} Score": round(np.mean(home_scores), 2),
        f"Average {away_team} Score": round(np.mean(away_scores), 2),
        "Average Total Score": round(np.mean(home_scores + away_scores), 2),
        f"Score Differential ({home_team} - {away_team})": round(np.mean(score_diff), 2)
    }
    
    return results

# ===========================
# 9. Display Functions
# ===========================

def display_results(results, home_team, away_team):
    if results:
        st.markdown('''
            <div class="results-section">
                <h3>Simulation Results</h3>
                <div class="metric-container">
        ''', unsafe_allow_html=True)

        st.markdown(f'''
            <div class="metric">
                <h4>{home_team} Win Probability</h4>
                <p>{results[f'{home_team} Win Percentage']}%</p>
            </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
            <div class="metric">
                <h4>{away_team} Win Probability</h4>
                <p>{results[f'{away_team} Win Percentage']}%</p>
            </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
            <div class="metric">
                <h4>Average Total Score</h4>
                <p>{results["Average Total Score"]}</p>
            </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
            <div class="metric">
                <h4>Score Differential</h4>
                <p>{results[f'Score Differential ({home_team} - {away_team})']}</p>
            </div>
        ''', unsafe_allow_html=True)

        st.markdown('''
                </div>
            </div>
        ''', unsafe_allow_html=True)

# ===========================
# 10. Summary Table Function
# ===========================

def create_summary_table(all_results):
    summary_data = []
    for game in all_results:
        summary_data.append({
            "Home Team": game['home_team'],
            "Away Team": game['away_team'],
            "Home Win %": game['results'][f"{game['home_team']} Win Percentage"],
            "Away Win %": game['results'][f"{game['away_team']} Win Percentage"],
            "Avg Total Score": game['results']["Average Total Score"],
            "Score Differential": game['results'][f"Score Differential ({game['home_team']} - {game['away_team']})"]
        })
    summary_df = pd.DataFrame(summary_data)
    st.markdown('''
        <div class="data-section">
            <h2>Summary of All Predictions</h2>
        </div>
    ''', unsafe_allow_html=True)
    st.dataframe(summary_df.style.set_properties(**{
        'background-color': 'rgba(255, 255, 255, 0.05)',
        'color': 'var(--primary-text-color)',
        'border-color': 'rgba(255, 255, 255, 0.1)'
    }))

# ===========================
# 11. Main Functionality
# ===========================

# Initialize session state for caching
if 'nfl_team_stats' not in st.session_state:
    st.session_state.nfl_team_stats = calculate_team_stats()

# Sidebar for controls
st.markdown('''
    <div class</antArtifact><div class="controls-section">
        <h3>Simulation Controls</h3>
    ''', unsafe_allow_html=True)

upcoming_games = get_upcoming_games()

if not upcoming_games.empty:
    game_options = [
        f"{row['formatted_time']} - {row['home_team']} vs {row['away_team']}"
        for _, row in upcoming_games.iterrows()
    ]
    selected_game = st.selectbox("Select Game", game_options)
    
    # Parse selected game info
    game_info = selected_game.split(' - ')[1]  # Get the part after the datetime
    home_team = game_info.split(' vs ')[0]
    away_team = game_info.split(' vs ')[1]
    
    spread_adjustment = st.slider(
        "Home Team Spread Adjustment",
        -10.0, 10.0, 0.0, step=0.5,
        help="Positive values favor home team, negative values favor away team"
    )
    
    num_simulations = st.selectbox(
        "Number of Simulations",
        [1000, 10000, 100000],
        help="More simulations = more accurate results but slower processing"
    )
    
    run_simulation = st.button("Run Simulation")
    predict_all = st.button("Predict All Upcoming Games")

st.markdown('</div>', unsafe_allow_html=True)

# Main content area
if run_simulation:
    with st.spinner("Running simulation..."):
        team_stats = st.session_state.nfl_team_stats
        results = quantum_monte_carlo_simulation(
            home_team, away_team, spread_adjustment, num_simulations,
            team_stats
        )
        display_results(results, home_team, away_team)

if predict_all:
    all_results = []
    with st.spinner("Running simulations for all games..."):
        team_stats = st.session_state.nfl_team_stats
        for _, row in upcoming_games.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            game_results = quantum_monte_carlo_simulation(
                home_team, away_team, spread_adjustment, num_simulations,
                team_stats
            )
            all_results.append({
                'home_team': home_team,
                'away_team': away_team,
                'results': game_results
            })
    
    # Display summary table
    create_summary_table(all_results)

    # Display individual game results
    for game in all_results:
        st.markdown(f'''
            <div class="results-section">
                <h3>{game['home_team']} vs {game['away_team']}</h3>
        ''', unsafe_allow_html=True)
        display_results(game['results'], game['home_team'], game['away_team'])
        st.markdown('</div>', unsafe_allow_html=True)

# ===========================
# 12. Footer Section
# ===========================

st.markdown('''
    <div class="footer">
        &copy; 2024 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
