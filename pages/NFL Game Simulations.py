import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.auth import hash_password, check_password
from utils.user_database import initialize_database
from utils.database import save_model, get_saved_models, load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz

# Initialize the database for user and model management
initialize_database()

# Set page configuration
st.set_page_config(
    page_title="FoxEdge - NFL Game Simulations",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Restrict access if not logged in
if "user" not in st.session_state:
    st.error("Access denied. Please log in to view this page.")
    st.stop()

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

        /* Simulation Controls */
        .controls-section {
            padding: 2em 1em;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            margin-bottom: 2em;
        }

        .controls-section h3 {
            font-size: 2em;
            margin-bottom: 0.5em;
            color: var(--accent-color-teal);
        }

        .controls-section label {
            font-size: 1.1em;
            color: var(--primary-text-color);
        }

        /* Prediction Results */
        .results-section {
            padding: 2em 1em;
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            margin-bottom: 2em;
        }

        .results-section h3 {
            font-size: 2em;
            margin-bottom: 0.5em;
            color: var(--accent-color-purple);
        }

        .metric-container {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin-top: 1em;
        }

        .metric {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 1em;
            border-radius: 10px;
            margin: 0.5em;
            flex: 1 1 200px;
            text-align: center;
        }

        .metric h4 {
            font-size: 1.2em;
            margin-bottom: 0.3em;
            color: var(--highlight-color);
        }

        .metric p {
            font-size: 1.5em;
            margin: 0;
            color: var(--primary-text-color);
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

        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 2em 1em;
            border-radius: 15px;
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

            .metric-container {
                flex-direction: column;
                align-items: center;
            }

            .metric {
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
        <p>NFL Game Simulations</p>
    </div>
''', unsafe_allow_html=True)

# Functionality

# Data Visualizations and Insights Section
st.markdown('''
    <div class="data-section">
        <h2>Run Thousands of Simulations for Smarter Betting Strategies</h2>
    </div>
''', unsafe_allow_html=True)

# Function to calculate team stats
def calculate_team_stats():
    current_year = datetime.now().year
    games = nfl.import_schedules([current_year])

    past_games = games.dropna(subset=['home_score', 'away_score'])
    team_stats = {}

    for team in pd.concat([past_games['home_team'], past_games['away_team']]).unique():
        home_games = past_games[past_games['home_team'] == team]
        home_scores = home_games['home_score']

        away_games = past_games[past_games['away_team'] == team]
        away_scores = away_games['away_score']

        all_scores = pd.concat([home_scores, away_scores])

        team_stats[team] = {
            'avg_score': all_scores.mean(),
            'std_dev': all_scores.std(),
            'min_score': all_scores.min()
        }

    return team_stats

# Updated function to get upcoming games based on current day
def get_upcoming_games():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])
    schedule['game_datetime'] = pd.to_datetime(schedule['gameday']).dt.tz_localize('UTC')
    now = datetime.now(pytz.UTC)
    today_weekday = now.weekday()

    # Set target game days based on the current weekday
    if today_weekday == 3:  # Thursday
        target_days = [3, 6, 0]
    elif today_weekday == 6:  # Sunday
        target_days = [6, 0, 3]
    elif today_weekday == 0:  # Monday
        target_days = [0, 3, 6]
    else:
        target_days = [3, 6, 0]

    upcoming_game_dates = [
        now + timedelta(days=(d - today_weekday + 7) % 7) for d in target_days
    ]

    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') &
        (schedule['game_datetime'].dt.date.isin([date.date() for date in upcoming_game_dates]))
    ].sort_values('game_datetime')

    return upcoming_games[['game_datetime', 'home_team', 'away_team']]

# Monte Carlo simulation function
def monte_carlo_simulation(home_team, away_team, spread_adjustment, num_simulations, team_stats):
    home_score_avg = team_stats[home_team]['avg_score']
    home_score_std = team_stats[home_team]['std_dev']
    home_min_score = team_stats[home_team]['min_score']

    away_score_avg = team_stats[away_team]['avg_score']
    away_score_std = team_stats[away_team]['std_dev']
    away_min_score = team_stats[away_team]['min_score']

    home_wins, away_wins = 0, 0
    total_home_scores, total_away_scores = [], []

    for _ in range(num_simulations):
        home_score = max(home_min_score, np.random.normal(home_score_avg + spread_adjustment, home_score_std))
        away_score = max(away_min_score, np.random.normal(away_score_avg, away_score_std))

        if home_score > away_score:
            home_wins += 1
        else:
            away_wins += 1

        total_home_scores.append(home_score)
        total_away_scores.append(away_score)

    avg_home_score = np.mean(total_home_scores)
    avg_away_score = np.mean(total_away_scores)
    avg_total_score = avg_home_score + avg_away_score

    return {
        f"{home_team} Win Percentage": round(home_wins / num_simulations * 100, 2),
        f"{away_team} Win Percentage": round(away_wins / num_simulations * 100, 2),
        f"Average {home_team} Score": round(avg_home_score, 2),
        f"Average {away_team} Score": round(avg_away_score, 2),
        "Average Total Score": round(avg_total_score, 2),
        f"Score Differential ({home_team} - {away_team})": round(np.mean(np.array(total_home_scores) - np.array(total_away_scores)), 2),
        "Simulation Data": {
            "home_scores": total_home_scores,
            "away_scores": total_away_scores
        }
    }

# Function to create visualizations of the simulation results
def create_simulation_visualizations(results, home_team, away_team):
    home_scores = results["Simulation Data"]["home_scores"]
    away_scores = results["Simulation Data"]["away_scores"]
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Score Distribution', 'Score Differential Distribution')
    )
    
    # Score distribution
    fig.add_trace(
        go.Histogram(x=home_scores, name=f"{home_team} Scores", opacity=0.75),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=away_scores, name=f"{away_team} Scores", opacity=0.75),
        row=1, col=1
    )
    
    # Score differential distribution
    score_diff = np.array(home_scores) - np.array(away_scores)
    fig.add_trace(
        go.Histogram(x=score_diff, name="Score Differential", opacity=0.75),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text="Simulation Results Analysis",
        showlegend=True
    )
    
    return fig

# Sidebar for controls
st.markdown('''
    <div class="controls-section">
        <h3>Simulation Controls</h3>
''', unsafe_allow_html=True)

upcoming_games = get_upcoming_games()

if not upcoming_games.empty:
    game_options = [
        f"{row['game_datetime'].date()} - {row['home_team']} vs {row['away_team']}"
        for _, row in upcoming_games.iterrows()
    ]
    selected_game = st.selectbox("Select Game", game_options)
    
    # Extract teams from selection
    home_team = selected_game.split(' vs ')[0].split(' - ')[1]
    away_team = selected_game.split(' vs ')[1]
    
    spread_adjustment = st.slider(
        "Home Team Spread Adjustment",
        -10.0, 10.0, 0.0,
        help="Positive values favor home team, negative values favor away team"
    )
    
    num_simulations = st.selectbox(
        "Number of Simulations",
        [1000, 10000, 100000],
        help="More simulations = more accurate results but slower processing"
    )
    
    run_simulation = st.button("Run Simulation")
else:
    st.error("No upcoming games found in the schedule.")
    run_simulation = False

st.markdown('</div>', unsafe_allow_html=True)

# Main content area
if run_simulation:
    with st.spinner("Running simulation..."):
        team_stats = calculate_team_stats()
        results = monte_carlo_simulation(
            home_team, away_team,
            spread_adjustment, num_simulations,
            team_stats
        )
        
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
            
            # Display visualizations
            st.markdown('''
                <div class="data-section">
                    <h2>Visualization</h2>
                </div>
            ''', unsafe_allow_html=True)
            fig = create_simulation_visualizations(results, home_team, away_team)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Display team statistics
            st.markdown('''
                <div class="data-section">
                    <h2>Team Statistics</h2>
                </div>
            ''', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{home_team} Recent Stats**")
                st.write({k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in team_stats[home_team].items()})
            with col2:
                st.write(f"**{away_team} Recent Stats**")
                st.write({k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in team_stats[away_team].items()})

# Footer
st.markdown('''
    <div class="footer">
        &copy; 2023 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
