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

# Title and Description
st.set_page_config(
    page_title="NFL Game Simulations",
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
        <h1 class="header-title">NFL Game Simulations</h1>
        <p style="color: #9CA3AF; font-size: 1.2em;">
            Run thousands of simulations for smarter betting strategies.
        </p>
    </div>
''', unsafe_allow_html=True)

# Data Visualizations
st.markdown('''
    <h2>Simulation Results</h2>
    <div class="gradient-bar"></div>
    <p style="color: var(--primary-color); font-weight: 700;">GB Win Probability: 68.3% vs CHI: 31.7%</p>
''', unsafe_allow_html=True)

# Functionality
st.write("Customize simulations for upcoming NFL games.")

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

# Header for simulation controls
st.title("NFL Simulation Dashboard")
st.markdown("**Predict outcomes, adjust spreads, and visualize your simulation results.**")

# Section: Simulation Controls
st.header("Simulation Controls")
upcoming_games = get_upcoming_games()

if not upcoming_games.empty:
    # Expander for simulation controls
    with st.expander("Customize Simulation", expanded=True):
        # Dropdown for selecting a game
        game_options = [
            f"{row['game_datetime'].date()} - {row['home_team']} vs {row['away_team']}"
            for _, row in upcoming_games.iterrows()
        ]
        selected_game = st.selectbox("Select Game", game_options, help="Choose a game from the upcoming schedule.")

        # Extract home and away teams
        home_team = selected_game.split(' vs ')[0].split(' - ')[1]
        away_team = selected_game.split(' vs ')[1]

        # Spread adjustment slider
        spread_adjustment = st.slider(
            "Home Team Spread Adjustment",
            -20.0, 20.0, 0.0, step=0.5,
            help="Adjust the spread in favor of the home or away team. Positive values favor the home team."
        )

        # Number of simulations dropdown
        num_simulations = st.selectbox(
            "Number of Simulations",
            [1000, 10000, 100000],
            help="Select the number of Monte Carlo simulations to run. Higher values increase accuracy but take longer."
        )

        # Run simulation button
        run_simulation = st.button("Run Simulation", help="Start the simulation for the selected game.")
else:
    st.error("No upcoming games found in the schedule.")
    run_simulation = False

# Section: Simulation Results
if run_simulation:
    with st.spinner("Running simulation..."):
        # Fetch team statistics
        team_stats = calculate_team_stats()

        # Run Monte Carlo simulation
        results = monte_carlo_simulation(
            home_team, away_team,
            spread_adjustment, num_simulations,
            team_stats
        )

        if results:
            # Display key win probability metrics
            st.subheader("Simulation Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{home_team} Win Probability", f"{results[f'{home_team} Win Percentage']}%")
            with col2:
                st.metric(f"{away_team} Win Probability", f"{results[f'{away_team} Win Percentage']}%")

            # Detailed predictions in a collapsible section
            with st.expander("Detailed Predictions", expanded=False):
                metrics = {k: round(v, 2) if isinstance(v, (float, int)) else v
                           for k, v in results.items() if k != "Simulation Data"}
                st.json(metrics)

            # Visualization
            st.subheader("Visualization")
            fig = create_simulation_visualizations(results, home_team, away_team)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Team statistics
            st.subheader("Team Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{home_team} Recent Stats**")
                st.write({k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in team_stats[home_team].items()})
            with col2:
                st.markdown(f"**{away_team} Recent Stats**")
                st.write({k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in team_stats[away_team].items()})
else:
    st.info("Configure the simulation above and click 'Run Simulation' to view results.")


