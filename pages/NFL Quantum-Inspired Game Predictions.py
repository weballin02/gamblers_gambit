import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz

# Streamlit App Title
st.set_page_config(
    page_title="NFL Quantum Predictions",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# General Styling and High Contrast Toggle
st.markdown("""
    <style>
        /* Include shared CSS here */
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
        <h1 class="header-title">NFL Quantum-Inspired Predictions</h1>
        <p style="color: #9CA3AF; font-size: 1.2em;">
            Leverage quantum simulations for smarter betting strategies.
        </p>
    </div>
''', unsafe_allow_html=True)

# Data Visualizations
st.markdown('''
    <h2>Simulation Results</h2>
    <div class="gradient-bar"></div>
    <p style="color: #3B82F6; font-weight: 700;">Win Probability: GB 68.3% vs CHI 31.7%</p>
''', unsafe_allow_html=True)

# Functionality
st.write("Run simulations and explore probabilities.")

# Cache the data loading to improve performance
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

def calculate_team_stats():
    games = load_nfl_data()
    if games is None:
        return {}

    past_games = games.dropna(subset=['home_score', 'away_score'])
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

@st.cache_data(ttl=3600)
def get_upcoming_games():
    games = load_nfl_data()
    if games is None:
        return pd.DataFrame()

    schedule = games.copy()
    schedule['game_datetime'] = pd.to_datetime(schedule['gameday']).dt.tz_localize('UTC')
    now = datetime.now().astimezone(pytz.UTC)
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

def display_results(results, home_team, away_team):
    if results:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{home_team} Win Probability", f"{results[f'{home_team} Win Percentage']:.2f}%")
        with col2:
            st.metric(f"{away_team} Win Probability", f"{results[f'{away_team} Win Percentage']:.2f}%")

        st.subheader("Detailed Predictions")
        metrics = {k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in results.items()}
        st.json(metrics)

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
    st.write("### Summary of All Predictions")
    st.dataframe(summary_df)

# Streamlit UI
st.title("NFL Game Prediction System")
st.markdown("### Using Quantum-Inspired Monte Carlo Simulation")

# Initialize session state for caching
if 'nfl_team_stats' not in st.session_state:
    st.session_state.nfl_team_stats = calculate_team_stats()

# Sidebar for controls
with st.sidebar:
    st.header("Simulation Controls")
    upcoming_games = get_upcoming_games()

    if not upcoming_games.empty:
        game_options = [
            f"{row['game_datetime'].date()} - {row['home_team']} vs {row['away_team']}"
            for _, row in upcoming_games.iterrows()
        ]
        selected_game = st.selectbox("Select Game", game_options)
        
        home_team = selected_game.split(' vs ')[0].split(' - ')[1]
        away_team = selected_game.split(' vs ')[1]
        
        spread_adjustment = st.slider(
            "Home Team Spread Adjustment",
            -10.0, 10.0, 0.0, step=0.5
        )
        
        num_simulations = st.selectbox(
            "Number of Simulations",
            [1000, 10000, 100000]
        )
        
        run_simulation = st.button("Run Simulation")
        predict_all = st.button("Predict All Upcoming Games")

if run_simulation:
    with st.spinner("Running simulation..."):
        results = quantum_monte_carlo_simulation(
            home_team, away_team, spread_adjustment, num_simulations,
            st.session_state.nfl_team_stats
        )
        display_results(results, home_team, away_team)

if predict_all:
    all_results = []
    with st.spinner("Running simulations for all games..."):
        for _, row in upcoming_games.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            game_results = quantum_monte_carlo_simulation(
                home_team, away_team, spread_adjustment, num_simulations,
                st.session_state.nfl_team_stats
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
        with st.expander(f"{game['home_team']} vs {game['away_team']}"):
            display_results(game['results'], game['home_team'], game['away_team'])
