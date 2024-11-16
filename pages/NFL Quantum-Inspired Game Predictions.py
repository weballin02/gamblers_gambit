import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz

# Set page configuration
st.set_page_config(
    page_title="FoxEdge - NFL Quantum Predictions",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
        <p>NFL Quantum-Inspired Predictions</p>
    </div>
''', unsafe_allow_html=True)

# High Contrast Toggle (Optional)
if st.button("Toggle High Contrast Mode"):
    st.markdown("""
        <style>
            body {
                background: #000;
                color: #FFF;
            }

            .hero::before {
                background: radial-gradient(circle at center, rgba(255, 255, 255, 0.1), transparent);
            }

            .stButton > button {
                background: linear-gradient(45deg, #FFF, #AAA);
                color: #000;
            }

            .footer {
                color: #CCC;
            }

            .footer a {
                color: #FFF;
            }
        </style>
    """, unsafe_allow_html=True)

# Functionality

# Data Visualizations and Insights Section
st.markdown('''
    <div class="data-section">
        <h2>Leverage Quantum Simulations for Smarter Betting Strategies</h2>
    </div>
''', unsafe_allow_html=True)

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

# Initialize session state for caching
if 'nfl_team_stats' not in st.session_state:
    st.session_state.nfl_team_stats = calculate_team_stats()

# Sidebar for controls
st.sidebar.markdown('''
    <div class="controls-section">
        <h3>Simulation Controls</h3>
    ''', unsafe_allow_html=True)

upcoming_games = get_upcoming_games()

if not upcoming_games.empty:
    game_options = [
        f"{row['game_datetime'].date()} - {row['home_team']} vs {row['away_team']}"
        for _, row in upcoming_games.iterrows()
    ]
    selected_game = st.sidebar.selectbox("Select Game", game_options)
    
    home_team = selected_game.split(' vs ')[0].split(' - ')[1]
    away_team = selected_game.split(' vs ')[1]
    
    spread_adjustment = st.sidebar.slider(
        "Home Team Spread Adjustment",
        -10.0, 10.0, 0.0, step=0.5
    )
    
    num_simulations = st.sidebar.selectbox(
        "Number of Simulations",
        [1000, 10000, 100000]
    )
    
    run_simulation = st.sidebar.button("Run Simulation")
    predict_all = st.sidebar.button("Predict All Upcoming Games")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown(f'''
            <div class="results-section">
                <h3>{game['home_team']} vs {game['away_team']}</h3>
        ''', unsafe_allow_html=True)
        display_results(game['results'], game['home_team'], game['away_team'])
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('''
    <div class="footer">
        &copy; 2023 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
