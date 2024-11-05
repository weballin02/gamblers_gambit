import streamlit as st
import nfl_data_py as nfl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    # Using correct column names: home_score, away_score, and gameday
    past_games = games.dropna(subset=['home_score', 'away_score'])
    team_stats = {}

    for team in pd.concat([past_games['home_team'], past_games['away_team']]).unique():
        home_games = past_games[past_games['home_team'] == team]
        away_games = past_games[past_games['away_team'] == team]
        
        # Calculate recent form (last 5 games)
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
                
                # Calculate recent form score
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

def get_upcoming_games():
    games = load_nfl_data()
    if games is None:
        return pd.DataFrame()

    today = datetime.now().date()
    next_month = today + timedelta(days=30)
    
    schedule = games.copy()
    schedule['gameday'] = pd.to_datetime(schedule['gameday']).dt.date
    upcoming_games = schedule[
        (schedule['gameday'] >= today) & 
        (schedule['gameday'] <= next_month)
    ].sort_values('gameday')
    
    return upcoming_games[['gameday', 'home_team', 'away_team']]

def quantum_monte_carlo_simulation(home_team, away_team, spread_adjustment, num_simulations, team_stats):
    if home_team not in team_stats or away_team not in team_stats:
        st.error("Team stats not available for selected teams")
        return None

    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    # Initialize arrays for vectorized operations
    home_cluster_scores = np.array(home_stats['cluster_avg_scores'])
    away_cluster_scores = np.array(away_stats['cluster_avg_scores'])
    
    # Vectorized simulation
    home_score_avg = np.random.choice(home_cluster_scores, num_simulations) + spread_adjustment
    away_score_avg = np.random.choice(away_cluster_scores, num_simulations)
    
    # Generate scores with recent form influence
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
    
    # Calculate results with rounding to 2 decimal places
    score_diff = home_scores - away_scores
    home_wins = np.sum(score_diff > 0)
    
    results = {
        f"{home_team} Win Percentage": round(home_wins / num_simulations * 100, 2),
        f"{away_team} Win Percentage": round((num_simulations - home_wins) / num_simulations * 100, 2),
        f"Average {home_team} Score": round(np.mean(home_scores), 2),
        f"Average {away_team} Score": round(np.mean(away_scores), 2),
        "Average Total Score": round(np.mean(home_scores + away_scores), 2),
        f"Score Differential ({home_team} - {away_team})": round(np.mean(score_diff), 2),
        "Simulation Data": {
            "home_scores": home_scores,
            "away_scores": away_scores
        }
    }
    
    return results

def create_simulation_visualizations(results, home_team, away_team):
    if not results:
        return None
    
    home_scores = results["Simulation Data"]["home_scores"]
    away_scores = results["Simulation Data"]["away_scores"]
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Score Distribution', 'Win Probability Distribution')
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
    score_diff = home_scores - away_scores
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

# Streamlit UI
st.title("NFL Game Prediction System")
st.markdown("### Using Quantum-Inspired Monte Carlo Simulation")

# Initialize session state for caching
if 'team_stats' not in st.session_state:
    st.session_state.team_stats = calculate_team_stats()

# Sidebar for controls
with st.sidebar:
    st.header("Simulation Controls")
    upcoming_games = get_upcoming_games()
    
    if not upcoming_games.empty:
        game_options = [
            f"{row['gameday']} - {row['home_team']} vs {row['away_team']}"
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

# Main content area
if run_simulation:
    with st.spinner("Running simulation..."):
        results = quantum_monte_carlo_simulation(
            home_team, away_team,
            spread_adjustment, num_simulations,
            st.session_state.team_stats
        )
        
        if results:
            # Display key metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{home_team} Win Probability",
                          f"{results[f'{home_team} Win Percentage']:.2f}%")
            with col2:
                st.metric(f"{away_team} Win Probability",
                          f"{results[f'{away_team} Win Percentage']:.2f}%")
            
            # Display detailed results
            st.subheader("Detailed Predictions")
            metrics = {k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in results.items() if k != "Simulation Data"}
            st.json(metrics)
            
            # Display visualizations
            st.subheader("Visualization")
            fig = create_simulation_visualizations(results, home_team, away_team)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Display team statistics with rounding to 2 decimal places
            st.subheader("Team Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"{home_team} Recent Stats")
                home_team_stats = {k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in st.session_state.team_stats[home_team].items()}
                st.write(home_team_stats)
            with col2:
                st.write(f"{away_team} Recent Stats")
                away_team_stats = {k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in st.session_state.team_stats[away_team].items()}
                st.write(away_team_stats)
