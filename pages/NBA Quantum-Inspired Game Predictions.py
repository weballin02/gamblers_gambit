# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Streamlit App Title
st.title("NBA Quantum-Inspired Game Predictions")
st.markdown("Experience an advanced, quantum-inspired simulation of NBA games. Select a game or simulate all matchups for the day. Customize spread and simulation settings to see win probabilities, average scores, and predictions that will guide your next moves.")

# Team Name Mapping - Corrected and Cleaned
def find_team_full_name(abbrev):
    team_name_mapping = {
        'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
        'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
        'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
        'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
        'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
        'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
        'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
    }
    return team_name_mapping.get(abbrev, abbrev)

# Fetch team list and create abbreviation mappings
nba_team_list = nba_teams.get_teams()
abbrev_to_full = {team['abbreviation']: team['full_name'] for team in nba_team_list}
id_to_abbrev = {team['id']: team['abbreviation'] for team in nba_team_list}

# Cache the data loading to improve performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_nba_game_logs(season=['2023-24', '2022-23']):
    try:
        game_logs = LeagueGameLog(season=season, season_type_all_star='Regular Season', player_or_team_abbreviation='T')
        games = game_logs.get_data_frames()[0]
        if games.empty:
            st.error(f"No game logs available for the {season} season.")
            return None
        return games
    except Exception as e:
        st.error(f"Error loading NBA game logs: {str(e)}")
        return None

def calculate_team_stats(game_logs):
    if game_logs is None:
        return {}
    
    team_stats = {}
    past_games = game_logs.dropna(subset=['PTS'])
    
    for index, row in past_games.iterrows():
        team_abbrev = row['TEAM_ABBREVIATION']
        team_full = find_team_full_name(team_abbrev)
        pts = row['PTS']
        game_date = pd.to_datetime(row['GAME_DATE']).date()
        
        if team_abbrev not in team_stats:
            team_stats[team_abbrev] = {'scores': [], 'dates': []}
        team_stats[team_abbrev]['scores'].append(pts)
        team_stats[team_abbrev]['dates'].append(game_date)
    
    for team, stats in team_stats.items():
        scores = np.array(stats['scores']).reshape(-1, 1)
        if len(scores) < 3:
            team_stats[team]['avg_score'] = np.mean(scores)
            team_stats[team]['std_dev'] = np.std(scores)
            team_stats[team]['min_score'] = np.min(scores)
            team_stats[team]['max_score'] = np.max(scores)
            team_stats[team]['cluster_avg_scores'] = [team_stats[team]['avg_score']]
            team_stats[team]['recent_form'] = team_stats[team]['avg_score']
            team_stats[team]['games_played'] = len(scores)
            continue
        
        try:
            n_clusters = min(3, len(scores))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scores)
            avg_scores_by_cluster = kmeans.cluster_centers_.flatten()
            
            recent_scores = scores.flatten()[-5:]
            recent_form = np.mean(recent_scores) if len(recent_scores) > 0 else np.mean(scores)
            
            team_stats[team]['avg_score'] = round(np.mean(scores), 2)
            team_stats[team]['std_dev'] = round(np.std(scores), 2)
            team_stats[team]['min_score'] = round(np.min(scores), 2)
            team_stats[team]['max_score'] = round(np.max(scores), 2)
            team_stats[team]['cluster_avg_scores'] = [round(score, 2) for score in avg_scores_by_cluster]
            team_stats[team]['recent_form'] = round(recent_form, 2)
            team_stats[team]['games_played'] = len(scores)
        except Exception as e:
            st.warning(f"Error calculating stats for {team}: {str(e)}")
            continue
    
    return team_stats

def get_upcoming_games():
    today = datetime.now().date()
    next_day = today + timedelta(days=1)
    
    try:
        today_scoreboard = ScoreboardV2(game_date=today.strftime('%Y-%m-%d'))
        tomorrow_scoreboard = ScoreboardV2(game_date=next_day.strftime('%Y-%m-%d'))
        combined_games = pd.concat([today_scoreboard.get_data_frames()[0], tomorrow_scoreboard.get_data_frames()[0]], ignore_index=True)
        
        if combined_games.empty:
            st.info(f"No upcoming games scheduled for today ({today}) and tomorrow ({next_day}).")
            return pd.DataFrame()
        
        game_list = []
        for index, row in combined_games.iterrows():
            game_id = row['GAME_ID']
            home_team_id = row['HOME_TEAM_ID']
            away_team_id = row['VISITOR_TEAM_ID']
            home_team_abbrev = id_to_abbrev.get(home_team_id, 'UNK')
            away_team_abbrev = id_to_abbrev.get(away_team_id, 'UNK')
            home_team_full = find_team_full_name(home_team_abbrev)
            away_team_full = find_team_full_name(away_team_abbrev)
            
            game_list.append({
                'Game ID': game_id,
                'Game Label': f"{away_team_full} at {home_team_full}",
                'Home Team Abbrev': home_team_abbrev,
                'Away Team Abbrev': away_team_abbrev,
                'Home Team Full': home_team_full,
                'Away Team Full': away_team_full
            })
        
        return pd.DataFrame(game_list)
    except Exception as e:
        st.error(f"Error fetching upcoming games: {str(e)}")
        return pd.DataFrame()

def quantum_monte_carlo_simulation(home_team_abbrev, away_team_abbrev, home_team_full, away_team_full, spread_adjustment, num_simulations, team_stats):
    if home_team_abbrev not in team_stats or away_team_abbrev not in team_stats:
        st.error("Team stats not available for selected teams")
        return None
    
    home_stats = team_stats[home_team_abbrev]
    away_stats = team_stats[away_team_abbrev]
    
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
        f"{home_team_full} Win Percentage": round(home_wins / num_simulations * 100, 2),
        f"{away_team_full} Win Percentage": round((num_simulations - home_wins) / num_simulations * 100, 2),
        f"Average {home_team_full} Score": round(np.mean(home_scores), 2),
        f"Average {away_team_full} Score": round(np.mean(away_scores), 2),
        "Average Total Score": round(np.mean(home_scores + away_scores), 2),
        f"Score Differential ({home_team_full} - {away_team_full})": round(np.mean(score_diff), 2)
    }
    
    return results

def display_results(results, home_team_full, away_team_full):
    if results:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{home_team_full} Win Probability", f"{results[f'{home_team_full} Win Percentage']:.2f}%")
        with col2:
            st.metric(f"{away_team_full} Win Probability", f"{results[f'{away_team_full} Win Percentage']:.2f}%")
        
        st.subheader("Detailed Predictions")
        metrics = {k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in results.items()}
        st.json(metrics)

def create_summary_table(all_results):
    summary_data = []
    for game in all_results:
        summary_data.append({
            "Home Team": game['home_team_full'],
            "Away Team": game['away_team_full'],
            "Home Win %": game['results'][f"{game['home_team_full']} Win Percentage"],
            "Away Win %": game['results'][f"{game['away_team_full']} Win Percentage"],
            "Avg Total Score": game['results']["Average Total Score"],
            "Score Differential": game['results'][f"Score Differential ({game['home_team_full']} - {game['away_team_full']})"]
        })
    summary_df = pd.DataFrame(summary_data)
    st.write("### Summary of All Predictions")
    st.dataframe(summary_df)

# Initialize session state for caching
if 'nba_team_stats' not in st.session_state:
    current_season = "2024-25"
    game_logs = load_nba_game_logs(season=current_season)
    st.session_state.nba_team_stats = calculate_team_stats(game_logs)

# Sidebar for controls
with st.sidebar:
    st.header("Simulation Controls")
    upcoming_games = get_upcoming_games()
    
    if not upcoming_games.empty:
        game_options = [
            f"{row['Game Label']} on {row['Game ID']}"
            for _, row in upcoming_games.iterrows()
        ]
        selected_game = st.selectbox("Select Game", game_options)
        
        selected_game_row = upcoming_games[upcoming_games['Game ID'] == selected_game.split(' on ')[1]].iloc[0]
        home_team = selected_game_row['Home Team Full']
        away_team = selected_game_row['Away Team Full']
        
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

# Button to refresh data, update models, and predict
if st.button("Refresh Data & Predict"):
    with st.spinner("Refreshing data and updating models..."):
        current_season = "2024-25"
        game_logs = load_nba_game_logs(season=current_season)
        st.session_state.nba_team_stats = calculate_team_stats(game_logs)
        st.success("Data refreshed and models updated.")

if run_simulation:
    with st.spinner("Running simulation..."):
        results = quantum_monte_carlo_simulation(
            selected_game_row['Home Team Abbrev'],
            selected_game_row['Away Team Abbrev'],
            home_team, away_team,
            spread_adjustment, num_simulations,
            st.session_state.nba_team_stats
        )
        display_results(results, home_team, away_team)

if predict_all:
    all_results = []
    with st.spinner("Running simulations for all games..."):
        for _, row in upcoming_games.iterrows():
            home_team_abbrev = row['Home Team Abbrev']
            away_team_abbrev = row['Away Team Abbrev']
            home_team_full = row['Home Team Full']
            away_team_full = row['Away Team Full']
            game_results = quantum_monte_carlo_simulation(
                home_team_abbrev, away_team_abbrev,
                home_team_full, away_team_full,
                spread_adjustment, num_simulations,
                st.session_state.nba_team_stats
            )
            all_results.append({
                'home_team_full': home_team_full,
                'away_team_full': away_team_full,
                'results': game_results
            })
    
    create_summary_table(all_results)

    for game in all_results:
        with st.expander(f"{game['home_team_full']} vs {game['away_team_full']}"):
            display_results(game['results'], game['home_team_full'], game['away_team_full'])

