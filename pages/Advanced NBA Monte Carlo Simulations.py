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
st.title("NBA Game Prediction System")
st.markdown("### Using Quantum-Inspired Monte Carlo Simulation")

# Team Name Mapping - Corrected and Cleaned
team_name_mapping = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards',
}

# Utility function to find team full name by abbreviation
def find_team_full_name(abbrev):
    return team_name_mapping.get(abbrev, abbrev)

# Fetch team list and create abbreviation mappings
nba_team_list = nba_teams.get_teams()
abbrev_to_full = {team['abbreviation']: team['full_name'] for team in nba_team_list}
id_to_abbrev = {team['id']: team['abbreviation'] for team in nba_team_list}

# Cache the data loading to improve performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_nba_game_logs(season='2022-23'):
    try:
        # Fetch game logs for all teams for the specified season and regular season
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
    
    # Initialize team stats dictionary
    team_stats = {}
    
    # Aggregate points by team and date
    # 'TEAM_ID', 'TEAM_ABBREVIATION', 'PTS', 'GAME_DATE', etc.
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
    
    # Calculate statistics using KMeans clustering
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
            
            # Calculate recent form (last 5 games)
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
        # Fetch games for today
        today_scoreboard = ScoreboardV2(game_date=today.strftime('%Y-%m-%d'))
        today_games = today_scoreboard.get_data_frames()[0]
        
        # Fetch games for tomorrow
        tomorrow_scoreboard = ScoreboardV2(game_date=next_day.strftime('%Y-%m-%d'))
        tomorrow_games = tomorrow_scoreboard.get_data_frames()[0]
        
        # Combine today's and tomorrow's games
        combined_games = pd.concat([today_games, tomorrow_games], ignore_index=True)
        
        if combined_games.empty:
            st.info(f"No upcoming games scheduled for today ({today}) and tomorrow ({next_day}).")
            return pd.DataFrame()
        
        # Mapping from team ID to abbreviation
        nba_team_list = nba_teams.get_teams()
        id_to_abbrev = {team['id']: team['abbreviation'] for team in nba_team_list}
        abbrev_to_full = {team['abbreviation']: team['full_name'] for team in nba_team_list}
        
        # Process game data
        game_list = []
        for index, row in combined_games.iterrows():
            game_id = row['GAME_ID']
            home_team_id = row['HOME_TEAM_ID']
            away_team_id = row['VISITOR_TEAM_ID']
            home_team_abbrev = id_to_abbrev.get(home_team_id, 'UNK')
            away_team_abbrev = id_to_abbrev.get(away_team_id, 'UNK')
            home_team_full = find_team_full_name(home_team_abbrev)
            away_team_full = find_team_full_name(away_team_abbrev)
            
            game_label = f"{away_team_full} at {home_team_full}"
            game_list.append({
                'Game ID': game_id,
                'Game Label': game_label,
                'Home Team Abbrev': home_team_abbrev,
                'Away Team Abbrev': away_team_abbrev,
                'Home Team Full': home_team_full,
                'Away Team Full': away_team_full
            })
        
        games_df = pd.DataFrame(game_list)
        
        return games_df
    
    except Exception as e:
        st.error(f"Error fetching upcoming games: {str(e)}")
        return pd.DataFrame()

def get_current_season():
    """
    Determine the current NBA season based on the current date.
    NBA seasons typically start in October and end in June.
    """
    now = datetime.now()
    year = now.year
    if now.month >= 10:
        season_start = year
        season_end = year + 1
    else:
        season_start = year - 1
        season_end = year
    return f"{season_start}-{str(season_end)[-2:]}"

def quantum_monte_carlo_simulation(home_team_abbrev, away_team_abbrev, home_team_full, away_team_full, spread_adjustment, num_simulations, team_stats):
    if home_team_abbrev not in team_stats or away_team_abbrev not in team_stats:
        st.error("Team stats not available for selected teams")
        return None
    
    home_stats = team_stats[home_team_abbrev]
    away_stats = team_stats[away_team_abbrev]
    
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
        f"{home_team_full} Win Percentage": round(home_wins / num_simulations * 100, 2),
        f"{away_team_full} Win Percentage": round((num_simulations - home_wins) / num_simulations * 100, 2),
        f"Average {home_team_full} Score": round(np.mean(home_scores), 2),
        f"Average {away_team_full} Score": round(np.mean(away_scores), 2),
        "Average Total Score": round(np.mean(home_scores + away_scores), 2),
        f"Score Differential ({home_team_full} - {away_team_full})": round(np.mean(score_diff), 2),
        "Simulation Data": {
            "home_scores": home_scores,
            "away_scores": away_scores
        }
    }
    
    return results

def create_simulation_visualizations(results, home_team_full, away_team_full):
    if not results:
        return None
    
    home_scores = results["Simulation Data"]["home_scores"]
    away_scores = results["Simulation Data"]["away_scores"]
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Score Distribution', 'Score Differential Distribution')
    )
    
    # Score distribution
    fig.add_trace(
        go.Histogram(x=home_scores, name=f"{home_team_full} Scores", opacity=0.75),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=away_scores, name=f"{away_team_full} Scores", opacity=0.75),
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

# Initialize session state for caching team stats
if 'team_stats' not in st.session_state:
    current_season = get_current_season()
    game_logs = load_nba_game_logs(season=current_season)
    st.session_state.team_stats = calculate_team_stats(game_logs)

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
        
        # Extract teams from selection
        selected_game_split = selected_game.split(' on ')
        if len(selected_game_split) == 2:
            game_label, game_id = selected_game_split
            selected_game_row = upcoming_games[upcoming_games['Game ID'] == game_id]
            if not selected_game_row.empty:
                selected_game_row = selected_game_row.iloc[0]
                home_team = selected_game_row['Home Team Full']
                away_team = selected_game_row['Away Team Full']
            else:
                home_team = None
                away_team = None
        else:
            home_team = None
            away_team = None
        
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
        st.error("No upcoming games found for today and tomorrow.")
        run_simulation = False

# Main content area
if run_simulation and home_team and away_team:
    with st.spinner("Running simulation..."):
        # Extract abbreviations and full names
        home_team_abbrev = selected_game_row['Home Team Abbrev']
        away_team_abbrev = selected_game_row['Away Team Abbrev']
        home_team_full = selected_game_row['Home Team Full']
        away_team_full = selected_game_row['Away Team Full']
        
        results = quantum_monte_carlo_simulation(
            home_team_abbrev, away_team_abbrev,
            home_team_full, away_team_full,
            spread_adjustment, num_simulations,
            st.session_state.team_stats
        )
        
        if results:
            # Display key metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{home_team_full} Win Probability",
                          f"{results[f'{home_team_full} Win Percentage']:.2f}%")
            with col2:
                st.metric(f"{away_team_full} Win Probability",
                          f"{results[f'{away_team_full} Win Percentage']:.2f}%")
            
            # Display detailed results
            st.subheader("Detailed Predictions")
            metrics = {k: v for k, v in results.items() if k != "Simulation Data"}
            st.json(metrics)
            
            # Display visualizations
            st.subheader("Visualization")
            fig = create_simulation_visualizations(results, home_team_full, away_team_full)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Display team statistics with rounding to 2 decimal places
            st.subheader("Team Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"{home_team_full} Recent Stats")
                if home_team_abbrev and home_team_abbrev in st.session_state.team_stats:
                    home_team_stats = st.session_state.team_stats.get(home_team_abbrev, {})
                    if home_team_stats:
                        home_team_stats_display = {k: (round(v, 2) if isinstance(v, (float, int)) else v) for k, v in home_team_stats.items()}
                        st.write(home_team_stats_display)
                    else:
                        st.write("No stats available.")
                else:
                    st.write("No stats available.")
            with col2:
                st.write(f"{away_team_full} Recent Stats")
                if away_team_abbrev and away_team_abbrev in st.session_state.team_stats:
                    away_team_stats = st.session_state.team_stats.get(away_team_abbrev, {})
                    if away_team_stats:
                        away_team_stats_display = {k: (round(v, 2) if isinstance(v, (float, int)) else v) for k, v in away_team_stats.items()}
                        st.write(away_team_stats_display)
                    else:
                        st.write("No stats available.")
                else:
                    st.write("No stats available.")

# Hide Streamlit style elements if necessary
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
