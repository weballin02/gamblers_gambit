# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import warnings
from nba_api.stats.endpoints import LeagueGameLog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Streamlit App Title and Configuration
st.set_page_config(
    page_title="NBA Quantum-Inspired Simulations",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&family=Open+Sans:wght@400;600&display=swap');

/* General Styling */
html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
    background: linear-gradient(135deg, #1a1c2c 0%, #0f111a 100%);
    color: #E5E7EB;
}

/* Header Section */
.header-container {
    text-align: center;
    margin-bottom: 1.5em;
}

.header-title {
    font-family: 'Montserrat', sans-serif;
    background: linear-gradient(120deg, #FFA500, #FF6B00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: 800;
}

.header-subtitle {
    color: #9CA3AF;
    font-size: 1.2em;
}

/* Trust Indicators */
.trust-badges {
    display: flex;
    justify-content: center;
    gap: 2em;
    margin-top: 1em;
}

.trust-badge {
    text-align: center;
    font-size: 1.1em;
    color: #FFA500;
    display: flex;
    align-items: center;
    gap: 0.5em;
}

/* Button Styling */
div.stButton > button {
    background: linear-gradient(90deg, #FF6B00, #FFA500);
    color: white;
    border: none;
    padding: 1em 2em;
    border-radius: 8px;
    font-size: 1.1em;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('''
<div class="header-container">
    <h1 class="header-title">NBA Quantum Predictions</h1>
    <p class="header-subtitle">Leverage quantum-inspired simulations for smarter decisions.</p>
</div>
''', unsafe_allow_html=True)

# Trust Indicators
st.markdown('''
<div class="trust-badges">
    <div class="trust-badge"><span>🔮</span> Quantum Algorithms</div>
    <div class="trust-badge"><span>📈</span> Predictive Analytics</div>
    <div class="trust-badge"><span>🎯</span> Proven Results</div>
</div>
''', unsafe_allow_html=True)

# Informational Text
st.write("Select a game to view quantum-inspired predictions.")

# Team Name Mapping Function
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
full_to_abbrev = {team['full_name']: team['abbreviation'] for team in nba_team_list}

# Load NBA Game Logs with Caching
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

# Calculate Team Stats
def calculate_team_stats(game_logs):
    if game_logs is None:
        return {}
    
    team_stats = {}
    past_games = game_logs.dropna(subset=['PTS'])
    
    # Initialize all teams with empty stats
    for team_abbrev in abbrev_to_full.keys():
        team_stats[team_abbrev] = {
            'scores': [],
            'dates': [],
            'avg_score': 0.0,
            'std_dev': 0.0,
            'min_score': 0.0,
            'max_score': 0.0,
            'cluster_avg_scores': [0.0],
            'recent_form': 0.0,
            'games_played': 0
        }
    
    # Populate scores and dates
    for index, row in past_games.iterrows():
        team_abbrev = row['TEAM_ABBREVIATION']
        pts = row['PTS']
        game_date = pd.to_datetime(row['GAME_DATE']).date()
        
        if team_abbrev in team_stats:
            team_stats[team_abbrev]['scores'].append(pts)
            team_stats[team_abbrev]['dates'].append(game_date)
        else:
            # In case there's a team abbreviation not in the mapping
            team_stats[team_abbrev] = {
                'scores': [pts],
                'dates': [game_date],
                'avg_score': 0.0,
                'std_dev': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'cluster_avg_scores': [0.0],
                'recent_form': 0.0,
                'games_played': 0
            }
    
    # Calculate statistics
    for team, stats in team_stats.items():
        scores = np.array(stats['scores'])
        if len(scores) == 0:
            # Team has no games; assign default values
            team_stats[team]['avg_score'] = 0.0
            team_stats[team]['std_dev'] = 0.0
            team_stats[team]['min_score'] = 0.0
            team_stats[team]['max_score'] = 0.0
            team_stats[team]['cluster_avg_scores'] = [0.0]
            team_stats[team]['recent_form'] = 0.0
            team_stats[team]['games_played'] = 0
            continue
        
        if len(scores) < 3:
            # Assign stats without clustering
            avg_score = np.mean(scores)
            team_stats[team]['avg_score'] = round(avg_score, 2)
            team_stats[team]['std_dev'] = round(np.std(scores), 2)
            team_stats[team]['min_score'] = round(np.min(scores), 2)
            team_stats[team]['max_score'] = round(np.max(scores), 2)
            team_stats[team]['cluster_avg_scores'] = [round(avg_score, 2)]
            team_stats[team]['recent_form'] = round(avg_score, 2)
            team_stats[team]['games_played'] = len(scores)
            continue
        
        try:
            n_clusters = min(3, len(scores))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scores.reshape(-1, 1))
            avg_scores_by_cluster = kmeans.cluster_centers_.flatten()
            
            recent_scores = scores[-5:]
            recent_form = np.mean(recent_scores) if len(recent_scores) > 0 else np.mean(scores)
            
            team_stats[team]['avg_score'] = round(np.mean(scores), 2)
            team_stats[team]['std_dev'] = round(np.std(scores), 2)
            team_stats[team]['min_score'] = round(np.min(scores), 2)
            team_stats[team]['max_score'] = round(np.max(scores), 2)
            team_stats[team]['cluster_avg_scores'] = [round(score, 2) for score in avg_scores_by_cluster]
            team_stats[team]['recent_form'] = round(recent_form, 2)
            team_stats[team]['games_played'] = len(scores)
        except Exception as e:
            st.warning(f"Error calculating stats for {team}: {str(e)}. Assigning average score as cluster_avg_scores.")
            avg_score = np.mean(scores)
            team_stats[team]['avg_score'] = round(avg_score, 2)
            team_stats[team]['std_dev'] = round(np.std(scores), 2)
            team_stats[team]['min_score'] = round(np.min(scores), 2)
            team_stats[team]['max_score'] = round(np.max(scores), 2)
            team_stats[team]['cluster_avg_scores'] = [round(avg_score, 2)]
            team_stats[team]['recent_form'] = round(avg_score, 2)
            team_stats[team]['games_played'] = len(scores)
    
    # Debugging: Display team_stats for inspection
    # Uncomment the following line to see team_stats in the Streamlit app
    # st.json(team_stats)
    
    return team_stats

# Fetch Upcoming Games
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
            home_team_full = abbrev_to_full.get(home_team_abbrev, home_team_abbrev)
            away_team_full = abbrev_to_full.get(away_team_abbrev, away_team_abbrev)
            
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

# Quantum Monte Carlo Simulation Function
def quantum_monte_carlo_simulation(home_team_abbrev, away_team_abbrev, home_team_full, away_team_full, spread_adjustment, num_simulations, team_stats):
    if home_team_abbrev not in team_stats or away_team_abbrev not in team_stats:
        st.error("Team stats not available for selected teams")
        return None
    
    home_stats = team_stats[home_team_abbrev]
    away_stats = team_stats[away_team_abbrev]
    
    # Debugging: Check if 'cluster_avg_scores' exists
    if 'cluster_avg_scores' not in home_stats:
        st.error(f"'cluster_avg_scores' missing for home team: {home_team_full} ({home_team_abbrev})")
        return None
    if 'cluster_avg_scores' not in away_stats:
        st.error(f"'cluster_avg_scores' missing for away team: {away_team_full} ({away_team_abbrev})")
        return None
    
    home_cluster_scores = np.array(home_stats['cluster_avg_scores'])
    away_cluster_scores = np.array(away_stats['cluster_avg_scores'])
    
    # Check if cluster_avg_scores are not empty
    if home_cluster_scores.size == 0:
        st.error(f"'cluster_avg_scores' is empty for home team: {home_team_full} ({home_team_abbrev})")
        return None
    if away_cluster_scores.size == 0:
        st.error(f"'cluster_avg_scores' is empty for away team: {away_team_full} ({away_team_abbrev})")
        return None
    
    # Handle cases where avg_score is 0 to prevent NaN
    home_score_avg = np.random.choice(home_cluster_scores, num_simulations) + spread_adjustment if home_stats['avg_score'] > 0 else spread_adjustment
    away_score_avg = np.random.choice(away_cluster_scores, num_simulations) if away_stats['avg_score'] > 0 else 0.0
    
    home_form_factor = (home_stats['recent_form'] / home_stats['avg_score']) if home_stats['avg_score'] != 0 else 1
    away_form_factor = (away_stats['recent_form'] / away_stats['avg_score']) if away_stats['avg_score'] != 0 else 1
    
    # Prevent std_dev from being zero to avoid identical scores
    home_std_dev = home_stats['std_dev'] if home_stats['std_dev'] > 0 else 1.0
    away_std_dev = away_stats['std_dev'] if away_stats['std_dev'] > 0 else 1.0
    
    home_scores = np.maximum(
        home_stats['min_score'],
        np.random.normal(home_score_avg * home_form_factor, home_std_dev, num_simulations)
    )
    away_scores = np.maximum(
        away_stats['min_score'],
        np.random.normal(away_score_avg * away_form_factor, away_std_dev, num_simulations)
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

# Display Results with Deltas
def display_results(results, home_team_full, away_team_full, team_stats):
    if results:
        # Retrieve abbreviations using the reverse mapping
        home_team_abbrev = full_to_abbrev.get(home_team_full)
        away_team_abbrev = full_to_abbrev.get(away_team_full)
        
        if not home_team_abbrev or not away_team_abbrev:
            st.error("Invalid team name provided.")
            return
        
        # Access team statistics using abbreviations
        home_recent_avg = team_stats[home_team_abbrev]['recent_form']
        away_recent_avg = team_stats[away_team_abbrev]['recent_form']
        
        home_stats = team_stats[home_team_abbrev]
        away_stats = team_stats[away_team_abbrev]
        
        col1, col2 = st.columns(2)
        with col1:
            # Calculate deltas for home team
            expected_home_win = (home_stats['recent_form'] / home_stats['avg_score'] * 100) if home_stats['avg_score'] != 0 else 0
            home_win_delta = results[f'{home_team_full} Win Percentage'] - expected_home_win
            home_score_delta = results[f'Average {home_team_full} Score'] - home_recent_avg
            
            st.metric(
                label=f"{home_team_full} Win Probability",
                value=f"{results[f'{home_team_full} Win Percentage']}%",
                delta=f"{home_win_delta:.2f}%",
                delta_color="normal"
            )
            st.metric(
                label="Projected Home Score",
                value=f"{results[f'Average {home_team_full} Score']:.2f}",
                delta=f"{home_score_delta:.2f}",
                delta_color="normal"
            )
            st.metric(
                label="Projected Total Score",
                value=f"{results['Average Total Score']:.2f}"
            )
            st.write(f"- **Score Differential:** {results[f'Score Differential ({home_team_full} - {away_team_full})']:.2f}")
        
        with col2:
            # Calculate deltas for away team
            expected_away_win = (away_stats['recent_form'] / away_stats['avg_score'] * 100) if away_stats['avg_score'] != 0 else 0
            away_win_delta = results[f'{away_team_full} Win Percentage'] - expected_away_win
            away_score_delta = results[f'Average {away_team_full} Score'] - away_recent_avg
            
            st.metric(
                label=f"{away_team_full} Win Probability",
                value=f"{results[f'{away_team_full} Win Percentage']}%",
                delta=f"{away_win_delta:.2f}%",
                delta_color="normal"
            )
            st.metric(
                label="Projected Away Score",
                value=f"{results[f'Average {away_team_full} Score']:.2f}",
                delta=f"{away_score_delta:.2f}",
                delta_color="normal"
            )
        
        st.subheader("Detailed Predictions")
        metrics = {k: round(v, 2) if isinstance(v, (float, int)) else v for k, v in results.items()}
        st.json(metrics)

# Create Summary Table for All Predictions
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

# Main Area for Controls
st.header("Simulation Controls")
upcoming_games = get_upcoming_games()

if not upcoming_games.empty:
    game_options = [
        f"{row['Game Label']} on {row['Game ID']}"
        for _, row in upcoming_games.iterrows()
    ]
    selected_game = st.selectbox("Select Game", game_options)
    
    # Extract Game ID to fetch the corresponding row
    selected_game_id = selected_game.split(' on ')[1]
    selected_game_row = upcoming_games[upcoming_games['Game ID'] == selected_game_id].iloc[0]
    home_team = selected_game_row['Home Team Full']
    away_team = selected_game_row['Away Team Full']
    
    spread_adjustment = st.slider(
        "Home Team Spread Adjustment",
        -30.0, 30.0, 0.0, step=0.5
    )
    
    num_simulations = st.selectbox(
        "Number of Simulations",
        [1000, 10000, 100000, 1000000]
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

# Run Simulation Buttons
if 'nba_team_stats' in st.session_state:
    if run_simulation and not upcoming_games.empty:
        with st.spinner("Running simulation..."):
            results = quantum_monte_carlo_simulation(
                selected_game_row['Home Team Abbrev'],
                selected_game_row['Away Team Abbrev'],
                home_team, away_team,
                spread_adjustment, num_simulations,
                st.session_state.nba_team_stats
            )
            if results:
                display_results(results, home_team, away_team, st.session_state.nba_team_stats)
    
    if predict_all and not upcoming_games.empty:
        all_results = []
        with st.spinner("Running simulations for all games..."):
            for _, row in upcoming_games.iterrows():
                game_results = quantum_monte_carlo_simulation(
                    row['Home Team Abbrev'],
                    row['Away Team Abbrev'],
                    row['Home Team Full'],
                    row['Away Team Full'],
                    spread_adjustment, num_simulations,
                    st.session_state.nba_team_stats
                )
                if game_results:
                    all_results.append({
                        'home_team_full': row['Home Team Full'],
                        'away_team_full': row['Away Team Full'],
                        'results': game_results
                    })
        
        if all_results:
            create_summary_table(all_results)
            
            for game in all_results:
                with st.expander(f"{game['home_team_full']} vs {game['away_team_full']}"):
                    display_results(game['results'], game['home_team_full'], game['away_team_full'], st.session_state.nba_team_stats)
