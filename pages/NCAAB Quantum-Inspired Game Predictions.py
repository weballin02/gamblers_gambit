# ncaa.py

# ===========================
# 1. Import Libraries
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import pytz
import cbbpy.mens_scraper as s  # Ensure this import is resolved
import warnings
from typing import Dict, Optional, Tuple, List
import multiprocessing as mp
import logging

# Initialize logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ===========================
# 2. Set Page Configuration
# ===========================

st.set_page_config(
    page_title="NCAAB Basketball Quantum-Inspired Simulator",
    page_icon="🏀",
    layout="wide"
)

# ===========================
# 3. App Title and Description
# ===========================

st.title("NCAAB Basketball Quantum-Inspired Simulator")
# Removed the Features section as per user request

# ===========================
# 4. Team Name Mapping
# ===========================

def find_team_full_name(game_info: pd.Series) -> Tuple[str, str]:
    """
    Extracts full team names from game_info.

    Args:
        game_info (pd.Series): A row from the game_info DataFrame.

    Returns:
        Tuple[str, str]: Home team and Away team full names.
    """
    home_team = game_info['home_team']
    away_team = game_info['away_team']
    return home_team, away_team

# ===========================
# 5. Team Performance Metrics Class
# ===========================

class TeamPerformanceMetrics:
    def __init__(self):
        self.games_played = 0
        self.total_points = 0
        self.points_allowed = 0
        self.home_wins = 0
        self.away_wins = 0
        self.home_games = 0
        self.away_games = 0
        self.recent_scores = []
        self.recent_points_allowed = []
        self.conference_record = 0
        self.ranked_wins = 0

    def add_game(self, game_row: pd.Series, is_home: bool) -> None:
        self.games_played += 1

        if is_home:
            self.home_games += 1
            self.total_points += game_row['home_score']
            self.points_allowed += game_row['away_score']
            self.recent_scores.append(game_row['home_score'])
            self.recent_points_allowed.append(game_row['away_score'])
            if game_row.get('home_win', False):
                self.home_wins += 1
                if pd.notna(game_row.get('away_rank')) and game_row['away_rank'] <= 25:
                    self.ranked_wins += 1
        else:
            self.away_games += 1
            self.total_points += game_row['away_score']
            self.points_allowed += game_row['home_score']
            self.recent_scores.append(game_row['away_score'])
            self.recent_points_allowed.append(game_row['home_score'])
            if not game_row.get('home_win', True):
                self.away_wins += 1
                if pd.notna(game_row.get('home_rank')) and game_row['home_rank'] <= 25:
                    self.ranked_wins += 1

        # Keep only last 5 games for recent performance
        self.recent_scores = self.recent_scores[-5:]
        self.recent_points_allowed = self.recent_points_allowed[-5:]

    @property
    def points_per_game(self) -> float:
        return self.total_points / self.games_played if self.games_played > 0 else 0

    @property
    def points_allowed_per_game(self) -> float:
        return self.points_allowed / self.games_played if self.games_played > 0 else 0

    @property
    def home_win_pct(self) -> float:
        return self.home_wins / self.home_games if self.home_games > 0 else 0

    @property
    def away_win_pct(self) -> float:
        return self.away_wins / self.away_games if self.away_games > 0 else 0

    @property
    def recent_performance_std(self) -> float:
        return np.std(self.recent_scores) if len(self.recent_scores) > 1 else 10

    def to_dict(self) -> Dict[str, any]:
        """
        Converts the TeamPerformanceMetrics instance to a dictionary.
        """
        return {
            'games_played': self.games_played,
            'total_points': self.total_points,
            'points_allowed': self.points_allowed,
            'home_wins': self.home_wins,
            'away_wins': self.away_wins,
            'home_games': self.home_games,
            'away_games': self.away_games,
            'recent_scores': self.recent_scores,
            'recent_points_allowed': self.recent_points_allowed,
            'conference_record': self.conference_record,
            'ranked_wins': self.ranked_wins,
            'points_per_game': self.points_per_game,
            'points_allowed_per_game': self.points_allowed_per_game,
            'home_win_pct': self.home_win_pct,
            'away_win_pct': self.away_win_pct,
            'recent_performance_std': self.recent_performance_std
        }

# ===========================
# 6. Load Season Data with Caching
# ===========================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_season_data(season: int = 2025) -> Tuple[Dict[str, Dict[str, any]], pd.DataFrame]:
    """
    Loads the season data using cbbpy's get_games_season function.

    Args:
        season (int): The season year (e.g., 2024 for the 2023-24 season).

    Returns:
        Tuple containing:
            - team_metrics: Dictionary of serialized TeamPerformanceMetrics dictionaries keyed by team name.
            - game_info_df: DataFrame containing game metadata.
    """
    try:
        # Fetch all game information for the specified season
        game_info_df, _, _ = s.get_games_season(season=season, info=True, box=False, pbp=False)

        if game_info_df.empty:
            raise ValueError(f"No game data available for the {season} season.")

        # Initialize team statistics
        team_metrics: Dict[str, TeamPerformanceMetrics] = {}

        # Process each game
        for _, game in game_info_df.iterrows():
            # Extract home and away team names
            home_team, away_team = find_team_full_name(game)

            # Initialize TeamPerformanceMetrics if not already
            if home_team not in team_metrics:
                team_metrics[home_team] = TeamPerformanceMetrics()
            if away_team not in team_metrics:
                team_metrics[away_team] = TeamPerformanceMetrics()

            # Add game to home and away teams
            team_metrics[home_team].add_game(game, is_home=True)
            team_metrics[away_team].add_game(game, is_home=False)

        # Serialize team_metrics
        serialized_team_metrics = {team: metrics.to_dict() for team, metrics in team_metrics.items()}

        return serialized_team_metrics, game_info_df

    except AttributeError:
        st.error("The 'cbbpy' library does not have the required function. Please ensure you have the latest version installed.")
        logging.error("AttributeError: 'cbbpy' library may be missing required functions.", exc_info=True)
        return {}, pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading season data: {str(e)}")
        logging.error(f"Error loading season data: {str(e)}", exc_info=True)
        return {}, pd.DataFrame()

# ===========================
# 7. Fetch Upcoming Games
# ===========================

def fetch_upcoming_games() -> pd.DataFrame:
    """
    Fetches upcoming NCAA men's basketball games for today using ESPN API.

    Returns:
        pd.DataFrame: DataFrame containing game details.
    """
    # Define the timezone
    timezone = pytz.timezone('America/Los_Angeles')

    # Current date and time
    current_time = datetime.now(timezone)

    # Fetch dates for today only
    dates = [current_time]

    all_games = []

    for date in dates:
        # Format the date as YYYYMMDD
        date_str = date.strftime('%Y%m%d')

        # ESPN API endpoint
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"

        # Parameters
        params = {
            'dates': date_str,
            'groups': '50',
            'limit': '357'
        }

        # Fetch the data
        response = requests.get(url, params=params)
        if response.status_code != 200:
            st.error(f"Error: API request failed with status code {response.status_code} for date {date_str}")
            logging.error(f"API request failed with status code {response.status_code} for date {date_str}")
            continue  # Skip to the next date

        data = response.json()

        # Extract games
        games = data.get('events', [])
        if not games:
            st.info(f"No games scheduled for {date.strftime('%Y-%m-%d')}.")
            continue  # No games for this date

        for game in games:
            try:
                # Extract game details
                game_time_str = game['date']
                game_time = datetime.fromisoformat(game_time_str[:-1]).astimezone(timezone)

                # Extract teams
                competitors = game['competitions'][0]['competitors']
                home_team = next(team for team in competitors if team['homeAway'] == 'home')['team']['displayName']
                away_team = next(team for team in competitors if team['homeAway'] == 'away')['team']['displayName']

                # Extract venue
                venue = game.get('competitions')[0].get('venue', {}).get('fullName', 'Unknown Venue')

                # Append to list
                all_games.append({
                    'Game_ID': game['id'],
                    'Game_Label': f"{away_team} at {home_team}",
                    'Home_Team': home_team,
                    'Away_Team': away_team,
                    'Game_Day': game_time.strftime('%Y-%m-%d'),
                    'Game_Time': game_time.strftime('%I:%M %p %Z'),
                    'Arena': venue,
                    'Tournament': game.get('competitions')[0].get('tournament', {}).get('name', 'N/A')
                })

            except KeyError as e:
                st.warning(f"Missing key in game data: {e}")
                logging.warning(f"Missing key in game data: {e}")
            except Exception as e:
                st.warning(f"Unexpected error: {e}")
                logging.warning(f"Unexpected error: {e}")

    if not all_games:
        st.info("No upcoming games fetched for today.")
        return pd.DataFrame()

    # Convert to DataFrame
    upcoming_games_df = pd.DataFrame(all_games)

    return upcoming_games_df

# ===========================
# 8. Preprocess Data Vectorized
# ===========================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def preprocess_data_vectorized(games_info_df: pd.DataFrame) -> Dict[str, Dict[str, any]]:
    """
    Preprocesses and engineers features from game info DataFrame.

    Args:
        games_info_df (pd.DataFrame): DataFrame containing game metadata.

    Returns:
        Dict[str, Dict[str, any]]: Dictionary of serialized team performance metrics.
    """
    team_metrics: Dict[str, TeamPerformanceMetrics] = {}

    # Iterate through games and populate team metrics
    for _, game in games_info_df.iterrows():
        home_team, away_team = find_team_full_name(game)

        # Initialize metrics if not present
        if home_team not in team_metrics:
            team_metrics[home_team] = TeamPerformanceMetrics()
        if away_team not in team_metrics:
            team_metrics[away_team] = TeamPerformanceMetrics()

        # Add game to metrics
        team_metrics[home_team].add_game(game, is_home=True)
        team_metrics[away_team].add_game(game, is_home=False)

    # Serialize team_metrics
    serialized_team_metrics = {team: metrics.to_dict() for team, metrics in team_metrics.items()}

    return serialized_team_metrics

# ===========================
# 9. Simulation Function
# ===========================

def simulate_game(
    home_team: str,
    away_team: str,
    spread: float,
    num_sims: int,
    team_metrics: Dict[str, Dict[str, any]]
) -> Optional[Dict[str, float]]:
    """
    Simulates a game between home_team and away_team.

    Args:
        home_team (str): Full name of the home team.
        away_team (str): Full name of the away team.
        spread (float): Spread adjustment for the home team.
        num_sims (int): Number of simulations to run.
        team_metrics (Dict[str, Dict[str, any]]): Serialized team performance metrics.

    Returns:
        Optional[Dict[str, float]]: Simulation results or None if error occurs.
    """
    try:
        home = team_metrics[home_team]
        away = team_metrics[away_team]

        # Base scoring metrics
        home_base = home['points_per_game']
        away_base = away['points_per_game']

        # Adjust for home/away performance
        home_adj = home_base * (1 + 0.05 * home['home_win_pct'])
        away_adj = away_base * (1 + 0.05 * away['away_win_pct'])

        # Generate scores using multiple factors
        home_scores = np.maximum(0, np.random.normal(
            home_adj + spread,
            home['recent_performance_std'],
            num_sims
        ))

        away_scores = np.maximum(0, np.random.normal(
            away_adj,
            away['recent_performance_std'],
            num_sims
        ))

        # Calculate outcomes
        score_diff = home_scores - away_scores
        home_wins = np.sum(score_diff > 0)

        return {
            "win_prob": home_wins / num_sims * 100,  # Stored as float
            "home_score": np.mean(home_scores),
            "away_score": np.mean(away_scores),
            "total_score": np.mean(home_scores + away_scores),
            "spread": np.mean(score_diff),
            "home_std": home['recent_performance_std'],
            "away_std": away['recent_performance_std'],
            "home_recent_avg": np.mean(home['recent_scores']) if home['recent_scores'] else 0,
            "away_recent_avg": np.mean(away['recent_scores']) if away['recent_scores'] else 0
        }

    except KeyError as e:
        st.error(f"Missing team metrics for {e}")
        logging.error(f"Missing team metrics for {e}", exc_info=True)
        return None
    except Exception as e:
        st.error(f"Simulation error: {str(e)}")
        logging.error(f"Simulation error: {str(e)}", exc_info=True)
        return None

# ===========================
# 10. Display Functions
# ===========================

def display_team_stats(team: str, metrics: Dict[str, any], is_home: bool):
    """
    Displays team statistics.

    Args:
        team (str): Team name.
        metrics (Dict[str, any]): Serialized team metrics.
        is_home (bool): Whether the team is playing at home.
    """
    location = "Home" if is_home else "Away"
    st.subheader(f"{team} ({location})")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Points Per Game", f"{metrics['points_per_game']:.1f}")
        st.metric("Points Allowed Per Game", f"{metrics['points_allowed_per_game']:.1f}")
    with col2:
        win_pct = metrics['home_win_pct'] if is_home else metrics['away_win_pct']
        st.metric(f"{location} Win %", f"{win_pct*100:.1f}%")
        st.metric("Ranked Wins", metrics['ranked_wins'])

def display_prediction(results: Dict[str, float], home_team: str, away_team: str):
    """
    Displays simulation prediction results.

    Args:
        results (Dict[str, float]): Simulation results.
        home_team (str): Home team name.
        away_team (str): Away team name.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label=f"{home_team} Win Probability",
            value=f"{results['win_prob']:.1f}%",
            delta=f"{results['spread']:.1f} pt spread"
        )
        st.metric(
            label="Projected Score",
            value=f"{results['home_score']:.1f}",
            delta=f"{results['home_score'] - results['home_recent_avg']:.1f} vs recent"
        )

    with col2:
        st.metric(
            label=f"{away_team} Win Probability",
            value=f"{(100 - results['win_prob']):.1f}%",
        )
        st.metric(
            label="Projected Score",
            value=f"{results['away_score']:.1f}",
            delta=f"{results['away_score'] - results['away_recent_avg']:.1f} vs recent"
        )

    st.metric(
        label="Projected Total Score",
        value=f"{results['total_score']:.1f}"
    )

# ===========================
# 11. Parallel Simulation Function
# ===========================

def run_single_simulation(args):
    home_team, away_team, spread, num_sims, team_metrics = args
    return simulate_game(home_team, away_team, spread, num_sims, team_metrics)

def run_parallel_simulations(upcoming_games: pd.DataFrame, spread: float, num_sims: int, team_metrics: Dict[str, Dict[str, any]]) -> List[Optional[Dict[str, float]]]:
    """
    Runs simulations in parallel for all upcoming games.

    Args:
        upcoming_games (pd.DataFrame): DataFrame of upcoming games.
        spread (float): Spread adjustment.
        num_sims (int): Number of simulations per game.
        team_metrics (Dict[str, Dict[str, any]]): Serialized team performance metrics.

    Returns:
        List[Optional[Dict[str, float]]]: List of simulation results for each game.
    """
    game_params = [
        (row['Home_Team'], row['Away_Team'], spread, num_sims, team_metrics)
        for _, row in upcoming_games.iterrows()
    ]

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(run_single_simulation, game_params)

    return results

# ===========================
# 12. Main Function
# ===========================

def main():
    # Load Season Data
    with st.spinner("Loading season data..."):
        team_metrics, game_info_df = load_season_data(season=2025)
    
    if not team_metrics or game_info_df.empty:
        st.error("Unable to load team statistics. Please check your data source and try again.")
        return

    # Fetch Upcoming Games
    upcoming_games_df = fetch_upcoming_games()

    # Game Settings Section
    st.header("Game Settings")
    if not upcoming_games_df.empty:
        # Sort games alphabetically for better navigation
        game_options = sorted(upcoming_games_df['Game_Label'].tolist())
        selected_game = st.selectbox("Select Game", options=["Select a Game"] + game_options)

        if selected_game != "Select a Game":
            selected_game_info = upcoming_games_df[upcoming_games_df['Game_Label'] == selected_game].iloc[0]
            home_team = selected_game_info['Home_Team']
            away_team = selected_game_info['Away_Team']
        else:
            home_team = None
            away_team = None
    else:
        st.info("No upcoming games to display.")
        home_team = None
        away_team = None

    # Simulation Settings Section
    st.header("Simulation Settings")
    spread_adjustment = st.slider(
        "Home Team Spread Adjustment",
        -30.0, 30.0, 0.0, step=0.5
    )
    num_simulations = st.selectbox(
        "Number of Simulations",
        [1000, 10000, 100000, 1000000],
        index=1
    )

    # Best Bets Criteria Section
    st.markdown("### Best Bets Criteria")
    col1, col2 = st.columns(2)
    with col1:
        high_win_threshold = st.number_input(
            "Win Probability Threshold (%)",
            min_value=50.0,
            max_value=100.0,
            value=70.0,
            step=1.0,
            help="Minimum win probability to qualify as a Best Bet."
        )
    with col2:
        large_spread_threshold = st.number_input(
            "Minimum Spread (points)",
            min_value=0.0,
            max_value=30.0,
            value=5.0,
            step=0.5,
            help="Minimum point spread to qualify as a Best Bet."
        )

    # Simulation Actions Section
    st.header("Simulation Actions")
    run_simulation = st.button("Run Simulation for Selected Game")
    run_all_simulations = st.button("Run Simulations for All Upcoming Games")

    # Display and Run Simulation for Selected Game
    if run_simulation and home_team and away_team:
        st.subheader(f"Simulation: {away_team} at {home_team}")
        
        # Display Team Statistics
        col1, col2 = st.columns(2)
        with col1:
            display_team_stats(home_team, team_metrics[home_team], is_home=True)
        with col2:
            display_team_stats(away_team, team_metrics[away_team], is_home=False)
        
        # Run Simulation
        with st.spinner("Running simulation..."):
            results = simulate_game(
                home_team=home_team,
                away_team=away_team,
                spread=spread_adjustment,
                num_sims=num_simulations,
                team_metrics=team_metrics
            )
            if results:
                st.divider()
                display_prediction(results, home_team, away_team)
                
                # Analysis based on actual performance metrics
                st.subheader("Game Analysis")
                
                insights = []
                
                # Scoring trend analysis
                if abs(results['home_score'] - team_metrics[home_team]['points_per_game']) > 5:
                    insights.append(f"Model projects significant deviation from {home_team}'s average scoring.")
                if abs(results['away_score'] - team_metrics[away_team]['points_per_game']) > 5:
                    insights.append(f"Model projects significant deviation from {away_team}'s average scoring.")
                
                # Matchup analysis
                if results['win_prob'] > 75:
                    insights.append(f"Strong favorite: {home_team} projected for a convincing home win.")
                elif results['win_prob'] < 25:
                    insights.append(f"Strong favorite: {away_team} projected for a road victory.")
                else:
                    insights.append("Competitive matchup projected with no clear favorite.")
                
                # Display insights
                for insight in insights:
                    st.write(f"- {insight}")

    # Display and Run Simulations for All Upcoming Games
    if run_all_simulations and not upcoming_games_df.empty:
        st.subheader("Simulations for All Upcoming Games")
        simulation_results = []
        
        with st.spinner("Running simulations for all upcoming games..."):
            try:
                # Run simulations in parallel
                results = run_parallel_simulations(upcoming_games_df, spread_adjustment, num_simulations, team_metrics)

                for (_, game_row), result in zip(upcoming_games_df.iterrows(), results):
                    if result:
                        simulation_results.append({
                            'Game_Label': game_row['Game_Label'],
                            'Home_Win_%': result['win_prob'],           # Stored as float
                            'Away_Win_%': 100 - result['win_prob'],     # Stored as float
                            'Projected_Home_Score': f"{result['home_score']:.1f}",
                            'Projected_Away_Score': f"{result['away_score']:.1f}",
                            'Projected_Total_Score': f"{result['total_score']:.1f}",
                            'Spread': f"{result['spread']:.1f} pts"
                        })
            except Exception as e:
                st.error(f"Error during parallel simulations: {str(e)}")
                logging.error(f"Error during parallel simulations: {str(e)}", exc_info=True)
                results = []

        if simulation_results:
            # Create DataFrame for Results
            results_df = pd.DataFrame(simulation_results)
            
            # Debugging statement to verify data
            st.write("### Simulation Results Preview")
            st.write(results_df.head())

            # ===========================
            # 13. Best Bets Section
            # ===========================

            # Debugging statement to check if we reach this section
            st.write("Reached Best Bets Section")  # Debugging output

            # Check if results_df is populated
            if 'results_df' in locals() and not results_df.empty:
                st.subheader("Best Bets")

                # Convert 'Spread' from string to float for comparison
                results_df['Spread_Value'] = results_df['Spread'].str.replace(' pts', '').astype(float)

                # Define criteria:
                # 1. Home Win % > high_win_threshold OR Away Win % > high_win_threshold
                # 2. Spread > large_spread_threshold (absolute value)
                best_bets_df = results_df[
                    (
                        (results_df['Home_Win_%'] > high_win_threshold) |
                        (results_df['Away_Win_%'] > high_win_threshold)
                    ) &
                    (results_df['Spread_Value'].abs() > large_spread_threshold)
                ]

                # Debugging statement to check the number of best bets found
                st.write(f"Number of Best Bets Found: {len(best_bets_df)}")  # Debugging output

                if not best_bets_df.empty:
                    # Format the DataFrame for display by appending '%' symbol
                    display_best_bets_df = best_bets_df.copy()
                    display_best_bets_df['Home_Win_%'] = display_best_bets_df['Home_Win_%'].astype(float).map("{:.1f}%".format)
                    display_best_bets_df['Away_Win_%'] = display_best_bets_df['Away_Win_%'].astype(float).map("{:.1f}%".format)
                    display_best_bets_df['Spread'] = display_best_bets_df['Spread'].astype(str)
                    
                    st.markdown("### Top Recommended Bets")
                    st.dataframe(display_best_bets_df[['Game_Label', 'Home_Win_%', 'Away_Win_%', 'Spread']])
                    
                    # Download Best Bets as CSV
                    csv = best_bets_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Best Bets as CSV",
                        data=csv,
                        file_name='best_bets.csv',
                        mime='text/csv',
                    )
                    
                    # Visualization: Bar Chart for Win Probabilities
                    st.markdown("#### Win Probabilities")
                    st.bar_chart(best_bets_df[['Home_Win_%', 'Away_Win_%']])
                    
                    # Visualization: Scatter Plot for Spread vs. Win Probability
                    st.markdown("#### Spread vs. Win Probability")
                    scatter_data = best_bets_df.copy()
                    scatter_data['Home_Win_%'] = scatter_data['Home_Win_%'].astype(float)
                    scatter_data['Away_Win_%'] = scatter_data['Away_Win_%'].astype(float)
                    scatter_data['Spread_Value'] = scatter_data['Spread_Value'].astype(float)
                    st.scatter_chart(scatter_data[['Spread_Value', 'Home_Win_%', 'Away_Win_%']])
                else:
                    # Display message indicating no bets qualify
                    st.info("No best bets found based on the current criteria.")
            else:
                st.warning("Results DataFrame is empty or not defined.")

# ===========================
# 13. Run the App
# ===========================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logging.error(f"Unexpected error in main: {str(e)}", exc_info=True)
