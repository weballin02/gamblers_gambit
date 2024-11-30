# Import Libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from nba_api.stats.endpoints import teamgamelog, ScoreboardV2, LeagueGameLog
from nba_api.stats.static import teams as nba_teams
from pmdarima import auto_arima
import joblib
import os
import warnings
import plotly.graph_objects as go
import seaborn as sns

warnings.filterwarnings('ignore')

# Streamlit App Title
st.set_page_config(
    page_title="NBA FoxEdge",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Fetch NBA Teams
nba_team_list = nba_teams.get_teams()
team_abbreviations = [team['abbreviation'] for team in nba_team_list]
team_name_mapping = {team['abbreviation']: team['full_name'] for team in nba_team_list}

# Utility: ROI Calculator
def calculate_roi(bet_amount, odds):
    """
    Calculate potential ROI for a given bet amount and odds.
    """
    if odds > 0:
        return bet_amount * (odds / 100)
    else:
        return bet_amount * (100 / abs(odds))

# Fetch and Preprocess Data
@st.cache_data
def fetch_and_preprocess_data(season):
    """
    Fetch data for the specified season from NBA API and preprocess it.
    """
    all_data = []
    failed_teams = []
    for team in nba_team_list:
        team_id = team['id']
        team_abbrev = team['abbreviation']
        try:
            team_logs = teamgamelog.TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]
            team_logs = team_logs[['Game_ID', 'GAME_DATE', 'PTS']]
            team_logs['TEAM_ABBREV'] = team_abbrev
            all_data.append(team_logs)
        except Exception as e:
            st.warning(f"Error fetching data for {team_name_mapping[team_abbrev]}: {e}")
            failed_teams.append(team_abbrev)
            continue
    if all_data:
        data = pd.concat(all_data, ignore_index=True)
        data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'])
        return data, failed_teams
    else:
        return pd.DataFrame(), failed_teams

# Aggregate Points by Team and Date
def aggregate_points_by_team(data):
    team_data = data.groupby(['GAME_DATE', 'TEAM_ABBREV'])['PTS'].sum().reset_index()
    team_data.set_index('GAME_DATE', inplace=True)
    return team_data

# Train ARIMA Models
@st.cache_resource
def train_arima_models(team_data):
    model_dir = 'models/nba'
    os.makedirs(model_dir, exist_ok=True)
    team_models = {}
    for team_abbrev in team_data['TEAM_ABBREV'].unique():
        model_path = os.path.join(model_dir, f"{team_abbrev}_arima_model.pkl")
        team_points = team_data[team_data['TEAM_ABBREV'] == team_abbrev]['PTS']
        team_points.reset_index(drop=True, inplace=True)
        model = auto_arima(
            team_points,
            seasonal=False,
            trace=False,
            error_action='ignore',
            suppress_warnings=True
        )
        model.fit(team_points)
        joblib.dump(model, model_path)
        team_models[team_abbrev] = model
    return team_models

# Compute Team Forecasts
@st.cache_data
def compute_team_forecasts(_team_models, team_data):
    team_forecasts = {}
    forecast_periods = 5
    for team_abbrev, model in _team_models.items():
        team_points = team_data[team_data['TEAM_ABBREV'] == team_abbrev]['PTS']
        if team_points.empty:
            continue
        last_date = team_points.index.max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        forecast = model.predict(n_periods=forecast_periods)
        predictions = pd.DataFrame({'Date': future_dates, 'Predicted_PTS': forecast, 'Team': team_abbrev})
        team_forecasts[team_abbrev] = predictions
    return pd.concat(team_forecasts.values(), ignore_index=True) if team_forecasts else pd.DataFrame(columns=['Date', 'Predicted_PTS', 'Team'])

# Fetch Today's Games
def fetch_todays_games():
    today = datetime.now().strftime("%Y-%m-%d")
    scoreboard = ScoreboardV2(game_date=today)
    games = scoreboard.get_data_frames()[0]
    id_to_abbrev = {team['id']: team['abbreviation'] for team in nba_team_list}
    if 'HOME_TEAM_ID' in games.columns and 'VISITOR_TEAM_ID' in games.columns:
        games['HOME_TEAM_ABBREVIATION'] = games['HOME_TEAM_ID'].map(id_to_abbrev)
        games['VISITOR_TEAM_ABBREVIATION'] = games['VISITOR_TEAM_ID'].map(id_to_abbrev)
        return games[['HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION']]
    else:
        st.error("Expected columns 'HOME_TEAM_ID' and 'VISITOR_TEAM_ID' not found in games data.")
        return pd.DataFrame(columns=['HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION'])

# Predict Matchups
def predict_game_outcome_with_arima(home_team, away_team, team_forecasts, current_season_stats):
    home_forecast = team_forecasts[team_forecasts['Team'] == home_team]['Predicted_PTS'].mean()
    away_forecast = team_forecasts[team_forecasts['Team'] == away_team]['Predicted_PTS'].mean()
    home_stats = current_season_stats.get(home_team, {})
    away_stats = current_season_stats.get(away_team, {})
    if home_stats and away_stats:
        home_team_rating = (
            0.3 * home_stats['avg_score'] +
            0.2 * home_stats['max_score'] +
            0.3 * home_stats['recent_form'] +
            0.2 * home_forecast -
            0.1 * home_stats['std_dev']
        )
        away_team_rating = (
            0.3 * away_stats['avg_score'] +
            0.2 * away_stats['max_score'] +
            0.3 * away_stats['recent_form'] +
            0.2 * away_forecast -
            0.1 * away_stats['std_dev']
        )
        rating_diff = home_team_rating - away_team_rating
        confidence = min(100, max(0, 50 + (rating_diff - max(home_stats['std_dev'], away_stats['std_dev'])) * 3))
        predicted_winner = home_team if rating_diff > 0 else away_team
        predicted_score_diff = abs(rating_diff)
        return predicted_winner, predicted_score_diff, confidence, home_team_rating, away_team_rating
    else:
        return "Unavailable", "N/A", "N/A", None, None

# Display Enhanced Matchup Details
def display_matchup_details(home_team, away_team, result, team_name_mapping, team_forecasts):
    """
    Display detailed matchup insights with enhanced visuals.
    """
    winner, score_diff, confidence, home_rating, away_rating = result

    if pd.isna(score_diff) or pd.isna(confidence) or winner == "Unavailable":
        st.warning(f"Data unavailable for matchup: {team_name_mapping[home_team]} vs {team_name_mapping[away_team]}")
        return

    home_forecast = team_forecasts[team_forecasts["Team"] == home_team]["Predicted_PTS"].mean()
    away_forecast = team_forecasts[team_forecasts["Team"] == away_team]["Predicted_PTS"].mean()

    st.write(f"## {team_name_mapping[home_team]} vs {team_name_mapping[away_team]}")
    st.write(f"- **Predicted Winner:** {team_name_mapping.get(winner, 'Unavailable')}")
    st.write(f"- **Predicted Final Scores:** {team_name_mapping[home_team]} {home_forecast:.2f} - {team_name_mapping[away_team]} {away_forecast:.2f}")
    st.write(f"- **Score Difference:** {score_diff:.2f}")
    st.write(f"- **Confidence:** {confidence:.2f}%")

    # Betting Insights Section
    st.write("### Betting Insights")

    # Spread Lean
    spread_lean = f"Lean {team_name_mapping[winner]} -{int(score_diff)}" if score_diff > 0 else f"Lean {team_name_mapping[away_team]} +{int(abs(score_diff))}"
    st.write(f"#### Spread Lean: {spread_lean}")

    # Total Lean
    total_points = home_forecast + away_forecast
    total_recommendation = f"Lean Over if the line is < {total_points - 5:.2f}"
    st.write(f"#### Total Lean: {total_recommendation}")

    # Radar Chart
    categories = ["Average Score", "Max Score", "Consistency", "Recent Form", "Predicted PTS"]
    home_stats = [home_rating, home_forecast, 1, 1, home_forecast]
    away_stats = [away_rating, away_forecast, 1, 1, away_forecast]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=home_stats, theta=categories, fill='toself', name=f"{team_name_mapping[home_team]}"))
    fig.add_trace(go.Scatterpolar(r=away_stats, theta=categories, fill='toself', name=f"{team_name_mapping[away_team]}"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Team Comparison Radar Chart"
    )
    st.plotly_chart(fig)

# Function to create a confidence heatmap
def plot_confidence_heatmap(results):
    confidence_data = pd.DataFrame(results)
    heatmap_data = confidence_data.pivot("Home_Team", "Away_Team", "Confidence")
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Confidence Level'}, fmt=".1f")
    plt.title("Confidence Heatmap for Today's Matchups", fontsize=16)
    plt.xlabel("Away Team", fontsize=12)
    plt.ylabel("Home Team", fontsize=12)
    st.pyplot(plt)

# Function to create a summary table for recommended bets
def display_betting_summary(results):
    summary_df = pd.DataFrame(results)
    st.write("### Recommended Bets Summary")
    st.dataframe(summary_df[['Predicted Winner', 'Spread Lean', 'Total Lean', 'Confidence']].style.highlight_max(axis=0, color='lightgreen'))

# Function to create a confidence-level breakdown pie chart
def plot_confidence_breakdown(results):
    confidence_counts = pd.Series([result['Confidence'] for result in results]).value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(confidence_counts, labels=confidence_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    plt.title("Confidence Level Breakdown", fontsize=16)
    st.pyplot(plt)

# Function to visualize predicted spreads and totals
def plot_spread_total_overlays(results):
    spreads = [result['Spread Lean'] for result in results]
    totals = [result['Total Lean'] for result in results]
    teams = [result['Home_Team'] for result in results]
    
    plt.figure(figsize=(12, 6))
    x = range(len(teams))
    plt.bar(x, spreads, width=0.4, label='Predicted Spread', alpha=0.6, color='blue', align='center')
    plt.bar([p + 0.4 for p in x], totals, width=0.4, label='Predicted Total', alpha=0.6, color='orange', align='center')
    plt.xticks([p + 0.2 for p in x], teams)
    plt.title("Predicted Spreads and Totals", fontsize=16)
    plt.xlabel("Teams", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.legend()
    st.pyplot(plt)

# Function to create radar charts for matchup comparison
def plot_matchup_radar(home_stats, away_stats, home_team, away_team):
    categories = ["Average Score", "Max Score", "Consistency", "Recent Form", "Predicted PTS"]
    home_values = [home_stats['avg_score'], home_stats['max_score'], home_stats['std_dev'], home_stats['recent_form'], home_stats['Predicted_PTS']]
    away_values = [away_stats['avg_score'], away_stats['max_score'], away_stats['std_dev'], away_stats['recent_form'], away_stats['Predicted_PTS']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=home_values, theta=categories, fill='toself', name=home_team))
    fig.add_trace(go.Scatterpolar(r=away_values, theta=categories, fill='toself', name=away_team))
    fig.update_layout(title="Matchup Radar Comparison", polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig)

# Function to highlight best bets
def highlight_best_bets(results):
    best_bets = [result for result in results if result['Confidence'] > 75]  # Example threshold
    st.write("### Best Bets")
    for bet in best_bets:
        st.write(f"**{bet['Predicted Winner']}** - Spread Lean: {bet['Spread Lean']}, Total Lean: {bet['Total Lean']}")

# Function to display key matchup insights
def display_key_insights(home_team_name, away_team_name, home_stats, away_stats, recent_performance):
    st.write(f"### Key Insights for {home_team_name} vs {away_team_name}")
    st.write(f"- Recent Performance (Last 5 Games): {recent_performance}")
    st.write(f"- **Home Team Average Points:** {home_stats.get('avg_score', 'N/A'):.2f}")
    st.write(f"- **Away Team Average Points:** {away_stats.get('avg_score', 'N/A'):.2f}")

# Function to plot team performance trends
def plot_team_performance_trends(team_data, team_abbrev):
    plt.figure(figsize=(10, 5))
    plt.plot(team_data['GAME_DATE'], team_data['PTS'], label='Points Scored', marker='o', color='purple')
    plt.title(f"{team_abbrev} Performance Trends Over Last 5 Games", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Points", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

# Streamlit Interface
st.title("NBA FoxEdge Betting Insights with Spread and Total Leans")

# Add this function before the button logic
@st.cache_data
def fetch_current_season_stats(season):
    """
    Fetch game logs for the current season and calculate team stats.
    """
    try:
        # Fetch all game logs for the current season
        game_logs = LeagueGameLog(season=season, season_type_all_star='Regular Season', player_or_team_abbreviation='T')
        games = game_logs.get_data_frames()[0]

        # Convert game dates to datetime
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        
        # Filter for recent games (last 30 days)
        recent_games = games[games['GAME_DATE'] >= datetime.now() - timedelta(days=30)]

        # Calculate team stats
        team_stats = games.groupby('TEAM_ABBREVIATION').apply(
            lambda x: pd.Series({
                'avg_score': x['PTS'].mean(),
                'max_score': x['PTS'].max(),
                'std_dev': x['PTS'].std(),
                'recent_form': recent_games[recent_games['TEAM_ABBREVIATION'] == x.name]['PTS'].mean()
            })
        ).to_dict(orient='index')

        return team_stats
    except Exception as e:
        st.error(f"Error fetching current season stats: {e}")
        return {}

# Main logic for predicting today's games
if st.button("Predict Today's Games"):
    season = "2024-25"
    data, failed_teams = fetch_and_preprocess_data(season)

    if data.empty:
        st.error("No data available for analysis.")
    else:
        team_data = aggregate_points_by_team(data)
        team_models = train_arima_models(team_data)
        team_forecasts = compute_team_forecasts(team_models, team_data)
        current_season_stats = fetch_current_season_stats(season)
        todays_games = fetch_todays_games()

        results = []
        if not todays_games.empty:
            for _, game in todays_games.iterrows():
                home_team = game["HOME_TEAM_ABBREVIATION"]
                away_team = game["VISITOR_TEAM_ABBREVIATION"]
                result = predict_game_outcome_with_arima(home_team, away_team, team_forecasts, current_season_stats)

                home_forecast = team_forecasts[team_forecasts['Team'] == home_team]['Predicted_PTS'].mean()
                away_forecast = team_forecasts[team_forecasts['Team'] == away_team]['Predicted_PTS'].mean()

                display_matchup_details(home_team, away_team, result, team_name_mapping, team_forecasts)

                results.append({
                    'Home_Team': home_team,
                    'Away_Team': away_team,
                    'Predicted Winner': result[0],
                    'Spread Lean': f"Lean {home_team} -{int(result[2])}" if result[2] > 0 else f"Lean {away_team} +{int(abs(result[2]))}",
                    'Total Lean': f"Lean Over if the line is < {home_forecast + away_forecast - 5:.2f}",
                    'Confidence': result[3]
                })

            # Visualizations
            plot_confidence_heatmap(results)
            display_betting_summary(results)
            plot_confidence_breakdown(results)
            plot_spread_total_overlays(results)
            highlight_best_bets(results)

            # Display key insights and trends
            for result in results:
                # Retrieve the stats for the home and away teams
                home_stats = current_season_stats.get(result['Home_Team'], {})
                away_stats = current_season_stats.get(result['Away_Team'], {})
                
                # Get team names from the mapping
                home_team_name = team_name_mapping.get(result['Home_Team'], "Unknown Home Team")
                away_team_name = team_name_mapping.get(result['Away_Team'], "Unknown Away Team")
                
                # Pass the stats and team names to the display function
                display_key_insights(home_team_name, away_team_name, home_stats, away_stats, "Recent performance data here")  # Replace with actual data
                plot_team_performance_trends(data[data['TEAM_ABBREV'] == result['Home_Team']], result['Home_Team'])
                plot_team_performance_trends(data[data['TEAM_ABBREV'] == result['Away_Team']], result['Away_Team'])

        if failed_teams:
            st.warning(f"Data could not be fetched for the following teams: {', '.join(failed_teams)}")

# CSS for Color Palette
st.markdown('''
    <style>
        /* Root Variables */
        :root {
            --background-gradient-start: #1A1A1A; /* Dark Gray */
            --background-gradient-end: #2C2C2C; /* Slightly Lighter Gray */
            --primary-text-color: #FFFFFF; /* Crisp White */
            --heading-text-color: #FFD700; /* Gold */
            --accent-color-teal: #00BFFF; /* Deep Sky Blue */
            --accent-color-purple: #8A2BE2; /* Blue Violet */
            --highlight-color: #FF4500; /* Orange Red */
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

        h1, h2, h3 {
            font-family: var(--font-heading);
            color: var(--heading-text-color);
        }

        /* Button Styles */
        .stButton > button {
            background-color: var(--accent-color-teal);
            color: var(--primary-text-color);
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }

        .stButton > button:hover {
            background-color: var(--highlight-color);
        }
    </style>
''', unsafe_allow_html=True)
