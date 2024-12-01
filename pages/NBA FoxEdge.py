# NBA FoxEdge

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

# Streamlit App Configuration
st.set_page_config(
    page_title="FoxEdge NBA Insights",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Session State for Theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Function to Toggle Theme
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Theme Toggle Button
st.sidebar.button("🌗 Toggle Theme", on_click=toggle_theme)

# Apply Theme Based on Dark Mode
if st.session_state.dark_mode:
    primary_bg = "#121212"
    primary_text = "#FFFFFF"
    secondary_bg = "#1E1E1E"
    accent_color = "#BB86FC"
    highlight_color = "#03DAC6"
    chart_template = "plotly_dark"
else:
    primary_bg = "#FFFFFF"
    primary_text = "#000000"
    secondary_bg = "#F5F5F5"
    accent_color = "#6200EE"
    highlight_color = "#03DAC6"
    chart_template = "plotly_white"

# General Styling and Applied FoxEdge Colors
st.markdown(f"""
    <style>
    /* Overall Page Styling */
    body {{
        background-color: {primary_bg};
        color: {primary_text};
        font-family: 'Open Sans', sans-serif;
    }}

    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Header Styling */
    .header-title {{
        font-family: 'Montserrat', sans-serif;
        background: linear-gradient(120deg, {highlight_color}, {accent_color});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5em;
    }}

    /* Subheader Styling */
    .subheader-text {{
        color: {primary_text};
        opacity: 0.7;
        font-size: 1.5em;
        text-align: center;
        margin-bottom: 1.5em;
    }}

    /* Button Styling */
    .button {{
        background: linear-gradient(45deg, {highlight_color}, {accent_color});
        border: none;
        padding: 0.8em 2em;
        color: #FFFFFF;
        font-size: 1.1em;
        border-radius: 30px;
        cursor: pointer;
        transition: transform 0.3s ease, background 0.3s ease;
        text-decoration: none;
        display: inline-block;
        margin-top: 1em;
    }}

    /* Button Hover Effect */
    .button:hover {{
        transform: translateY(-5px);
        background: linear-gradient(45deg, {accent_color}, {highlight_color});
    }}

    /* Data Section Styling */
    .data-section {{
        padding: 2em 1em;
        text-align: center;
        background-color: {secondary_bg};
        border-radius: 15px;
        margin: 2em 0;
    }}

    .data-section h2 {{
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }}

    .data-section p {{
        font-size: 1.2em;
        color: {primary_text};
        opacity: 0.8;
        margin-bottom: 2em;
    }}

    /* Summary Section Styling */
    .summary-section {{
        padding: 2em 1em;
        background-color: {secondary_bg};
        border-radius: 15px;
        margin-bottom: 2em;
    }}

    .summary-section h3 {{
        font-size: 2em;
        margin-bottom: 0.5em;
        color: {highlight_color};
    }}

    .summary-section p {{
        font-size: 1.1em;
        color: {primary_text};
        opacity: 0.9;
        line-height: 1.6;
    }}

    /* Team Comparison Card Styling */
    .team-comparison {{
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem auto;
        max-width: 1200px;
        flex-wrap: wrap;
    }}

    .team-card {{
        flex: 1;
        background-color: {secondary_bg};
        border-radius: 15px;
        padding: 1.5rem;
        max-width: 500px;
        margin-bottom: 1em;
        transition: transform 0.3s ease;
    }}

    .team-card:hover {{
        transform: translateY(-5px);
    }}

    .stats-grid {{
        display: grid;
        gap: 1rem;
        margin: 1rem 0;
    }}

    .stat-item {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        background-color: {primary_bg};
        border-radius: 8px;
    }}

    /* Plotly Chart Styling */
    .plotly-graph-div {{
        background-color: {secondary_bg} !important;
    }}

    /* Footer Styling */
    .footer {{
        text-align: center;
        padding: 2em 1em;
        color: {primary_text};
        opacity: 0.6;
        font-size: 0.9em;
    }}

    .footer a {{
        color: {accent_color};
        text-decoration: none;
    }}

    /* Highlighted Text */
    .highlight {{
        color: {highlight_color};
        font-weight: bold;
    }}

    /* Responsive Design */
    @media (max-width: 768px) {{
        .header-title {{
            font-size: 2em;
        }}

        .subheader-text {{
            font-size: 1em;
        }}

        .team-comparison {{
            flex-direction: column;
            align-items: center;
        }}

        .team-card {{
            width: 90%;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

# Title and Subheader
st.markdown('<div class="header-title">FoxEdge NBA Betting Insights with Spread and Total Leans</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">Predict game outcomes and uncover key trends to inform your betting strategies.</div>', unsafe_allow_html=True)

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
            team_logs['TEAM_ABBREVIATION'] = team_abbrev
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
    team_data = data.groupby(['GAME_DATE', 'TEAM_ABBREVIATION'])['PTS'].sum().reset_index()
    team_data.set_index('GAME_DATE', inplace=True)
    return team_data

# Train ARIMA Models
@st.cache_resource
def train_arima_models(team_data):
    model_dir = 'models/nba'
    os.makedirs(model_dir, exist_ok=True)
    team_models = {}
    for team_abbrev in team_data['TEAM_ABBREVIATION'].unique():
        model_path = os.path.join(model_dir, f"{team_abbrev}_arima_model.pkl")
        team_points = team_data[team_data['TEAM_ABBREVIATION'] == team_abbrev]['PTS']
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
        team_points = team_data[team_data['TEAM_ABBREVIATION'] == team_abbrev]['PTS']
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

    st.markdown(f"""
        <div class="summary-section">
            <h3>{team_name_mapping[home_team]} vs {team_name_mapping[away_team]}</h3>
            <p><strong>Predicted Winner:</strong> {team_name_mapping.get(winner, 'Unavailable')}</p>
            <p><strong>Predicted Final Scores:</strong> {team_name_mapping[home_team]} {home_forecast:.2f} - {team_name_mapping[away_team]} {away_forecast:.2f}</p>
            <p><strong>Score Difference:</strong> {score_diff:.2f}</p>
            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
        </div>
    """, unsafe_allow_html=True)

    # Betting Insights Section
    st.markdown('### Betting Insights')
    
    # Spread Lean
    spread_lean = f"Lean {team_name_mapping[winner]} -{int(score_diff)}" if score_diff > 0 else f"Lean {team_name_mapping[away_team]} +{int(abs(score_diff))}"
    st.markdown(f"#### Spread Lean: **{spread_lean}**")

    # Total Lean
    total_points = home_forecast + away_forecast
    total_recommendation = f"Lean Over if the line is < {total_points - 5:.2f}"
    st.markdown(f"#### Total Lean: **{total_recommendation}**")

    # Radar Chart using Plotly
    categories = ["Average Score", "Max Score", "Consistency", "Recent Form", "Predicted PTS"]
    home_stats_values = [
        home_rating,
        current_season_stats[home_team]['max_score'],
        100 - current_season_stats[home_team]['std_dev'],  # Inverse of standard deviation for consistency
        current_season_stats[home_team]['recent_form'],
        home_forecast
    ]
    away_stats_values = [
        away_rating,
        current_season_stats[away_team]['max_score'],
        100 - current_season_stats[away_team]['std_dev'],
        current_season_stats[away_team]['recent_form'],
        away_forecast
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=home_stats_values, theta=categories, fill='toself', name=f"{team_name_mapping[home_team]}"))
    fig.add_trace(go.Scatterpolar(r=away_stats_values, theta=categories, fill='toself', name=f"{team_name_mapping[away_team]}"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Team Comparison Radar Chart",
        template=chart_template
    )
    st.plotly_chart(fig)
    
    # ROI Suggestion
    st.markdown(f"""
        <div class="summary-section">
            <h3>Betting Recommendations</h3>
            <p>Based on the analysis, consider the following betting strategies:</p>
            <ul>
                <li><strong>Spread Bet:</strong> {spread_lean}</li>
                <li><strong>Total Bet:</strong> {total_recommendation}</li>
            </ul>
            <p class="highlight">Ensure to consider the <strong>confidence level</strong> and perform your own research before placing bets.</p>
        </div>
    """, unsafe_allow_html=True)

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
                    'Confidence': result[2]
                })

            if failed_teams:
                st.warning(f"Data could not be fetched for the following teams: {', '.join(failed_teams)}")

# Footer
st.markdown(f"""
    <div class="footer">
        &copy; {datetime.now().year} <a href="#">FoxEdge</a>. All rights reserved.
    </div>
""", unsafe_allow_html=True)
