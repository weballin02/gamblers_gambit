# Import Libraries
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import streamlit as st
from pmdarima import auto_arima
import joblib
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="FoxEdge - NFL Team Scoring Predictions",
    page_icon="🏈",
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
        <p>NFL Team Scoring Predictions</p>
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
        <h2>Explore Scoring Predictions and Trends to Refine Your Strategy</h2>
    </div>
''', unsafe_allow_html=True)

# Load and Preprocess Data
@st.cache_data
def fetch_and_preprocess_data():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])

    # Prepare home and away data
    home_df = schedule[['gameday', 'home_team', 'home_score']].copy().rename(columns={'home_team': 'team', 'home_score': 'score'})
    away_df = schedule[['gameday', 'away_team', 'away_score']].copy().rename(columns={'away_team': 'team', 'away_score': 'score'})

    # Combine home and away data
    team_data = pd.concat([home_df, away_df], ignore_index=True)
    team_data['gameday'] = pd.to_datetime(team_data['gameday'], errors='coerce')
    team_data.dropna(subset=['score'], inplace=True)
    team_data.set_index('gameday', inplace=True)
    team_data.sort_index(inplace=True)

    return team_data

# Fetch data
team_data = fetch_and_preprocess_data()

# Train or Load ARIMA Models
@st.cache_resource
def get_team_models(team_data):
    model_dir = 'models/nfl/'
    os.makedirs(model_dir, exist_ok=True)

    team_models = {}
    teams_list = team_data['team'].unique()

    for team in teams_list:
        model_path = os.path.join(model_dir, f'{team}_arima_model.pkl')
        team_scores = team_data[team_data['team'] == team]['score']
        team_scores.reset_index(drop=True, inplace=True)

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            team_models[team] = model
        else:
            if len(team_scores) < 10:
                continue
            model = auto_arima(
                team_scores,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True
            )
            model.fit(team_scores)
            joblib.dump(model, model_path)
            team_models[team] = model

    return team_models

# Get team models
team_models = get_team_models(team_data)

# Function to Predict Team Score
def predict_team_score(team, periods=1):
    """Predict the score for a given team and number of future periods."""
    model = team_models.get(team)
    if model:
        forecast = model.predict(n_periods=periods)
        if isinstance(forecast, pd.Series):
            forecast = forecast.values
        return forecast[0]
    else:
        return None

# Forecast the Next 5 Games for Each Team
@st.cache_data
def compute_team_forecasts(_team_models, team_data):
    team_forecasts = {}
    forecast_periods = 5

    for team, model in _team_models.items():
        team_scores = team_data[team_data['team'] == team]['score']
        if team_scores.empty:
            continue
        last_date = team_scores.index.max()

        # Generate future dates for weekly games
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=forecast_periods, freq='7D')
        forecast = model.predict(n_periods=forecast_periods)

        predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Score': forecast,
            'Team': team
        })
        team_forecasts[team] = predictions

    return pd.concat(team_forecasts.values(), ignore_index=True) if team_forecasts else pd.DataFrame(columns=['Date', 'Predicted_Score', 'Team'])

# Compute forecasts
all_forecasts = compute_team_forecasts(team_models, team_data)

# Fetch Upcoming Games
@st.cache_data(ttl=3600)
def fetch_upcoming_games():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])

    schedule['game_datetime'] = pd.to_datetime(
        schedule['gameday'].astype(str) + ' ' + schedule['gametime'].astype(str),
        errors='coerce',
        utc=True
    )

    schedule.dropna(subset=['game_datetime'], inplace=True)

    now = datetime.now(pytz.UTC)
    today_weekday = now.weekday()

    if today_weekday == 3:
        target_days = [3, 6, 0]
    elif today_weekday == 6:
        target_days = [6, 0, 3]
    elif today_weekday == 0:
        target_days = [0, 3, 6]
    else:
        target_days = [3, 6, 0]

    upcoming_game_dates = [
        now + timedelta(days=(d - today_weekday + 7) % 7)
        for d in target_days
    ]

    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') &
        (schedule['game_datetime'].dt.date.isin([date.date() for date in upcoming_game_dates]))
    ]

    upcoming_games = upcoming_games[['game_id', 'game_datetime', 'home_team', 'away_team']]

    team_abbrev_mapping = {
        'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills',
        'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns',
        'DAL': 'Dallas Cowboys', 'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
        'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'KC': 'Kansas City Chiefs',
        'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams', 'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins',
        'MIN': 'Minnesota Vikings', 'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
        'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers', 'SEA': 'Seattle Seahawks',
        'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
    }

    upcoming_games['home_team_full'] = upcoming_games['home_team'].map(team_abbrev_mapping)
    upcoming_games['away_team_full'] = upcoming_games['away_team'].map(team_abbrev_mapping)

    upcoming_games.dropna(subset=['home_team_full', 'away_team_full'], inplace=True)
    upcoming_games.reset_index(drop=True, inplace=True)

    return upcoming_games

# Fetch updated upcoming games
upcoming_games = fetch_upcoming_games()

# Team abbreviation to full name mapping
team_abbrev_mapping = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys', 'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'KC': 'Kansas City Chiefs',
    'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams', 'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings', 'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers', 'SEA': 'Seattle Seahawks',
    'SF': 'San Francisco 49ers', 'TB': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
}
inverse_team_abbrev_mapping = {v: k for k, v in team_abbrev_mapping.items()}

# Streamlit App UI

# Sidebar for team selection
st.sidebar.markdown('''
    <div class="controls-section">
        <h3>Select a Team</h3>
    ''', unsafe_allow_html=True)

teams_list = sorted(team_data['team'].unique())
team_full_names = [team_abbrev_mapping.get(team, team) for team in teams_list]
team_selection = st.sidebar.selectbox('Choose a team for prediction:', team_full_names)

st.sidebar.markdown('</div>', unsafe_allow_html=True)

if team_selection:
    team_abbrev = inverse_team_abbrev_mapping.get(team_selection, team_selection)
    team_scores = team_data[team_data['team'] == team_abbrev]['score']
    team_scores.index = pd.to_datetime(team_scores.index)

    st.markdown(f'''
        <div class="data-section">
            <h2>Historical Scores for {team_selection}</h2>
        </div>
    ''', unsafe_allow_html=True)
    st.line_chart(team_scores)

    # Display future predictions
    team_forecast = all_forecasts[all_forecasts['Team'] == team_abbrev]
    if not team_forecast.empty:
        st.markdown(f'''
            <div class="data-section">
                <h2>Predicted Scores for Next 5 Games ({team_selection})</h2>
            </div>
        ''', unsafe_allow_html=True)
        st.dataframe(team_forecast[['Date', 'Predicted_Score']].set_index('Date'))

        # Plot the historical and predicted scores
        fig, ax = plt.subplots(figsize=(10, 6))
        forecast_dates = mdates.date2num(team_forecast['Date'])
        historical_dates = mdates.date2num(team_scores.index)

        ax.plot(historical_dates, team_scores.values, label=f'Historical Scores for {team_selection}', color='blue')
        ax.plot(forecast_dates, team_forecast['Predicted_Score'], label='Predicted Scores', color='red')
        lower_bound = team_forecast['Predicted_Score'] - 5
        upper_bound = team_forecast['Predicted_Score'] + 5
        finite_indices = np.isfinite(forecast_dates) & np.isfinite(lower_bound) & np.isfinite(upper_bound)
        ax.fill_between(forecast_dates[finite_indices], lower_bound.values[finite_indices], upper_bound.values[finite_indices], color='gray', alpha=0.2, label='Confidence Interval')
        ax.xaxis_date()
        fig.autofmt_xdate()
        ax.set_title(f'Score Prediction for {team_selection}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.warning(f"No forecast available for {team_selection}.")

# Game Predictions
st.markdown('''
    <div class="data-section">
        <h2>NFL Game Predictions for Upcoming Games</h2>
    </div>
''', unsafe_allow_html=True)

if not upcoming_games.empty:
    game_options = upcoming_games['home_team_full'] + " vs " + upcoming_games['away_team_full']
    game_selection = st.selectbox('Select an upcoming game:', game_options)
    selected_game = upcoming_games[
        (upcoming_games['home_team_full'] == game_selection.split(" vs ")[0]) &
        (upcoming_games['away_team_full'] == game_selection.split(" vs ")[1])
    ].iloc[0]

    home_team = selected_game['home_team_full']
    away_team = selected_game['away_team_full']

    home_team_abbrev = inverse_team_abbrev_mapping.get(home_team)
    away_team_abbrev = inverse_team_abbrev_mapping.get(away_team)

    home_team_score = predict_team_score(home_team_abbrev)
    away_team_score = predict_team_score(away_team_abbrev)

    if home_team_score is not None and away_team_score is not None:
        st.markdown('''
            <div class="results-section">
                <h3>Predicted Scores</h3>
                <div class="metric-container">
        ''', unsafe_allow_html=True)

        st.markdown(f'''
            <div class="metric">
                <h4>{home_team}</h4>
                <p>{home_team_score:.2f}</p>
            </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'''
            <div class="metric">
                <h4>{away_team}</h4>
                <p>{away_team_score:.2f}</p>
            </div>
        ''', unsafe_allow_html=True)

        if home_team_score > away_team_score:
            winner = home_team
        elif away_team_score > home_team_score:
            winner = away_team
        else:
            winner = "Tie"

        st.markdown(f'''
                </div>
                <h3>Predicted Winner: {winner}</h3>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.error("Prediction models for one or both teams are not available.")
else:
    st.write("No upcoming games found.")

# Footer
st.markdown('''
    <div class="footer">
        &copy; 2023 <a href="#">FoxEdge</a>. All rights reserved.
    </div>
''', unsafe_allow_html=True)
