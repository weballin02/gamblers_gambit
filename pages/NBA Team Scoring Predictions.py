# NBA Team Scoring Predictions

# Import Libraries
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import joblib
import os
import numpy as np
import streamlit as st
from nba_api.stats.endpoints import teamgamelog, ScoreboardV2
from nba_api.stats.static import teams as nba_teams
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings('ignore')

# Streamlit App Configuration
st.set_page_config(
    page_title="NBA Team Scoring Predictions",
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
st.button("🌗 Toggle Theme", on_click=toggle_theme)

# Apply Theme Based on Dark Mode
if st.session_state.dark_mode:
    primary_bg = "#121212"
    secondary_bg = "#1E1E1E"
    primary_text = "#FFFFFF"
    secondary_text = "#B0B0B0"
    accent_color = "#BB86FC"
    highlight_color = "#03DAC6"
    chart_color = "#BB86FC"
    chart_template = "plotly_dark"
else:
    primary_bg = "#FFFFFF"
    secondary_bg = "#1E1E1E"
    primary_text = "#FFFFFF"
    secondary_text = "#B0B0B0"
    accent_color = "#6200EE"
    highlight_color = "#03DAC6"
    chart_color = "#6200EE"
    chart_template = "plotly_white"

# General Styling and High Contrast Toggle
st.markdown(f"""
    <style>
    /* Overall Page Styling */
    body {{
        background-color: {primary_bg};
        color: {primary_text};
        font-family: 'Open Sans', sans-serif;
    }}

    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Header Section */
    .header-title {{
        font-family: 'Montserrat', sans-serif;
        background: linear-gradient(90deg, {highlight_color}, {accent_color});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5em;
    }}

    .subheader-text {{
        color: {secondary_text};
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 1.5em;
    }}

    /* Gradient Bar */
    .gradient-bar {{
        height: 10px;
        background: linear-gradient(90deg, {highlight_color}, {accent_color});
        border-radius: 5px;
    }}

    /* Button Styling */
    div.stButton > button {{
        background: linear-gradient(90deg, {highlight_color}, {accent_color});
        color: {primary_text};
        border: none;
        padding: 0.8em 1.5em;
        border-radius: 30px;
        font-size: 1em;
        cursor: pointer;
        transition: transform 0.3s ease, background 0.3s ease;
    }}

    div.stButton > button:hover {{
        transform: translateY(-5px);
        background: linear-gradient(90deg, {accent_color}, {highlight_color});
    }}

    /* High Contrast Toggle */
    .high-contrast {{
        background-color: #000000;
        color: #FFFFFF;
    }}

    /* Data Section Styling */
    .data-section {{
        padding: 2em 1em;
        background-color: {secondary_bg};
        border-radius: 15px;
        margin: 2em 0;
    }}

    .data-section h2 {{
        font-size: 2em;
        margin-bottom: 0.5em;
        color: {accent_color};
    }}

    .data-section p {{
        font-size: 1em;
        color: {secondary_text};
        margin-bottom: 1em;
    }}

    /* Plotly Chart Styling */
    .plotly-graph-div {{
        background-color: {secondary_bg} !important;
    }}

    /* Footer Styling */
    .footer {{
        text-align: center;
        padding: 2em 1em;
        color: {secondary_text};
        font-size: 0.9em;
    }}

    .footer a {{
        color: {accent_color};
        text-decoration: none;
    }}

    /* Responsive Design */
    @media (max-width: 768px) {{
        .header-title {{
            font-size: 2em;
        }}

        .subheader-text {{
            font-size: 1em;
        }}

        .data-section {{
            padding: 1em 0.5em;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown(f'''
    <div>
        <h1 class="header-title">NBA Team Scoring Predictions</h1>
        <p class="subheader-text">Explore team scoring trends and forecasts for smarter decisions.</p>
    </div>
''', unsafe_allow_html=True)

# Fetch NBA team abbreviations and IDs
nba_team_list = nba_teams.get_teams()
team_abbreviations = [team['abbreviation'] for team in nba_team_list]

# Define Team Name Mapping Globally
team_name_mapping = {team['abbreviation']: team['full_name'] for team in nba_team_list}
inverse_team_name_mapping = {v: k for k, v in team_name_mapping.items()}

# Fetch and Preprocess Data from NBA API
@st.cache_data
def fetch_and_preprocess_data(season):
    """Fetch data for the specified season from NBA API and preprocess it for model training."""
    all_data = []

    # Iterate over each team to get their game logs
    for team in nba_team_list:
        team_id = team['id']
        team_abbrev = team['abbreviation']

        # Fetch team game logs for this team and season
        try:
            team_logs = teamgamelog.TeamGameLog(team_id=team_id, season=season).get_data_frames()[0]

            # Filter and rename columns to match our target structure
            team_logs = team_logs[['Game_ID', 'GAME_DATE', 'PTS']]
            team_logs['TEAM_ABBREVIATION'] = team_abbrev
            all_data.append(team_logs)
        
        except Exception as e:
            st.error(f"Error fetching data for team {team_abbrev}: {e}")
            continue

    # Concatenate all teams' data
    data = pd.concat(all_data, ignore_index=True)
    data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'])
    return data

# Aggregate Points by Team and Date
def aggregate_points_by_team(data):
    team_data = data.groupby(['GAME_DATE', 'TEAM_ABBREVIATION'])['PTS'].sum().reset_index()
    team_data.set_index('GAME_DATE', inplace=True)
    return team_data

# Initialize, Train, Save, and Load ARIMA Models for Each Team
@st.cache_resource
def train_arima_models(team_data):
    model_dir = 'models/nba'
    os.makedirs(model_dir, exist_ok=True)

    team_models = {}
    teams_list = team_data['TEAM_ABBREVIATION'].unique()

    for team_abbrev in teams_list:
        model_filename = f'{team_abbrev}_arima_model.pkl'
        model_path = os.path.join(model_dir, model_filename)
        team_points = team_data[team_data['TEAM_ABBREVIATION'] == team_abbrev]['PTS']
        team_points.reset_index(drop=True, inplace=True)

        # Train a new ARIMA model
        model = auto_arima(
            team_points,
            seasonal=False,
            trace=False,
            error_action='ignore',
            suppress_warnings=True
        )
        model.fit(team_points)

        # Save the trained model
        joblib.dump(model, model_path)
        team_models[team_abbrev] = model

    return team_models

# Button to fetch data, retrain models, and predict
if st.button("Refresh Data & Retrain Models"):
    with st.spinner("Fetching data and retraining models..."):
        # Fetch and preprocess data
        season = "2024-25"  # Use the current season or update dynamically
        data = fetch_and_preprocess_data(season)
        team_data = aggregate_points_by_team(data)

        # Train ARIMA models with the latest data
        st.session_state['team_models'] = train_arima_models(team_data)
        st.session_state['team_data'] = team_data
        st.success("Data refreshed and models retrained.")

# Retrieve models and data from session state on first load
if 'team_models' not in st.session_state or 'team_data' not in st.session_state:
    season = "2024-25"
    data = fetch_and_preprocess_data(season)
    team_data = aggregate_points_by_team(data)
    st.session_state['team_models'] = train_arima_models(team_data)
    st.session_state['team_data'] = team_data

team_models = st.session_state['team_models']
team_data = st.session_state['team_data']

# Forecast the Next 5 Games for Each Team
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
        if isinstance(forecast, pd.Series):
            forecast = forecast.values
        predictions = pd.DataFrame({'Date': future_dates, 'Predicted_PTS': forecast, 'Team': team_abbrev})
        team_forecasts[team_abbrev] = predictions

    return pd.concat(team_forecasts.values(), ignore_index=True) if team_forecasts else pd.DataFrame(columns=['Date', 'Predicted_PTS', 'Team'])

# Compute Team Forecasts
all_forecasts = compute_team_forecasts(team_models, team_data)

# Streamlit App for Team Points Prediction
st.markdown('<div class="data-section"><h2>Select a Team for Prediction</h2></div>', unsafe_allow_html=True)

# Dropdown menu for selecting a team
teams_list = sorted(team_data['TEAM_ABBREVIATION'].unique())
team_abbrev = st.selectbox('Select a team for prediction:', teams_list)

if team_abbrev:
    team_full_name = team_name_mapping.get(team_abbrev, "Unknown Team")
    team_points = team_data[team_data['TEAM_ABBREVIATION'] == team_abbrev]['PTS']
    team_points.index = pd.to_datetime(team_points.index)

    st.markdown(f'<div class="data-section"><h2>Historical Points for {team_full_name}</h2></div>', unsafe_allow_html=True)
    st.line_chart(team_points)

    team_forecast = all_forecasts[all_forecasts['Team'] == team_abbrev]
    st.markdown(f'<div class="data-section"><h2>Predicted Points for Next 5 Games ({team_full_name})</h2></div>', unsafe_allow_html=True)
    st.write(team_forecast[['Date', 'Predicted_PTS']])

    st.markdown(f'<div class="data-section"><h2>Points Prediction for {team_full_name}</h2></div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    forecast_dates = mdates.date2num(team_forecast['Date'])
    historical_dates = mdates.date2num(team_points.index)

    ax.plot(historical_dates, team_points.values, label=f'Historical Points for {team_full_name}', color=chart_color)
    ax.plot(forecast_dates, team_forecast['Predicted_PTS'], label='Predicted Points', color=highlight_color)
    lower_bound = team_forecast['Predicted_PTS'] - 5
    upper_bound = team_forecast['Predicted_PTS'] + 5
    finite_indices = np.isfinite(forecast_dates) & np.isfinite(lower_bound) & np.isfinite(upper_bound)
    ax.fill_between(forecast_dates[finite_indices], lower_bound.values[finite_indices], upper_bound.values[finite_indices], color='gray', alpha=0.2, label='Confidence Interval')
    ax.xaxis_date()
    fig.autofmt_xdate()
    ax.set_title(f'Points Prediction for {team_full_name}', color=primary_text)
    ax.set_xlabel('Date', color=primary_text)
    ax.set_ylabel('Points', color=primary_text)
    ax.legend()
    ax.grid(True)
    ax.set_facecolor(secondary_bg)
    fig.patch.set_facecolor(primary_bg)
    st.pyplot(fig)

# Fetch upcoming games using nba_api and get predictions
st.markdown('<div class="data-section"><h2>NBA Game Predictions for Today</h2></div>', unsafe_allow_html=True)

# Fetch upcoming games
@st.cache_data(ttl=3600)
def fetch_nba_games():
    today = datetime.now().strftime('%m/%d/%Y')
    scoreboard = ScoreboardV2(game_date=today)
    games = scoreboard.get_data_frames()[0]
    return games

games = fetch_nba_games()

# Map team IDs to abbreviations
nba_team_dict = {int(team['id']): team['abbreviation'] for team in nba_team_list}

# Map team IDs to your dataset's abbreviations
games['HOME_TEAM_ABBREVIATION'] = games['HOME_TEAM_ID'].map(nba_team_dict)
games['VISITOR_TEAM_ABBREVIATION'] = games['VISITOR_TEAM_ID'].map(nba_team_dict)

# Filter out any games where teams could not be mapped
games = games.dropna(subset=['HOME_TEAM_ABBREVIATION', 'VISITOR_TEAM_ABBREVIATION'])

# Process game data
game_list = []
for index, row in games.iterrows():
    game_id = row['GAME_ID']
    home_team_abbrev = row['HOME_TEAM_ABBREVIATION']
    away_team_abbrev = row['VISITOR_TEAM_ABBREVIATION']
    home_team_full = team_name_mapping.get(home_team_abbrev)
    away_team_full = team_name_mapping.get(away_team_abbrev)

    # Create game label
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

# Check if there are games today
if not games_df.empty:
    st.markdown('<div class="data-section"><h2>Select a Game for Prediction</h2></div>', unsafe_allow_html=True)
    game_selection = st.selectbox('Select a game to get predictions:', games_df['Game Label'])

    # Get selected game details
    selected_game = games_df[games_df['Game Label'] == game_selection].iloc[0]
    home_team_abbrev = selected_game['Home Team Abbrev']
    away_team_abbrev = selected_game['Away Team Abbrev']
    home_team_full = selected_game['Home Team Full']
    away_team_full = selected_game['Away Team Full']

    # Get models
    home_team_model = team_models.get(home_team_abbrev)
    away_team_model = team_models.get(away_team_abbrev)

    if home_team_model and away_team_model:
        # Predict points
        home_team_forecast = home_team_model.predict(n_periods=1)
        away_team_forecast = away_team_model.predict(n_periods=1)

        # Access the prediction value
        if isinstance(home_team_forecast, pd.Series):
            home_team_forecast = home_team_forecast.iloc[0]
        else:
            home_team_forecast = home_team_forecast[0]

        if isinstance(away_team_forecast, pd.Series):
            away_team_forecast = away_team_forecast.iloc[0]
        else:
            away_team_forecast = away_team_forecast[0]

        st.markdown(f'''
            <div class="data-section">
                <h2>Predicted Points</h2>
                <p><strong>{home_team_full} ({home_team_abbrev}):</strong> {home_team_forecast:.2f}</p>
                <p><strong>{away_team_full} ({away_team_abbrev}):</strong> {away_team_forecast:.2f}</p>
        ''', unsafe_allow_html=True)

        if home_team_forecast > away_team_forecast:
            st.markdown(f'<p style="color: {highlight_color}; font-weight: bold;">Predicted Winner: {home_team_full}</p>', unsafe_allow_html=True)
        elif away_team_forecast > home_team_forecast:
            st.markdown(f'<p style="color: {highlight_color}; font-weight: bold;">Predicted Winner: {away_team_full}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="font-weight: bold;">Predicted Outcome: Tie</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Prediction models for one or both teams are not available.")
else:
    st.write("No games scheduled for today.")

# Footer
st.markdown(f'''
    <div class="footer">
        &copy; {datetime.now().year} NBA Team Scoring Predictions. All rights reserved.
    </div>
''', unsafe_allow_html=True)
