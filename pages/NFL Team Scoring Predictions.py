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

# Streamlit App Title
st.title('NFL Team Scoring Predictions')
st.markdown("View projected NFL team scores for upcoming games based on recent stats. Select a team to see its scoring history and forecasts for the next five games. Use the insights on daily matchups and expected scores to refine your bets.")

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
                # Log missing model for teams with insufficient data
                st.warning(f"Insufficient data to train ARIMA model for {team}. Skipping.")
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
        st.warning(f"No model available for team {team}.")
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

# Fetch Upcoming Games (dynamically select based on current day of the week)
@st.cache_data(ttl=3600)
def fetch_upcoming_games():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year])

    # Combine 'gameday' and 'gametime' to create 'game_datetime'
    schedule['game_datetime'] = pd.to_datetime(
        schedule['gameday'].astype(str) + ' ' + schedule['gametime'].astype(str),
        errors='coerce',
        utc=True
    )

    # Drop rows where 'game_datetime' could not be parsed
    schedule.dropna(subset=['game_datetime'], inplace=True)

    # Get current day and upcoming key game days (Thursday, Sunday, Monday)
    now = datetime.now(pytz.UTC)
    today_weekday = now.weekday()  # Monday=0, Sunday=6

    # Determine which days to fetch based on the current day of the week
    if today_weekday == 3:  # Thursday
        target_days = [3, 6, 0]  # Thursday, Sunday, Monday
    elif today_weekday == 6:  # Sunday
        target_days = [6, 0, 3]  # Sunday, Monday, next Thursday
    elif today_weekday == 0:  # Monday
        target_days = [0, 3, 6]  # Monday, next Thursday, next Sunday
    else:
        # For non-gamedays (Tue, Wed, Fri, Sat), find next Thurs, Sun, Mon
        target_days = [3, 6, 0]  # Next Thursday, Sunday, Monday

    # Get game dates for each of the target days (within a week from today)
    upcoming_game_dates = [
        now + timedelta(days=(d - today_weekday + 7) % 7)
        for d in target_days
    ]

    # Filter games scheduled on the target dates and within the regular season
    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') &
        (schedule['game_datetime'].dt.date.isin([date.date() for date in upcoming_game_dates]))
    ]

    # Select necessary columns
    upcoming_games = upcoming_games[['game_id', 'game_datetime', 'home_team', 'away_team']]

    # Map team abbreviations to full names
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

    # Apply mapping for readability
    upcoming_games['home_team_full'] = upcoming_games['home_team'].map(team_abbrev_mapping)
    upcoming_games['away_team_full'] = upcoming_games['away_team'].map(team_abbrev_mapping)

    # Drop games with unmapped teams
    upcoming_games.dropna(subset=['home_team_full', 'away_team_full'], inplace=True)
    upcoming_games.reset_index(drop=True, inplace=True)

    return upcoming_games

# Fetch updated upcoming games
upcoming_games = fetch_upcoming_games()


# Streamlit App UI
teams_list = sorted(team_data['team'].unique())
team = st.selectbox('Select a team for prediction:', teams_list)

if team:
    team_scores = team_data[team_data['team'] == team]['score']
    team_scores.index = pd.to_datetime(team_scores.index)

    st.write(f'### Historical Scores for {team}')
    st.line_chart(team_scores)

    # Display future predictions
    team_forecast = all_forecasts[all_forecasts['Team'] == team]
    st.write(f'### Predicted Scores for Next 5 Games ({team})')
    st.write(team_forecast[['Date', 'Predicted_Score']])

    # Plot the historical and predicted scores
    fig, ax = plt.subplots(figsize=(10, 6))
    forecast_dates = mdates.date2num(team_forecast['Date'])
    historical_dates = mdates.date2num(team_scores.index)

    ax.plot(historical_dates, team_scores.values, label=f'Historical Scores for {team}', color='blue')
    ax.plot(forecast_dates, team_forecast['Predicted_Score'], label='Predicted Scores', color='red')
    lower_bound = team_forecast['Predicted_Score'] - 5
    upper_bound = team_forecast['Predicted_Score'] + 5
    finite_indices = np.isfinite(forecast_dates) & np.isfinite(lower_bound) & np.isfinite(upper_bound)
    ax.fill_between(forecast_dates[finite_indices], lower_bound.values[finite_indices], upper_bound.values[finite_indices], color='gray', alpha=0.2, label='Confidence Interval')
    ax.xaxis_date()
    fig.autofmt_xdate()
    ax.set_title(f'Score Prediction for {team}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Team abbreviation to full name mapping (for accessing models correctly)
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
inverse_team_abbrev_mapping = {v: k for k, v in team_abbrev_mapping.items()}  # Reverse mapping for full name to abbrev

# Game Predictions
st.header('NFL Game Predictions for Upcoming Games')
if not upcoming_games.empty:
    game_selection = st.selectbox(
        'Select an upcoming game:',
        upcoming_games['home_team_full'] + " vs " + upcoming_games['away_team_full']
    )
    selected_game = upcoming_games[
        (upcoming_games['home_team_full'] == game_selection.split(" vs ")[0]) &
        (upcoming_games['away_team_full'] == game_selection.split(" vs ")[1])
    ].iloc[0]

    home_team = selected_game['home_team_full']
    away_team = selected_game['away_team_full']

    # Convert full names to abbreviations for model access
    home_team_abbrev = inverse_team_abbrev_mapping.get(home_team)
    away_team_abbrev = inverse_team_abbrev_mapping.get(away_team)

    # Predict scores using abbreviations
    home_team_score = predict_team_score(home_team_abbrev)
    away_team_score = predict_team_score(away_team_abbrev)

    if home_team_score is not None and away_team_score is not None:
        st.write(f"### Predicted Scores")
        st.write(f"**{home_team}:** {home_team_score:.2f}")
        st.write(f"**{away_team}:** {away_team_score:.2f}")

        if home_team_score > away_team_score:
            st.success(f"**Predicted Winner:** {home_team}")
        elif away_team_score > home_team_score:
            st.success(f"**Predicted Winner:** {away_team}")
        else:
            st.info("**Predicted Outcome:** Tie")
    else:
        st.error("Prediction models for one or both teams are not available.")
else:
    st.write("No upcoming games found.")

