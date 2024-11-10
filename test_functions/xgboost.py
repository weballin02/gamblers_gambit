# Import Libraries
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

def nfl_predictions_xgboost():
    st.title('NFL Team Points Prediction with Gradient Boosting Machines (XGBoost)')

# Load and Preprocess Data
@st.cache_data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['schedule_date'] = pd.to_datetime(data['schedule_date'], errors='coerce')

    # Prepare home and away data
    home_df = data[['schedule_date', 'team_home', 'score_home']].copy()
    home_df.rename(columns={'team_home': 'team', 'score_home': 'score'}, inplace=True)

    away_df = data[['schedule_date', 'team_away', 'score_away']].copy()
    away_df.rename(columns={'team_away': 'team', 'score_away': 'score'}, inplace=True)

    # Combine both DataFrames
    team_data = pd.concat([home_df, away_df], ignore_index=True)
    team_data.dropna(subset=['score'], inplace=True)
    team_data['score'] = pd.to_numeric(team_data['score'], errors='coerce')
    team_data.set_index('schedule_date', inplace=True)
    team_data.sort_index(inplace=True)

    return team_data

# Load Data
file_path = 'data/nfl_data.csv'  # Update this path if necessary
team_data = load_and_preprocess_data(file_path)

# Get list of teams
teams_list = team_data['team'].unique()

# Feature Engineering
@st.cache_data
def feature_engineering(team_data):
    # Create lag features
    lag_features = 3  # Number of lag weeks
    feature_list = []
    for lag in range(1, lag_features + 1):
        team_data[f'score_lag_{lag}'] = team_data.groupby('team')['score'].shift(lag)
        feature_list.append(f'score_lag_{lag}')

    # Create rolling statistics
    team_data['score_roll_mean_3'] = team_data.groupby('team')['score'].rolling(window=3).mean().reset_index(0, drop=True)
    team_data['score_roll_std_3'] = team_data.groupby('team')['score'].rolling(window=3).std().reset_index(0, drop=True)

    feature_list.extend(['score_roll_mean_3', 'score_roll_std_3'])

    # Drop rows with NaN values created by lagging
    team_data = team_data.dropna()

    return team_data, feature_list

team_data, feature_list = feature_engineering(team_data)

# Train or Load XGBoost Models
@st.cache_resource
def get_team_models_xgboost(team_data, feature_list):
    model_dir = 'models/nfl_xgboost/'  # Directory for XGBoost models
    os.makedirs(model_dir, exist_ok=True)

    team_models = {}
    teams_list = team_data['team'].unique()

    for team in teams_list:
        model_path = os.path.join(model_dir, f'{team}_xgboost_model.pkl')

        team_df = team_data[team_data['team'] == team].copy()

        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            # Check if there are enough data points to train the model
            if len(team_df) < 20:
                # Skip teams with insufficient data
                continue
            try:
                X = team_df[feature_list]
                y = team_df['score']

                # Split into training and testing sets (e.g., last 5 weeks as test)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                # Initialize and train the model
                model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
                model.fit(X_train, y_train)

                # Evaluate the model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                st.write(f'Team: {team} | Test MSE: {mse:.2f}')

                # Save the model
                joblib.dump(model, model_path)
            except:
                continue

        team_models[team] = model

    return team_models

# Get Team XGBoost Models
team_models = get_team_models_xgboost(team_data, feature_list)

# Function to Predict Team Score using XGBoost
def predict_team_score_xgboost(team, team_data, feature_list, periods=1):
    model = team_models.get(team)
    if model:
        # Get the latest available data
        last_entry = team_data[team_data['team'] == team].iloc[-1]
        input_features = last_entry[feature_list].values.reshape(1, -1)

        # Predict
        forecast = model.predict(input_features)
        return forecast[0]
    else:
        return None

# Forecast the Next 5 Games for Each Team
@st.cache_data
def compute_team_forecasts_xgboost(team_models, team_data, feature_list):
    team_forecasts = {}
    forecast_periods = 5

    for team, model in team_models.items():
        team_scores = team_data[team_data['team'] == team].copy()
        predictions = []
        current_team_data = team_scores.copy()

        for _ in range(forecast_periods):
            # Get the latest features
            last_entry = current_team_data.iloc[-1]
            input_features = last_entry[feature_list].values.reshape(1, -1)

            # Predict
            pred_score = model.predict(input_features)[0]
            predictions.append(pred_score)

            # Update the DataFrame with the prediction
            new_date = current_team_data.index[-1] + timedelta(days=7)
            new_entry = pd.DataFrame({
                'score': pred_score,
                'score_lag_1': last_entry['score'],
                'score_lag_2': last_entry['score_lag_1'],
                'score_lag_3': last_entry['score_lag_2'],
                'score_roll_mean_3': np.mean([last_entry['score'], last_entry['score_lag_1'], last_entry['score_lag_2']]),
                'score_roll_std_3': np.std([last_entry['score'], last_entry['score_lag_1'], last_entry['score_lag_2']])
            }, index=[new_date])

            # Append the new entry
            current_team_data = pd.concat([current_team_data, new_entry])

        # Create a DataFrame for predictions
        prediction_dates = pd.date_range(start=current_team_data.index[-forecast_periods], periods=forecast_periods, freq='7D')
        predictions_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Score': predictions,
            'Team': team
        })
        team_forecasts[team] = predictions_df

    # Combine all forecasts
    if team_forecasts:
        all_forecasts = pd.concat(team_forecasts.values(), ignore_index=True)
    else:
        all_forecasts = pd.DataFrame(columns=['Date', 'Predicted_Score', 'Team'])
    return all_forecasts

# Compute Team Forecasts
all_forecasts = compute_team_forecasts_xgboost(team_models, team_data, feature_list)

# Fetch Upcoming Games
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

    # Get current time in UTC
    now = datetime.now(pytz.UTC)

    # Filter for upcoming regular-season games
    upcoming_games = schedule[
        (schedule['game_type'] == 'REG') &
        (schedule['game_datetime'] >= now)
    ]

    # Select necessary columns
    upcoming_games = upcoming_games[['game_id', 'game_datetime', 'home_team', 'away_team']]

    # Mapping from team abbreviations to full names
    team_abbrev_mapping = {
        'ARI': 'Arizona Cardinals',
        'ATL': 'Atlanta Falcons',
        'BAL': 'Baltimore Ravens',
        'BUF': 'Buffalo Bills',
        'CAR': 'Carolina Panthers',
        'CHI': 'Chicago Bears',
        'CIN': 'Cincinnati Bengals',
        'CLE': 'Cleveland Browns',
        'DAL': 'Dallas Cowboys',
        'DEN': 'Denver Broncos',
        'DET': 'Detroit Lions',
        'GB': 'Green Bay Packers',
        'HOU': 'Houston Texans',
        'IND': 'Indianapolis Colts',
        'JAX': 'Jacksonville Jaguars',
        'KC': 'Kansas City Chiefs',
        'LAC': 'Los Angeles Chargers',
        'LAR': 'Los Angeles Rams',
        'LV': 'Las Vegas Raiders',
        'MIA': 'Miami Dolphins',
        'MIN': 'Minnesota Vikings',
        'NE': 'New England Patriots',
        'NO': 'New Orleans Saints',
        'NYG': 'New York Giants',
        'NYJ': 'New York Jets',
        'PHI': 'Philadelphia Eagles',
        'PIT': 'Pittsburgh Steelers',
        'SEA': 'Seattle Seahawks',
        'SF': 'San Francisco 49ers',
        'TB': 'Tampa Bay Buccaneers',
        'TEN': 'Tennessee Titans',
        'WAS': 'Washington Commanders',
    }

    # Apply mapping
    upcoming_games['home_team_full'] = upcoming_games['home_team'].map(team_abbrev_mapping)
    upcoming_games['away_team_full'] = upcoming_games['away_team'].map(team_abbrev_mapping)

    # Remove games where team names couldn't be mapped
    upcoming_games.dropna(subset=['home_team_full', 'away_team_full'], inplace=True)

    # Reset index
    upcoming_games.reset_index(drop=True, inplace=True)

    return upcoming_games

# Fetch upcoming games
upcoming_games = fetch_upcoming_games()

# Streamlit App
st.title('NFL Team Points Prediction with Gradient Boosting Machines (XGBoost)')

# Dropdown menu for selecting a team
teams_list = sorted(team_data['team'].unique())
team = st.selectbox('Select a team for prediction:', teams_list)

if team:
    team_scores = team_data[team_data['team'] == team].copy()
    team_scores = team_scores.asfreq('W').fillna(method='ffill')

    st.write(f'### Historical Scores for {team}')
    st.line_chart(team_scores['score'])

    # Display future predictions
    team_forecast = all_forecasts[all_forecasts['Team'] == team]
    st.write(f'### Predicted Scores for Next 5 Games ({team})')
    st.write(team_forecast[['Date', 'Predicted_Score']])

    # Plot the historical and predicted scores
    st.write(f'### Score Prediction for {team}')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert dates to Matplotlib's numeric format
    forecast_dates = mdates.date2num(team_forecast['Date'])
    historical_dates = mdates.date2num(team_scores.index)

    # Plot historical scores
    ax.plot(
        historical_dates,
        team_scores['score'].values,
        label=f'Historical Scores for {team}',
        color='blue'
    )
    # Plot predicted scores
    ax.plot(
        forecast_dates,
        team_forecast['Predicted_Score'],
        label='Predicted Scores',
        color='red'
    )
    # Plot confidence interval (using +/- 5 as placeholder)
    lower_bound = team_forecast['Predicted_Score'] - 5
    upper_bound = team_forecast['Predicted_Score'] + 5

    # Ensure no non-finite values
    finite_indices = np.isfinite(forecast_dates) & np.isfinite(lower_bound) & np.isfinite(upper_bound)

    ax.fill_between(
        forecast_dates[finite_indices],
        lower_bound.values[finite_indices],
        upper_bound.values[finite_indices],
        color='gray',
        alpha=0.2,
        label='Confidence Interval'
    )

    ax.xaxis_date()
    fig.autofmt_xdate()

    ax.set_title(f'Score Prediction for {team}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# New functionality: NFL Game Predictions
st.write('---')
st.header('NFL Game Predictions for Upcoming Games')

# Create game labels
upcoming_games['game_label'] = [
    f"{row['away_team_full']} at {row['home_team_full']} ({row['game_datetime'].strftime('%Y-%m-%d %H:%M %Z')})"
    for _, row in upcoming_games.iterrows()
]

# Let the user select a game
if not upcoming_games.empty:
    game_selection = st.selectbox('Select an upcoming game:', upcoming_games['game_label'])
    selected_game = upcoming_games[upcoming_games['game_label'] == game_selection].iloc[0]

    home_team = selected_game['home_team_full']
    away_team = selected_game['away_team_full']

    # Predict scores
    home_team_score = predict_team_score_xgboost(home_team, team_data, feature_list)
    away_team_score = predict_team_score_xgboost(away_team, team_data, feature_list)

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
    st.info("No upcoming games available.")

