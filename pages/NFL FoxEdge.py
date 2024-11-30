# Import Libraries
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz
from pmdarima import auto_arima
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from scipy import stats
from scipy.special import expit  # Sigmoid function

warnings.filterwarnings('ignore')

# Streamlit App Title and Configuration
st.set_page_config(
    page_title="🏈 NFL FoxEdge",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("🏈 NFL FoxEdge")
st.markdown("## Matchup Predictions and Betting Recommendations")
st.markdown("""
Welcome to the Enhanced NFL Betting Insights app! Here, you'll find game predictions, betting leans, and in-depth analysis to help you make informed betting decisions. Whether you're a casual bettor or a seasoned professional, our insights are tailored to provide value to all.
""")

# Synesthetic Interface CSS
st.markdown('''
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@400;700&family=Open+Sans:wght@400;600&display=swap');

        /* Root Variables */
        :root {
            --background-gradient-start: #1A252F; /* Darker Charcoal Gray */
            --background-gradient-end: #2C3E50; /* Enhanced Dark Gray */
            --primary-text-color: #FFFFFF; /* Crisp White */
            --heading-text-color: #F0F0F0; /* Slightly Brighter Light Gray */
            --accent-color-teal: #2ED573; /* Brighter Lime Green */
            --accent-color-purple: #A56BFF; /* Keep Electric Blue */
            --highlight-color: #FF4C4C; /* Brighter Fiery Red */
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

        /* Loading Animation */
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-size: 2em;
            color: var(--accent-color-teal);
        }

        /* MAE Display */
        .mae-card {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1.5em;
            margin: 1em;
            text-align: center;
            transition: transform 0.3s;
        }

        .mae-card:hover {
            transform: scale(1.05);
            background-color: rgba(255, 255, 255, 0.2);
        }

        /* Other styles remain unchanged... */
    </style>
''', unsafe_allow_html=True)

# Fetch and Preprocess Data
@st.cache_data
def fetch_and_preprocess_data(season_years):
    """
    Fetch data for the specified seasons and preprocess it.
    """
    all_team_data = []
    failed_teams = []
    for year in season_years:
        try:
            schedule = nfl.import_schedules([year])
            schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
            schedule['Week_Number'] = schedule['week']

            # Prepare home and away data
            home_df = schedule[['gameday', 'home_team', 'home_score', 'away_score', 'Week_Number']].copy()
            home_df.rename(columns={
                'gameday': 'GAME_DATE',
                'home_team': 'TEAM_ABBREV',
                'home_score': 'PTS',
                'away_score': 'OPP_PTS'
            }, inplace=True)
            home_df['Home_Away'] = 'Home'

            away_df = schedule[['gameday', 'away_team', 'away_score', 'home_score', 'Week_Number']].copy()
            away_df.rename(columns={
                'gameday': 'GAME_DATE',
                'away_team': 'TEAM_ABBREV',
                'away_score': 'PTS',
                'home_score': 'OPP_PTS'
            }, inplace=True)
            away_df['Home_Away'] = 'Away'

            # Combine both DataFrames
            season_data = pd.concat([home_df, away_df], ignore_index=True)
            season_data.dropna(subset=['PTS'], inplace=True)
            season_data['PTS'] = pd.to_numeric(season_data['PTS'], errors='coerce')
            season_data['OPP_PTS'] = pd.to_numeric(season_data['OPP_PTS'], errors='coerce')
            season_data['SEASON'] = year

            # Calculate Points Allowed (Defensive Stat)
            season_data['Points_Allowed'] = season_data['OPP_PTS']

            # Calculate Days Since Last Game (Rest Days)
            season_data.sort_values(['TEAM_ABBREV', 'GAME_DATE'], inplace=True)
            season_data['Days_Since_Last_Game'] = season_data.groupby('TEAM_ABBREV')['GAME_DATE'].diff().dt.days.fillna(7)

            all_team_data.append(season_data)
        except Exception as e:
            st.warning(f"Error fetching data for season {year}: {e}")
            continue
    if all_team_data:
        data = pd.concat(all_team_data, ignore_index=True)
        data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'])
        return data, failed_teams
    else:
        return pd.DataFrame(), failed_teams

# Fetch Data
current_year = datetime.now().year
season_years = [current_year - 2, current_year - 1, current_year]
data, failed_teams = fetch_and_preprocess_data(season_years)

# Train or Load Models with Cross-Validation and Fine-Tuning
@st.cache_resource
def train_models(team_data):
    model_dir = 'models/nfl'
    os.makedirs(model_dir, exist_ok=True)
    team_models = {}
    team_mae = {}
    teams = team_data['TEAM_ABBREV'].unique()

    for team in teams:
        model_path = os.path.join(model_dir, f"{team}_arima_model.pkl")
        team_df = team_data[team_data['TEAM_ABBREV'] == team]
        team_points = team_df['PTS']
        team_points.index = team_df.index  # Ensure index is date for time series

        if len(team_points) < 10:
            st.warning(f"Not enough data to train model for {team}.")
            continue

        # Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=3)
        errors = []
        for train_index, test_index in tscv.split(team_points):
            train, test = team_points.iloc[train_index], team_points.iloc[test_index]

            # ARIMA Model
            try:
                arima_model = auto_arima(
                    train,
                    start_p=0, max_p=3,
                    start_q=0, max_q=3,
                    seasonal=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                arima_forecast = arima_model.predict(n_periods=len(test))
                mae = mean_absolute_error(test, arima_forecast)
                errors.append(mae)
            except Exception as e:
                st.warning(f"Error training ARIMA model for {team}: {e}")
                continue

        # Average MAE
        avg_mae = np.mean(errors) if errors else None
        if avg_mae is not None:
            # Display MAE in a visually appealing card
            st.markdown(f'<div class="mae-card"><h3>{team_name_mapping[team]}</h3><p>Average MAE: <strong>{avg_mae:.2f}</strong></p></div>', unsafe_allow_html=True)
            team_mae[team] = avg_mae
        else:
            st.warning(f"No MAE calculated for {team} due to training errors.")

        # Retrain on full data
        try:
            final_arima_model = auto_arima(
                team_points,
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            # Save the model
            joblib.dump(final_arima_model, model_path)
            team_models[team] = final_arima_model
        except Exception as e:
            st.warning(f"Error retraining ARIMA model for {team}: {e}")
            continue

    return team_models, team_mae

team_models, team_mae_dict = train_models(team_data)

# Main logic for predicting upcoming games
if st.button("Predict Upcoming Games"):
    if data.empty:
        st.error("No data available for analysis.")
    else:
        # Loading animation while predictions are being made
        with st.spinner("Loading predictions..."):
            results = []
            if not upcoming_games.empty:
                for _, game in upcoming_games.iterrows():
                    home_team = game["home_team"]
                    away_team = game["away_team"]
                    game_datetime = game["game_datetime"]

                    result = predict_game_outcome(home_team, away_team, team_forecasts, current_season_stats, team_data, team_mae_dict)

                    winner, score_diff, confidence, home_forecast, away_forecast = result

                    if pd.isna(score_diff) or pd.isna(confidence) or winner == "Unavailable":
                        st.warning(f"Data unavailable for matchup: {team_name_mapping.get(home_team, home_team)} vs {team_name_mapping.get(away_team, away_team)}")
                        continue

                    # Display Matchup Details
                    st.markdown(f"### {team_name_mapping.get(home_team, home_team)} vs {team_name_mapping.get(away_team, away_team)}")
                    st.markdown(f"- **Game Date:** {game_datetime.strftime('%Y-%m-%d %H:%M %Z')}")
                    st.markdown(f"- **Predicted Winner:** **{team_name_mapping.get(winner, 'Unavailable')}**")
                    st.markdown(f"- **Predicted Final Scores:** {team_name_mapping.get(home_team, home_team)} **{home_forecast:.2f}** - {team_name_mapping.get(away_team, away_team)} **{away_forecast:.2f}**")
                    st.markdown(f"- **Score Difference:** **{score_diff:.2f}**")
                    st.markdown(f"- **Confidence Level:** **{confidence:.2f}%**")

                    # Betting Recommendations
                    spread_lean = f"Lean {team_name_mapping.get(winner, winner)} -{int(score_diff)}"
                    total_points = home_forecast + away_forecast
                    total_recommendation = f"Lean Over if the line is < {total_points:.2f}"
                    st.markdown(f"- **Spread Bet:** {spread_lean}")
                    st.markdown(f"- **Total Points Bet:** {total_recommendation}")
                    if confidence > 75:
                        st.success(f"High Confidence: {confidence:.2f}%")
                    elif confidence > 50:
                        st.warning(f"Moderate Confidence: {confidence:.2f}%")
                    else:
                        st.error(f"Low Confidence: {confidence:.2f}%")

                    # Radar Chart
                    categories = ["Average Score", "Max Score", "Consistency", "Recent Form", "Predicted PTS"]
                    home_stats = current_season_stats.get(home_team, {})
                    away_stats = current_season_stats.get(away_team, {})
                    home_stats_values = [
                        home_stats.get('avg_score', 0),
                        home_stats.get('max_score', 0),
                        1 / home_stats.get('std_dev', 1) if home_stats.get('std_dev', 1) != 0 else 1,  # Consistency
                        home_stats.get('recent_form', 0),
                        home_forecast
                    ]
                    away_stats_values = [
                        away_stats.get('avg_score', 0),
                        away_stats.get('max_score', 0),
                        1 / away_stats.get('std_dev', 1) if away_stats.get('std_dev', 1) != 0 else 1,  # Consistency
                        away_stats.get('recent_form', 0),
                        away_forecast
                    ]
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(r=home_stats_values, theta=categories, fill='toself', name=f"{team_name_mapping.get(home_team, home_team)}"))
                    fig.add_trace(go.Scatterpolar(r=away_stats_values, theta=categories, fill='toself', name=f"{team_name_mapping.get(away_team, away_team)}"))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        showlegend=True,
                        title=f"Team Comparison: {team_name_mapping.get(home_team, home_team)} vs {team_name_mapping.get(away_team, away_team)}"
                    )
                    st.plotly_chart(fig)

                    # Append to results
                    results.append({
                        'Home_Team': home_team,
                        'Away_Team': away_team,
                        'Predicted Winner': winner,
                        'Spread Lean': spread_lean,
                        'Total Lean': total_recommendation,
                        'Confidence': confidence
                    })

            # Visualizations
            if results:
                # Confidence Heatmap
                st.header("Confidence Heatmap for Upcoming Matchups")
                confidence_data = pd.DataFrame(results)
                heatmap_data = confidence_data.pivot("Home_Team", "Away_Team", "Confidence")
                plt.figure(figsize=(12, 8))
                sns.heatmap(heatmap_data, annot=True, cmap="RdYlGn", cbar_kws={'label': 'Confidence Level'}, fmt=".1f")
                plt.title("Confidence Heatmap for Upcoming Matchups")
                plt.xlabel("Away Team")
                plt.ylabel("Home Team")
                st.pyplot(plt)

                # Betting Summary
                st.header("📊 Recommended Bets Summary")
                summary_df = pd.DataFrame(results)
                st.dataframe(summary_df[['Home_Team', 'Away_Team', 'Predicted Winner', 'Spread Lean', 'Total Lean', 'Confidence']].style.highlight_max(subset=['Confidence'], color='lightgreen'))

                # Confidence Breakdown
                st.header("Confidence Level Breakdown")
                confidence_counts = pd.Series([result['Confidence'] for result in results]).value_counts(bins=[0,50,75,100], sort=False)
                labels = ['Low (0-50%)', 'Moderate (50-75%)', 'High (75-100%)']
                plt.figure(figsize=(8, 6))
                plt.pie(confidence_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
                plt.title("Confidence Level Breakdown")
                st.pyplot(plt)

                # Best Bets
                st.header("🔥 Top Betting Opportunities")
                best_bets = [result for result in results if result['Confidence'] > 80]
                if best_bets:
                    for bet in best_bets:
                        opponent_team = bet['Away_Team'] if bet['Predicted Winner'] == bet['Home_Team'] else bet['Home_Team']
                        st.markdown(f"**{team_name_mapping.get(bet['Predicted Winner'], bet['Predicted Winner'])}** over **{team_name_mapping.get(opponent_team, opponent_team)}**")
                        st.markdown(f"- **Spread Bet:** {bet['Spread Lean']}")
                        st.markdown(f"- **Total Points Bet:** {bet['Total Lean']}")
                        st.markdown(f"- **Confidence Level:** **{bet['Confidence']:.2f}%**")
                else:
                    st.write("No high confidence bets found.")

                # Additional Insights
                st.header("Detailed Analysis for Professional Bettors")
                st.write("Here are the in-depth stats and trends for each matchup.")
                for result in results:
                    home_team = result['Home_Team']
                    away_team = result['Away_Team']
                    home_stats = current_season_stats.get(home_team, {})
                    away_stats = current_season_stats.get(away_team, {})
                    home_team_name = team_name_mapping.get(home_team, home_team)
                    away_team_name = team_name_mapping.get(away_team, away_team)
                    st.subheader(f"{home_team_name} vs {away_team_name}")
                    st.write("**Home Team Stats:**")
                    st.write(home_stats)
                    st.write("**Away Team Stats:**")
                    st.write(away_stats)
                    # Plot team performance trends
                    team_data_home = data[data['TEAM_ABBREV'] == home_team]
                    team_data_away = data[data['TEAM_ABBREV'] == away_team]
                    st.write(f"**{home_team_name} Performance Trends:**")
                    plt.figure(figsize=(10, 5))
                    plt.plot(team_data_home['GAME_DATE'], team_data_home['PTS'], label='Points Scored', marker='o', color='blue')
                    plt.title(f"{home_team_name} Performance Trends")
                    plt.xlabel("Date")
                    plt.ylabel("Points")
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot(plt)
                    st.write(f"**{away_team_name} Performance Trends:**")
                    plt.figure(figsize=(10, 5))
                    plt.plot(team_data_away['GAME_DATE'], team_data_away['PTS'], label='Points Scored', marker='o', color='red')
                    plt.title(f"{away_team_name} Performance Trends")
                    plt.xlabel("Date")
                    plt.ylabel("Points")
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot(plt)

            else:
                st.warning("No matchups to display.")

        if failed_teams:
            st.warning(f"Data could not be fetched for the following teams: {', '.join(failed_teams)}")

    # User Education and Responsible Gambling Reminder
    st.markdown(f"""
    **Understanding the Metrics:**
    - **Predicted Winner:** The team our model predicts will win the game.
    - **Spread Bet:** Our suggested bet against the point spread.
    - **Total Points Bet:** Our recommendation for the over/under total points.
    - **Confidence Level:** How confident we are in our prediction (0-100%).
    
    **Please Gamble Responsibly:**
    - Set limits and stick to them.
    - Don't chase losses.
    - If you need help, visit [Gambling Support](https://www.example.com/gambling-support).
    
    **New to sports betting?**
    - [Understanding Point Spreads](https://www.example.com/point-spreads)
    - [Guide to Over/Under Betting](https://www.example.com/over-under)
    - [Betting Strategies for Beginners](https://www.example.com/betting-strategies)
    
    **Data last updated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """)

    # Footer
    st.markdown("""
    ---
    &copy; 2023 **Enhanced NFL Betting Insights**. All rights reserved.
    """)
