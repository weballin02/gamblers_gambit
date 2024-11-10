# Import Libraries
import nfl_data_py as nfl
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

# Streamlit App Title
st.title("Enhanced NFL Betting Insights")
st.markdown("Get detailed insights on NFL team trends and betting opportunities. This page combines multi-season stats, recent form, and consistency to highlight spread, moneyline, and over/under suggestions. Just select a game to dive into specific betting angles.")

# Define Seasons and Weights
current_year = datetime.now().year
previous_years = [current_year - 1, current_year - 2]
season_weights = {current_year: 1.0, current_year - 1: 0.7, current_year - 2: 0.5}  # Higher weight for recent data

# Load and Preprocess Data from Multiple Seasons
@st.cache_data
def load_and_preprocess_data():
    all_team_data = []
    for year in [current_year] + previous_years:
        schedule = nfl.import_schedules([year])
        
        # Add date and weights for seasons
        schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
        schedule['season_weight'] = season_weights[year]
        
        # Split into home and away data and standardize columns
        home_df = schedule[['gameday', 'home_team', 'home_score', 'season_weight']].copy().rename(columns={'home_team': 'team', 'home_score': 'score'})
        away_df = schedule[['gameday', 'away_team', 'away_score', 'season_weight']].copy().rename(columns={'away_team': 'team', 'away_score': 'score'})
        
        # Combine home and away data, filter out null scores
        season_data = pd.concat([home_df, away_df], ignore_index=True)
        season_data.dropna(subset=['score'], inplace=True)
        season_data.set_index('gameday', inplace=True)
        
        all_team_data.append(season_data)

    # Concatenate data for all seasons
    return pd.concat(all_team_data, ignore_index=False)

# Load the team data for training purposes
team_data = load_and_preprocess_data()

# Aggregate Team Stats with Weights (Training on All Seasons)
def aggregate_team_stats(team_data):
    team_stats = team_data.groupby('team').apply(
        lambda x: pd.Series({
            'avg_score': np.average(x['score'], weights=x['season_weight']),
            'min_score': x['score'].min(),
            'max_score': x['score'].max(),
            'std_dev': x['score'].std(),
            'games_played': x['score'].count()
        })
    ).to_dict(orient='index')
    
    return team_stats

# Train using multi-season aggregated data
team_stats = aggregate_team_stats(team_data)

# Filter Current Season Data Only for Predictions
def get_current_season_stats():
    schedule = nfl.import_schedules([current_year])
    schedule['gameday'] = pd.to_datetime(schedule['gameday'], errors='coerce')
    
    home_df = schedule[['gameday', 'home_team', 'home_score']].copy().rename(columns={'home_team': 'team', 'home_score': 'score'})
    away_df = schedule[['gameday', 'away_team', 'away_score']].copy().rename(columns={'away_team': 'team', 'away_score': 'score'})
    
    current_season_data = pd.concat([home_df, away_df], ignore_index=True)
    current_season_data.dropna(subset=['score'], inplace=True)
    current_season_data.set_index('gameday', inplace=True)
    current_season_data.sort_index(inplace=True)
    
    return current_season_data

# Load current season data for prediction
current_season_data = get_current_season_stats()

# Calculate current season stats only
def calculate_current_season_stats():
    current_season_stats = current_season_data.groupby('team').apply(
        lambda x: pd.Series({
            'avg_score': x['score'].mean(),
            'min_score': x['score'].min(),
            'max_score': x['score'].max(),
            'std_dev': x['score'].std(),
            'games_played': x['score'].count(),
            'recent_form': x['score'].tail(5).mean() if len(x) >= 5 else x['score'].mean()
        })
    ).to_dict(orient='index')
    return current_season_stats

# Calculate current season stats for all teams
current_season_stats = calculate_current_season_stats()

# Injury Data Retrieval Function
@st.cache_data(ttl=3600)
def fetch_injury_data():
    injury_data = nfl.import_injuries([current_year])
    
    # Define key positions and filter for players who are 'Out'
    key_positions = ['QB', 'RB', 'WR', 'OL']
    key_injuries = injury_data[
        (injury_data['position'].isin(key_positions)) &
        (injury_data['report_status'] == 'Out')
    ]

    # Filter injuries to only include updates within the past week
    today = datetime.now(pytz.UTC)  # Set to UTC
    one_week_ago = today - timedelta(days=7)
    key_injuries['date_modified'] = pd.to_datetime(key_injuries['date_modified'], errors='coerce')
    recent_injuries = key_injuries[key_injuries['date_modified'] >= one_week_ago]

    return recent_injuries

# Adjust Team Rating Based on Injury Impact
def adjust_rating_for_injuries(team, base_rating, injury_data):
    team_injuries = injury_data[(injury_data['team'] == team)]
    impact_score = 0
    impact_summary = []
    # Define impact coefficients for each key position based on expected scoring impact
    position_impacts = {
        'QB': 0.15,  # Estimated impact: 2.5-3.5 points
        'RB': 0.07,  # Estimated impact: 1.0-1.5 points
        'WR': 0.08,  # Estimated impact: 1.5-2 points
        'OL': 0.05,  # Estimated impact: 0.5-1 point
        'DEF': 0.02  # Indirect impact, affects opponent
    }

    for _, row in team_injuries.iterrows():
        position = row['position']
        position_impact = position_impacts.get(position, 0.02)  # Default impact for other positions
        impact_score += position_impact
        impact_summary.append(f"{row['full_name']} ({row['position']}) - {row['report_status']}")

    # Estimate point decrease based on accumulated impact score
    point_decrease = round(impact_score * 3, 2)  # Apply a multiplier to estimate points impact
    adjusted_rating = base_rating * (1 - impact_score)
    return adjusted_rating, impact_summary, point_decrease

# Predict Outcome with Optional Injury Adjustments
def predict_game_outcome(home_team, away_team, use_injury_impact):
    home_stats = current_season_stats.get(home_team, {})
    away_stats = current_season_stats.get(away_team, {})

    if home_stats and away_stats:
        home_team_rating = (
            home_stats['avg_score'] * 0.5 +
            home_stats['max_score'] * 0.2 +
            home_stats['recent_form'] * 0.3
        )
        away_team_rating = (
            away_stats['avg_score'] * 0.5 +
            away_stats['max_score'] * 0.2 +
            away_stats['recent_form'] * 0.3
        )

        if use_injury_impact:
            injury_data = fetch_injury_data()
            home_team_rating, home_injury_summary, home_point_decrease = adjust_rating_for_injuries(
                home_team, home_team_rating, injury_data
            )
            away_team_rating, away_injury_summary, away_point_decrease = adjust_rating_for_injuries(
                away_team, away_team_rating, injury_data
            )
        else:
            home_injury_summary, away_injury_summary = [], []
            home_point_decrease = away_point_decrease = 0

        rating_diff = abs(home_team_rating - away_team_rating)
        confidence = min(100, max(0, 50 + rating_diff * 5))

        if home_team_rating > away_team_rating:
            predicted_winner = home_team
            predicted_score_diff = home_team_rating - away_team_rating
        else:
            predicted_winner = away_team
            predicted_score_diff = away_team_rating - home_team_rating

        return (
            predicted_winner,
            predicted_score_diff,
            confidence,
            home_team_rating,
            away_team_rating,
            home_injury_summary,
            away_injury_summary,
            home_point_decrease,
            away_point_decrease
        )
    else:
        return "Unavailable", "N/A", "N/A", None, None, [], [], 0, 0

# Helper function to check for positions in injury summary
def has_position_injury(injury_summary, position):
    return any(position in injury for injury in injury_summary)

# Enhanced Summary for Betting Insights
def enhanced_summary(home_team, away_team, home_stats, away_stats, home_injury_summary, away_injury_summary, home_team_rating, away_team_rating, home_point_decrease, away_point_decrease, use_injury_impact):
    st.subheader("Enhanced Betting Insights Summary")

    # Key Players Missing (only shown if injury impact is included)
    if use_injury_impact:
        st.write(f"### Key Players Missing for {home_team}")
        st.write(", ".join(home_injury_summary) if home_injury_summary else "No key injuries.")
        st.write(f"### Key Players Missing for {away_team}")
        st.write(", ".join(away_injury_summary) if away_injury_summary else "No key injuries.")
        st.write("_Impact_: Key player absences can reduce a team’s scoring potential and overall performance. Bettors might consider this when assessing point spreads and moneyline bets.")

        # Injury Impact Score (only shown if injury impact is included)
        st.subheader("Injury Impact on Team Strength")
        home_impact_score = round(1 - home_team_rating / (home_stats['avg_score'] * 0.5 + home_stats['max_score'] * 0.2 + home_stats['recent_form'] * 0.3), 2)
        away_impact_score = round(1 - away_team_rating / (away_stats['avg_score'] * 0.5 + away_stats['max_score'] * 0.2 + away_stats['recent_form'] * 0.3), 2)
        st.write(f"**{home_team} Injury Impact Score:** {home_impact_score}")
        st.write(f"**{away_team} Injury Impact Score:** {away_impact_score}")
        st.write("_Impact_: Higher impact scores suggest a bigger negative effect from injuries. For close games, a high injury impact score on one team may tip the balance, making the healthier team a better bet.")

        # Offensive & Defensive Impact Based on Injuries (only shown if injury impact is included)
        st.subheader("Expected Offensive & Defensive Impact from Injuries")
        
        st.write(f"**{home_team} Offense:**")
        if has_position_injury(home_injury_summary, 'QB'):
            st.write("- Likely decline in passing game due to absence of QB")
        if has_position_injury(home_injury_summary, 'RB'):
            st.write("- Potential reduction in rushing game due to absence of RB")
        if has_position_injury(home_injury_summary, 'WR'):
            st.write("- Reduced passing game options due to absence of WR")
        st.write(f"- **Estimated Point Decrease Due to Injuries:** {home_point_decrease}")
        
        st.write(f"**{away_team} Offense:**")
        if has_position_injury(away_injury_summary, 'QB'):
            st.write("- Likely decline in passing game due to absence of QB")
        if has_position_injury(away_injury_summary, 'RB'):
            st.write("- Potential reduction in rushing game due to absence of RB")
        if has_position_injury(away_injury_summary, 'WR'):
            st.write("- Reduced passing game options due to absence of WR")
        st.write(f"- **Estimated Point Decrease Due to Injuries:** {away_point_decrease}")

        st.write("_Insight_: Injuries to key offensive players can lead to reduced scoring. This might make the under bet more attractive if a team is expected to score less due to missing players.")

    # Team Trends and Strengths
    st.subheader("Team Performance Trends")
    st.write(f"### {home_team} Trends:")
    st.write(f"- **Recent Form (Last 5 Games):** {round(home_stats['recent_form'], 2)}")
    st.write(f"- **Consistency (Std Dev):** {round(home_stats['std_dev'], 2)}")
    st.write("_Tip_: A higher recent form score suggests the team is on a good streak, which may indicate better performance in the upcoming game. Consistency is also key—lower values mean more reliable scoring.")

    st.write(f"### {away_team} Trends:")
    st.write(f"- **Recent Form (Last 5 Games):** {round(away_stats['recent_form'], 2)}")
    st.write(f"- **Consistency (Std Dev):** {round(away_stats['std_dev'], 2)}")
    st.write("_Tip_: For betting totals (over/under), look at consistency. Highly consistent teams can make predicting total points easier, while erratic scores suggest less predictable outcomes.")

    # Confidence Score with Injury Adjustment
    st.subheader("Overall Prediction and Confidence with Injury Adjustments")
    likely_advantage = home_team if home_team_rating > away_team_rating else away_team
    st.write(f"**Predicted Advantage:** {likely_advantage} is expected to have an edge, with adjusted ratings reflecting recent performance and injury impact.")
    st.write(f"_Confidence Boost_: If betting on {likely_advantage}, the injury impact and recent form support this choice. Use this insight for moneyline bets or spreads if the adjusted ratings favor a team by a solid margin.")


# Fetch Upcoming Games
@st.cache_data(ttl=3600)
def fetch_upcoming_games():
    schedule = nfl.import_schedules([current_year])
    schedule['game_datetime'] = pd.to_datetime(schedule['gameday'].astype(str) + ' ' + schedule['gametime'].astype(str), errors='coerce', utc=True)
    upcoming_games = schedule[(schedule['game_type'] == 'REG') & (schedule['game_datetime'] >= datetime.now(pytz.UTC))]
    return upcoming_games[['game_id', 'game_datetime', 'home_team', 'away_team']]

upcoming_games = fetch_upcoming_games()

# Streamlit UI for Team Prediction
st.header('NFL Game Predictions with Detailed Analysis')

# Checkbox to include injury impact in predictions
use_injury_impact = st.checkbox("Include Injury Impact in Prediction")

# Create game labels for selection
upcoming_games['game_label'] = [
    f"{row['away_team']} at {row['home_team']} ({row['game_datetime'].strftime('%Y-%m-%d %H:%M %Z')})"
    for _, row in upcoming_games.iterrows()
]

game_selection = st.selectbox('Select an upcoming game:', upcoming_games['game_label'])
selected_game = upcoming_games[upcoming_games['game_label'] == game_selection].iloc[0]

home_team = selected_game['home_team']
away_team = selected_game['away_team']

# Predict Outcome with optional Injury Impact
predicted_winner, predicted_score_diff, confidence, home_team_rating, away_team_rating, home_injury_summary, away_injury_summary, home_point_decrease, away_point_decrease = predict_game_outcome(home_team, away_team, use_injury_impact)

# Display Prediction Results with Betting Insights
if predicted_winner != "Unavailable":
    st.write(f"### Predicted Outcome for {home_team} vs. {away_team}")
    st.write(f"**Predicted Winner:** {predicted_winner} with a confidence of {round(confidence, 2)}%")
    st.write(f"**Expected Score Difference:** {round(predicted_score_diff, 2)}")

    # Call Enhanced Summary for Detailed Insights, passing the `use_injury_impact` parameter
    enhanced_summary(home_team, away_team, current_season_stats.get(home_team, {}), current_season_stats.get(away_team, {}),
                     home_injury_summary, away_injury_summary, home_team_rating, away_team_rating, home_point_decrease, away_point_decrease, use_injury_impact)
else:
    st.error("Prediction data for one or both teams is unavailable.")
