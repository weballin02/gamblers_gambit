import streamlit as st
from utils.database import init_db

# Initialize the database at startup
init_db()

# Title and introduction for the main landing page
st.title("Welcome to Gambler's Gambit")
st.write("Enter the Gambit: Conduct Monte Carlo Simulations and Correlation Analysis, Explore Game Predictions, and more.")
