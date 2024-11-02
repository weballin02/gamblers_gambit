import streamlit as st
from utils.database import init_db

# Initialize the database at startup
init_db()

# Title and introduction for the main landing page
st.title("Welcome to NFL Monte Carlo Simulation and Analysis")
st.write("Explore different pages using the sidebar to run simulations, view correlations, and more.")
