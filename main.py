import streamlit as st
from utils.database import init_db

# Initialize the database at startup
init_db()

# Set page configuration
st.set_page_config(
    page_title="FoxEdge",
    page_icon="🦊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Enhanced CSS for visuals and branding
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;800&family=Open+Sans:wght@400;600&family=Playfair+Display:wght@700&display=swap');

        /* General Page Styling */
        html, body, [class*="css"] {
            font-family: 'Open Sans', sans-serif;
            background: linear-gradient(135deg, #1a1c2c 0%, #0f111a 100%);
            color: #E5E7EB;
        }

        /* Hero Section Styling */
        .logo-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-bottom: 1em;
        }

        .fox-icon {
            font-size: 3em;
            color: #FFA500;
            animation: glow 3s infinite alternate;
        }

        .title h1 {
            font-family: 'Montserrat', sans-serif;
            background: linear-gradient(120deg, #FFA500, #FF6B00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5em;
            text-align: center;
            font-weight: 800;
            margin: 0.2em;
            text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.2);
            animation: shimmer 2s infinite;
        }

        .subtitle p {
            color: #9CA3AF;
            font-size: 1.3em;
            text-align: center;
            margin-top: 0.5em;
            line-height: 1.6;
            font-family: 'Playfair Display', serif;
        }

        /* Trust Indicators */
        .trust-badges {
            display: flex;
            justify-content: center;
            gap: 2em;
            margin-top: 2em;
        }

        .trust-badge {
            text-align: center;
            font-size: 1.2em;
            color: #FFA500;
            display: flex;
            align-items: center;
            gap: 0.5em;
            animation: fadeIn 2s ease;
        }

        .trust-badge span {
            font-size: 1.5em;
            color: #FF6B00;
        }

        /* Strategic Advantage Banner */
        .advantage-banner {
            background: linear-gradient(90deg, rgba(255,107,0,0.2) 0%, rgba(255,142,83,0.2) 100%);
            border-left: 5px solid #FF6B00;
            padding: 1em;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: bold;
            color: #E5E7EB;
            margin: 2em 0;
            display: flex;
            align-items: center;
            gap: 1em;
            animation: slideIn 1.5s ease;
        }

        .advantage-banner span {
            font-size: 1.5em;
        }

        /* CTA Button Styling */
        div.stButton > button {
            background: linear-gradient(90deg, #FF6B00, #FFA500);
            color: white;
            border: none;
            padding: 1em 2em;
            font-size: 1.3em;
            font-weight: 700;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 6px 12px rgba(255, 107, 0, 0.2);
        }

        div.stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #FFA500, #FF6B00);
        }

        /* Footer Styling */
        .footer {
            margin-top: 3em;
            text-align: center;
            font-size: 0.9em;
            color: #9CA3AF;
        }

        .footer a {
            color: #FFA500;
            text-decoration: none;
            font-weight: bold;
        }

        /* Animations */
        @keyframes shimmer {
            0% { background-position: 0% 50%; }
            100% { background-position: 100% 50%; }
        }

        @keyframes glow {
            from { text-shadow: 0 0 10px #FFA500; }
            to { text-shadow: 0 0 20px #FF6B00; }
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideIn {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# Main Page Content
st.markdown('''
    <div class="logo-container">
        <span class="fox-icon">🦊</span>
        <div class="title"><h1>FoxEdge</h1></div>
    </div>
''', unsafe_allow_html=True)

st.markdown('''
    <div class="subtitle">
        <p>Join <strong>10,000+</strong> analysts leveraging AI-driven predictions<br>
        for smarter NBA & NFL betting decisions.</p>
    </div>
''', unsafe_allow_html=True)

# Trust Badges
st.markdown('''
    <div class="trust-badges">
        <div class="trust-badge"><span>🎯</span> 94% Prediction Accuracy</div>
        <div class="trust-badge"><span>⚡</span> Real-Time Insights</div>
        <div class="trust-badge"><span>📊</span> Data-Driven Analysis</div>
    </div>
''', unsafe_allow_html=True)

# Strategic Advantage Banner
st.markdown('''
    <div class="advantage-banner">
        <span>🦊</span> Unlock Your Edge: Free Premium Analytics for 7 Days!
    </div>
''', unsafe_allow_html=True)

# Call-to-Action Button
if st.button('GAIN YOUR EDGE TODAY'):
    st.write("Preparing your strategic dashboard...")

# Footer
st.markdown('''
    <div class="footer">
        Built with ❤️ by <a href="#">FoxEdge</a>. Follow us on <a href="#">Twitter</a> and <a href="#">Instagram</a>.
    </div>
''', unsafe_allow_html=True)
