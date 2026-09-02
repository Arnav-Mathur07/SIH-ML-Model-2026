import streamlit as st

st.set_page_config(
    page_title="Sonar Debris Detection AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark marine theme
st.markdown("""
<style>
    .stApp {
        background-color: #0a1628;
        color: #e2e8f0;
    }
    .stSidebar {
        background-color: #0d2137;
    }
    .stCard {
        background-color: #0d2137;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    /* Accent colors */
    .stButton>button {
        background-color: #1e4d7b;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2b6cb0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌊 Underwater Marine Debris Detection System")
st.markdown("Welcome to the AI-powered Side-Scan Sonar analysis dashboard.")

st.info("👈 Select a page from the sidebar to begin.")
