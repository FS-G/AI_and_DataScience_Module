# Import dependencies
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Streamlit app settings
st.set_page_config(page_title="ASKSIDEWAYS", page_icon=":bar_chart:")


USERNAME ="Farhan"
PASSWORD = "123456"

# Function to add background color
def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #a0bfb9;
            color: #895051;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

# Add background color
add_bg_from_url()

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login logic
if not st.session_state.logged_in:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()  # Refresh the page after login
        else:
            st.error("Invalid username or password")
else:
    st.title("Welcome to the Envoy App!")
    st.write("You are logged in!")

