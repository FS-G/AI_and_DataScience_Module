import os
from dotenv import load_dotenv
# import google.generativeai as genai
from google import genai
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
gemni_key = os.getenv('GEMINI_API')

client = genai.Client(api_key=gemni_key)



query = st.text_input("Enter a Question to be asked from the chatbot")

if st.button('Submit'):
    response = client.models.generate_content(model="gemini-2.5-flash", contents=query)
    st.write(response.text)
