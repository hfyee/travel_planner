import streamlit as st
from crewai import Crew
import os
import asyncio
from PIL import Image
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List

icon_img = Image.open("travel_icon.png")
st.set_page_config(layout="centered")
col1, col2 = st.columns([1, 10])
with col1:
    st.image(icon_img, width=50)
with col2:
    st.title("Travel Planner Assistant")   
st.markdown("Hi, let me help you with your trip planning!")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Your inputs")
    origin = st.text_input("From where will you be traveling from?", placeholder="e.g., Singapore")
    cities = st.text_input("What are the cities options you are interested in visiting?", placeholder="e.g., Tokyo, Kyoto, Osaka")
    date_range = st.text_input("What is your date range for the trip?", placeholder="e.g., 2026-11-25 to 2026-12-08")
    interests = st.text_input("What are your interests for the trip?", placeholder="e.g., culture, food, nature")
    st.divider()
    st.info("Version v0.1 (c) HF 2026")

# Access secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
tavily_api_key = st.secrets["TAVILY_API_KEY"]
serper_api_key = st.secrets["SERPER_API_KEY"]

# Check for API keys in environment variables
if not openai_api_key or not tavily_api_key or not serper_api_key:
    st.warning("Please enter your API keys in the sidebar to proceed.")
    st.stop()

# Set environment variables
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["SERPER_API_KEY"] = serper_api_key

import crewai_logic
from crewai_logic import trip_crew

async def run_crew_async(crew: Crew, inputs: dict):
    return crew.kickoff(inputs=inputs)

class InputValidator(BaseModel):
    origin: str
    cities: List[str]
    date_range: str
    interests: List[str]

    @field_validator('origin', 'cities', 'date_range', 'interests')
    @classmethod
    def check_not_empty(cls, v) -> str:
        if v is None or len(v) == 0:
            raise ValueError('Field cannot be null or empty')
        return v

if st.button("Run"):
    if not origin.strip():
        st.warning("Please enter a travel origin.")
    elif not cities.strip():
        st.warning("Please enter the cities you are interested in.")
    elif not date_range.strip():
        st.warning("Please enter your date range for the trip.")
    elif not interests.strip():
        st.warning("Please enter your interests for the trip.")
    else:

        # Validate inputs before passing to crew
        raw_data = {
            "origin": origin,
            "cities": [city.strip() for city in cities.split(",") if city.strip()],
            "date_range": date_range,
            "interests": [interest.strip() for interest in interests.split(",") if interest.strip()]
        }

        try:
            validated_data = InputValidator(**raw_data)
        except ValidationError as e:
            st.warning(f"Validation Error: {e}")
            st.stop()

        with st.spinner("Our agents are working on it. One moment please..."):
            # Run the crew
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(run_crew_async(trip_crew, inputs=validated_data.model_dump()))

        st.success("✅ Task Completed!")
        #st.markdown("### ✨ Here is your travel plan:")
        st.write(result.raw)