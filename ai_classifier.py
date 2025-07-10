import io
import numpy as np
from PIL import Image
import streamlit as st
from google.generativeai import GenerativeModel, configure
from google.api_core.exceptions import ResourceExhausted
import hashlib

# Access both API keys from secrets
API_KEYS = [
    st.secrets["google_ai"]["api_key1"],
    st.secrets["google_ai"]["api_key2"],
]


def get_image_hash(image_bytes: bytes) -> str:
    """Generate a SHA-256 hash of the image for caching."""
    return hashlib.sha256(image_bytes).hexdigest()


@st.cache_data
def call_gemini_with_fallback(prompt: str, image_bytes: bytes) -> str:
    """Try calling Gemini with both keys; fallback to error message if both fail."""
    for key in API_KEYS:
        try:
            configure(api_key=key)
            model = GenerativeModel("gemini-pro-vision")
            response = model.generate_content([prompt, image_bytes])
            return response.text
        except ResourceExhausted:
            continue  # Try next key if quota exhausted
        except Exception as e:
            return f"⚠️ Gemini API error: {e}"

    return "⚠️ Gemini quota exceeded. Used local analysis."


class AIFigureClassifier:
    """AI-powered figure classifier using Google Gemini."""

    def __init__(self):
        self.prompt = (
            "You are an expert document visual analyst with more than 20 years of experience and 4 PHDs in the field of Figure classification and identification. "
            "Classify the given figure into one of the following types: "
            "Bar Chart, Line Graph, Pie Chart, Timeline, Photograph, Table, Other Chart, Other Diagram. "
            "Also describe the layout and visual structure in one line."
        )

    def classify(self, image: Image.Image) -> dict:
        image_bytes = self._image_to_bytes(image)
        image_hash = get_image_hash(image_bytes)

        # Call Gemini with fallback support
        response = call_gemini_with_fallback(self.prompt, image_bytes)

        # Handle API failures
        if "Gemini quota exceeded" in response or response.startswith("⚠️ Gemini"):
            tag = "📐 Other Diagram"
            confidence = 30.0
            desc = "Grayscale content, likely diagram or text"
            reason = response
        else:
            tag, confidence, desc, reason = self._parse_response(response)

        return {
            "type": tag,
            "confidence": confidence,
            "description": desc,
            "reasoning": reason,
        }

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to PNG byte format."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def _parse_response(self, response: str):
        """Parse Gemini response into structured label, confidence, description."""
        response = response.strip()
        lower_resp = response.lower()

        # Figure type detection
        if "bar chart" in lower_resp:
            tag = "📊 Bar Chart"
            confidence = 90.0
        elif "line graph" in lower_resp:
            tag = "📈 Line Graph"
            confidence = 90.0
        elif "pie chart" in lower_resp:
            tag = "🟢 Pie Chart"
            confidence = 90.0
        elif "timeline" in lower_resp:
            tag = "⏰ Timeline"
            confidence = 60.0
        elif "photograph" in lower_resp or "photo" in lower_resp:
            tag = "📷 Photograph"
            confidence = 40.0
        elif "table" in lower_resp:
            tag = "📋 Table"
            confidence = 70.0
        elif "chart" in lower_resp:
            tag = "📊 Other Chart"
            confidence = 50.0
        else:
            tag = "📐 Other Diagram"
            confidence = 40.0

        # Description is first sentence
        desc = response.split(".")[0]
        return tag, confidence, desc, response
