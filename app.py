"""Streamlit interface for testing the trained research-abstract classifier."""

import pickle
from pathlib import Path

import streamlit as st

from src.text_preprocessing import ensure_nltk_resources


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "model.pkl"


@st.cache_resource
def load_model():
    """Load the fitted training pipeline once per Streamlit server process."""
    ensure_nltk_resources()
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def display_label(label):
    """Format labels from the training dataset for a reader-friendly result."""
    return str(label).replace("_", " ").title()


st.set_page_config(page_title="Research Abstract Classifier", page_icon="🔬")
st.title("Research Abstract Classifier")
st.write(
    "Paste a research abstract below to classify it using the saved tuned Naive Bayes model."
)

abstract = st.text_area(
    "Enter research abstract:",
    height=220,
    placeholder="Paste a research abstract here...",
)

if st.button("Classify Abstract", type="primary"):
    if not abstract.strip():
        st.warning("Enter a research abstract before classifying it.")
    elif not MODEL_PATH.exists():
        st.error("The trained model file was not found at model/model.pkl.")
    else:
        try:
            with st.spinner("Classifying abstract..."):
                model = load_model()
                prediction = model.predict([abstract])[0]
            st.subheader("Prediction")
            st.success(f"The abstract is classified as: {display_label(prediction)}")
        except Exception as error:
            st.error(f"The abstract could not be classified: {error}")
