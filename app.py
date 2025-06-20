
import streamlit as st
import pandas as pd
import numpy as np
import random

from constructive_predict_engine import predict_top_sets as predict_b
from predict_engine import predict_top_sets as predict_a
from psi_star_engine import generate_predictions as psi_star_generate
from visual_diagnostics import (
    plot_psi_score_distribution,
    plot_d1_score_distribution,
    plot_number_frequency_heatmap
)

st.set_page_config(page_title="Ψ★ Powerball Predictor", layout="wide")
st.title("Ψ(Ω) Powerball Predictor")

st.write("Upload your Powerball history CSV file (must include 7 main numbers and a Powerball column).")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

mode = st.radio("Select Prediction Mode:", [
    "Mode A - Random & Score",
    "Mode B - Ψ-Guided Construction",
    "Mode C - Ψ★ Model 2.0"
])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("File loaded successfully!")

        if st.button("Run Predictions"):
            if mode == "Mode A - Random & Score":
                result_df = predict_a(data)
                st.success("Predictions Generated Successfully!")

            elif mode == "Mode B - Ψ-Guided Construction":
                result_df = predict_b(data)
                st.success("Ψ(Ω) Predictions Generated Successfully!")

            elif mode == "Mode C - Ψ★ Model 2.0":
                historical_draws = data.iloc[:, 1:8].values.tolist()

                # Robust Powerball extraction
                try:
                    past_pbs = pd.to_numeric(data["Powerball"], errors='coerce').dropna().astype(int).tolist()
                    st.write("Sample Powerballs:", past_pbs[:10])
                except Exception as e:
                    st.warning("Could not read Powerball column correctly.")
                    past_pbs = []

                result_df = psi_star_generate(historical_draws, past_pbs, num_predictions=200)
                st.success("Ψ★(Ω) Predictions Generated Successfully!")

                # Visual diagnostics for Mode C
                plot_psi_score_distribution(result_df)
                plot_d1_score_distribution(result_df)
                plot_number_frequency_heatmap(result_df)

            st.dataframe(result_df)
            st.download_button("Download Predictions as CSV", result_df.to_csv(index=False), "predictions.csv")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Awaiting file upload.")
