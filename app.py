
import streamlit as st
import pandas as pd
from predict_engine import predict_top_sets
from constructive_predict_engine import construct_psi_optimized_sets

st.title("Ψ(Ω) Powerball Predictor")

st.markdown("Upload your Powerball history CSV file (must include 7 main numbers and a Powerball column).")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

mode = st.radio(
    "Select Prediction Mode:",
    ("Mode A - Random & Score", "Mode B - Ψ-Guided Construction")
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")

    if st.button("Generate 200 Predictions"):
        with st.spinner("Generating predictions..."):
            if mode == "Mode A - Random & Score":
                predictions = predict_top_sets(data, num_predictions=200)
            else:
                predictions = construct_psi_optimized_sets(data, num_sets=200)
            st.success("Done! Here are your top predicted sets:")
            st.dataframe(predictions)
            csv = predictions.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "powerball_predictions.csv", "text/csv")
else:
    st.info("Awaiting file upload.")
