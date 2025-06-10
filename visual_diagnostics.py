
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import streamlit as st

def plot_psi_score_distribution(df):
    if "Ψ Score" in df.columns:
        st.subheader("Ψ Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Ψ Score"], kde=True, bins=20, ax=ax)
        st.pyplot(fig)

def plot_d1_score_distribution(df):
    if "D1 Score" in df.columns:
        st.subheader("Division 1 Potential (D1 Score)")
        fig, ax = plt.subplots()
        sns.histplot(df["D1 Score"], kde=True, bins=20, ax=ax)
        st.pyplot(fig)

def plot_number_frequency_heatmap(df):
    if "Main Numbers" in df.columns:
        st.subheader("Main Number Frequency Heatmap")
        try:
            all_numbers = sum(df["Main Numbers"].apply(eval).tolist(), [])
        except:
            all_numbers = []
        num_counts = Counter(all_numbers)
        heat_data = [num_counts.get(i, 0) for i in range(1, 46)]
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.heatmap([heat_data], cmap="YlGnBu", cbar=True, xticklabels=range(1, 46), ax=ax)
        ax.set_yticklabels(["Freq"])
        st.pyplot(fig)
