
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import entropy
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
import random

def preprocess_data(df):
    # Identify the 7 main draw columns automatically
    draw_cols = [col for col in df.columns if "number" in col.lower()]
    draws = df[draw_cols[:7]].astype(int).values
    return draws

def fft_score(draws):
    sequence = [sum(draw) for draw in draws]
    fft_vals = np.real(fft(sequence))
    return np.sum(np.abs(fft_vals[:7]))

def entropy_score(draws):
    flat = draws.flatten()
    values, counts = np.unique(flat, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs))

def bayes_score(candidate, draws):
    transition_matrix = np.zeros((46, 46))
    for row in draws:
        for i in range(len(row)-1):
            transition_matrix[row[i], row[i+1]] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        trans_prob = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    score = 0
    for i in range(len(candidate)-1):
        a, b = candidate[i], candidate[i+1]
        if 0 <= a < 46 and 0 <= b < 46 and trans_prob[a, b] > 0:
            score += np.log(trans_prob[a, b])
    return score

def spacing_penalty(candidate):
    return -np.std(np.diff(np.sort(candidate)))

def mahalanobis_score(candidate, draws):
    mean_vec = np.mean(draws, axis=0)
    cov_matrix = np.cov(draws.T)
    cov_inv = inv(cov_matrix)
    return mahalanobis(candidate, mean_vec, cov_inv)

def random_adjustment():
    return random.uniform(0, 1)

def score_candidate(candidate, draws):
    return (
        fft_score(draws) +
        entropy_score(draws) +
        bayes_score(candidate, draws) +
        spacing_penalty(candidate) -
        mahalanobis_score(candidate, draws) +
        random_adjustment()
    )

def predict_top_sets(df, num_predictions=200):
    draws = preprocess_data(df)
    predictions = []
    for _ in range(num_predictions * 5):
        main_numbers = np.sort(np.random.choice(range(1, 46), 7, replace=False))
        powerball = np.random.choice(range(1, 21))
        score = score_candidate(main_numbers, draws)
        predictions.append((score, main_numbers.tolist(), powerball))
    predictions.sort(reverse=True, key=lambda x: x[0])
    top = predictions[:num_predictions]
    return pd.DataFrame([{
        "Main Numbers": row[1],
        "Powerball": row[2],
        "Ψ Score": round(row[0], 3)
    } for row in top])
