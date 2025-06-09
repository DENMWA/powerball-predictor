
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import entropy
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
import random

def preprocess_data(df):
    draw_cols = [col for col in df.columns if "number" in col.lower()]
    draws = df[draw_cols[:7]].astype(int).values
    return draws

def compute_frequency_weights(draws):
    flat = draws.flatten()
    counts = np.bincount(flat, minlength=46)
    weights = counts / counts.sum()
    return weights

def compute_entropy_map(draws):
    flat = draws.flatten()
    values, counts = np.unique(flat, return_counts=True)
    probs = counts / counts.sum()
    ent = -np.sum(probs * np.log(probs))
    entropy_map = np.full(46, ent)
    return entropy_map

def compute_fft_map(draws):
    sequence = [sum(draw) for draw in draws]
    fft_vals = np.real(fft(sequence))
    top_freqs = np.argsort(np.abs(fft_vals))[-10:]
    freq_map = np.zeros(46)
    for i in top_freqs:
        if i < 46:
            freq_map[i] += 1
    return freq_map

def compute_bayes_matrix(draws):
    transition_matrix = np.zeros((46, 46))
    for row in draws:
        for i in range(len(row)-1):
            transition_matrix[row[i], row[i+1]] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        trans_prob = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    return np.nan_to_num(trans_prob)

def spacing_penalty(candidate):
    if len(candidate) < 2:
        return 0
    return -np.std(np.diff(np.sort(candidate)))

def mahalanobis_conformity(candidate, draws):
    mean_vec = np.mean(draws, axis=0)
    cov_matrix = np.cov(draws.T)
    cov_inv = inv(cov_matrix)
    try:
        return -mahalanobis(candidate, mean_vec[:len(candidate)], cov_inv[:len(candidate), :len(candidate)])
    except:
        return 0

def select_next_number(current_set, draws, freq_w, ent_map, fft_map, bayes_mat):
    candidates = [n for n in range(1, 46) if n not in current_set]
    scores = []
    for c in candidates:
        extended = current_set + [c]
        score = (
            freq_w[c] +
            ent_map[c] +
            fft_map[c % len(fft_map)] +
            (bayes_mat[current_set[-1]][c] if current_set else 0) +
            spacing_penalty(extended) +
            mahalanobis_conformity(extended, draws) +
            random.uniform(0, 1)
        )
        scores.append((score, c))
    scores.sort(reverse=True)
    return scores[0][1]

def build_prediction_set(draws, freq_w, ent_map, fft_map, bayes_mat):
    chosen = []
    for _ in range(7):
        next_num = select_next_number(chosen, draws, freq_w, ent_map, fft_map, bayes_mat)
        chosen.append(next_num)
    powerball = random.choice(range(1, 21))
    return {"Main Numbers": sorted(chosen), "Powerball": powerball}

def construct_psi_optimized_sets(df, num_sets=200):
    draws = preprocess_data(df)
    freq_w = compute_frequency_weights(draws)
    ent_map = compute_entropy_map(draws)
    fft_map = compute_fft_map(draws)
    bayes_mat = compute_bayes_matrix(draws)
    results = [build_prediction_set(draws, freq_w, ent_map, fft_map, bayes_mat) for _ in range(num_sets)]
    return pd.DataFrame(results)
