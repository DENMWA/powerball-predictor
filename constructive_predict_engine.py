
import numpy as np
import pandas as pd
import random
from scipy.special import softmax

def compute_psi_score(candidate, freq_vector, entropy_weight=1.0):
    freq_score = np.sum(freq_vector[candidate])
    spacing_penalty = -np.std(np.diff(sorted(candidate)))
    return freq_score + entropy_weight * spacing_penalty

def predict_top_sets(data, num_predictions=100):
    main_numbers = data.iloc[:, 1:8].values.flatten()
    freq_vector = np.bincount(main_numbers, minlength=46)

    candidate_sets = [sorted(random.sample(range(1, 46), 7)) for _ in range(num_predictions * 3)]
    scores = [compute_psi_score(s, freq_vector) for s in candidate_sets]

    probs = softmax(scores)
    selected_indices = np.random.choice(range(len(candidate_sets)), size=num_predictions, p=probs)
    selected_sets = [candidate_sets[i] for i in selected_indices]

    final_output = [{
        "Main Numbers": s,
        "Powerball": random.randint(1, 20)
    } for s in selected_sets]

    return pd.DataFrame(final_output)
