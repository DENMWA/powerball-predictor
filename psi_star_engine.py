
import numpy as np
import pandas as pd
import random
from scipy.special import softmax

def psi_score(candidate_set, historical_draws):
    fft_score = np.sum(np.abs(np.fft.fft([sum(draw) for draw in historical_draws])[:7]))
    entropy_score = -np.sum(np.bincount(candidate_set, minlength=46)/len(candidate_set) *
                            np.log1p(np.bincount(candidate_set, minlength=46)/len(candidate_set)))
    spacing_penalty = -np.std(np.diff(sorted(candidate_set)))
    noise = random.uniform(0, 1)
    return fft_score + entropy_score + spacing_penalty + noise

def drake_d1_score(candidate_set):
    even_count = sum(1 for n in candidate_set if n % 2 == 0)
    balance_score = -abs(even_count - 3.5)
    gap_score = -np.std(np.diff(sorted(candidate_set)))
    return (len(set(candidate_set)) ** 1.2) + balance_score + gap_score

def optimize_powerball(past_powerballs):
    if not past_powerballs or len(past_powerballs) < 5:
        return random.randint(1, 20)

    freq = pd.Series(past_powerballs).value_counts().reindex(range(1, 21), fill_value=0)
    gaps = pd.Series({
        pb: (len(past_powerballs) - max(i for i, x in enumerate(past_powerballs) if x == pb))
        if pb in past_powerballs else len(past_powerballs)
        for pb in range(1, 21)
    })

    score = freq + 1.5 * gaps
    if score.sum() == 0 or score.isnull().any():
        return random.randint(1, 20)

    probs = softmax(score)
    return int(np.random.choice(range(1, 21), p=probs))

def softmax_sampler(scored_sets, temperature=3.0):
    scores = np.array([s[1] for s in scored_sets])
    probs = softmax(temperature * scores)
    selected_index = np.random.choice(len(scored_sets), p=probs)
    return scored_sets[selected_index][0]

def evolve_sets(base_sets, historical_draws, generations=2):
    population = base_sets[:]
    for _ in range(generations):
        scored = [(s, psi_score(s, historical_draws) + drake_d1_score(s)) for s in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_half = [s[0] for s in scored[:len(scored)//2]]
        offspring = []
        while len(offspring) < len(population) - len(top_half):
            p1, p2 = random.sample(top_half, 2)
            child = sorted(list(set(random.sample(p1, 4) + random.sample(p2, 3))))[:7]
            if len(child) < 7:
                child += random.sample([n for n in range(1, 46) if n not in child], 7 - len(child))
            offspring.append(child)
        population = top_half + offspring
    return population

def generate_predictions(historical_draws, past_powerballs, num_predictions=200):
    base_candidates = [sorted(random.sample(range(1, 46), 7)) for _ in range(num_predictions * 2)]
    scored = [(s, psi_score(s, historical_draws) + drake_d1_score(s)) for s in base_candidates]
    selected_sets = [softmax_sampler(scored) for _ in range(num_predictions)]
    evolved_sets = evolve_sets(selected_sets, historical_draws)

    predictions = []
    for s in evolved_sets[:num_predictions]:
        powerball = optimize_powerball(past_powerballs)
        record = {
            "Main Numbers": s,
            "Powerball": powerball,
            "Ψ Score": round(psi_score(s, historical_draws), 3),
            "D1 Score": round(drake_d1_score(s), 3)
        }
        predictions.append(record)

    df = pd.DataFrame(predictions)
    print(df.head())  # Final debug print to confirm integrity
    return df
