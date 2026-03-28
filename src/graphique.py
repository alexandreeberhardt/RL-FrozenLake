import json
import numpy as np
import matplotlib.pyplot as plt

POLICIES = ["QLearning", "SARSA"]
CASES = [1, 2, 3]
WINDOW = 500

def load_rewards(policy, cs):
    with open(f"results/cs{cs}_{policy}.json") as f:
        all_seeds = json.load(f)
    rewards = np.array([[ep["reward"] for ep in seed] for seed in all_seeds]) # moyenne lissée sur les seeds
    mean = rewards.mean(axis=0)
    smoothed = np.convolve(mean, np.ones(WINDOW)/WINDOW, mode="valid") # moyenne smooth sur WINDOW (500)
    return smoothed

def plot_training():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

    for ax, cs in zip(axes, CASES):
        for policy in POLICIES:
            smoothed = load_rewards(policy, cs)
            episodes = np.arange(WINDOW, len(smoothed) + WINDOW)
            ax.plot(episodes, smoothed, label=policy)

        ax.set_title(f"Cas {cs}")
        ax.set_xlabel("Épisode")
        ax.set_ylabel(f"Récompense moyenne")
        ax.legend()
        ax.grid(True)

    fig.suptitle("Évolution de la récompense pendant l'entraînement")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training()
