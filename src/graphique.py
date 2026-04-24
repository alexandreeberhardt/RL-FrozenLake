"""
graphique.py
Fonctions de visualisation des résultats d'entraînement.

Produit deux types de figures à partir des fichiers JSON sauvegardés
par experiment.py :

  - plot_training_rewards() : évolution de la récompense moyenne lissée
    sur les épisodes, pour chaque stratégie et chaque cas d'étude.

  - plot_training_holes() : évolution du taux de chutes lissé (%)
    sur les épisodes, indicateur clé de la prudence des stratégies.

Le lissage est réalisé par convolution avec une fenêtre glissante de
taille WINDOW (défaut : 500 épisodes) pour réduire le bruit stochastique.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

POLICIES = ["QLearning", "SARSA", "RiskQLearning"]
CASES = [1, 2, 3]
WINDOW = 500  # taille de la fenêtre de lissage (en épisodes)


def smooth_series(values):
    """
    Lisse une série temporelle par moyenne glissante (convolution uniforme).

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Série brute à lisser.

    Returns
    -------
    np.ndarray, shape (n - WINDOW + 1,)
        Série lissée. Les WINDOW-1 premiers points sont supprimés
        (artefact de bord de la convolution en mode "valid").
    """
    return np.convolve(values, np.ones(WINDOW) / WINDOW, mode="valid")


def load_rewards(policy, cs):
    """
    Charge et lisse la récompense épisodique pour chaque seed.

    Parameters
    ----------
    policy : str
        Nom de la stratégie (ex. "QLearning", "SARSA", "RiskQLearning").
    cs : int
        Numéro du cas d'étude (1, 2 ou 3).

    Returns
    -------
    mean : np.ndarray
        Récompense moyenne lissée sur les épisodes.
    std : np.ndarray
        Écart-type lissé sur les seeds.
    """
    with open(f"results/cs{cs}_{policy}.json") as f:
        all_seeds = json.load(f)
    # Matrice (n_seeds x n_episodes)
    rewards = np.array([[ep["reward"] for ep in seed] for seed in all_seeds])
    mean = np.array([smooth_series(rewards[i]) for i in range(len(rewards))])
    return mean.mean(axis=0), mean.std(axis=0)


def load_hole_rates(policy, cs):
    """
    Charge et lisse le taux de chutes épisodique pour chaque seed.

    Parameters
    ----------
    policy : str
        Nom de la stratégie (ex. "QLearning", "SARSA", "RiskQLearning").
    cs : int
        Numéro du cas d'étude (1, 2 ou 3).

    Returns
    -------
    mean : np.ndarray
        Taux de chutes moyen lissé (en %) sur les épisodes.
    std : np.ndarray
        Écart-type lissé sur les seeds.
    """
    with open(f"results/cs{cs}_{policy}.json") as f:
        all_seeds = json.load(f)
    holes = np.array(
        [[ep["fell_in_hole"] for ep in seed] for seed in all_seeds],
        dtype=float
    ) * 100  # conversion en pourcentage
    smoothed = np.array([smooth_series(holes[i]) for i in range(len(holes))])
    return smoothed.mean(axis=0), smoothed.std(axis=0)


def plot_training_rewards():
    """
    Trace l'évolution de la récompense moyenne lissée pendant l'entraînement
    pour toutes les stratégies et tous les cas d'étude.

    Produit une figure 1x3 (un sous-graphique par cas d'étude).
    La bande colorée représente la moyenne +/- écart-type sur 5 seeds.
    Les axes Y sont partagés pour permettre la comparaison inter-cas.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, cs in zip(axes, CASES):
        for policy in POLICIES:
            mean, std = load_rewards(policy, cs)
            episodes = np.arange(WINDOW, len(mean) + WINDOW)
            line, = ax.plot(episodes, mean, label=policy)
            ax.fill_between(episodes, mean - std, mean + std,
                            alpha=0.2, color=line.get_color())

        ax.set_title(f"Cas {cs}")
        ax.set_xlabel("Épisode")
        ax.set_ylabel("Récompense moyenne")
        ax.legend()
        ax.grid(True)

    fig.suptitle("Évolution de la récompense pendant l'entraînement")
    plt.tight_layout()
    return fig


def plot_training_holes():
    """
    Trace l'évolution du taux de chutes lissé pendant l'entraînement
    pour toutes les stratégies et tous les cas d'étude.

    Produit une figure 1x3 (un sous-graphique par cas d'étude).
    La bande colorée représente la moyenne +/- écart-type sur 5 seeds.
    L'axe Y est partagé (0-100 %) pour faciliter la comparaison inter-cas.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, cs in zip(axes, CASES):
        for policy in POLICIES:
            mean, std = load_hole_rates(policy, cs)
            episodes = np.arange(WINDOW, len(mean) + WINDOW)
            line, = ax.plot(episodes, mean, label=policy)
            ax.fill_between(episodes, mean - std, mean + std,
                            alpha=0.2, color=line.get_color())

        ax.set_title(f"Cas {cs}")
        ax.set_xlabel("Épisode")
        ax.set_ylabel("Taux de chutes (%)")
        ax.legend()
        ax.grid(True)

    fig.suptitle("Évolution du taux de chutes pendant l'entraînement")
    plt.tight_layout()
    return fig


def plot_training():
    """
    Affiche les deux figures (récompenses et taux de chutes) à l'écran.
    Appelle plot_training_rewards() et plot_training_holes() puis plt.show().
    """
    plot_training_rewards()
    plot_training_holes()
    plt.show()


if __name__ == "__main__":
    plot_training()
