"""
experiment.py
Protocole expérimental principal du projet.

Ce module implémente les algorithmes d'apprentissage par renforcement tabulaires
(Q-learning, SARSA, Risk-sensitive Q-learning) et les fonctions d'entraînement
et d'évaluation associées.

Hyperparamètres communs (définis comme constantes du module) :
  N_EPISODES : int - nombre d'épisodes d'entraînement par seed (100 000)
  ALPHA : float - taux d'apprentissage (0.1)
  GAMMA : float - facteur d'escompte (0.99)
  KAPPA : float - paramètre de sensibilité au risque pour RiskQLearning (-0.5)
  kappa < 0 -> risk-averse (amplifie les erreurs TD négatives)

Protocole de reproductibilité :
  SEEDS : list[int] - 5 graines aléatoires pour lisser la variance
  N_EVAL_EPISODES : int - épisodes d'évaluation greedy après entraînement (1000)
"""

import json
import os
import numpy as np

# Constantes du protocole expérimental

SEEDS = [0, 1, 2, 3, 4]
POLICIES = ["QLearning", "SARSA", "RiskQLearning"]
CASES = [1, 2, 3]

N_EPISODES = 100_000
ALPHA = 0.1
GAMMA = 0.99

# Paramètre de sensibilité au risque pour Risk-sensitive Q-learning.
# kappa in (-1, 0) -> risk-averse : les erreurs TD négatives sont sur-pondérées
# par un facteur (1 - kappa) > 1, ce qui pousse l'agent à éviter les chutes.
KAPPA = -0.5

RESULTS_DIR = "results"
N_EVAL_EPISODES = 1000


# Politique d'exploration

def epsilon(episode_idx):
    """
    Calcule le taux d'exploration epsilon selon une décroissance exponentielle.

    epsilon décroît de ~1.0 à 0.01 sur N_EPISODES épisodes selon la formule :
        epsilon(t) = max(0.9995^(t-1), 0.01)

    Parameters
    ----------
    episode_idx : int
        Indice de l'épisode courant (commence à 1).

    Returns
    -------
    float
        Valeur de epsilon dans [0.01, 1.0].
    """
    return max(0.9995 ** (episode_idx - 1), 0.01)


def choose_action(Q, state, eps, rng):
    """
    Sélectionne une action selon la politique epsilon-greedy.

    Avec probabilité epsilon : action aléatoire uniforme (exploration).
    Avec probabilité 1-epsilon : action de valeur Q maximale (exploitation).

    Parameters
    ----------
    Q : np.ndarray, shape (n_states, n_actions)
        Table des valeurs Q courante.
    state : int
        État courant de l'agent.
    eps : float
        Taux d'exploration epsilon dans [0, 1].
    rng : np.random.Generator
        Générateur aléatoire NumPy (pour la reproductibilité).

    Returns
    -------
    int
        Indice de l'action choisie.
    """
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[state]))


# Transformation de l'erreur TD pour Risk-sensitive Q-learning
def risk_transform(delta, kappa):
    """
    Applique la transformation asymétrique de Mihatsch & Neuneier (2002)
    à l'erreur TD pour le Risk-sensitive Q-learning.

    La transformation amplifie les erreurs négatives (mauvaises surprises,
    comme tomber dans un trou) et atténue les positives lorsque kappa < 0 :
        f_kappa(delta) = (1 + kappa) * delta si delta >= 0 -> réduit le gain perçu
        f_kappa(delta) = (1 - kappa) * delta si delta < 0 -> amplifie la perte perçue

    Avec kappa = -0.5 : les erreurs négatives sont multipliées par 1.5,
    rendant l'agent plus conservateur face aux situations dangereuses.

    Parameters
    ----------
    delta : float
        Erreur TD standard : R + gamma * max_a Q(S', a) - Q(S, A).
    kappa : float
        Paramètre de risque dans (-1, 1). Négatif -> risk-averse.

    Returns
    -------
    float
        Erreur TD transformée f_kappa(delta).
    """
    if delta >= 0:
        return (1 + kappa) * delta
    return (1 - kappa) * delta


# Boucle d'entraînement
def train(policy, env, seed, hole_reward, goal_reward=1.0):
    """
    Entraîne une stratégie tabulaire sur un environnement FrozenLake.

    Implémente trois algorithmes :
      - "QLearning" : off-policy, cible = R + gamma * max_a Q(S', a)
      - "SARSA" : on-policy, cible = R + gamma * Q(S', A') ou A' ~ pi
      - "RiskQLearning" : off-policy avec transformation f_kappa de l'erreur TD
        (Mihatsch & Neuneier, 2002)

    La politique d'exploration est epsilon-greedy avec décroissance exponentielle.
    La table Q est initialisée à zéro.

    Parameters
    ----------
    policy : str
        Nom de la stratégie ("QLearning", "SARSA", "RiskQLearning" ou "Random").
    env : gymnasium.Env
        Environnement FrozenLake configuré.
    seed : int
        Graine aléatoire pour la reproductibilité.
    hole_reward : float
        Récompense associée aux tuiles H (permet de détecter les chutes).
    goal_reward : float, optional
        Récompense associée à la tuile G (défaut : 1.0).

    Returns
    -------
    episodes : list[dict]
        Liste des épisodes avec clés : "episode", "reward",
        "fell_in_hole", "success".
    Q : np.ndarray, shape (n_states, n_actions)
        Table Q apprise à l'issue de l'entraînement.
    """
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    episodes = []
    obs, _ = env.reset(seed=seed)

    for ep in range(1, N_EPISODES + 1):
        if ep > 1:
            obs, _ = env.reset()

        eps = epsilon(ep)
        ep_reward = 0.0

        # SARSA choisit l'action initiale avant d'entrer dans la boucle
        # pour respecter le schéma S, A, R, S', A'
        if policy == "SARSA":
            action = choose_action(Q, obs, eps, rng)

        while True:
            # Sélection de l'action (sauf SARSA qui l'a déjà choisie)
            if policy == "Random":
                action = int(rng.integers(n_actions))
            elif policy in ("QLearning", "RiskQLearning"):
                action = choose_action(Q, obs, eps, rng)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            # Mise à jour de la table Q
            if policy != "Random":
                # Cible de la valeur suivante selon la stratégie
                if terminated or truncated:
                    # État terminal : pas de valeur future
                    next_value = 0.0
                elif policy == "QLearning":
                    # Off-policy : on suppose le meilleur choix futur (greedy)
                    next_value = GAMMA * np.max(Q[next_obs])
                elif policy == "SARSA":
                    # On-policy : l'action suivante est celle que la politique
                    # d'exploration choisira réellement
                    next_action = choose_action(Q, next_obs, eps, rng)
                    next_value = GAMMA * Q[next_obs, next_action]
                elif policy == "RiskQLearning":
                    # Off-policy comme Q-learning, mais l'erreur TD est
                    # transformée par f_kappa pour pénaliser davantage les chutes
                    next_value = GAMMA * np.max(Q[next_obs])

                # Erreur TD standard
                delta = reward + next_value - Q[obs, action]

                if policy == "RiskQLearning":
                    # Transformation asymétrique : amplifie delta < 0
                    # (mauvaises surprises) avec le facteur (1 - kappa) > 1
                    # lorsque kappa < 0
                    delta = risk_transform(delta, KAPPA)

                Q[obs, action] += ALPHA * delta

            obs = next_obs

            # SARSA : réutilise l'action déjà choisie pour l'état suivant
            if policy == "SARSA" and not (terminated or truncated):
                action = next_action

            if terminated or truncated:
                fell = terminated and abs(reward - hole_reward) < 1e-6
                success = terminated and abs(reward - goal_reward) < 1e-6
                episodes.append({
                    "episode": ep,
                    "reward": round(ep_reward, 4),
                    "fell_in_hole": int(fell),
                    "success": int(success),
                })
                break

    return episodes, Q


# Évaluation greedy (sans exploration)
def evaluate_greedy(Q, env, hole_reward, goal_reward=1.0, n_episodes=N_EVAL_EPISODES):
    """
    Évalue une politique greedy (sans exploration) sur un environnement.

    L'agent choisit toujours l'action de valeur Q maximale (argmax).
    Cette évaluation est séparée de l'entraînement pour mesurer la qualité
    de la politique finale indépendamment du taux d'exploration résiduel.

    Parameters
    ----------
    Q : np.ndarray, shape (n_states, n_actions)
        Table Q issue de l'entraînement.
    env : gymnasium.Env
        Environnement d'évaluation (peut être une instance distincte).
    hole_reward : float
        Récompense associée aux tuiles H (pour détecter les chutes).
    goal_reward : float, optional
        Récompense associée à la tuile G (défaut : 1.0).
    n_episodes : int, optional
        Nombre d'épisodes d'évaluation (défaut : N_EVAL_EPISODES = 1000).

    Returns
    -------
    dict avec les clés :
        "mean_reward" : float - récompense moyenne sur n_episodes épisodes
        "hole_rate" : float - pourcentage d'épisodes terminés par une chute
        "success_rate" : float - pourcentage d'épisodes où l'objectif est atteint
    """
    rewards, holes, successes = [], [], []
    obs, _ = env.reset()

    for ep in range(n_episodes):
        if ep > 0:
            obs, _ = env.reset()

        ep_reward = 0.0

        while True:
            action = int(np.argmax(Q[obs]))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                rewards.append(ep_reward)
                holes.append(int(terminated and abs(reward - hole_reward) < 1e-6))
                successes.append(int(terminated and abs(reward - goal_reward) < 1e-6))
                break

    return {
        "mean_reward": round(float(np.mean(rewards)), 4),
        "hole_rate": round(float(np.mean(holes)) * 100, 1),
        "success_rate": round(float(np.mean(successes)) * 100, 1),
    }


# Script principal : entraînement et évaluation de toutes les combinaisons
def main():
    """
    Lance l'entraînement et l'évaluation pour toutes les combinaisons
    (stratégie x cas d'étude x seed) et sauvegarde les résultats en JSON.

    Fichiers produits dans results/ :
      cs{n}_{policy}.json - historique épisode par épisode (5 seeds)
      cs{n}_{policy}_eval.json - métriques d'évaluation greedy (5 seeds)
    """
    from generate_case_study import generate_env, CASE_CONFIGS
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for cs in CASES:
        hole_reward = CASE_CONFIGS[cs]["rewards"][1]
        goal_reward = CASE_CONFIGS[cs]["rewards"][0]
        for policy in POLICIES:
            all_seeds = []
            evals = []
            for seed in SEEDS:
                print(f"\nCas {cs}, {policy}, Seed {seed}")
                env = generate_env(cs, render_mode=None)
                episodes, Q = train(policy, env, seed, hole_reward, goal_reward)
                env.close()
                all_seeds.append(episodes)
                print(f"{len(episodes)} épisodes, {sum(e['fell_in_hole'] for e in episodes)} chutes")

                eval_env = generate_env(cs, render_mode=None)
                evals.append(evaluate_greedy(Q, eval_env, hole_reward, goal_reward))
                eval_env.close()

            path = f"{RESULTS_DIR}/cs{cs}_{policy}.json"
            with open(path, "w") as f:
                json.dump(all_seeds, f)

            eval_path = f"{RESULTS_DIR}/cs{cs}_{policy}_eval.json"
            with open(eval_path, "w") as f:
                json.dump(evals, f)

    print("\nBILAN GÉNÉRAL")
    for cs in CASES:
        print(f"\nCas {cs}")
        for policy in POLICIES:
            path = f"{RESULTS_DIR}/cs{cs}_{policy}_eval.json"
            with open(path) as f:
                evals = json.load(f)
            rewards = [x["mean_reward"] for x in evals]
            holes = [x["hole_rate"] for x in evals]
            successes = [x["success_rate"] for x in evals]
            mean_r = round(float(np.mean(rewards)), 4)
            hole_rate = round(float(np.mean(holes)), 1)
            success_rate = round(float(np.mean(successes)), 1)
            print(f"{policy:15s} : recompense moyenne: {mean_r:7.4f}, "
                  f"chutes: {hole_rate}%, succès: {success_rate}%")


if __name__ == "__main__":
    main()
