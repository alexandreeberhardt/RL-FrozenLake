"""
generate_case_study.py
Définition des trois cas d'étude du projet et génération des environnements
FrozenLake correspondants via la bibliothèque Gymnasium.

Chaque cas d'étude isole une difficulté spécifique :
  - Cas 1 : Efficacité pure - grille clairsemée, pénalité de pas
  - Cas 2 : Évitement strict - couloir diagonal entouré de trous
  - Cas 3 : Compromis risque/récompense - deux chemins (court risqué, long sûr)

Constantes exportées :
  CASE_CONFIGS : dict - paramètres (success_rate, rewards) par cas
  TIMESTEPS : int - nombre de pas d'entraînement (100 000)
"""

import gymnasium as gym

# Cartes des environnements
# Légende des tuiles :
#   S - départ (Start)
#   F - glace (Frozen) : déplacement stochastique
#   H - trou (Hole) : état terminal négatif
#   G - objectif (Goal) : état terminal positif

maps = [
    # Cas 1 : Efficacité pure
    # Grille 7x7 clairsemée avec 3 trous isolés.
    # L'agent doit apprendre le chemin le plus court (pénalité -0.01 par pas).
    [
        "SFFFFFF",
        "FFHFFFF",
        "FFFFFFF",
        "FFFHFFF",
        "FFFFFFF",
        "FFFFFHF",
        "FFFFFFG",
    ],
    # Cas 2 : Évitement strict
    # Un unique couloir diagonal praticable, entouré de trous.
    # Toute glissade hors du couloir est fatale.
    [
        "SFHHHHH",
        "FFFHHHH",
        "HFFFHHH",
        "HHFFFHH",
        "HHHFFFH",
        "HHHHFFF",
        "HHHHHFG",
    ],
    # Cas 3 : Compromis risque/récompense
    # Deux chemins possibles vers G :
    #   - court (6 pas, ligne du haut) : risqué car glissade -> ligne de trous
    #   - long (~18 pas, contournement) : sûr mais coûteux en pénalités de pas
    [
        "SFFFFFG",
        "FHHHHHF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
        "FFFFFFF",
    ],
]

# Configuration des cas d'étude
# Chaque entrée est un dict contenant :
#   success_rate : float - probabilité de succès d'un déplacement (1 - taux de glissade)
#   rewards      : tuple (goal_reward, hole_reward, step_reward)
#                    goal_reward - récompense à l'arrivée sur G
#                    hole_reward - récompense (négative) à l'arrivée sur H
#                    step_reward - récompense (négative) à chaque pas sur F
CASE_CONFIGS = {
    1: {"success_rate": 0.90, "rewards": (1.0,  -0.5, -0.01)},
    2: {"success_rate": 0.90, "rewards": (1.0,  -1.0,  0.0)},
    3: {"success_rate": 0.75, "rewards": (1.75, -1.0, -0.05)},
}

# Nombre total de pas d'entraînement par stratégie et par cas d'étude
TIMESTEPS = 100_000


def generate_env(case_study_nb, render_mode="human"):
    """
    Crée et retourne un environnement FrozenLake-v1 configuré selon le cas d'étude.

    Parameters
    ----------
    case_study_nb : int
        Numéro du cas d'étude (1, 2 ou 3).
    render_mode : str or None
        Mode de rendu Gymnasium ("human", "rgb_array", None, etc.).

    Returns
    -------
    gymnasium.Env
        Environnement FrozenLake configuré avec la carte, le taux de succès
        et la distribution des récompenses du cas demandé.
    """
    config = CASE_CONFIGS[case_study_nb]
    print(f"Génération du cas d'étude No {case_study_nb} "
          f"(success_rate={config['success_rate']}, rewards={config['rewards']})")
    env = gym.make(
        'FrozenLake-v1',
        desc=maps[case_study_nb - 1],
        is_slippery=True,
        success_rate=config["success_rate"],
        reward_schedule=config["rewards"],
        render_mode=render_mode,
    )
    return env


def train_model(name, env, cs):
    """
    Entraîne un modèle Stable-Baselines3 (PPO ou DQN) et le sauvegarde sur disque.

    Parameters
    ----------
    name : str
        Nom de la stratégie ("PPO" ou "DQN").
    env : gymnasium.Env
        Environnement d'entraînement.
    cs : int
        Numéro du cas d'étude (utilisé dans le nom du fichier de sauvegarde).
    """
    if name == "PPO":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True, log_interval=4)
        model.save(f"ppo_frozenlake_cs{cs}")

    if name == "DQN":
        from stable_baselines3 import DQN
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True, log_interval=4)
        model.save(f"dqn_frozenlake_cs{cs}")
