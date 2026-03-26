import gymnasium as gym

# observation space
# starting state
"""
S for Start tile
G for Goal tile
F for frozen tile
H for a tile with a hole
"""

# Cas 1 : Efficacité pure
# Description et défi : Cette carte présente une grille assez vide avec quelques trous. Le but principal n’est pas la survie,
# mais l’optimisation. L’objectif est d’apprendre à l’agent à aller droit au but sans faire de détours inutiles.

# Cas 2 : Évitement d'obstacles

# Description et défi : Ce scénario propose un environnement plus dangereux. Il n'existe qu'un seul couloir viable, placé en diagonale et entièrement entouré de trous. 
# Il n'y a aucune autre possibilité : le défi est de ne pas tomber, ce qui nécessite une navigation d'une grande prudence.

# Cas 3 : Compromis risque/récompense
# Description et défi : Cette carte introduit un dilemme de pathfinding en proposant deux itinéraires distincts vers l’objectif.
# - Un chemin court (6 pas) allant tout droit le long de la première ligne. Il est très risqué car la moindre glissade vers le bas entraîne une chute dans une ligne de trous.
# - Un chemin long (18 pas) qui contourne entièrement la zone de danger. Il est parfaitement sûr, mais très inefficace en termes de distance.
 
maps = [
    [
        "SFFFFFF",
        "FFHFFFF",
        "FFFFFFF",
        "FFFHFFF",
        "FFFFFFF",
        "FFFFFHF",
        "FFFFFFG",
    ],
    [
        "SFHHHHH",
        "FFFHHHH",
        "HFFFHHH",
        "HHFFFHH",
        "HHHFFFH",
        "HHHHFFF",
        "HHHHHFG",
    ],
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

CASE_CONFIGS = {
    1: {"success_rate": 0.90, "rewards": (5.0, -0.5, -0.01)},
    2: {"success_rate": 0.90, "rewards": (5.0, -1.0, -0.01)},
    3: {"success_rate": 0.75, "rewards": (10.0, -1.0, -0.05)},
}

def generate_env(case_study_nb, render_mode="human"):
    config = CASE_CONFIGS[case_study_nb]
    print(f"Génération du cas d'étude N°{case_study_nb} "
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

TIMESTEPS = 100_000

def train_model(name, env, cs):
    if name == "PPO":
        from stable_baselines3 import PPO

        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True, log_interval=4)
        filename = f"ppo_frozenlake_cs{cs}"
        model.save(filename)
    if name == "DQN":
        from stable_baselines3 import DQN

        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True, log_interval=4)
        filename = f"dqn_frozenlake_cs{cs}"
        model.save(filename)
