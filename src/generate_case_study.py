import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


# observation space
# starting state
"""
S for Start tile
G for Goal tile
F for frozen tile
H for a tile with a hole
"""
maps = [
    [   "SF",
        "HG"
    ],# 2x2 par exemple
    [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ], # 4x4
    [],
]


def generate_env(case_study_nb):
    print(f"  Génération du cas d'étude N°{case_study_nb} ")
    env = gym.make('FrozenLake-v1',
                   desc=maps[case_study_nb -1], # modifier ici pour changer la map
                   is_slippery=True, # rend les déplacements stochastiques
                   success_rate= 3.0/4.0, # proba de réaliser l'action souhaitée
                   reward_schedule=(1.0,-1.0,0.0),
                   # les récompenses pour Reach Goal, Reach Hole, Reach Frozen respectivement
                )

    print_env_data(env)
    return env

def create_recorder(case_study_nb,prefix="eval"):
    env = gym.make('FrozenLake-v1',
                   desc=maps[case_study_nb - 1],
                   is_slippery=True,
                   success_rate=3.0 / 4.0,
                   reward_schedule=(1.0, -1.0, 0.0),
                   render_mode="rgb_array"
                   )
    env_rec = RecordVideo(
        env,
        video_folder="fl-agent",
        name_prefix=prefix,
        episode_trigger=lambda x: True  # Record every episode
    )

    return env_rec


def print_env_data(env):
    print(f"  Action space: {env.action_space}")
    print(f"    - 0 : Gauche")
    print(f"    - 1 : Bas")
    print(f"    - 2 : Droit")
    print(f"    - 3 : Haut")
    print(f"  Observation space: {env.observation_space} i.e number of tiles")
