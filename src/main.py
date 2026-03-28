import numpy as np
from generate_case_study import generate_env, CASE_CONFIGS
from experiment import train
from parser import generate_argparser

parser = generate_argparser()

TABULAR = ["Random", "QLearning", "SARSA"]
SB3 = ["DQN", "PPO"]

def save_qtable(Q, policy_name, cs):
    np.save(f"{policy_name.lower()}_cs{cs}.npy", Q)


def load_qtable(policy_name, cs):
    return np.load(f"{policy_name.lower()}_cs{cs}.npy")


def evaluate_tabular(Q, policy_name, env, hole_reward, n_episodes=10):
    rewards, holes = [], []
    obs, _ = env.reset()

    for ep in range(n_episodes):
        if ep > 0:
            obs, _ = env.reset()
        ep_reward = 0.0

        while True:
            if policy_name == "Random":
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[obs]))

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                rewards.append(ep_reward)
                holes.append(int(terminated and abs(reward - hole_reward) < 1e-6))
                break

    mean_r = round(float(np.mean(rewards)), 4)
    std_r = round(float(np.std(rewards)), 4)
    hole_rate = round(sum(holes) / n_episodes, 4)
    return mean_r, std_r, hole_rate

def train_sb3(policy_name, env, cs):
    from generate_case_study import TIMESTEPS
    if policy_name == "PPO":
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
        model.save(f"ppo_frozenlake_cs{cs}")
    elif policy_name == "DQN":
        from stable_baselines3 import DQN
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
        model.save(f"dqn_frozenlake_cs{cs}")


def load_sb3(policy_name, cs):
    if policy_name == "PPO":
        from stable_baselines3 import PPO
        return PPO.load(f"ppo_frozenlake_cs{cs}")
    elif policy_name == "DQN":
        from stable_baselines3 import DQN
        return DQN.load(f"dqn_frozenlake_cs{cs}")


def evaluate_sb3(model, env, n_episodes=10):
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    return round(mean_r, 4), round(std_r, 4)

def main():
    args = parser.parse_args()
    cs = int(args.case_study)
    policy = args.policy
    hole_reward = CASE_CONFIGS[cs]["rewards"][1]

    if policy in TABULAR:
        if args.train:
            print(f"Entraînement de {policy} sur le cas {cs}...")
            env = generate_env(cs, render_mode=None)
            _, Q = train(policy, env, seed=0, hole_reward=hole_reward)
            env.close()
            save_qtable(Q, policy, cs)
            print(f"Modèle sauvegardé : {policy.lower()}_cs{cs}.npy")

        print(f"Évaluation de {policy} sur le cas {cs}...")
        Q = load_qtable(policy, cs)
        env = generate_env(cs, render_mode="None")
        mean_r, std_r, hole_rate = evaluate_tabular(Q, policy, env, hole_reward)
        env.close()
        print(f"\nRécompense moyenne : {mean_r} ± {std_r}")
        print(f"Taux de chute dans les trous : {hole_rate * 100:.1f}%")

    elif policy in SB3:
        if args.train:
            print(f"Entraînement de {policy} sur le cas {cs}...")
            env = generate_env(cs, render_mode=None)
            train_sb3(policy, env, cs)
            env.close()
            print(f"Modèle sauvegardé : {policy.lower()}_frozenlake_cs{cs}.zip")

        print(f"Évaluation de {policy} sur le cas {cs}...")
        model = load_sb3(policy, cs)
        env = generate_env(cs, render_mode="human")
        mean_r, std_r = evaluate_sb3(model, env)
        env.close()
        print(f"\nRécompense moyenne : {mean_r} ± {std_r}")


if __name__ == "__main__":
    main()
