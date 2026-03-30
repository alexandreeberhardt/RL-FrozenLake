import json
import os
import numpy as np

SEEDS = [0, 1, 2, 3, 4]
POLICIES = ["QLearning", "SARSA"]
CASES = [1, 2, 3]

N_EPISODES = 100_000
ALPHA = 0.1
GAMMA = 0.99

RESULTS_DIR = "results"


def epsilon(episode_idx):
    return max(0.9995 ** (episode_idx - 1), 0.01)


def choose_action(Q, state, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[state]))


def train(policy, env, seed, hole_reward, goal_reward=1.0):
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

        if policy == "SARSA":
            action = choose_action(Q, obs, eps, rng)

        while True:
            if policy == "Random":
                action = int(rng.integers(n_actions))
            elif policy == "QLearning":
                action = choose_action(Q, obs, eps, rng)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if policy != "Random":
                target = reward
                if not (terminated or truncated):
                    if policy == "QLearning":
                        target += GAMMA * np.max(Q[next_obs])
                    elif policy == "SARSA":
                        next_action = choose_action(Q, next_obs, eps, rng)
                        target += GAMMA * Q[next_obs, next_action]
                Q[obs, action] += ALPHA * (target - Q[obs, action])
            obs = next_obs

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


def main():
    from generate_case_study import generate_env, CASE_CONFIGS
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for cs in CASES:
        hole_reward = CASE_CONFIGS[cs]["rewards"][1]
        goal_reward = CASE_CONFIGS[cs]["rewards"][0]
        for policy in POLICIES:
            all_seeds = []
            for seed in SEEDS:
                print(f"\nCas {cs}, {policy}, Seed {seed}")
                env = generate_env(cs, render_mode=None)
                episodes, _ = train(policy, env, seed, hole_reward, goal_reward)
                env.close()
                all_seeds.append(episodes)
                print(f"{len(episodes)} épisodes, {sum(e['fell_in_hole'] for e in episodes)} chutes")

            path = f"{RESULTS_DIR}/cs{cs}_{policy}.json"
            with open(path, "w") as f:
                json.dump(all_seeds, f)

    print("\nBILAN GÉNÉRAL")
    for cs in CASES:
        print(f"\nCas {cs}")
        for policy in POLICIES:
            path = f"{RESULTS_DIR}/cs{cs}_{policy}.json"
            with open(path) as f:
                all_seeds = json.load(f)
            rewards = [x["reward"] for seed in all_seeds for x in seed[-100:]]
            holes = [x["fell_in_hole"] for seed in all_seeds for x in seed]
            successes = [x["success"] for seed in all_seeds for x in seed[-100:]]
            mean_r = round(float(np.mean(rewards)), 4)
            hole_rate = round(sum(holes) / len(holes) * 100, 1)
            success_rate = round(sum(successes) / len(successes) * 100, 1)
            print(f"{policy} : recompense moyenne: {mean_r:7.4f}, chutes: {hole_rate}%, succès: {success_rate}%")


if __name__ == "__main__":
    main()
