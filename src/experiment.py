import numpy as np

SEEDS = [0, 1, 2, 3, 4]
POLICIES = ["Random","QLearning", "SARSA"]
CASES = [1, 2, 3]

N_EPISODES = 10_000
ALPHA = 0.1
GAMMA = 0.99

RESULTS_DIR = "results"


def epsilon(episode_idx):
    return max(0.9995 ** (episode_idx - 1), 0.01)


def choose_action(Q, state, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[state]))


def train(policy, env, seed, hole_reward):
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
                episodes.append({
                    "episode": ep,
                    "reward": round(ep_reward, 4),
                    "fell_in_hole": int(fell),
                })
                break

    return episodes, Q
