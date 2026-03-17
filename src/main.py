from generate_case_study import generate_env, create_recorder

def main():
   print("Génération de l'environnement")
   env = generate_env(1)
   env_rec = create_recorder(1)
   random_episode(env_rec)

def random_episode(env):
    obs, info = env.reset()
    episode_reward = 0
    step_count = 0

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # Random policy
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_count += 1

        episode_over = terminated or truncated

        print(f"{step_count} steps, reward = {episode_reward}")

    env.close()

if __name__ == "__main__":
    main()
