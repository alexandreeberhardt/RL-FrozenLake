from generate_case_study import generate_env, train_model
from parser import generate_argparser

parser = generate_argparser()


def load_model(policy_name, case_study):
    if policy_name == "PPO":
        from stable_baselines3 import PPO

        return PPO.load(f"ppo_frozenlake_cs{case_study}")
    if policy_name == "DQN":
        from stable_baselines3 import DQN

        return DQN.load(f"dqn_frozenlake_cs{case_study}")

    raise ValueError(f"Unsupported policy: {policy_name}")


def main():
    from stable_baselines3.common.evaluation import evaluate_policy

    args = parser.parse_args()
    print("Génération de l'environnement")
    cs = int(args.case_study)
    env = generate_env(cs)
    if args.train:
        train_model(args.policy, env, cs)

    model = load_model(args.policy, cs)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"\nRécompense moyenne sur le case study {cs} pour la politique {args.policy}: {mean_reward}\n")



if __name__ == "__main__":
    main()
