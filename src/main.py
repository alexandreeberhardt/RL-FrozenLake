from generate_case_study import generate_env, create_recorder, train_model
from stable_baselines3 import PPO, DQN
from parser import generate_argparser
from stable_baselines3.common.evaluation import evaluate_policy

parser = generate_argparser()

def main():
    args = parser.parse_args()
    print("Génération de l'environnement")
    cs = int(args.case_study)
    env = generate_env(cs)
    if args.train:
        train_model(args.policy,env,cs)
   
    if args.policy == "PPO":
        model = PPO.load(f"ppo_frozenlake_cs{cs}")
    elif args.policy == "DQN":
        model = DQN.load(f"dqn_frozenlake_cs{cs}")
    mean_reward, std_reward = evaluate_policy(model,env, n_eval_episodes=10)
    print(f"\nRécompense moyenne sur le case study {cs} pour la politique {args.policy}: {mean_reward}\n")



if __name__ == "__main__":
    main()
