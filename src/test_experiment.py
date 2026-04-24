from generate_case_study import generate_env, CASE_CONFIGS
from experiment import train
from main import train_sb3, evaluate_sb3, TABULAR, SB3

for cs in [1, 2, 3]:
    print(f"\n Cas {cs} ")
    hole_reward = CASE_CONFIGS[cs]["rewards"][1]

    for policy in TABULAR:
        env = generate_env(cs, render_mode=None)
        episodes, _ = train(policy, env, seed=0, hole_reward=hole_reward)
        env.close()
        fell = sum(e["fell_in_hole"] for e in episodes)
        last_100 = episodes[-100:]
        mean_reward = round(sum(e["reward"] for e in last_100) / len(last_100), 4)
        print(f"{policy} : {fell} chutes, reward moyenne (100 derniers): {mean_reward}")

    for policy in SB3:
        env = generate_env(cs, render_mode=None)
        train_sb3(policy, env, cs)
        model_env = generate_env(cs, render_mode=None)
        from main import load_sb3
        model = load_sb3(policy, cs)
        mean_r, std_r = evaluate_sb3(model, model_env)
        env.close()
        model_env.close()
        print(f"{policy} : reward moyenne: {mean_r} +/- {std_r}")
