from generate_case_study import generate_env, CASE_CONFIGS, maps

for cs in [1, 2, 3]:
    config = CASE_CONFIGS[cs]
    print(f"\nCas {cs} (success_rate={config['success_rate']}, rewards={config['rewards']})")
    for row in maps[cs - 1]:
        print(row)
    env = generate_env(cs, render_mode="human")
    env.reset()
    input(f"Press Enter pour l'env suivant")
    env.close()
