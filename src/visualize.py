import numpy as np

ARROWS = {0: "<", 1: "v", 2: ">", 3: "^"}

MAPS = {
    1: ["SFFFFFF", "FFHFFFF", "FFFFFFF", "FFFHFFF", "FFFFFFF", "FFFFFHF", "FFFFFFG"],
    2: ["SFHHHHH", "FFFHHHH", "HFFFHHH", "HHFFFHH", "HHHFFFH", "HHHHFFF", "HHHHHFG"],
    3: ["SFFFFFG", "FHHHHHF", "FFFFFFF", "FFFFFFF", "FFFFFFF", "FFFFFFF", "FFFFFFF"],
}


def display_policy(policy_name, cs):
    path = f"{policy_name.lower()}_cs{cs}.npy"
    Q = np.load(path)
    policy = np.argmax(Q, axis=1)
    desc = MAPS[cs]
    n = len(desc)

    print(f"{policy_name} : Cas {cs}")
    for i in range(n):
        row = ""
        for j in range(n):
            cell = desc[i][j]
            if cell == "H":
                row += " X "
            elif cell == "G":
                row += " G "
            else:
                row += f" {ARROWS[policy[i * n + j]]} "
        print(row)
    print()


if __name__ == "__main__":
    for policy in ["QLearning", "SARSA"]:
        for cs in [1, 2, 3]:
            display_policy(policy, cs)
