"""
visualize.py
Affichage textuel de la politique apprise par une stratégie tabulaire.

La politique est extraite de la table Q (argmax sur les actions) et
représentée dans la grille avec des flèches directionnelles :
  < - aller à gauche (action 0)
  v - aller en bas (action 1)
  > - aller à droite (action 2)
  ^ - aller en haut (action 3)
  X - trou (tuile H, état terminal)
  G - objectif (tuile G, état terminal)
"""

import numpy as np

# Correspondance action -> symbole directionnel
ARROWS = {0: "<", 1: "v", 2: ">", 3: "^"}

# Cartes des environnements (identiques à generate_case_study.py)
MAPS = {
    1: ["SFFFFFF", "FFHFFFF", "FFFFFFF", "FFFHFFF", "FFFFFFF", "FFFFFHF", "FFFFFFG"],
    2: ["SFHHHHH", "FFFHHHH", "HFFFHHH", "HHFFFHH", "HHHFFFH", "HHHHFFF", "HHHHHFG"],
    3: ["SFFFFFG", "FHHHHHF", "FFFFFFF", "FFFFFFF", "FFFFFFF", "FFFFFFF", "FFFFFFF"],
}


def display_policy(policy_name, cs):
    """
    Charge la table Q sauvegardée et affiche la politique greedy dans la grille.

    Pour chaque cellule de la grille :
      - tuile H -> affiche "X" (trou, pas d'action pertinente)
      - tuile G -> affiche "G" (objectif atteint)
      - autres -> affiche la flèche correspondant à l'action de Q maximale

    Parameters
    ----------
    policy_name : str
        Nom de la stratégie (ex. "QLearning", "SARSA", "RiskQLearning").
        Doit correspondre au fichier {policy_name.lower()}_cs{cs}.npy.
    cs : int
        Numéro du cas d'étude (1, 2 ou 3).
    """
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
    for policy in ["QLearning", "SARSA", "RiskQLearning"]:
        for cs in [1, 2, 3]:
            display_policy(policy, cs)
