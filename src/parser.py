"""
parser.py
Définition de l'interface en ligne de commande (CLI) pour main.py.

Arguments disponibles :
  -t / --train : entraîne le modèle et le sauvegarde avant évaluation
  -p / --policy : stratégie à utiliser (Random, QLearning, SARSA,
    RiskQLearning, DQN, PPO)
  -c / --case-study : numéro du cas d'étude à utiliser (1, 2 ou 3)
"""


def generate_argparser():
    """
    Crée et retourne le parseur d'arguments pour le script main.py.

    Returns
    -------
    argparse.ArgumentParser
        Parseur configuré avec les arguments --train, --policy et --case-study.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Entraîne et évalue une stratégie RL sur FrozenLake."
    )
    parser.add_argument(
        "-t", "--train",
        action="store_true",
        help="Entraîne et sauvegarde le modèle avant de l'évaluer."
    )
    parser.add_argument(
        "-p", "--policy",
        required=True,
        choices=["Random", "QLearning", "SARSA", "RiskQLearning", "DQN", "PPO"],
        help="Stratégie à utiliser."
    )
    parser.add_argument(
        "-c", "--case-study",
        required=True,
        choices=["1", "2", "3"],
        help="Numéro du cas d'étude (1, 2 ou 3)."
    )
    return parser
