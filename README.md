# RL FrozenLake

> **Ne pas utiliser Python 3.14** car pygame ne supporte pas le chargement d'images PNG dans cette version. Utiliser **Python 3.12**.

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
cd src

# Visualiser les 3 cas d'étude
python test_envs.py

# Tester un run rapide (1 seed, 3 cas, 5 stratégies)
python test_experiment.py

# Entraîner et évaluer une stratégie
python main.py -p QLearning -c 1 --train
python main.py -p SARSA -c 2 --train
python main.py -p DQN -c 3 --train

# Évaluer uniquement (modèle déjà entraîné)
python main.py -p QLearning -c 1

# Lancer toutes les expériences (5 seeds × 3 cas × 5 stratégies)
python experiment.py
```

**Options :**
- `-p` / `--policy` : `Random`, `QLearning`, `SARSA`, `DQN`, `PPO`
- `-c` / `--case-study` : `1`, `2` ou `3`
- `-t` / `--train` : entraîne et sauvegarde le modèle avant évaluation

## Stratégies

- **Random** : baseline aléatoire
- **QLearning** : apprentissage hors-politique (off-policy), tabulaire
- **SARSA** : apprentissage sur-politique (on-policy), tabulaire
- **DQN** : Deep Q-Network, réseau de neurones
- **PPO** : Proximal Policy Optimization, réseau de neurones

## Cas d'étude

| # | Composante | `success_rate` | Rewards (Goal / Hole / Frozen) |
|---|---|---|---|
| 1 | Efficacité | 0.90 | +1 / -0.5 / -0.01 |
| 2 | Évitement d'obstacles | 0.60 | +1 / -5.0 / 0.0 |
| 3 | Compromis risque/récompense | 0.75 | +1 / -1.0 / -0.05 |
