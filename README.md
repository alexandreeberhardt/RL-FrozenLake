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

# Entraîner et évaluer
python main.py -p DQN -c 1 --train
python main.py -p PPO -c 2 --train

# Évaluer uniquement (modèle déjà entraîné)
python main.py -p DQN -c 1
python main.py -p PPO -c 1
```

**Options :**
- `-p` / `--policy` : `DQN` ou `PPO`
- `-c` / `--case-study` : `1`, `2` ou `3`
- `-t` / `--train` : entraîne et sauvegarde le modèle avant évaluation

## Cas d'étude

| # | Composante | `success_rate` | Rewards (Goal / Hole / Frozen) |
|---|---|---|---|
| 1 | Efficacité | 0.90 | +1 / -0.5 / -0.01 |
| 2 | Évitement d'obstacles | 0.60 | +1 / -5.0 / 0.0 |
| 3 | Compromis risque/récompense | 0.75 | +1 / -1.0 / -0.05 |
