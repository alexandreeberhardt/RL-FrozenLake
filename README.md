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
