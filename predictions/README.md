# Predictions / Prédictions

Ce dossier contient les résultats des deux méthodes de détection sur des exemples de fichiers.

## Structure

```
predictions/
├── unsupervised/          # Résultats de la méthode non supervisée
│   ├── ball_data_1_predicted.json
│   ├── ball_data_10_predicted.json
│   └── ball_data_100_predicted.json
│
└── supervised/            # Résultats de la méthode supervisée (XGBoost)
    ├── ball_data_1_predicted.json
    ├── ball_data_10_predicted.json
    └── ball_data_100_predicted.json
```

## Format des fichiers

Les fichiers JSON enrichis contiennent la même structure que les fichiers d'entrée, avec le champ `"action"` mis à jour selon les prédictions :

```json
{
  "32500": {
    "x": 1024.5,
    "y": 768.2,
    "visible": true,
    "action": "hit"     // Prédiction: "air", "hit", ou "bounce"
  }
}
```

## Génération des prédictions

Pour générer vos propres prédictions :

```bash
# Méthode non supervisée
python main.py "Data hit & bounce/per_point_v2/ball_data_X.json" \
  --method unsupervised \
  --output predictions/unsupervised/ball_data_X_predicted.json

# Méthode supervisée
python main.py "Data hit & bounce/per_point_v2/ball_data_X.json" \
  --method supervised \
  --output predictions/supervised/ball_data_X_predicted.json
```

## Résultats sur les exemples

### ball_data_1.json
- **Unsupervised** : 5 hits, 1 bounce détectés
- **Supervised** : 6 hits, 3 bounces détectés

### ball_data_10.json
- **Unsupervised** : 5 hits, 4 bounces détectés
- **Supervised** : 2 hits, 0 bounces détectés

### ball_data_100.json
- **Unsupervised** : 3 hits, 2 bounces détectés
- **Supervised** : 7 hits, 3 bounces détectés

---

**Note** : Ces fichiers sont des exemples de sortie des deux méthodes. Vous pouvez générer des prédictions sur n'importe quel fichier du dataset en utilisant les commandes ci-dessus.
