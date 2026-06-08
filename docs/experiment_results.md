# Resultados experimentales

Resultados generados automaticamente con:

```bash
python3 -m othello_ai.experiments --selfplay-games 50 --selfplay-agent uct:12 --epochs 35 --tournament-games 12 --uct-iters 16 --seed 23
```

## Dataset

- Partidas de autojuego: 50
- Agente de autojuego: `uct:12`
- Ejemplos tras simetrias: 23792
- Fichero: `data/processed/selfplay_experiment.npz`

## Entrenamiento

- Modelo: `models/value_net_experiment.npz`
- Capas ocultas: `(64, 32)`
- Epocas: 35
- Loss entrenamiento final: 0.79606
- Loss validacion final: 0.80524

## Torneos

| Agente A | Agente B | Partidas | Victorias A | Victorias B | Empates | Dif. media A | Tiempo (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `greedy` | `random` | 12 | 9 | 3 | 0 | 8.83 | 0.3 |
| `heuristic` | `greedy` | 12 | 11 | 1 | 0 | 24.92 | 1.1 |
| `uct:16` | `greedy` | 12 | 8 | 4 | 0 | 6.00 | 36.3 |
| `uct:32` | `uct:16` | 12 | 7 | 4 | 1 | 9.33 | 109.2 |
| `uctnn:models/value_net_experiment.npz:16` | `random` | 12 | 11 | 0 | 1 | 17.33 | 2.5 |
| `uctnn:models/value_net_experiment.npz:16` | `greedy` | 12 | 9 | 3 | 0 | 9.50 | 2.7 |
| `uctnn:models/value_net_experiment.npz:16` | `uct:16` | 12 | 10 | 2 | 0 | 13.33 | 39.5 |

Nota: las partidas alternan el color inicial para reducir el sesgo de mover primero.
