# Aprendiendo a jugar a Otelo

Proyecto en Python para la convocatoria de junio: agente de Otelo basado en
Monte Carlo Tree Search con UCT y una red neuronal de valor entrenada por
autojuego.

## Contenido

- `othello_ai/game.py`: reglas completas de Otelo/Reversi.
- `othello_ai/mcts.py`: implementacion propia de MCTS con seleccion UCT.
- `othello_ai/agents.py`: agentes `random`, `greedy`, `heuristic`, `uct` y
  `uctnn`.
- `othello_ai/self_play.py`: generacion de datasets por partidas automaticas.
- `othello_ai/neural.py`: red neuronal de valor implementada con NumPy.
- `othello_ai/train.py`: entrenamiento de la red de valor.
- `othello_ai/tournament.py`: comparativas entre agentes.
- `othello_ai/cli.py`: interfaz de texto para jugar contra la maquina.
- `tests/`: pruebas unitarias con `unittest`.
- `docs/memoria-otelo.tex`: memoria IEEE del trabajo.
- `docs/memoria-otelo.pdf`: memoria compilada.
- `docs/uso_ia_generativa.md`: declaracion separada sobre uso de IA generativa.

## Requisitos

El codigo solo necesita NumPy.

```bash
python3 -m pip install -r requirements.txt
```

## Pruebas

```bash
python3 -m unittest discover -s tests -v
```

## Documentacion

La memoria final esta en `docs/`:

- `docs/memoria-otelo.tex`
- `docs/memoria-otelo.pdf`
- `docs/IEEEtran.cls`

Para recompilarla:

```bash
cd docs
tectonic memoria-otelo.tex
```

## Jugar contra la maquina

```bash
python3 -m othello_ai.cli --agent uct:300 --human black
```

Coordenadas aceptadas: `d3`, `c4`, etc. Tambien se acepta `fila columna`, por
ejemplo `3 4`.

## Generar datos por autojuego

```bash
python3 -m othello_ai.self_play \
  --games 500 \
  --agent uct:80 \
  --output data/processed/selfplay_uct80_500.npz \
  --seed 11
```

Cada estado se etiqueta con `+1`, `0` o `-1` segun el resultado final desde la
perspectiva del jugador activo. Por defecto se aplican las 8 simetrias del
tablero para aumentar el dataset.

## Entrenar la red de valor

```bash
python3 -m othello_ai.train \
  --dataset data/processed/selfplay_uct80_500.npz \
  --output models/value_net_uct80_500.npz \
  --hidden 128,64 \
  --epochs 40 \
  --batch-size 128 \
  --lr 0.001 \
  --seed 11
```

La red recibe 128 atributos: 64 casillas propias y 64 casillas rivales, siempre
desde la perspectiva del jugador activo. La salida usa `tanh`, por lo que queda
en el rango `[-1, 1]`.

## Torneos

```bash
python3 -m othello_ai.tournament --agent-a uct:100 --agent-b greedy --games 50
python3 -m othello_ai.tournament --agent-a uctnn:models/value_net_uct80_500.npz:100 --agent-b uct:100 --games 50
```

Los jugadores alternan color en cada partida para evitar sesgos por mover
primero.

## Experimento reproducible

Para generar un dataset, entrenar la red y ejecutar una liga de agentes en un
solo paso:

```bash
python3 -m othello_ai.experiments \
  --selfplay-games 50 \
  --selfplay-agent uct:12 \
  --epochs 35 \
  --tournament-games 12 \
  --uct-iters 16 \
  --seed 23
```

La ejecucion validada genero:

- `data/processed/selfplay_experiment.npz`: 23.792 ejemplos.
- `models/value_net_experiment.npz`: modelo de valor entrenado.
- `docs/experiment_results.md`: resumen de resultados.
- `docs/experiment_results.json`: resultados estructurados.

Resumen de resultados obtenidos:

| Agente A | Agente B | Partidas | Victorias A | Victorias B | Empates | Dif. media A |
|---|---:|---:|---:|---:|---:|---:|
| `greedy` | `random` | 12 | 9 | 3 | 0 | 8.83 |
| `heuristic` | `greedy` | 12 | 11 | 1 | 0 | 24.92 |
| `uct:16` | `greedy` | 12 | 8 | 4 | 0 | 6.00 |
| `uct:32` | `uct:16` | 12 | 7 | 4 | 1 | 9.33 |
| `uctnn:...:16` | `random` | 12 | 11 | 0 | 1 | 17.33 |
| `uctnn:...:16` | `greedy` | 12 | 9 | 3 | 0 | 9.50 |
| `uctnn:...:16` | `uct:16` | 12 | 10 | 2 | 0 | 13.33 |

Estos numeros muestran que la ampliacion del dataset mejora claramente al agente
neuronal: UCTNN-16 supera a `random`, `greedy` y `uct:16` en esta ejecucion.
