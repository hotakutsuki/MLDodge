# MLDodge — versión Python (2026)

Reescritura del proyecto en **Python + NumPy puro**, sin motor de juego ni frameworks de ML.
Todo a mano: el *forward* de la red, el algoritmo evolutivo, el aprendizaje por refuerzo y el
renderizado (pygame). Ver la [historia completa](../README.md) en el README raíz.

## Correr

```bash
pip install -r requirements.txt
python -m src.main
```

Se abre una ventana con menú: entrenar (evolutivo o RL), ver demos, o jugar **versus** la IA.
Los cerebros campeones ya entrenados están en `saved_brains/`.

## Mapa del código (`src/`)

| Módulo | Qué hace |
|---|---|
| `game/engine.py` | Motor: física, spawns, colisiones, y la **codificación de la observación** (visión egocéntrica 7×7 + previsión). |
| `neural/mlp.py` | La red: MLP con *forward*, y backprop (para el módulo RL). Serialización de cerebros. |
| `evolution/trainer.py` | Algoritmo genético: selección por torneo, cruza, mutación, elitismo, semillas comunes (CRN), bono de diversidad, curriculum adaptativo. |
| `rl/reinforce.py` | Aprendizaje por refuerzo: REINFORCE → A2C (policy gradient con *critic*). El experimento que perdió contra la evolución. |
| `render_game.py` | Dibujo de la arena y de la “visión” de la red. |
| `params.py` | Todos los hiperparámetros, comentados. |
| `main.py` | UI, bucle principal, modos (entrenar / demo / versus). |

## La demo web (`web/`)

`web/index.html` es autocontenido: el motor y la red portados a JavaScript, con los pesos del
campeón embebidos. Es lo que se publica en GitHub Pages. Ver `web/README.md` para cómo se generó
y cómo re-exportar el cerebro si reentrenás.

## Decisiones de diseño (lo que aprendí)

- **Ruido de evaluación, no de mutación.** El aprendizaje se veía errático porque cada red se medía
  contra enemigos distintos. Fijar semillas comunes (CRN) por generación lo estabilizó.
- **La representación importa más que la capacidad.** Agrandar la red no ayudó; cambiar a un mapa de
  peligro egocéntrico sí — de ~5 s a ~15 s.
- **Evolución > RL en un terreno de recompensa plano.** Cuando moverse al azar ya sobrevive, el
  gradiente de política casi no informa; la búsqueda evolutiva encuentra mejores soluciones.
- **Más información no es gratis.** Tres horizontes de previsión rindieron *peor* que uno: dimensiones
  extra que diluyen la señal.
