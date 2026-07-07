# MLDodge — una IA que aprende a esquivar, desde cero

Una red neuronal que **nadie programó para jugar**: aprende sola a esquivar proyectiles que la
persiguen desde los bordes. Empezó como un experimento en Unity (2021) y lo reescribí entero en
**Python con NumPy puro** (2026) — sin motores de juego, sin frameworks de ML, con el *forward* de
la red y el algoritmo evolutivo escritos a mano.

### ▶ Demos jugables (en el navegador)

| | |
|---|---|
| **[Portada con ambas demos](https://hotakutsuki.github.io/MLDodge/)** | El hub del proyecto |
| **[Versión Python 2026](https://hotakutsuki.github.io/MLDodge/neural-dodger/)** | Ver a la IA · jugar · versus · la historia |
| **[Versión Unity 2021](https://hotakutsuki.github.io/MLDodge/unity/)** | El WebGL original |

La demo Python corre el **cerebro campeón** (una MLP de 9 573 pesos) 100% en tu navegador:
el motor del juego y la inferencia de la red están portados a JavaScript y **verificados 1:1**
contra el original en Python (error < 1e-6, cero decisiones divergentes).

---

## El problema

El jugador es un punto en una arena. Cada 0.6 s aparece un proyectil en un borde y viaja en línea
recta hacia donde estaba el jugador. Tocar un proyectil o una pared = muerte. Cinco acciones:
arriba, abajo, izquierda, derecha, quieto. El objetivo: sobrevivir el mayor tiempo posible.

## Cómo aprende (sin backprop)

Un **algoritmo genético**, no descenso de gradiente:

1. **Población** de 200 redes con pesos aleatorios.
2. Cada una **juega varias partidas**; su *fitness* = cuánto sobrevive (+ bonos por alejarse de las
   paredes y por diversidad de movimiento).
3. Los mejores se **seleccionan** (torneo), se **cruzan** (mezcla de pesos) y **mutan** (ruido gaussiano).
4. Repetir por cientos de generaciones. La estrategia **emerge** — nunca se escribió a mano.

## Lo que la red “ve”: visión egocéntrica

El mayor salto del proyecto fue la **representación de la entrada**. En vez de una lista de enemigos
(cuyo orden salta de frame a frame), la red recibe un **mapa de calor 7×7 de peligro centrado en
ella misma** y alineado con sus acciones: cada celda significa *siempre* lo mismo (“peligro hacia
aquí”). Incluye un canal de **previsión** — dónde estará cada proyectil a 0.5 s — porque como van en
línea recta, su futuro es exacto. Eso llevó la supervivencia de ~5 s a ~15 s.

*(Activá “Mostrar la visión de la red” en la demo para verlo dibujado sobre la arena.)*

## Resultados y experimentos

**El experimento que zanjó el proyecto — ¿más previsión = mejor?** Medición cara a cara,
60 semillas idénticas:

| Cerebro | Entrada | Supervivencia media |
|---|---:|---:|
| Sin previsión | 104 | 14.4 s |
| **1 ventana de futuro (0.5 s)** ★ | 153 | **16.9 s** |
| 3 ventanas (0.3/0.6/0.9 s) | 251 | 13.7 s |

**Más futuro no fue mejor.** El punto dulce fue *una sola ventana*: los proyectiles van en recta, así
que 0.5 s ya dice casi todo, y los canales extra solo diluyen la señal. Ese es el cerebro de la demo.

**Evolución vs. Aprendizaje por Refuerzo.** Entrené un segundo cerebro con RL (REINFORCE y luego
A2C con *critic*) para compararlos. **La evolución ganó con claridad.** El terreno de recompensa es
plano (moverse al azar ya sobrevive un rato), así que el gradiente de política casi no apunta a
ningún lado y converge a algo casi aleatorio. Un resultado negativo, pero legítimo y documentado.

**El techo honesto.** Una red puramente *reactiva* — que responde a menos de 1 s de futuro — no
planifica rutas de escape como un humano. Su límite real ronda los ~15 s de mediana. Llegar más
lejos pediría otra clase de modelo (memoria/planificación), no más ajuste del mismo.

## Estructura

```
python/            La reescritura 2026 (ver python/README.md)
  src/             Motor del juego, red (MLP), evolución, RL — NumPy puro
  saved_brains/    Cerebros campeones archivados (.brain + meta)
  web/             La demo web (motor + red portados a JS)
docs/              GitHub Pages
  index.html       Portada con ambas demos
  neural-dodger/   Demo Python (build de python/web)
  unity/           WebGL original de 2021
Assets/ ...        Proyecto Unity original (2021, C#)
```

## Correr la versión Python localmente

```bash
cd python
pip install -r requirements.txt
python -m src.main
```

---

*Proyecto personal de aprendizaje. La versión Python está construida a mano a propósito —
para entender de verdad qué pasa dentro de una red que aprende.*
