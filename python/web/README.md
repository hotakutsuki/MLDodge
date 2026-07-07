# Neural Dodger — versión web

Página estática que corre el **cerebro evolutivo campeón (gen 590)** directo en el navegador:
motor del juego + *forward* del MLP portados a JavaScript. Sin servidor, sin dependencias.

- **Ver a la IA** jugar (con su "visión" 7×7 dibujada encima)
- **Jugar** vos mismo (flechas / WASD)
- **Versus** IA vs humano, misma semilla y dificultad
- **La historia** del proyecto (de ~5s a ~15s, por qué RL y el multi-horizonte no ganaron)

## Archivos
- `index.html` — **autocontenido** (pesos + motor embebidos). Es lo único que necesitás para publicar.
- `sim.src.js` — el motor JS legible (referencia; ya está inline en index.html).
- `brain.json` — pesos del campeón (referencia; ya está inline en index.html).

## Publicarlo
Cualquier host estático sirve. Opciones gratis:
- **GitHub Pages**: subí `index.html` a un repo, activá Pages → queda en `usuario.github.io/repo`.
- **Netlify / Vercel**: arrastrá la carpeta `web/` a su panel (drag & drop).
- **itch.io**: subí `index.html` como HTML game (zip con index.html en la raíz).
- **Local**: doble clic en `index.html` (funciona sin servidor).

## Re-exportar el cerebro (si reentrenás)
El JSON de pesos se genera desde el .brain de Python:

```python
import json
from src.neural.mlp import load_brain_from_file
net = load_brain_from_file('saved_brains/evo_pred1h_gen590.brain')
layers = [{'W':[[round(float(v),5) for v in row] for row in W],
           'b':[round(float(v),5) for v in b]}
          for W,b in zip(net.weight_matrices, net.bias_vectors)]
open('web/brain.json','w').write(
    json.dumps({'layer_sizes':list(net.layer_sizes),'layers':layers}, separators=(',',':')))
```

Luego re-embebé en index.html reemplazando el bloque `window.__BRAIN__=...`.

**Fidelidad:** el port JS se validó 1:1 contra el motor Python (error de observación < 1e-6,
0 acciones divergentes en 25 estados de prueba). La IA en la web juega idéntico al campeón.
El motor asume la config del campeón: entrada 153 (rejilla 7×7×3, previsión 1 ventana a 0.5s),
red 153→48→32→16→5, spawn 0.6s, velocidad 120. Si cambiás esos parámetros, actualizá `sim.js`.
