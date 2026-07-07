"""
Parámetros del proyecto — edita este archivo para cambiar juego, red y evolución.

Juego (GameConfig)
-----------------
- arena_w / arena_h: tamaño lógico del tablero (unidades).
- player_radius / enemy_radius: radios de colisión.
- speed: velocidad del jugador y enemigos (unidades/segundo).
- spawn_interval: segundos entre apariciones de enemigos en un borde.
- obs_enemy_slots: cuántos enemigos codifica la observación (top-K por distancia).
- enemy_max_lifetime: segundos máximos de vida de un enemigo (libera lista).
- enemy_despawn_margin: si el centro del enemigo supera el rectángulo del nivel
  ampliado en este margen, se elimina (enemigos que ya “se fueron” lejos).

Red neuronal
------------
- NN_HIDDEN: tuplas con neuronas por capa OCULTA. La entrada es observation_dim()
  (depende de obs_enemy_slots) y la salida es NUM_ACTIONS (5).
  Ejemplo: (32, 24, 16) = 3 capas ocultas.

Evolución (EvolutionConfig)
---------------------------
- population_size: individuos por generación (pool).
- elite_count: mejores que pasan COPIADOS a la siguiente gen. (sin cruce ni mutación).
- tournament_k: tamaño del torneo al elegir padres (mayor k = más presión de selección).
- mutation_rate / mutation_sigma: valores iniciales de mutación.
- mutation_rate_end / mutation_sigma_end: hacia donde desciende con el annealing.
- mutation_anneal_generations: en cuántas generaciones se interpola (0 = sin annealing).
- eval_episodes: partidas por individuo al medir fitness (se usa la media).
- eval_episode_seed_stride: separación de semillas entre esas partidas.
- max_episode_seconds / fixed_dt: tope y paso de simulación al evaluar fitness.

Reproducción (resumen)
----------------------
1) Evaluar fitness (tiempo vivo) de todos.
2) Ordenar; los elite_count mejores entran intactos en la nueva población.
3) Hasta llenar: elegir 2 padres por torneo, cruce uniforme de vectores de pesos,
   mutación gaussiana esparsa, añadir hijo.
"""

from __future__ import annotations

from typing import List, Tuple

from src.evolution.trainer import EvolutionConfig
from src.game.engine import GameConfig, NUM_ACTIONS, observation_dim
from src.rl.reinforce import RLConfig

# --- Juego ---
GAME: GameConfig = GameConfig(
    arena_w=420.0,
    arena_h=420.0,
    player_radius=10.0,
    enemy_radius=9.0,
    speed=120.0,
    # spawn_interval 1.0 → 0.6 (jul 2026): a spawn 1.0s el campeón ya "vivía casi para
    # siempre" (28% de partidas topaban el episodio de 45s) y la señal de selección se
    # saturaba por arriba. Con más enemigos (~60% más en pantalla) la supervivencia baja
    # bajo el cap → señal afilada de nuevo Y episodios más cortos = entrenamiento más rápido.
    # La rejilla de visión los maneja gratis (mapa fijo 7×7, sin importar cuántos enemigos) y
    # el curriculum ADAPTATIVO evita pasarse (regresa si <40% sobrevive → busca la frontera
    # jugable sola). Se CONTINÚA la población actual (no reinicio): su competencia es un buen
    # punto de partida para el nivel más difícil. Objetivo final: que supere a un humano.
    spawn_interval=0.6,
    obs_enemy_slots=8,
    enemy_max_lifetime=35.0,
    enemy_despawn_margin=72.0,
    # Features de observación (cambian la dimensión de entrada → invalidan cerebros viejos):
    # - closing speed: escalar por enemigo "se acerca (>0) / se aleja (<0)".
    # - sort_by_threat: ordenar slots por amenaza (cercanía+acercamiento), no por distancia.
    obs_include_closing_speed=True,
    obs_sort_by_threat=True,
    # --- Visión por REJILLA EGOCÉNTRICA (jun 2026) ---
    # Reemplaza la lista de enemigos por un "mapa de calor" de peligro centrado en el
    # jugador y alineado con las 4 acciones (estable, no se reordena, capta múltiples
    # amenazas a la vez; las paredes = celdas fuera de la arena con peligro máximo).
    # Entrada = 6 (paredes) + rows*cols*2 canales. 7×7 con celda 60 cubre la arena (420).
    # Se DIBUJA en la demo (la "visión" del jugador) para confirmar que ve lo que debe.
    obs_use_vision_grid=True,
    vision_grid_rows=7,
    vision_grid_cols=7,
    vision_cell_size=60.0,
    vision_influence=70.0,
    # Canal de PREVISIÓN (jul 2026): 3er canal con el peligro FUTURO (enemigos extrapolados
    # en línea recta a t+dt), centrado en la posición actual del jugador. Le da anticipación
    # —la ventaja del humano— sin cambiar la red. Entrada = 6 + 7·7·3 = 153. Se ve en la demo
    # como contorno naranja (dónde va a estar el peligro).
    # Previsión: se probó MULTI-HORIZONTE (3 canales a 0.3/0.6/0.9s, entrada 251) como último
    # intento de superar el techo. Resultado (jul 2026): NO ayudó — empeoró. En medición cara a
    # cara con semillas idénticas, 1 sola ventana (gen590) ganó 16.9s vs 13.7s de 3 ventanas, y
    # el fitness de 3 ventanas topó en el MISMO techo (~42) que 1 ventana. Los enemigos van en
    # línea recta: el futuro a ~0.5s ya dice casi todo; los canales extra solo diluyen la señal
    # (+98 dims de ruido). Punto dulce = UNA ventana. Campeón definitivo: saved_brains/evo_pred1h_gen590.
    # Entrada = 6 + 7·7·(2+1) = 153. En la demo se ve como contorno naranja (dónde irá el peligro).
    vision_future_channel=True,
    vision_future_dts=(0.5,),
)

# --- Red: solo capas ocultas; entrada y salida se fijan automáticamente ---
# (48, 32, 16) desde jun 2026 (antes (24, 16), ~2000 pesos). Experimento #2: tras descartar
# ruido (CRN) y confirmar meseta real (+0.2s de supervivencia en 800 gens limpias), el cuello
# era CAPACIDAD: la red no integraba múltiples amenazas simultáneas y la 1ª capa (24) comprimía
# por debajo de la entrada (62). Ahora 1ª capa 48 > entrada, +capa intermedia 32. ~5000 pesos.
NN_HIDDEN: Tuple[int, ...] = (48, 32, 16)


def neural_layer_sizes(game: GameConfig | None = None) -> List[int]:
    g = game or GAME
    d = observation_dim(g)
    return [d, *NN_HIDDEN, NUM_ACTIONS]


# --- Evolución ---
EVOLUTION: EvolutionConfig = EvolutionConfig(
    population_size=200,
    elite_count=8,
    tournament_k=6,
    mutation_rate=0.12,
    mutation_sigma=0.35,
    mutation_rate_end=0.03,
    mutation_sigma_end=0.06,
    mutation_anneal_generations=1000,
    crossover_uniform_prob=0.5,
    episode_seed_stride=91,
    eval_episodes=15,
    eval_episode_seed_stride=9973,
    # --- Common Random Numbers (jun 2026): banco de semillas fijo y compartido ---
    # El fitness por individuo era ruidísimo (sd ~14 por partida) porque cada individuo
    # peleaba contra enemigos distintos: la selección elegía suerte, no habilidad, y la
    # media se estancaba (~21) con el best rebotando. Ahora todos los individuos de una
    # generación enfrentan el MISMO banco de 15 semillas (comparación justa) y el banco
    # rota cada 25 gens (no se sobreajusta a patrones fijos). Una sola variable cambiada.
    eval_common_seeds=True,
    eval_seed_rotation_gens=25,
    max_episode_seconds=45.0,
    fixed_dt=1.0 / 128.0,
    # --- Fitness shaping: penalizar proximidad a bordes y a enemigos ---
    # k_wall SUBIDO 0.4→1.0 y wall_danger_dist 40→50 (jun 2026): el castigo viejo no bastaba
    # y la red se acorralaba arriba (nunca usaba DOWN). Con bordes más caros, bajar para
    # re-centrarse MEJORA el fitness → la acción DOWN deja de ser un "valle" y se revive.
    # NO penalizamos quedarse quieto (es buen comportamiento). Solo proximidad a paredes.
    wall_danger_dist=50.0,
    k_wall=1.0,
    enemy_danger_dist=70.0,
    k_enemy=0.3,
    # --- Premio por HOLGURA (jun 2026): reemplaza el castigo de enemigo cercano ---
    # El castigo viejo (k_enemy) se saturaba a 0 más allá de 70u, así que a la red le
    # bastaba esquivar "lo mínimo" para salir de la línea de fuego: jugaba raspando y
    # frágil (se estrellaba de frente ante geometrías un pelo peores). Este premio crece
    # con la distancia al enemigo más cercano hasta saturar en safety_target_dist → estar
    # lejos es ACTIVAMENTE mejor, empujando a esquivar con margen / alejarse por seguridad.
    safety_reward_enabled=True,
    safety_target_dist=160.0,
    k_safety=0.6,
    # --- Curriculum learning: dificultad creciente ---
    # Warmup: gen 0-30 sin enemigos (aprender a no chocar paredes), episodios de 15s.
    # Rampa: gen 30-430 sube spawn 8s→1s y episodio 15s→45s (los finales = valores de arriba).
    # Pleno: gen 430+ dificultad completa. Al reanudar desde un checkpoint avanzado
    # (p. ej. gen 1640), arranca directo en pleno porque va indexado por generación.
    curriculum_enabled=True,
    curriculum_warmup_generations=30,
    curriculum_ramp_generations=400,  # solo se usa si curriculum_adaptive=False
    curriculum_spawn_interval_start=8.0,
    curriculum_episode_seconds_start=15.0,
    # Curriculum ADAPTATIVO: la rampa sube solo cuando la población domina el nivel
    # actual (top-50% sobrevive ≥70% del episodio), no por número de generación. Si
    # rinde mal (<40%) retrocede. Evita "subir a 3 enemigos sin dominar 2".
    curriculum_adaptive=True,
    curriculum_advance_threshold=0.70,
    curriculum_regress_threshold=0.40,
    curriculum_advance_step=0.02,
    curriculum_regress_step=0.01,
    # --- Inyección de diversidad: anti-estancamiento tras converger ---
    # Respaldo periódico: desde gen 1000, cada 50 gen reemplaza ~8% (no-elite) por aleatorios.
    diversity_injection_enabled=True,
    diversity_injection_start_gen=1000,
    diversity_injection_interval=50,
    diversity_injection_fraction=0.08,
    # --- Anti-estancamiento por estancamiento REAL ---
    # Si el mejor fitness no mejora en 40 gen, inyecta 15% de aleatorios y sube la
    # mutación a un suelo alto (18% / σ=0.25) para reactivar la búsqueda, sin esperar
    # a la gen 1000. El baseline se reinicia al cambiar de dificultad el curriculum.
    stagnation_enabled=True,
    stagnation_patience=40,
    stagnation_inject_cooldown=25,
    stagnation_inject_fraction=0.15,
    # Hipermutación: al estancarse, los slots de inyección se llenan con copias de los
    # élite mutadas fuerte (rate 0.4 / σ 0.45) en vez de cerebros aleatorios. Los aleatorios
    # eran demasiado tontos para ganar torneos y reproducirse; las copias de élite sí.
    stagnation_hypermutate=True,
    stagnation_hypermut_rate=0.4,
    stagnation_hypermut_sigma=0.45,
    # Boost de mutación global al estancarse: OFF (perturbaba a toda la población y hundía
    # la media). La exploración la hace ahora la hipermutación.
    stagnation_boost_global_mut=False,
    stagnation_mut_rate=0.18,
    stagnation_mut_sigma=0.25,
    # --- Bonus de diversidad de direcciones: anti-colapso (jun 2026) ---
    # La red colapsaba a una política de 3 direcciones (una se extinguía por efecto
    # fundador) y se acorralaba contra la pared de la dirección no usada — primero murió
    # DOWN, tras subir k_wall murió UP (mismo óptimo local, reflejado). Este bonus premia
    # mantener vivas las 4 direcciones MIENTRAS la política se forma (cobertura por piso del
    # histograma de movimientos; idle NO cuenta → no penaliza quedarse quieto) y se anula con
    # el annealing para que la red madura se especialice. Es el "exploration bonus" de RL.
    # floor=0.10: cada dirección debe ser ≥10% de los movimientos para dar bonus pleno; si una
    # se extingue (0%), el bonus cae a 0 → presión fuerte y directa contra la extinción.
    # REFUERZO jun 2026: con la rejilla la red volvió a colapsar (solo izq/der, eje vertical
    # muerto) porque el bonus se annealaba por GENERACIÓN y a la gen ~300 ya era ~1.4, débil
    # frente a un fitness ~20-26 → la evolución lo dejaba sobre la mesa para optimizar el juke
    # lateral. Ahora: pico MÁS ALTO (6.0) y el anneal se ata al CURRICULUM, no a la generación
    # — se mantiene pleno toda la fase fácil/media (donde colapsa) y solo baja hacia un PISO
    # permanente (1.5) cuando el curriculum ya está alto (≥0.85), donde el vertical es
    # obligatorio de todos modos. Así nunca se "apaga" del todo en niveles fáciles.
    diversity_bonus_enabled=True,
    diversity_bonus_start=6.0,
    diversity_bonus_floor=0.10,
    diversity_bonus_anneal_progress=0.85,  # progreso de curriculum donde EMPIEZA a bajar
    diversity_bonus_floor_coef=1.5,        # piso permanente del coeficiente (no baja de aquí)
    diversity_bonus_anneal_generations=500,  # (obsoleto: ya no se usa, anneal por curriculum)
)

# --- RL (REINFORCE): el "cerebro matemático" (jul 2026) ---
# Entrena la MISMA red por descenso del gradiente en vez de mutación+selección. Política
# estocástica (softmax), recompensa +1/frame vivo, retorno descontado γ, baseline por lote,
# entropía para explorar, Adam. Tiene su propio curriculum (arranca fácil). Se entrena y guarda
# en paralelo al evolutivo (archivos latest_rl.brain / latest_evo.brain). Ver src/rl/reinforce.py.
RL: RLConfig = RLConfig(
    # lr 0.01→0.003 y gradient clipping (jul 2026): con REINFORCE puro el aprendizaje era
    # errático y hasta regresaba (18s→13s) — un lote ruidoso daba un paso grande que destruía
    # la política. lr más bajo + recorte de norma del gradiente domestican esa oscilación.
    lr=0.003,
    gamma=0.99,
    batch_episodes=24,
    max_episode_seconds=45.0,
    entropy_coef=0.01,
    max_grad_norm=1.0,
    # Actor-critic (A2C) + anneal de entropía (jul 2026, "última chance"): REINFORCE puro
    # topó a ~½ del evolutivo y su política GREEDY era degenerada (solo servía muestreando).
    # El critic V(s) da una ventaja de baja varianza (mejor que el promedio del lote), y
    # annealar la entropía (0.01→0.0005 con el curriculum) hace que la política argmax cristalice.
    entropy_coef_end=0.0005,
    use_critic=True,
    value_lr=0.01,
    curriculum_enabled=True,
    curriculum_warmup_updates=15,
)
