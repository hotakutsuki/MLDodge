"""
Algoritmo evolutivo: población de MLP, evaluación, selección, cruce, mutación, elitismo.
Checkpoints en disco (numpy comprimido + metadatos JSON opcional).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.game.engine import (
    GameConfig,
    GameState,
    NUM_ACTIONS,
    encode_observation,
    observation_dim,
    step_game,
)
from src.neural.mlp import MLP, deserialize_brain, serialize_brain


@dataclass
class EvolutionConfig:
    population_size: int = 40
    elite_count: int = 2
    tournament_k: int = 4
    # Mutación inicial (exploración); con annealing tiende hacia *_end.
    mutation_rate: float = 0.12
    mutation_sigma: float = 0.35
    mutation_rate_end: float = 0.03
    mutation_sigma_end: float = 0.06
    # Generaciones en las que se interpola linealmente hasta *_end; 0 = sin annealing.
    mutation_anneal_generations: int = 500
    crossover_uniform_prob: float = 0.5
    episode_seed_stride: int = 91
    # Varios episodios por individuo; fitness = media de tiempos (menos ruido).
    eval_episodes: int = 1
    eval_episode_seed_stride: int = 9973
    # --- Common Random Numbers (CRN): banco de semillas FIJO y COMPARTIDO ---
    # Antes cada individuo (y cada generación) peleaba contra enemigos DISTINTOS, así que
    # comparábamos cerebros en exámenes diferentes y el fitness por individuo tenía un ruido
    # enorme (sd por partida ~14): la selección elegía suerte, no habilidad, y el best
    # rebotaba sin que la media subiera (meseta ~21). Con CRN todos los individuos de una
    # generación enfrentan EL MISMO banco de semillas → la diferencia de fitness es pura
    # habilidad. El banco se mantiene estable y rota despacio (cada eval_seed_rotation_gens)
    # para no sobreajustar a unos pocos patrones de spawn. Es reducción de varianza clásica
    # en optimización evolutiva con fitness ruidoso.
    eval_common_seeds: bool = True
    eval_seed_rotation_gens: int = 25  # cada cuántas generaciones rota el banco (0/1 = cada gen)
    eval_seed_base: int = 770_000      # offset del banco (separa de las semillas viejas)
    # Límite por episodio: evita evaluaciones eternas; sube el techo cuando mejore la IA.
    max_episode_seconds: float = 45.0
    # Paso de simulación más grande = evolución más rápida (menos fidelidad que 240 Hz).
    fixed_dt: float = 1.0 / 128.0
    # Fitness shaping: penalizar proximidad a bordes y a enemigos.
    # La penalización se acumula como integral de tiempo (mismas unidades que time_alive).
    wall_danger_dist: float = 40.0   # unidades; penalización lineal desde aquí hasta el borde
    k_wall: float = 0.4              # escala de la penalización de bordes
    enemy_danger_dist: float = 70.0  # unidades; penalización lineal desde aquí hasta contacto
    k_enemy: float = 0.3             # escala de la penalización de enemigo cercano (fallback)
    # Premio por HOLGURA (margen de seguridad). Reemplaza al castigo de enemigo cercano
    # (que saturaba a 0 más allá de enemy_danger_dist → la red esquivaba "lo mínimo").
    # Aquí estar lejos del enemigo más cercano es ACTIVAMENTE mejor (no solo "no castigado"),
    # con gradiente hasta safety_target_dist → empuja a alejarse por seguridad, no a raspar.
    safety_reward_enabled: bool = True
    safety_target_dist: float = 160.0  # unidades; el premio satura (pleno) a esta distancia
    k_safety: float = 0.6              # escala del premio por holgura

    # --- Curriculum learning: dificultad creciente con las generaciones ---
    # La idea: empezar fácil (sin enemigos, episodios cortos) para que la población
    # aprenda lo básico (no chocar paredes), y subir la dificultad GRADUALMENTE para
    # no revolver el ranking de selección con cambios bruscos.
    curriculum_enabled: bool = True
    # Fase warmup: 0 enemigos y episodios cortos → aprender a no salir del arena.
    curriculum_warmup_generations: int = 30
    # Tras el warmup, se interpola linealmente la dificultad durante estas generaciones.
    curriculum_ramp_generations: int = 400
    # spawn_interval al empezar la rampa (grande = pocos enemigos); el valor FINAL es
    # game_cfg.spawn_interval (normal). Interpolamos de start → final.
    curriculum_spawn_interval_start: float = 8.0
    # Tope de episodio al empezar la rampa; el valor FINAL es max_episode_seconds.
    curriculum_episode_seconds_start: float = 15.0

    # --- Curriculum ADAPTATIVO: subir dificultad por DOMINIO, no por reloj ---
    # Con curriculum_adaptive, la rampa no avanza por número de generación sino según
    # qué tan bien sobrevive la población al nivel actual. Así nunca sube de dificultad
    # antes de haber dominado la actual (era el problema: "no dominaba 2 y ya había 3").
    # `progreso` ∈ [0,1] reemplaza al factor de interpolación t de la rampa.
    curriculum_adaptive: bool = True
    # Fracción del tope del episodio que el top-50% debe sobrevivir para AVANZAR.
    curriculum_advance_threshold: float = 0.70
    # Si cae por debajo de esto, RETROCEDE un poco (recupera un nivel ya pasado).
    curriculum_regress_threshold: float = 0.40
    # Cuánto sube/baja el progreso por generación cuando domina / fracasa.
    curriculum_advance_step: float = 0.02
    curriculum_regress_step: float = 0.01

    # --- Inyección de diversidad: anti-estancamiento tras converger ---
    # Cuando el annealing termina la población se parece demasiado a sí misma y la
    # evolución se estanca. Cada N generaciones reemplazamos una fracción de la
    # población (nunca los elite) por individuos aleatorios para reactivar exploración.
    diversity_injection_enabled: bool = True
    diversity_injection_start_gen: int = 1000
    diversity_injection_interval: int = 50
    diversity_injection_fraction: float = 0.08

    # --- Anti-estancamiento por ESTANCAMIENTO REAL (no por gen fija) ---
    # Si el mejor fitness no mejora durante `patience` generaciones, disparamos una
    # inyección de diversidad más fuerte y subimos temporalmente la mutación, sin
    # esperar a la gen 1000. El baseline se reinicia cuando cambia la dificultad del
    # curriculum (la bajada de fitness al subir nivel no cuenta como estancamiento).
    stagnation_enabled: bool = True
    stagnation_patience: int = 40
    stagnation_inject_cooldown: int = 25   # gens mínimas entre inyecciones por estancamiento
    stagnation_inject_fraction: float = 0.15
    # --- Hipermutación: explorar desde los MEJORES, no desde cero ---
    # Inyectar cerebros aleatorios no sirve en poblaciones convergidas: son tan tontos
    # que la selección los elimina antes de que se reproduzcan. En su lugar, los slots de
    # inyección se llenan con COPIAS de los élite mutadas fuerte: conservan el "andamiaje"
    # que ya sabe sobrevivir (ganan torneos) y exploran conductas nuevas a su alrededor.
    stagnation_hypermutate: bool = True
    stagnation_hypermut_rate: float = 0.4    # fracción de pesos perturbados (agresivo)
    stagnation_hypermut_sigma: float = 0.45  # magnitud de la perturbación
    # Boost de mutación GLOBAL al estancarse: perturba a TODA la población y suele hundir
    # la media sin romper el estancamiento. Desactivado: la exploración va por hipermutación.
    stagnation_boost_global_mut: bool = False
    stagnation_mut_rate: float = 0.18      # solo si stagnation_boost_global_mut=True
    stagnation_mut_sigma: float = 0.25

    # --- Bonus de diversidad de direcciones (anti-colapso) ---
    # Problema observado: la red converge a una política que usa solo 3 de las 4
    # direcciones (una se EXTINGUE por efecto fundador) y se acorrala contra la pared
    # de la dirección que no usa. Con argmax determinista, esa acción nunca gana y se
    # vuelve invisible para la selección; revivirla es un "valle de fitness".
    # Solución: premiar a los individuos que mantienen las 4 DIRECCIONES vivas en su
    # repertorio, para que ninguna se extinga mientras la política se forma. Idle (acción 4)
    # NO entra en el cálculo → no penalizamos quedarse quieto. Se ANNEALEA a 0 para que la
    # red madura pueda especializarse. (Es la idea del "exploration bonus" de RL.)
    # Métrica: cobertura por PISO. Cada dirección debe usarse al menos `floor` (fracción de
    # los movimientos) para contar como "viva"; bonus = coef · min(1, min_frac/floor).
    # Elegida sobre la entropía porque la entropía es muy plana cerca del uniforme (perder
    # una dirección entera solo baja ~21% el bonus); con el piso, una dirección extinta
    # manda el bonus a 0 → presión fuerte y directa contra la extinción, pero permite
    # asimetría natural (no exige uso uniforme).
    diversity_bonus_enabled: bool = True
    diversity_bonus_start: float = 3.0   # pico del bonus (en "segundos" de fitness) cuando
                                         # las 4 direcciones superan el piso de uso
    diversity_bonus_floor: float = 0.10  # fracción mínima de movimientos por dirección para
                                         # contar como "viva" (permite hasta 10%/dir de asimetría)
    diversity_bonus_anneal_generations: int = 500  # OBSOLETO (anneal viejo por generación)
    # Anneal atado al CURRICULUM en vez de a la generación: el bonus se mantiene pleno toda la
    # fase fácil/media (donde la red colapsa de direcciones) y solo decae hacia un PISO cuando
    # el curriculum ya está alto (donde usar las 4 direcciones es obligatorio igual).
    diversity_bonus_anneal_progress: float = 0.85  # progreso de curriculum donde empieza a bajar
    diversity_bonus_floor_coef: float = 1.5        # piso permanente del coeficiente (no baja de aquí)
    diversity_bonus_coef: float = 0.0    # coef EFECTIVO inyectado por el trainer cada gen
                                         # (no editar a mano; lo fija diversity_coef_schedule)


def _worker_run_individual(args: tuple) -> float:
    """Worker para multiprocessing: reconstruye el cerebro desde el vector de parámetros
    y evalúa todos los episodios. Función de módulo (no método) para que pickle funcione."""
    param_vec, layer_sizes, game_cfg, evo_cfg, seeds = args
    brain = MLP(layer_sizes)
    brain.set_parameter_vector(param_vec)
    total = sum(run_episode(brain, game_cfg, seed, evo_cfg) for seed in seeds)
    return float(total / len(seeds))


def run_episode(
    brain: MLP,
    game_cfg: GameConfig,
    rng_or_seed: Optional[int],
    evo_cfg: EvolutionConfig,
) -> float:
    """
    Ejecuta un episodio hasta game over o max_episode_seconds.
    Devuelve fitness con shaping: time_alive menos penalizaciones acumuladas por
    proximidad a bordes y al enemigo más cercano.
    """
    seed = None if rng_or_seed is None else int(rng_or_seed)
    state = GameState.new_game(game_cfg, seed=seed)
    dt = evo_cfg.fixed_dt
    max_steps = int(evo_cfg.max_episode_seconds / dt) + 2

    wall_acc = 0.0
    enemy_acc = 0.0
    safety_acc = 0.0
    wall_danger = evo_cfg.wall_danger_dist
    enemy_danger = evo_cfg.enemy_danger_dist
    safety_on = evo_cfg.safety_reward_enabled
    safety_target = max(1e-6, evo_cfg.safety_target_dist)
    pr = game_cfg.player_radius
    W = game_cfg.arena_w
    H = game_cfg.arena_h

    # Conteo de movimientos por dirección (0=arriba,1=abajo,2=izq,3=der) para el bonus
    # de entropía. La acción 4 (quieto) NO se cuenta: idle es neutral, no se premia ni castiga.
    move_counts = [0, 0, 0, 0]

    for _ in range(max_steps):
        if state.game_over:
            break
        obs = encode_observation(state)
        action, _, _ = brain.decide_action(obs, return_trace=False)
        if action < 4:
            move_counts[action] += 1
        step_game(state, action, dt)

        if state.game_over:
            break

        px, py = state.player_x, state.player_y

        margin = min(px - pr, W - px - pr, py - pr, H - py - pr)
        if margin < wall_danger:
            wall_acc += (1.0 - margin / wall_danger) * dt

        if state.enemies:
            min_d2 = min((e.x - px) ** 2 + (e.y - py) ** 2 for e in state.enemies)
            min_d = min_d2 ** 0.5
            if safety_on:
                # Premio por HOLGURA: crece con la distancia al enemigo más cercano hasta
                # saturar en safety_target. Alejarse es ACTIVAMENTE mejor, no solo "no
                # castigado" → empuja a salir de la línea de fuego con margen, no a raspar.
                safety_acc += min(1.0, min_d / safety_target) * dt
            elif min_d < enemy_danger:
                enemy_acc += (1.0 - min_d / enemy_danger) * dt

    # Bonus de diversidad: cobertura por piso sobre los movimientos REALES (idle excluido).
    # Cada dirección debe usarse >= floor (fracción de movimientos) para contar como "viva".
    # bonus = coef · min(1, min_frac/floor): si una dirección se extingue (frac 0) → 0.
    div_bonus = 0.0
    coef = evo_cfg.diversity_bonus_coef
    if coef > 0.0:
        total_moves = move_counts[0] + move_counts[1] + move_counts[2] + move_counts[3]
        if total_moves > 0:
            min_frac = min(move_counts) / total_moves
            floor = evo_cfg.diversity_bonus_floor
            coverage = 1.0 if floor <= 0.0 else min(1.0, min_frac / floor)
            div_bonus = coef * coverage

    fitness = (state.time_alive
               - evo_cfg.k_wall * wall_acc
               - evo_cfg.k_enemy * enemy_acc
               + evo_cfg.k_safety * safety_acc
               + div_bonus)
    return float(fitness)


class EvolutionTrainer:
    def __init__(
        self,
        game_cfg: GameConfig,
        evo_cfg: EvolutionConfig,
        layer_sizes: Optional[List[int]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.game_cfg = game_cfg
        self.evo_cfg = evo_cfg
        self.rng = rng if rng is not None else np.random.default_rng()

        in_dim = observation_dim(game_cfg)
        if layer_sizes is None:
            layer_sizes = [in_dim, 24, 16, NUM_ACTIONS]
        else:
            layer_sizes = list(layer_sizes)
            if layer_sizes[0] != in_dim or layer_sizes[-1] != NUM_ACTIONS:
                raise ValueError(
                    f"Capas deben empezar en in_dim={in_dim} y terminar en {NUM_ACTIONS} acciones"
                )
        self.layer_sizes = layer_sizes

        self.population: List[MLP] = [
            MLP(self.layer_sizes, rng=self.rng) for _ in range(evo_cfg.population_size)
        ]
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.mean_fitness_history: List[float] = []
        # Evento del curriculum por generación, paralelo a las historias de fitness:
        # "warmup" | "flat" | "up" (subió nivel) | "down" (bajó) | "stagnant" | "inject".
        self.curriculum_event_history: List[str] = []
        self.best_brain: Optional[MLP] = None
        self.best_fitness_ever: float = -1.0
        self.last_champion: Optional[MLP] = None  # mejor de la última gen evaluada (para guardar/demo)
        # Estado del curriculum adaptativo: progreso de la rampa ∈ [0,1] por dominio.
        self.curriculum_progress: float = 0.0
        # Estado anti-estancamiento: mejor fitness al nivel de dificultad actual y
        # generaciones sin superarlo (se reinicia al cambiar de dificultad).
        self.stagnation_best: float = float("-inf")
        self.gens_since_improvement: int = 0
        self.last_injection_gen: int = -10 ** 9
        self.last_injection_type: str = ""  # "hyper" | "random" (para el evento del gráfico)

    def curriculum_difficulty(self) -> Tuple[float, float]:
        """Devuelve (spawn_interval, max_episode_seconds) efectivos para la generación
        actual según el curriculum. Sin curriculum, devuelve los valores plenos.

        - Warmup: spawn_interval = inf (ningún enemigo aparece) y episodio corto.
        - Rampa: interpola linealmente spawn_interval y tope de episodio hasta lo normal.
        - Después de la rampa: dificultad plena.
        """
        cfg = self.evo_cfg
        spawn_final = self.game_cfg.spawn_interval
        ep_final = cfg.max_episode_seconds
        if not cfg.curriculum_enabled:
            return spawn_final, ep_final

        g = self.generation
        warm = cfg.curriculum_warmup_generations
        if g < warm:
            # Sin enemigos: spawn_timer nunca alcanza inf, así que no aparece ninguno.
            return float("inf"), float(cfg.curriculum_episode_seconds_start)

        if cfg.curriculum_adaptive:
            # Avance por dominio: el progreso lo gobierna _update_curriculum_progress.
            t = min(1.0, max(0.0, self.curriculum_progress))
        else:
            # Avance por reloj (comportamiento antiguo): interpola por generación.
            ramp = max(1, cfg.curriculum_ramp_generations)
            t = min(1.0, float(g - warm) / float(ramp))
        spawn = cfg.curriculum_spawn_interval_start + (spawn_final - cfg.curriculum_spawn_interval_start) * t
        ep = cfg.curriculum_episode_seconds_start + (ep_final - cfg.curriculum_episode_seconds_start) * t
        return float(spawn), float(ep)

    def current_game_cfg(self) -> GameConfig:
        """GameConfig con el spawn_interval ajustado por el curriculum (o el original)."""
        spawn, _ = self.curriculum_difficulty()
        if spawn == self.game_cfg.spawn_interval:
            return self.game_cfg
        return replace(self.game_cfg, spawn_interval=spawn)

    def diversity_coef_schedule(self) -> float:
        """Coeficiente EFECTIVO del bonus de diversidad, atado al CURRICULUM (no a la gen).

        Se mantiene en `diversity_bonus_start` mientras el curriculum esté por debajo de
        `diversity_bonus_anneal_progress` (toda la fase fácil/media, donde la red colapsa de
        direcciones) y luego decae linealmente hasta `diversity_bonus_floor_coef` —un piso
        permanente— a medida que el curriculum se acerca a 1.0. Nunca se apaga del todo: en
        este juego usar las 4 direcciones es siempre necesario, así que conviene mantener una
        presión mínima en vez de dejar que la política madura abandone un eje.
        """
        cfg = self.evo_cfg
        if not cfg.diversity_bonus_enabled:
            return 0.0
        start = cfg.diversity_bonus_start
        floor = cfg.diversity_bonus_floor_coef
        p0 = cfg.diversity_bonus_anneal_progress
        prog = self.curriculum_progress
        if prog <= p0:
            return float(start)
        t = min(1.0, max(0.0, (prog - p0) / max(1e-6, 1.0 - p0)))
        return float(start + (floor - start) * t)

    def current_evo_cfg(self) -> EvolutionConfig:
        """EvolutionConfig con max_episode_seconds y el coef de diversidad del curriculum/gen."""
        _, ep = self.curriculum_difficulty()
        coef = self.diversity_coef_schedule()
        if ep == self.evo_cfg.max_episode_seconds and coef == self.evo_cfg.diversity_bonus_coef:
            return self.evo_cfg
        return replace(self.evo_cfg, max_episode_seconds=ep, diversity_bonus_coef=coef)

    def _update_curriculum_progress(self, fitness: List[float]) -> bool:
        """Ajusta `self.curriculum_progress` según el dominio del nivel actual.

        Mide la supervivencia del top-50% como fracción del tope del episodio: si
        supera el umbral de avance, sube el progreso; si cae por debajo del de
        retroceso, baja. Devuelve True si el progreso (y por tanto la dificultad)
        cambió, para reiniciar la detección de estancamiento.
        """
        cfg = self.evo_cfg
        if not (cfg.curriculum_enabled and cfg.curriculum_adaptive):
            return False
        if self.generation < cfg.curriculum_warmup_generations:
            return False
        _, cap = self.curriculum_difficulty()
        if not np.isfinite(cap) or cap <= 0:
            return False

        arr = np.sort(np.asarray(fitness, dtype=np.float64))[::-1]
        half = max(1, len(arr) // 2)
        survival = float(np.mean(arr[:half])) / cap
        survival = min(1.0, max(0.0, survival))

        prev = self.curriculum_progress
        if survival >= cfg.curriculum_advance_threshold:
            self.curriculum_progress = min(1.0, prev + cfg.curriculum_advance_step)
        elif survival <= cfg.curriculum_regress_threshold:
            self.curriculum_progress = max(0.0, prev - cfg.curriculum_regress_step)
        return self.curriculum_progress != prev

    def _eval_seeds(self, cfg: EvolutionConfig, index: int) -> List[int]:
        """Semillas de los episodios de evaluación de un individuo.

        Con CRN (eval_common_seeds): banco COMPARTIDO por todos los individuos de la gen
        e independiente del índice; rota despacio (cada eval_seed_rotation_gens) para no
        sobreajustar a patrones fijos. Sin CRN: comportamiento viejo (un set por individuo
        y por generación → mucho ruido en la comparación)."""
        n_ep = max(1, int(cfg.eval_episodes))
        if cfg.eval_common_seeds:
            rot = max(1, int(cfg.eval_seed_rotation_gens))
            block = self.generation // rot
            base = cfg.eval_seed_base + block * 1_000_003  # primo grande: bloques bien separados
        else:
            base = self.generation * 10000 + index * cfg.episode_seed_stride
        return [base + e * cfg.eval_episode_seed_stride for e in range(n_ep)]

    def evaluate_one(self, index: int) -> float:
        """Fitness = media de `eval_episodes` partidas (banco CRN compartido por defecto)."""
        brain = self.population[index]
        cfg = self.current_evo_cfg()
        game_cfg = self.current_game_cfg()
        total = 0.0
        seeds = self._eval_seeds(cfg, index)
        for seed in seeds:
            total += run_episode(brain, game_cfg, seed, cfg)
        return float(total / len(seeds))

    def mutation_schedule(self) -> Tuple[float, float]:
        """(mutation_rate, mutation_sigma) efectivos según generación y annealing."""
        cfg = self.evo_cfg
        if cfg.mutation_anneal_generations <= 0:
            rate, sigma = cfg.mutation_rate, cfg.mutation_sigma
        else:
            t = min(1.0, float(self.generation) / float(cfg.mutation_anneal_generations))
            rate = cfg.mutation_rate_end + (cfg.mutation_rate - cfg.mutation_rate_end) * (1.0 - t)
            sigma = cfg.mutation_sigma_end + (cfg.mutation_sigma - cfg.mutation_sigma_end) * (1.0 - t)
        # Estancado: opcionalmente sube la mutación GLOBAL. Por defecto OFF — perturbar a
        # toda la población hunde la media; la exploración la hace la hipermutación de élites.
        if (cfg.stagnation_enabled and cfg.stagnation_boost_global_mut
                and self.gens_since_improvement >= cfg.stagnation_patience):
            rate = max(rate, cfg.stagnation_mut_rate)
            sigma = max(sigma, cfg.stagnation_mut_sigma)
        return float(rate), float(sigma)

    def evaluate_all(self) -> List[float]:
        cfg = self.current_evo_cfg()
        game_cfg = self.current_game_cfg()
        args = []
        for i, brain in enumerate(self.population):
            # CRN: con eval_common_seeds todos comparten el mismo banco (index es ignorado).
            seeds = self._eval_seeds(cfg, i)
            args.append((brain.get_parameter_vector(), self.layer_sizes, game_cfg, cfg, seeds))
        with Pool() as pool:
            return pool.map(_worker_run_individual, args)

    def _tournament_pick(self, fitness: List[float]) -> int:
        k = min(self.evo_cfg.tournament_k, len(fitness))
        idxs = self.rng.choice(len(fitness), size=k, replace=False)
        best = int(idxs[0])
        for j in idxs[1:]:
            j = int(j)
            if fitness[j] > fitness[best]:
                best = j
        return best

    def _crossover(self, a: MLP, b: MLP) -> MLP:
        child = MLP(self.layer_sizes, rng=self.rng)
        va = a.get_parameter_vector()
        vb = b.get_parameter_vector()
        mask = self.rng.random(va.shape) < self.evo_cfg.crossover_uniform_prob
        child_vec = np.where(mask, va, vb).astype(np.float32)
        child.set_parameter_vector(child_vec)
        return child

    def _mutate(self, brain: MLP) -> None:
        mut_rate, mut_sigma = self.mutation_schedule()
        v = brain.get_parameter_vector()
        noise = self.rng.normal(0.0, mut_sigma, size=v.shape).astype(np.float32)
        apply = self.rng.random(v.shape) < mut_rate
        v = v + np.where(apply, noise, 0.0).astype(np.float32)
        brain.set_parameter_vector(v)

    def _hypermutated_copy(self, parent: MLP) -> MLP:
        """Copia de un cerebro competente con mutación agresiva: explora conductas nuevas
        sin tirar el 'andamiaje' que ya sabe sobrevivir."""
        cfg = self.evo_cfg
        child = MLP(self.layer_sizes, rng=self.rng)
        v = parent.get_parameter_vector().copy()
        noise = self.rng.normal(0.0, cfg.stagnation_hypermut_sigma, size=v.shape).astype(np.float32)
        apply = self.rng.random(v.shape) < cfg.stagnation_hypermut_rate
        v = (v + np.where(apply, noise, 0.0)).astype(np.float32)
        child.set_parameter_vector(v)
        return child

    def step_generation(self) -> Tuple[float, float, MLP]:
        """
        Una generación completa: evalúa, crea nueva población, avanza contador.
        Devuelve (mejor fitness, media fitness, mejor cerebro de esta generación).
        """
        fitness = self.evaluate_all()
        fit_arr = np.array(fitness, dtype=np.float64)
        best_idx = int(np.argmax(fit_arr))
        best_fit = float(fit_arr[best_idx])
        mean_fit = float(np.mean(fit_arr))

        best_brain_this_gen = self.population[best_idx].copy()
        self.last_champion = best_brain_this_gen  # campeón de ESTA gen (el que ves en la demo)

        # Curriculum adaptativo: ¿domina el nivel actual? (puede cambiar la dificultad).
        prev_progress = self.curriculum_progress
        difficulty_changed = self._update_curriculum_progress(fitness)
        went_up = self.curriculum_progress > prev_progress

        # Mejor de la historia: el fitness NO es comparable entre niveles del curriculum
        # (más enemigos → menos fitness), así que al cambiar de dificultad reiniciamos el
        # récord. Si no, best_brain se queda CONGELADO en una marca de un nivel fácil y el
        # checkpoint guarda ese cerebro viejo en vez del campeón actual.
        if difficulty_changed or best_fit > self.best_fitness_ever:
            self.best_fitness_ever = best_fit
            self.best_brain = best_brain_this_gen

        # Detección de estancamiento: el baseline se reinicia al cambiar de dificultad
        # (la caída de fitness al subir de nivel no debe contar como estancamiento).
        if difficulty_changed:
            self.stagnation_best = best_fit
            self.gens_since_improvement = 0
        elif best_fit > self.stagnation_best + 1e-6:
            self.stagnation_best = best_fit
            self.gens_since_improvement = 0
        else:
            self.gens_since_improvement += 1

        ranked = np.argsort(-fit_arr)
        elite_n = min(self.evo_cfg.elite_count, len(self.population))
        new_pop: List[MLP] = []

        for i in range(elite_n):
            new_pop.append(self.population[int(ranked[i])].copy())

        while len(new_pop) < self.evo_cfg.population_size:
            p1 = self.population[self._tournament_pick(fitness)]
            p2 = self.population[self._tournament_pick(fitness)]
            child = self._crossover(p1, p2)
            self._mutate(child)
            new_pop.append(child)

        self._maybe_inject_diversity(new_pop, elite_n)

        # Clasifica el estado de esta generación (para el panel y los gráficos).
        # _maybe_inject_diversity marca last_injection_gen con la gen actual si inyectó.
        cfg = self.evo_cfg
        injected = self.last_injection_gen == self.generation
        if cfg.curriculum_enabled and self.generation < cfg.curriculum_warmup_generations:
            event = "warmup"
        elif injected:
            event = "hyper" if self.last_injection_type == "hyper" else "inject"
        elif difficulty_changed and went_up:
            event = "up"
        elif difficulty_changed and not went_up:
            event = "down"
        elif cfg.stagnation_enabled and self.gens_since_improvement >= cfg.stagnation_patience:
            event = "stagnant"
        else:
            event = "flat"
        self.curriculum_event_history.append(event)

        self.population = new_pop[: self.evo_cfg.population_size]
        self.generation += 1
        self.best_fitness_history.append(best_fit)
        self.mean_fitness_history.append(mean_fit)

        return best_fit, mean_fit, best_brain_this_gen

    def _maybe_inject_diversity(self, new_pop: List[MLP], elite_n: int) -> None:
        """Reemplaza una fracción de los individuos no-elite para reactivar la exploración
        cuando la población converge. Por estancamiento usa HIPERMUTACIÓN de los élite
        (copias mutadas fuerte, que sí se reproducen); el respaldo periódico usa aleatorios."""
        cfg = self.evo_cfg
        if not cfg.diversity_injection_enabled:
            return

        frac = cfg.diversity_injection_fraction
        trigger = False
        by_stagnation = False
        # Disparo principal: estancamiento real (sin esperar a una generación fija).
        if (cfg.stagnation_enabled
                and self.gens_since_improvement >= cfg.stagnation_patience
                and (self.generation - self.last_injection_gen) >= cfg.stagnation_inject_cooldown):
            trigger = True
            by_stagnation = True
            frac = cfg.stagnation_inject_fraction
        # Disparo secundario: periódico clásico tras `start_gen` (respaldo).
        elif (cfg.diversity_injection_interval > 0
                and self.generation >= cfg.diversity_injection_start_gen
                and (self.generation % cfg.diversity_injection_interval) == 0):
            trigger = True
        if not trigger:
            return

        n_inject = int(round(len(new_pop) * frac))
        n_inject = min(n_inject, max(0, len(new_pop) - elite_n))
        if n_inject <= 0:
            return
        self.last_injection_gen = self.generation

        # Hipermutación: partir de los élite (frente de new_pop, los mejores de la gen).
        # Se usa para CUALQUIER inyección (estancamiento o respaldo periódico): inyectar
        # cerebros aleatorios no sirve en población convergida. `by_stagnation` queda solo
        # como info; aleatorio solo si se desactiva la hipermutación a mano.
        use_hyper = cfg.stagnation_hypermutate and elite_n > 0
        self.last_injection_type = "hyper" if use_hyper else "random"
        # Reemplaza los últimos (recién creados, jamás los elite del frente de la lista).
        for j in range(n_inject):
            idx = len(new_pop) - 1 - j
            if idx < elite_n:
                break
            if use_hyper:
                parent = new_pop[int(self.rng.integers(0, elite_n))]
                new_pop[idx] = self._hypermutated_copy(parent)
            else:
                new_pop[idx] = MLP(self.layer_sizes, rng=self.rng)

    def seed_population_from_brain(self, brain: MLP) -> None:
        """Siembra la población mutando desde un cerebro dado (para reanudar desde checkpoint).
        El primer individuo es copia exacta; el resto lleva mutación inicial para diversidad."""
        mr = self.evo_cfg.mutation_rate
        ms = self.evo_cfg.mutation_sigma
        base = brain.get_parameter_vector()
        exact = MLP(self.layer_sizes, rng=self.rng)
        exact.set_parameter_vector(base.copy())
        self.population[0] = exact
        for i in range(1, len(self.population)):
            mutant = MLP(self.layer_sizes, rng=self.rng)
            vec = base.copy()
            noise = self.rng.normal(0.0, ms, size=vec.shape).astype(np.float32)
            mask = self.rng.random(vec.shape) < mr
            vec = (vec + np.where(mask, noise, 0.0)).astype(np.float32)
            mutant.set_parameter_vector(vec)
            self.population[i] = mutant

    def save_checkpoint(self, path: Path, extra: Optional[dict] = None,
                        brain: Optional[MLP] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if brain is not None:
            best = brain.copy()              # cerebro explícito (p. ej. el campeón actual)
        elif self.best_brain is not None:
            best = self.best_brain.copy()
        else:
            # Salva el mejor de la población actual
            fitness = self.evaluate_all()
            idx = int(np.argmax(np.array(fitness)))
            best = self.population[idx].copy()

        meta = {
            "generation": self.generation,
            "best_fitness_ever": self.best_fitness_ever,
            "curriculum_progress": self.curriculum_progress,
            "layer_sizes": self.layer_sizes,
            "game": {
                "arena_w": self.game_cfg.arena_w,
                "arena_h": self.game_cfg.arena_h,
                "speed": self.game_cfg.speed,
                "spawn_interval": self.game_cfg.spawn_interval,
            },
        }
        if extra:
            meta["extra"] = extra

        with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with open(path, "wb") as f:
            f.write(serialize_brain(best))

    def save_history_csv(self, path: Path) -> None:
        """Vuelca el historial por generación (mejor, media, evento del curriculum) a
        un CSV. Se reescribe completo en cada llamada (es pequeño) para poder comparar
        el crecimiento entre distintos entrenamientos."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.best_fitness_history)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("generation,best_fitness,mean_fitness,event\n")
            for i in range(n):
                ev = self.curriculum_event_history[i] if i < len(self.curriculum_event_history) else ""
                f.write(f"{i},{self.best_fitness_history[i]:.4f},{self.mean_fitness_history[i]:.4f},{ev}\n")

    @staticmethod
    def load_brain_checkpoint(path: Path) -> Tuple[MLP, dict]:
        path = Path(path)
        with open(path, "rb") as f:
            brain = deserialize_brain(f.read())
        meta_path = path.with_suffix(".meta.json")
        meta = {}
        if meta_path.is_file():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        return brain, meta


def default_checkpoint_path(generation: int) -> str:
    return f"checkpoints/gen_{generation:05d}.brain.npz"
