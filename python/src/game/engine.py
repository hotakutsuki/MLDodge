"""
Motor del juego de evasión: jugador, enemigos, spawns en bordes, colisiones.
Coordenadas en un rectángulo [0, arena_w] x [0, arena_h]; mismo SPEED para todos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Acciones alineadas con argmax de salida de la red:
# 0=arriba, 1=abajo, 2=izq, 3=der, 4=quieto (sin movimiento)
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_IDLE = 4

NUM_ACTIONS = 5


@dataclass
class GameConfig:
    arena_w: float = 420.0
    arena_h: float = 420.0
    player_radius: float = 10.0
    enemy_radius: float = 9.0
    speed: float = 120.0  # unidades por segundo (misma para jugador y enemigos)
    spawn_interval: float = 1.0  # segundos entre spawns (parametrizable)
    # Cuántos enemigos máximos codificamos en el vector de observación
    obs_enemy_slots: int = 8
    # Enemigos se eliminan tras esta vida o si se alejan del nivel (ahorra CPU/memoria).
    enemy_max_lifetime: float = 35.0
    enemy_despawn_margin: float = 72.0
    # --- Features de observación (afectan la dimensión de entrada de la red) ---
    # Velocidad de acercamiento por enemigo: escalar +/- que dice si se acerca (>0) o
    # se aleja (<0). Le ahorra a la red tener que deducir el producto escalar pos·vel.
    obs_include_closing_speed: bool = True
    # Ordenar los slots por AMENAZA (cercanía + acercamiento) en vez de solo por
    # distancia. El más peligroso va siempre al slot 0; estabiliza qué enemigo "mira".
    obs_sort_by_threat: bool = True
    # --- Visión por REJILLA EGOCÉNTRICA (jun 2026, reemplaza la lista de enemigos) ---
    # En vez de una lista top-K ordenada (cuyo orden salta de frame a frame), un "mapa de
    # calor" de peligro centrado en el jugador y alineado con las 4 acciones: cada celda
    # significa SIEMPRE lo mismo ("peligro hacia esta dirección"), es estable y enciende
    # varias celdas a la vez ante múltiples amenazas. Las paredes entran gratis (celdas
    # fuera de la arena = peligro máximo). 2 canales por celda: peligro y acercamiento.
    obs_use_vision_grid: bool = True
    vision_grid_rows: int = 7
    vision_grid_cols: int = 7
    vision_cell_size: float = 60.0    # unidades por celda (7×60 = 420 = ancho de arena)
    vision_influence: float = 70.0    # radio de "brillo" de un enemigo sobre las celdas
    # --- Canal de PREVISIÓN (jul 2026): dónde estarán los enemigos en el futuro ---
    # Como van en línea recta, su futuro es exacto: pos + vel·dt. Añadimos un 3er canal con
    # el mapa de peligro FUTURO (posiciones extrapoladas), centrado en la posición ACTUAL del
    # jugador ("si me quedo quieto, esto se me viene"). Le da anticipación —la ventaja que hoy
    # tiene el humano— sin cambiar la red (solo más input); la red sigue siendo reactiva.
    vision_future_channel: bool = True
    # Multi-horizonte: un canal de peligro futuro por cada dt → "película" del futuro, no una
    # foto. (0.3/0.6/0.9s a 120u/s ≈ 36/72/108u = ~0.6/1.2/1.8 celdas de corrimiento.)
    vision_future_dts: Tuple[float, ...] = (0.5,)


@dataclass
class Enemy:
    x: float
    y: float
    vx: float
    vy: float
    age: float = 0.0  # segundos desde el spawn


@dataclass
class GameState:
    cfg: GameConfig
    player_x: float = 0.0
    player_y: float = 0.0
    enemies: List[Enemy] = field(default_factory=list)
    time_alive: float = 0.0
    spawn_timer: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    game_over: bool = False
    game_over_reason: str = ""

    @classmethod
    def new_game(cls, cfg: GameConfig, seed: Optional[int] = None) -> GameState:
        rng = np.random.default_rng(seed)
        px = cfg.arena_w / 2.0
        py = cfg.arena_h / 2.0
        return cls(
            cfg=cfg,
            player_x=px,
            player_y=py,
            enemies=[],
            time_alive=0.0,
            spawn_timer=0.0,
            rng=rng,
            game_over=False,
            game_over_reason="",
        )


def _clamp_circle_in_arena(
    x: float, y: float, r: float, cfg: GameConfig
) -> Tuple[float, float]:
    x = max(r, min(cfg.arena_w - r, x))
    y = max(r, min(cfg.arena_h - r, y))
    return x, y


def _dist_sq(ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = ax - bx, ay - by
    return dx * dx + dy * dy


def _circle_out_of_arena(x: float, y: float, r: float, cfg: GameConfig) -> bool:
    return x - r < 0 or x + r > cfg.arena_w or y - r < 0 or y + r > cfg.arena_h


def _enemy_outside_despawn_zone(ex: float, ey: float, cfg: GameConfig) -> bool:
    """Centro del enemigo fuera del rectángulo ampliado → ya no afecta al juego."""
    m = cfg.enemy_despawn_margin
    return ex < -m or ex > cfg.arena_w + m or ey < -m or ey > cfg.arena_h + m


def _spawn_enemy(state: GameState) -> Enemy:
    """
    Aparece en un borde, ligeramente fuera del área jugable; rumbo al jugador
    en el instante del spawn (vector fijo normalizado).
    """
    cfg = state.cfg
    side = int(state.rng.integers(0, 4))
    # Posición sobre el borde exterior, con el centro del círculo fuera del rect de juego
    margin = cfg.enemy_radius + 2.0
    if side == 0:  # arriba
        ex = float(state.rng.uniform(margin, cfg.arena_w - margin))
        ey = -margin
    elif side == 1:  # abajo
        ex = float(state.rng.uniform(margin, cfg.arena_w - margin))
        ey = cfg.arena_h + margin
    elif side == 2:  # izquierda
        ex = -margin
        ey = float(state.rng.uniform(margin, cfg.arena_h - margin))
    else:  # derecha
        ex = cfg.arena_w + margin
        ey = float(state.rng.uniform(margin, cfg.arena_h - margin))

    dx = state.player_x - ex
    dy = state.player_y - ey
    dist = np.sqrt(dx * dx + dy * dy) + 1e-6
    vx = (dx / dist) * cfg.speed
    vy = (dy / dist) * cfg.speed
    return Enemy(x=ex, y=ey, vx=vx, vy=vy)


def step_game(state: GameState, action: int, dt: float) -> None:
    """
    Avanza un paso de simulación; muta `state`. `dt` en segundos.
    """
    if state.game_over or dt <= 0:
        return

    cfg = state.cfg

    # Movimiento del jugador (velocidad fija, ortogonal)
    dx = dy = 0.0
    if action == ACTION_UP:
        dy = -cfg.speed
    elif action == ACTION_DOWN:
        dy = cfg.speed
    elif action == ACTION_LEFT:
        dx = -cfg.speed
    elif action == ACTION_RIGHT:
        dx = cfg.speed
    # ACTION_IDLE (y valores no usados): dx, dy siguen en 0

    state.player_x += dx * dt
    state.player_y += dy * dt

    if _circle_out_of_arena(state.player_x, state.player_y, cfg.player_radius, cfg):
        state.game_over = True
        state.game_over_reason = "boundary"
        return

    # Enemigos en línea recta
    p_r = cfg.player_radius
    e_r = cfg.enemy_radius
    touch_r = p_r + e_r
    touch_sq = touch_r * touch_r

    survivors: List[Enemy] = []
    for e in state.enemies:
        e.age += dt
        e.x += e.vx * dt
        e.y += e.vy * dt
        if _dist_sq(e.x, e.y, state.player_x, state.player_y) <= touch_sq:
            state.game_over = True
            state.game_over_reason = "enemy"
            return
        if e.age >= cfg.enemy_max_lifetime or _enemy_outside_despawn_zone(e.x, e.y, cfg):
            continue
        survivors.append(e)
    state.enemies = survivors

    state.time_alive += dt
    state.spawn_timer += dt
    while state.spawn_timer >= cfg.spawn_interval:
        state.spawn_timer -= cfg.spawn_interval
        state.enemies.append(_spawn_enemy(state))


# Dimensiones base por enemigo: (dx, dy, vx, vy, dist, máscara). Si la config activa
# la velocidad de acercamiento se añade un escalar más (ver enemy_slot_dims).
# - La dirección (vx, vy): la red sabe hacia dónde va el enemigo, no solo dónde está.
# - El acercamiento (closing): le damos masticado el "se acerca o se aleja" (producto
#   escalar pos·vel), que una red con ReLU no calcula fácil por su cuenta.
_ENEMY_SLOT_BASE_DIMS = 6


def enemy_slot_dims(cfg: GameConfig) -> int:
    """Dimensiones por enemigo en la observación según las features activadas."""
    return _ENEMY_SLOT_BASE_DIMS + (1 if cfg.obs_include_closing_speed else 0)


# Canales por celda: (peligro, acercamiento) + un canal de peligro futuro por horizonte.
def vision_channels(cfg: GameConfig) -> int:
    return 2 + (len(cfg.vision_future_dts) if cfg.vision_future_channel else 0)


def vision_grid_dims(cfg: GameConfig) -> int:
    """Nº de números que aporta la rejilla de visión a la observación."""
    return cfg.vision_grid_rows * cfg.vision_grid_cols * vision_channels(cfg)


def compute_vision_grid(state: GameState) -> np.ndarray:
    """
    "Mapa de calor" de peligro EGOCÉNTRICO: una rejilla rows×cols centrada en el jugador
    y alineada con los ejes (fila 0 = arriba, col 0 = izquierda), con 2 canales por celda:
      canal 0 (peligro):     cercanía del enemigo más próximo a esa celda (0..1); las celdas
                             FUERA de la arena valen 1.0 (las paredes son peligro máximo).
      canal 1 (acercamiento): de ese enemigo, su componente de velocidad HACIA el jugador
                             (0 = se aleja / ninguno, 1 = viene de lleno).
      canal 2 (peligro futuro, si vision_future_channel): cercanía a la posición EXTRAPOLADA
                             del enemigo a t+vision_future_dt (línea recta) → anticipación.
    Devuelve un array (rows, cols, C) con C=2 o 3. Esta MISMA función la usan la red (aplanada
    en la observación) y el dibujo de la demo, así lo que se ve es exactamente lo que entra.
    """
    cfg = state.cfg
    rows, cols = cfg.vision_grid_rows, cfg.vision_grid_cols
    cell = cfg.vision_cell_size
    cr, cc = rows // 2, cols // 2
    px, py = state.player_x, state.player_y

    # Centros de celda en coordenadas del juego (vectorizado).
    cxs = px + (np.arange(cols) - cc) * cell          # (cols,)
    cys = py + (np.arange(rows) - cr) * cell          # (rows,)
    GX, GY = np.meshgrid(cxs, cys)                     # (rows, cols) cada uno

    danger = np.zeros((rows, cols), dtype=np.float32)
    closing = np.zeros((rows, cols), dtype=np.float32)
    want_future = cfg.vision_future_channel
    fdts = cfg.vision_future_dts if want_future else ()
    futures = [np.zeros((rows, cols), dtype=np.float32) for _ in fdts]  # uno por horizonte

    influence = max(1e-6, cfg.vision_influence)
    inv_speed = 1.0 / cfg.speed if cfg.speed > 0 else 0.0
    for e in state.enemies:
        d = np.sqrt((GX - e.x) ** 2 + (GY - e.y) ** 2)
        prox = np.clip(1.0 - d / influence, 0.0, 1.0).astype(np.float32)
        # Acercamiento de este enemigo hacia el jugador (escalar, igual para toda la celda).
        ddx, ddy = e.x - px, e.y - py
        dist = float(np.sqrt(ddx * ddx + ddy * ddy))
        cl = 0.0 if dist <= 1e-6 else max(0.0, -(e.vx * ddx + e.vy * ddy) / dist * inv_speed)
        upd = prox > danger
        danger = np.where(upd, prox, danger)
        closing = np.where(upd, np.float32(cl), closing)
        # Peligro futuro por horizonte: posición extrapolada en recta pos + vel·dt.
        for k, fdt in enumerate(fdts):
            fx, fy = e.x + e.vx * fdt, e.y + e.vy * fdt
            df = np.sqrt((GX - fx) ** 2 + (GY - fy) ** 2)
            fprox = np.clip(1.0 - df / influence, 0.0, 1.0).astype(np.float32)
            futures[k] = np.maximum(futures[k], fprox)

    # Paredes: celdas cuyo centro cae fuera de la arena = peligro máximo, sin acercamiento.
    outside = (GX < 0) | (GX > cfg.arena_w) | (GY < 0) | (GY > cfg.arena_h)
    danger = np.where(outside, np.float32(1.0), danger)
    closing = np.where(outside, np.float32(0.0), closing)

    return np.stack([danger, closing, *futures], axis=-1).astype(np.float32)


def encode_observation(state: GameState) -> np.ndarray:
    """
    Vector fijo: posición normalizada, márgenes a paredes, top-K enemigos.
    Cada enemigo lleva su dirección (vx, vy) y, si está activado, su velocidad de
    ACERCAMIENTO (closing): >0 se acerca, <0 se aleja. Los slots se ordenan por AMENAZA
    (cercanía + acercamiento) si la config lo pide, así el más peligroso va al slot 0.
    """
    cfg = state.cfg
    w, h = cfg.arena_w, cfg.arena_h
    px, py = state.player_x, state.player_y
    # Centro del rectángulo = referencia [-1,1] aproximadamente
    cx, cy = w / 2.0, h / 2.0
    half_w = max(w / 2.0, 1e-6)
    half_h = max(h / 2.0, 1e-6)
    nx = (px - cx) / half_w
    ny = (py - cy) / half_h

    pr = cfg.player_radius
    margin_left = (px - pr) / w
    margin_right = (w - px - pr) / w
    margin_top = (py - pr) / h
    margin_bottom = (h - py - pr) / h

    base = np.array(
        [nx, ny, margin_left, margin_right, margin_top, margin_bottom],
        dtype=np.float32,
    )

    # Modo REJILLA EGOCÉNTRICA: base (paredes) + mapa de calor de peligro aplanado.
    if cfg.obs_use_vision_grid:
        return np.concatenate([base, compute_vision_grid(state).ravel()])

    k = cfg.obs_enemy_slots
    diag = np.sqrt(w * w + h * h)
    inv_speed = 1.0 / cfg.speed if cfg.speed > 0 else 0.0
    include_closing = cfg.obs_include_closing_speed
    slot_dims = enemy_slot_dims(cfg)

    if not state.enemies:
        enemy_block = np.zeros(k * slot_dims, dtype=np.float32)
    else:
        # Features por enemigo: (amenaza, dist, edx, edy, evx, evy, d_norm, closing).
        feats = []
        for e in state.enemies:
            ddx = e.x - px
            ddy = e.y - py
            dist = float(np.sqrt(ddx * ddx + ddy * ddy))
            d = dist / diag
            if dist > 1e-6:
                inv_d = 1.0 / dist
                # Componente de la velocidad del enemigo hacia el jugador; >0 = se acerca.
                closing = -(e.vx * ddx + e.vy * ddy) * inv_d * inv_speed
            else:
                closing = 0.0
            # Amenaza: estar cerca (1 - d) y acercándose (closing>0) suman.
            threat = max(0.0, 1.0 - d) + max(0.0, closing)
            feats.append((threat, dist, ddx / half_w, ddy / half_h,
                          e.vx * inv_speed, e.vy * inv_speed, d, closing))

        if cfg.obs_sort_by_threat:
            feats.sort(key=lambda f: f[0], reverse=True)  # más amenazante primero
        else:
            feats.sort(key=lambda f: f[1])                # más cercano primero

        slots = []
        for i in range(k):
            if i < len(feats):
                _, _, edx, edy, evx, evy, d, closing = feats[i]
                if include_closing:
                    slots.extend((edx, edy, evx, evy, d, closing, 1.0))
                else:
                    slots.extend((edx, edy, evx, evy, d, 1.0))
            elif include_closing:
                slots.extend((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            else:
                slots.extend((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        enemy_block = np.array(slots, dtype=np.float32)

    return np.concatenate([base, enemy_block])


def observation_dim(cfg: GameConfig) -> int:
    if cfg.obs_use_vision_grid:
        return 6 + vision_grid_dims(cfg)
    return 6 + cfg.obs_enemy_slots * enemy_slot_dims(cfg)
