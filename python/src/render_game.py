"""
Dibujo del tablero y utilidades de layout (Pygame).
"""

from __future__ import annotations

from typing import Optional, Tuple

import pygame

from src.game.engine import GameConfig, GameState, compute_vision_grid


def arena_inner_rect(outer: pygame.Rect) -> pygame.Rect:
    """Rect interior dejando margen para borde visual."""
    m = 4
    return pygame.Rect(outer.x + m, outer.y + m, outer.width - 2 * m, outer.height - 2 * m)


def game_to_screen(
    gx: float, gy: float, cfg: GameConfig, inner: pygame.Rect
) -> Tuple[int, int]:
    sx = inner.x + int(gx / cfg.arena_w * inner.width)
    sy = inner.y + int(gy / cfg.arena_h * inner.height)
    return sx, sy


def _danger_color(danger: float, closing: float) -> Tuple[int, int, int]:
    """Color de una celda de visión: verde (seguro) → amarillo → rojo (peligro). El
    acercamiento (canal 1) añade azul, para distinguir 'peligro que VIENE' del estático."""
    d = max(0.0, min(1.0, danger))
    if d < 0.5:
        r = int(2 * d * 255); g = 200
    else:
        r = 255; g = int((1.0 - 2 * (d - 0.5)) * 200)
    b = int(max(0.0, min(1.0, closing)) * 200)  # azul = viene hacia ti
    return (r, g, b)


def draw_vision_grid(
    surface: pygame.Surface,
    state: GameState,
    outer_rect: pygame.Rect,
) -> None:
    """Dibuja la "visión" del jugador: la rejilla egocéntrica de peligro que entra a la red,
    superpuesta y semitransparente sobre la arena (para confirmar que ve lo que debe). Usa la
    MISMA función que la observación, así lo dibujado es exactamente lo que ve la red."""
    cfg = state.cfg
    if not getattr(cfg, "obs_use_vision_grid", False):
        return
    inner = arena_inner_rect(outer_rect)
    grid = compute_vision_grid(state)  # (rows, cols, 2)
    rows, cols, _ = grid.shape
    cr, cc = rows // 2, cols // 2
    cell = cfg.vision_cell_size
    px, py = state.player_x, state.player_y
    sx = inner.width / cfg.arena_w
    sy = inner.height / cfg.arena_h
    cw = max(1, int(cell * sx) + 1)
    ch = max(1, int(cell * sy) + 1)

    n_future = grid.shape[2] - 2   # canales de futuro (multi-horizonte)
    overlay = pygame.Surface((inner.width, inner.height), pygame.SRCALPHA)
    for r in range(rows):
        for c in range(cols):
            danger = float(grid[r, c, 0]); closing = float(grid[r, c, 1])
            gx = px + (c - cc - 0.5) * cell  # esquina sup-izq de la celda (coords juego)
            gy = py + (r - cr - 0.5) * cell
            rx = int(gx * sx); ry = int(gy * sy)
            col = _danger_color(danger, closing)
            alpha = int(35 + 150 * max(0.0, min(1.0, danger)))  # más opaco = más peligro
            pygame.draw.rect(overlay, (*col, alpha), pygame.Rect(rx, ry, cw, ch))
            pygame.draw.rect(overlay, (255, 255, 255, 25), pygame.Rect(rx, ry, cw, ch), width=1)
            # PREVISIÓN multi-horizonte: contorno naranja = "estela" por donde pasará el
            # peligro (unión de los horizontes 0.3/0.6/0.9s).
            if n_future > 0:
                fut = max(float(grid[r, c, 2 + k]) for k in range(n_future))
                if fut > 0.15:
                    pygame.draw.rect(overlay, (255, 150, 40, int(230 * min(1.0, fut))),
                                     pygame.Rect(rx + 2, ry + 2, cw - 4, ch - 4), width=2)
    surface.blit(overlay, (inner.x, inner.y))


def draw_game_state(
    surface: pygame.Surface,
    state: GameState,
    outer_rect: pygame.Rect,
    title: str,
    font: pygame.font.Font,
    *,
    subtitle: Optional[str] = None,
    show_vision: bool = False,
) -> None:
    inner = arena_inner_rect(outer_rect)
    pygame.draw.rect(surface, (25, 30, 40), outer_rect, border_radius=6)
    pygame.draw.rect(surface, (120, 130, 160), inner, width=2, border_radius=4)

    # La "visión" del jugador va sobre el fondo pero DEBAJO de enemigos y jugador.
    if show_vision:
        draw_vision_grid(surface, state, outer_rect)

    t = font.render(title, True, (230, 235, 245))
    surface.blit(t, (outer_rect.x + 8, outer_rect.y + 4))
    if subtitle:
        st = font.render(subtitle, True, (160, 170, 190))
        surface.blit(st, (outer_rect.x + 8, outer_rect.y + 22))

    cfg = state.cfg
    pr = max(2, int(cfg.player_radius / cfg.arena_w * inner.width * 0.9))
    er = max(2, int(cfg.enemy_radius / cfg.arena_w * inner.width * 0.9))

    for e in state.enemies:
        sx, sy = game_to_screen(e.x, e.y, cfg, inner)
        pygame.draw.circle(surface, (220, 90, 90), (sx, sy), er)

    psx, psy = game_to_screen(state.player_x, state.player_y, cfg, inner)
    pygame.draw.circle(surface, (80, 200, 255), (psx, psy), pr)

    if state.game_over:
        overlay = pygame.Surface((inner.width, inner.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        surface.blit(overlay, (inner.x, inner.y))
        msg = f"Game over ({state.game_over_reason})"
        go = font.render(msg, True, (255, 200, 200))
        surface.blit(go, (inner.x + inner.width // 2 - go.get_width() // 2, inner.centery))


def draw_checkbox(
    surface: pygame.Surface,
    rect: pygame.Rect,
    checked: bool,
    label: str,
    font: pygame.font.Font,
) -> None:
    pygame.draw.rect(surface, (60, 65, 80), rect, border_radius=3)
    pygame.draw.rect(surface, (180, 190, 210), rect, width=1, border_radius=3)
    if checked:
        pygame.draw.rect(
            surface,
            (100, 220, 140),
            rect.inflate(-6, -6),
            border_radius=2,
        )
    txt = font.render(label, True, (220, 225, 235))
    surface.blit(txt, (rect.right + 10, rect.y + rect.height // 2 - txt.get_height() // 2))


def checkbox_contains_click(rect: pygame.Rect, margin_right_for_label: int) -> pygame.Rect:
    """Área clic más amplia (incluye etiqueta aproximada)."""
    return pygame.Rect(rect.x, rect.y, rect.width + margin_right_for_label, rect.height)
