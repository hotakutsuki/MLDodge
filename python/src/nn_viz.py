"""
Visualización esquemática de la red: entradas, logits, heatmap de una capa W.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pygame

from src.neural.mlp import ForwardTrace, MLP


def _color_for_weight(v: float, scale: float) -> pygame.Color:
    """Rojo negativo, azul positivo; intensidad por magnitud."""
    t = np.clip(abs(v) / (scale + 1e-9), 0.0, 1.0)
    if v >= 0:
        return pygame.Color(int(80 + 175 * t), int(120 + 80 * (1 - t)), 255)
    return pygame.Color(255, int(80 + 120 * (1 - t)), int(80 + 80 * (1 - t)))


def draw_network_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    mlp: MLP,
    obs: np.ndarray,
    trace: Optional[ForwardTrace],
    logits: np.ndarray,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    weight_layer_index: int,
) -> None:
    pygame.draw.rect(surface, (28, 32, 42), rect, border_radius=8)
    pygame.draw.rect(surface, (90, 100, 120), rect, width=1, border_radius=8)
    y = rect.y + 10
    x0 = rect.x + 10

    title = font.render("Red (IA)", True, (230, 235, 245))
    surface.blit(title, (x0, y))
    y += 28

    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    n_in = min(len(obs), 20)
    lab = small_font.render(f"Entradas ({len(obs)} dims, muestra {n_in})", True, (170, 180, 200))
    surface.blit(lab, (x0, y))
    y += 18

    bar_w = rect.width - 20
    bar_h = 6
    for i in range(n_in):
        val = float(obs[i])
        t = (np.clip(val, -2.0, 2.0) + 2.0) / 4.0
        bx = x0 + int(t * (bar_w - 40))
        pygame.draw.rect(surface, (50, 55, 70), pygame.Rect(x0, y, bar_w, bar_h))
        pygame.draw.rect(surface, (120, 200, 255), pygame.Rect(x0, y, bx - x0, bar_h))
        name = small_font.render(f"x{i}", True, (130, 140, 155))
        surface.blit(name, (x0 + bar_w - 36, y - 1))
        y += bar_h + 4

    y += 10
    log = np.asarray(logits, dtype=np.float32).reshape(-1)
    names = ["arriba", "abajo", "izq", "der", "quieto"]
    names = names[: len(log)]
    surface.blit(small_font.render("Salidas (logits)", True, (170, 180, 200)), (x0, y))
    y += 16
    mx = max(0.01, float(np.max(np.abs(log))))
    arg = int(np.argmax(log))
    for i, nm in enumerate(names):
        t = abs(float(log[i])) / mx
        w = int((bar_w - 50) * t)
        col = (180, 255, 140) if arg == i else (100, 110, 130)
        pygame.draw.rect(surface, (45, 50, 62), pygame.Rect(x0, y, bar_w - 50, bar_h + 2))
        pygame.draw.rect(surface, col, pygame.Rect(x0, y, w, bar_h + 2))
        surface.blit(small_font.render(nm, True, (210, 215, 225)), (x0 + bar_w - 48, y))
        y += bar_h + 6

    y += 8
    ws = mlp.weight_matrices
    if weight_layer_index < 0:
        weight_layer_index = 0
    if weight_layer_index >= len(ws):
        weight_layer_index = len(ws) - 1
    W = ws[weight_layer_index]
    h, wmat = W.shape
    surface.blit(
        small_font.render(f"Pesos capa {weight_layer_index} ({h}x{wmat}) [Tab cicla]", True, (170, 180, 200)),
        (x0, y),
    )
    y += 16

    max_pix_r = min(10, (rect.bottom - y - 10) // max(1, h))
    max_pix_c = min(8, (rect.width - 20) // max(1, wmat))
    cell = max(2, min(max_pix_r, max_pix_c))
    scale = float(np.max(np.abs(W))) + 1e-9
    for r in range(h):
        for c in range(wmat):
            clr = _color_for_weight(float(W[r, c]), scale)
            pygame.draw.rect(
                surface,
                clr,
                pygame.Rect(x0 + c * cell, y + r * cell, cell - 1, cell - 1),
            )
