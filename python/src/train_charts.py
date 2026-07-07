"""
Gráficos de barras para historial evolutivo (mejor y promedio por generación).
Incluye una franja de marcadores que indica el "estado" del curriculum por
generación (subió de nivel, bajó, estancado, inyección de diversidad, etc.).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import pygame

# Color por evento del curriculum (debe coincidir con trainer.curriculum_event_history).
EVENT_COLORS = {
    "warmup": (95, 115, 150),    # gris-azul: fase sin enemigos
    "flat": (70, 95, 120),       # azul apagado: aprendiendo, dificultad estable
    "up": (110, 225, 140),       # verde: subió de nivel (dominó el actual)
    "down": (235, 135, 95),      # naranja-rojo: bajó de nivel (rindió mal)
    "stagnant": (235, 200, 95),  # ámbar: estancado (exploración elevada)
    "inject": (200, 120, 220),   # violeta: inyección aleatoria (respaldo)
    "hyper": (120, 225, 225),    # turquesa: hipermutación de élites
}
# Qué entradas mostrar en la leyenda (las más informativas) y su texto.
_LEGEND = [("up", "subió"), ("down", "bajó"), ("stagnant", "estancado"), ("hyper", "hipermut.")]


def _slice_tail(seq: Sequence, max_visible: int) -> List:
    """Muestra solo las últimas generaciones para que las barras sean legibles."""
    if max_visible <= 0 or len(seq) <= max_visible:
        return list(seq)
    return list(seq[-max_visible:])


def draw_generation_history_charts(
    surface: pygame.Surface,
    outer: pygame.Rect,
    best_history: Sequence[float],
    mean_history: Sequence[float],
    font_title: pygame.font.Font,
    font_small: pygame.font.Font,
    *,
    max_visible: int = 72,
    event_history: Optional[Sequence[str]] = None,
) -> None:
    """
    Dos paneles apilados: arriba = mejor fitness por gen.; abajo = promedio por gen.
    Escala Y independiente en cada panel (el promedio suele ser menor que el máximo).
    Si se pasa `event_history`, el panel superior muestra una franja de estado del
    curriculum alineada con las barras.
    """
    pygame.draw.rect(surface, (24, 28, 38), outer, border_radius=8)
    pygame.draw.rect(surface, (70, 78, 98), outer, width=1, border_radius=8)

    cap = font_title.render("Historial por generación", True, (215, 220, 235))
    surface.blit(cap, (outer.x + 10, outer.y + 6))

    # Leyenda de la franja de estado, a la derecha del título.
    if event_history is not None:
        _draw_legend(surface, outer, font_small)

    best_v = _slice_tail(best_history, max_visible)
    mean_v = _slice_tail(mean_history, max_visible)
    n_pair = min(len(best_v), len(mean_v))
    best_v = best_v[-n_pair:] if n_pair else []
    mean_v = mean_v[-n_pair:] if n_pair else []

    events_v: Optional[List[str]] = None
    if event_history is not None:
        events_v = _slice_tail(event_history, max_visible)
        events_v = events_v[-n_pair:] if n_pair else []

    inner_h = outer.height - 44
    half = inner_h // 2
    pad = 8
    top_rect = pygame.Rect(outer.x + pad, outer.y + 36, outer.width - 2 * pad, half - pad // 2)
    bot_rect = pygame.Rect(
        outer.x + pad,
        outer.y + 36 + half + 2,
        outer.width - 2 * pad,
        half - pad // 2,
    )

    skipped = max(0, len(best_history) - len(best_v))
    if skipped > 0:
        hint = font_small.render(
            f"(últimas {len(best_v)} gen.; {skipped} anteriores recortadas)",
            True,
            (140, 145, 160),
        )
        surface.blit(hint, (outer.right - hint.get_width() - 12, outer.bottom - 16))

    _draw_bar_chart_panel(
        surface, top_rect, best_v, "Mejor individuo (s)", (110, 220, 150),
        font_title, font_small, events=events_v,
    )
    _draw_bar_chart_panel(
        surface, bot_rect, mean_v, "Promedio población (s)", (120, 170, 240),
        font_title, font_small,
    )


def _draw_legend(surface: pygame.Surface, outer: pygame.Rect, font_small: pygame.font.Font) -> None:
    """Cuadritos de color + etiqueta, alineados a la derecha del título superior."""
    chip = 9
    x = outer.right - 12
    items = []
    for key, label in _LEGEND:
        lbl = font_small.render(label, True, (170, 175, 190))
        items.append((EVENT_COLORS[key], lbl))
    total = sum(chip + 4 + lbl.get_width() + 12 for _, lbl in items)
    x = outer.right - 12 - total
    y = outer.y + 7
    for color, lbl in items:
        pygame.draw.rect(surface, color, pygame.Rect(x, y, chip, chip), border_radius=2)
        surface.blit(lbl, (x + chip + 4, y - 2))
        x += chip + 4 + lbl.get_width() + 12


def _draw_bar_chart_panel(
    surface: pygame.Surface,
    plot_outer: pygame.Rect,
    values: List[float],
    title: str,
    bar_color: tuple[int, int, int],
    font_title: pygame.font.Font,
    font_small: pygame.font.Font,
    *,
    events: Optional[List[str]] = None,
) -> None:
    pygame.draw.rect(surface, (30, 34, 44), plot_outer, border_radius=4)
    surface.blit(font_title.render(title, True, (185, 190, 205)), (plot_outer.x + 6, plot_outer.y + 4))

    inner = pygame.Rect(
        plot_outer.x + 6,
        plot_outer.y + 26,
        plot_outer.width - 12,
        plot_outer.height - 34,
    )
    pygame.draw.rect(surface, (18, 20, 28), inner, border_radius=2)

    n = len(values)
    if n == 0:
        t = font_small.render("(esperando primera generación…)", True, (110, 115, 130))
        surface.blit(t, (inner.x + 8, inner.y + inner.height // 2 - 8))
        return

    y_max = max(values)
    if y_max <= 1e-9:
        y_max = 1.0

    # Si hay eventos, reservamos una franja arriba para los marcadores de estado.
    strip_h = 6 if events else 0
    strip_gap = 2 if events else 0

    baseline_y = inner.bottom - 2
    top_y = inner.y + 4 + strip_h + strip_gap
    plot_h = baseline_y - top_y

    slot = inner.width / n
    gap_ratio = 0.15
    for i, v in enumerate(values):
        rel = min(1.0, max(0.0, float(v) / y_max))
        bar_h = int(rel * plot_h)
        bw = max(1, int(slot * (1.0 - gap_ratio)))
        bx = int(inner.x + i * slot + (slot - bw) * 0.5)
        by = baseline_y - bar_h
        pygame.draw.rect(surface, bar_color, pygame.Rect(bx, by, bw, bar_h))

    # Franja de estado del curriculum, una celda por generación alineada con su barra.
    if events:
        strip_y = inner.y + 1
        for i, ev in enumerate(events):
            color = EVENT_COLORS.get(ev, EVENT_COLORS["flat"])
            bw = max(1, int(slot * (1.0 - gap_ratio)))
            bx = int(inner.x + i * slot + (slot - bw) * 0.5)
            pygame.draw.rect(surface, color, pygame.Rect(bx, strip_y, bw, strip_h))

    # Etiqueta máximo y último
    mx_lbl = font_small.render(f"max {y_max:.2f}s", True, (150, 155, 170))
    surface.blit(mx_lbl, (inner.right - mx_lbl.get_width() - 4, inner.y + strip_h + 2))
    last_lbl = font_small.render(f"último {values[-1]:.2f}s", True, (160, 165, 180))
    surface.blit(last_lbl, (inner.x + 4, inner.y + strip_h + 2))
