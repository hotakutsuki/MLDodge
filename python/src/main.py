"""
Punto de entrada: menú Pygame — entrenamiento evolutivo, versus (IA vs humano), práctica.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple
from types import SimpleNamespace

import numpy as np
import pygame

from src.evolution.trainer import EvolutionTrainer
from src.game.engine import (
    ACTION_DOWN,
    ACTION_IDLE,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    NUM_ACTIONS,
    GameConfig,
    GameState,
    encode_observation,
    observation_dim,
    step_game,
)
from src.neural.mlp import MLP
from src.nn_viz import draw_network_panel
from src.render_game import checkbox_contains_click, draw_checkbox, draw_game_state
from src.rl.reinforce import RLTrainer
from src.params import EVOLUTION as EVO_CFG
from src.params import GAME as GAME_CFG
from src.params import RL as RL_CFG
from src.params import NN_HIDDEN, neural_layer_sizes
from src.train_charts import draw_generation_history_charts

INITIAL_W, INITIAL_H = 1750, 980
FPS = 60
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
# Autoguardado del campeón cada N generaciones (además del manual con 'S'), para no
# perder la población si la corrida se corta (apagón, cierre). 0 = desactivado.
AUTOSAVE_EVERY_GENS = 25


def _compute_layout(w: int, h: int) -> SimpleNamespace:
    """Calcula todos los Rects del layout a partir del tamaño actual de la ventana."""
    margin = 36
    gap = 22

    # Modo entrenamiento — layout vertical:
    #   [header + checkbox]  ← fila superior
    #   [arena] [NN panel]   ← columnas al mismo nivel
    #   [stats]              ← debajo de arena, ancho total
    #   [charts]             ← debajo de stats, ancho total
    header_h   = 74   # dos líneas de texto (título + checkbox)
    stats_h    = 90   # ~5 líneas de texto pequeño
    charts_min = 160
    nn_w       = 372
    board_side = min(420, max(150, h - header_h - gap - stats_h - gap - charts_min - gap - margin))
    arena_y    = header_h
    stats_y    = arena_y + board_side + gap
    charts_y   = stats_y + stats_h + gap
    charts_h   = max(charts_min, h - charts_y - margin)
    # Segunda columna de stats empieza donde empieza el panel NN
    stats_col2_x = margin + board_side + gap

    # Modo práctica
    prac_w, prac_h = 420, min(460, h - 160)

    # Modo versus
    vs_gap   = 28
    vs_nn_w  = 368
    vs_aw    = 460
    vs_ah    = min(520, h - 60)
    vs_left  = pygame.Rect(max(margin, (w - vs_aw * 2 - vs_gap - vs_nn_w - vs_gap) // 2), 52, vs_aw, vs_ah)
    vs_right = pygame.Rect(vs_left.right + vs_gap, 52, vs_aw, vs_ah)
    vs_nn    = pygame.Rect(vs_right.right + vs_gap, 52, max(vs_nn_w, w - vs_right.right - vs_gap - margin), vs_ah)

    return SimpleNamespace(
        margin=margin,
        gap=gap,
        # Entrenamiento
        train_arena   = pygame.Rect(margin, arena_y, board_side, board_side),
        train_nn      = pygame.Rect(margin + board_side + gap, arena_y, nn_w, board_side),
        train_charts  = pygame.Rect(margin, charts_y, max(10, w - 2 * margin), charts_h),
        train_stats_y = stats_y,
        train_stats_col2_x = stats_col2_x,
        cb_rect       = pygame.Rect(margin, 44, 22, 22),
        # Práctica
        practice_arena = pygame.Rect(w // 2 - prac_w // 2, 120, prac_w, prac_h),
        # Versus
        versus_left = vs_left,
        versus_right = vs_right,
        versus_nn   = vs_nn,
        # Título
        btn_train    = pygame.Rect(80, 200, 280, 48),
        btn_versus   = pygame.Rect(80, 270, 280, 48),
        btn_practice = pygame.Rect(80, 340, 280, 48),
    )


def _method_latest_path(method: str) -> Path:
    """Archivo latest por método de entrenamiento (cerebros separados: evo vs RL)."""
    return CHECKPOINT_DIR / ("latest_rl.brain" if method == "rl" else "latest_evo.brain")


def _find_latest_brain(method: str = "evo") -> Optional[Path]:
    """El latest del método pedido. Para 'evo' cae al legacy latest.brain si no hay uno nuevo."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    mp = _method_latest_path(method)
    if mp.is_file():
        return mp
    if method == "evo":
        legacy = CHECKPOINT_DIR / "latest.brain"   # cerebros evolutivos previos al split
        if legacy.is_file():
            return legacy
    return None


def _list_brains() -> List[Tuple[Path, dict]]:
    """Lista todos los gen_*.brain ordenados por mtime desc, con sus metadatos."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        CHECKPOINT_DIR.glob("gen_*.brain"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    result = []
    for p in paths:
        try:
            _, meta = EvolutionTrainer.load_brain_checkpoint(p)
        except Exception:
            meta = {}
        result.append((p, meta))
    return result


def _count_brains() -> int:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return len(list(CHECKPOINT_DIR.glob("*.brain")))


def _clear_brains() -> int:
    """Borra todos los checkpoints (*.brain y *.meta.json) de checkpoints/.
    Devuelve cuántos archivos .brain se eliminaron. Conserva .gitkeep."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in CHECKPOINT_DIR.glob("*.brain"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    for p in CHECKPOINT_DIR.glob("*.meta.json"):
        try:
            p.unlink()
        except OSError:
            pass
    return n


def _load_ckpt(path: Path) -> Tuple[Optional[MLP], dict, bool]:
    """Carga un checkpoint y devuelve (brain, meta, compatible)."""
    try:
        brain, meta = EvolutionTrainer.load_brain_checkpoint(path)
        compatible = list(meta.get("layer_sizes", [])) == neural_layer_sizes(GAME_CFG)
        return brain, meta, compatible
    except Exception:
        return None, {}, False


def run_demo_episode_step(
    state: GameState, brain: MLP, dt: float
) -> Tuple[np.ndarray, np.ndarray, object]:
    obs = encode_observation(state)
    action, logits, trace = brain.decide_action(obs, return_trace=True)
    step_game(state, action, dt)
    return obs, logits, trace


def _draw_button(
    screen: pygame.Surface,
    r: pygame.Rect,
    label: str,
    font: pygame.font.Font,
    mx: int,
    my: int,
    *,
    active: bool = True,
    highlight: bool = False,
) -> None:
    if active:
        col_bg  = (50, 56, 72) if r.collidepoint(mx, my) else (38, 42, 54)
        col_brd = (120, 130, 155)
        col_txt = (230, 233, 245)
    else:
        col_bg  = (30, 33, 42)
        col_brd = (65, 70, 85)
        col_txt = (100, 105, 120)
    if highlight:  # opción seleccionada (p. ej. método de entrenamiento activo)
        col_bg  = (46, 74, 62)
        col_brd = (110, 210, 150)
        col_txt = (215, 245, 225)
    pygame.draw.rect(screen, col_bg,  r, border_radius=8)
    pygame.draw.rect(screen, col_brd, r, width=2 if highlight else 1, border_radius=8)
    screen.blit(font.render(label, True, col_txt), (r.x + 16, r.y + 13))


def main() -> None:
    pygame.init()
    pygame.display.set_caption("NeuronalNetworkEvolution — evasión")
    screen = pygame.display.set_mode((INITIAL_W, INITIAL_H), pygame.RESIZABLE)
    clock  = pygame.time.Clock()
    font       = pygame.font.SysFont("consolas,courier,monospace", 18)
    font_large = pygame.font.SysFont("consolas,courier,monospace", 22)
    font_small = pygame.font.SysFont("consolas,courier,monospace", 14)

    mode = "title"

    # Estado entrenamiento
    trainer = None                 # EvolutionTrainer o RLTrainer
    train_method = "evo"           # "evo" (evolutivo) | "rl" (gradiente/REINFORCE)
    show_training_view = True
    evo_thread: Optional[threading.Thread] = None
    evo_result: Optional[Tuple[float, float, MLP]] = None
    evo_error: Optional[BaseException] = None
    demo_state: Optional[GameState] = None
    demo_brain: Optional[MLP] = None
    demo_accum   = 0.0
    train_phase  = "menu"
    train_thread_out: list = []
    last_gen_stats: Optional[Tuple[float, float]] = None
    weight_layer_tab = 0
    # Identificador del run actual; nombra el CSV de historial para comparar entrenamientos.
    run_id: Optional[str] = None

    # Checkpoint para reanudar entrenamiento
    ckpt_brain: Optional[MLP] = None
    ckpt_meta: dict = {}
    ckpt_compatible: bool = False

    # Lista de cerebros disponibles (pantalla pick_brain)
    pick_brain_list: List[Tuple[Path, dict]] = []

    # Feedback de guardado: frames restantes para mostrar el mensaje
    ckpt_msg_timer: int = 0
    ckpt_msg_name: str = ""

    # Práctica / versus
    human_state: Optional[GameState] = None
    ai_state: Optional[GameState] = None
    versus_brain: Optional[MLP] = None
    versus_loaded_label: str = ""
    versus_ai_accum = 0.0
    last_logits = np.zeros(NUM_ACTIONS,          dtype=np.float32)
    last_obs    = np.zeros(observation_dim(GAME_CFG), dtype=np.float32)
    last_trace  = None

    # Botones de choose_start (posiciones fijas, no dependen del layout)
    CS_BTN_EVO    = pygame.Rect(80, 168, 165, 44)   # toggle de método: evolutivo
    CS_BTN_RL     = pygame.Rect(255, 168, 165, 44)  # toggle de método: RL
    CS_BTN_NUEVO  = pygame.Rect(80, 240, 340, 48)
    CS_BTN_CONT   = pygame.Rect(80, 308, 340, 48)
    CS_BTN_ELEGIR = pygame.Rect(80, 414, 340, 48)
    CS_BTN_LIMPIAR = pygame.Rect(80, 482, 340, 48)
    clear_confirm = False  # botón "limpiar": requiere segundo clic para confirmar

    running = True
    while running:
        dt_frame = clock.tick(FPS) / 1000.0
        w, h = screen.get_size()
        lay  = _compute_layout(w, h)
        mx, my = pygame.mouse.get_pos()

        # ------------------------------------------------------------------ #
        # Eventos                                                              #
        # ------------------------------------------------------------------ #
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:

                if mode == "title":
                    if lay.btn_train.collidepoint(mx, my):
                        mode = "train"
                        evo_thread = None
                        evo_result = None
                        evo_error  = None
                        demo_state = None
                        demo_brain = None
                        trainer    = None
                        last_gen_stats = None
                        train_thread_out.clear()
                        ckpt_brain = None
                        ckpt_meta  = {}
                        ckpt_compatible = False
                        # Intentar cargar el latest del método actual para "Continuar"
                        path = _find_latest_brain(train_method)
                        if path:
                            ckpt_brain, ckpt_meta, ckpt_compatible = _load_ckpt(path)
                        # Si no hay ningún checkpoint, saltamos choose_start
                        if ckpt_brain is None and not _list_brains():
                            train_phase = "init"
                        else:
                            train_phase = "choose_start"

                    elif lay.btn_versus.collidepoint(mx, my):
                        mode = "versus"
                        path = _find_latest_brain()
                        versus_ai_accum = 0.0
                        if path:
                            versus_brain, _ = EvolutionTrainer.load_brain_checkpoint(path)
                            versus_loaded_label = path.name
                        else:
                            versus_brain = MLP(neural_layer_sizes(GAME_CFG), rng=np.random.default_rng())
                            versus_loaded_label = "(sin checkpoint — red aleatoria)"
                        seed = int(np.random.randint(0, 10_000_000))
                        ai_state    = GameState.new_game(GAME_CFG, seed=seed)
                        human_state = GameState.new_game(GAME_CFG, seed=seed)
                        weight_layer_tab = 0

                    elif lay.btn_practice.collidepoint(mx, my):
                        mode = "practice"
                        human_state = GameState.new_game(GAME_CFG, seed=np.random.randint(0, 10_000_000))

                elif mode == "train":
                    if train_phase == "choose_start":
                        if CS_BTN_EVO.collidepoint(mx, my) or CS_BTN_RL.collidepoint(mx, my):
                            # Cambiar de método: recargar el checkpoint del método elegido.
                            train_method = "rl" if CS_BTN_RL.collidepoint(mx, my) else "evo"
                            clear_confirm = False
                            ckpt_brain, ckpt_meta, ckpt_compatible = None, {}, False
                            p = _find_latest_brain(train_method)
                            if p:
                                ckpt_brain, ckpt_meta, ckpt_compatible = _load_ckpt(p)
                        elif CS_BTN_NUEVO.collidepoint(mx, my):
                            clear_confirm = False
                            ckpt_brain = None
                            train_phase = "init"
                        elif CS_BTN_CONT.collidepoint(mx, my) and ckpt_compatible:
                            clear_confirm = False
                            train_phase = "init"   # ckpt_brain ya está cargado
                        elif CS_BTN_ELEGIR.collidepoint(mx, my):
                            clear_confirm = False
                            pick_brain_list = _list_brains()
                            train_phase = "pick_brain"
                        elif CS_BTN_LIMPIAR.collidepoint(mx, my):
                            if clear_confirm:
                                n = _clear_brains()
                                clear_confirm = False
                                # Olvidar el checkpoint cargado para "Continuar"
                                ckpt_brain = None
                                ckpt_meta = {}
                                ckpt_compatible = False
                                ckpt_msg_name  = f"{n} cerebros borrados"
                                ckpt_msg_timer = 240
                            else:
                                clear_confirm = True
                        else:
                            clear_confirm = False

                    elif train_phase == "pick_brain":
                        entry_h = 46
                        list_y0 = 150
                        for i, (p, meta) in enumerate(pick_brain_list):
                            r = pygame.Rect(80, list_y0 + i * entry_h, w - 160, entry_h - 4)
                            if r.collidepoint(mx, my):
                                b, m, compat = _load_ckpt(p)
                                if b is not None and compat:
                                    ckpt_brain = b
                                    ckpt_meta  = m
                                    ckpt_compatible = True
                                    train_phase = "init"
                                break

                    else:
                        hit = checkbox_contains_click(lay.cb_rect, 420)
                        if hit.collidepoint(mx, my):
                            show_training_view = not show_training_view

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if mode == "title":
                        running = False
                    elif mode == "train" and train_phase == "pick_brain":
                        train_phase = "choose_start"   # volver a elegir, no al menú
                    else:
                        mode = "title"
                        train_phase = "menu"
                        evo_thread = None
                        demo_state = None
                        clear_confirm = False

                if mode == "versus" and event.key == pygame.K_TAB:
                    if versus_brain:
                        n = len(versus_brain.weight_matrices)
                        weight_layer_tab = (weight_layer_tab + 1) % max(1, n)

                if mode == "train" and event.key == pygame.K_TAB:
                    if demo_brain:
                        n = len(demo_brain.weight_matrices)
                        weight_layer_tab = (weight_layer_tab + 1) % max(1, n)

                if mode == "train" and event.key == pygame.K_s:
                    if trainer and (trainer.last_champion or trainer.best_brain):
                        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                        gen_name = f"gen_{train_method}_{trainer.generation:05d}.brain"
                        # Guarda el CAMPEÓN de la última generación (el que se ve jugando),
                        # no best_brain (que puede ser una marca vieja de un nivel más fácil).
                        champ = trainer.last_champion or trainer.best_brain
                        trainer.save_checkpoint(_method_latest_path(train_method), brain=champ)
                        trainer.save_checkpoint(CHECKPOINT_DIR / gen_name, brain=champ)
                        if run_id:
                            trainer.save_history_csv(CHECKPOINT_DIR / "history" / f"run_{run_id}.csv")
                        ckpt_msg_name  = gen_name
                        ckpt_msg_timer = 180  # ~3 segundos a 60 fps

        # ------------------------------------------------------------------ #
        # Render                                                               #
        # ------------------------------------------------------------------ #
        screen.fill((18, 20, 28))

        # --- Título ---
        if mode == "title":
            screen.blit(font_large.render("NeuronalNetworkEvolution", True, (240, 242, 250)), (80, 80))
            screen.blit(font.render("Evasión con red neuronal evolucionada (sin PyTorch)", True, (160, 165, 180)), (80, 118))

            for label, r in [
                ("Entrenamiento evolutivo", lay.btn_train),
                ("Versus (IA vs humano)",   lay.btn_versus),
                ("Práctica (solo humano)",  lay.btn_practice),
            ]:
                _draw_button(screen, r, label, font, mx, my)

            screen.blit(
                font_small.render("ESC salir · En entrenamiento: S guarda checkpoint en checkpoints/", True, (100, 110, 130)),
                (80, 420),
            )

        # --- Entrenamiento ---
        elif mode == "train":

            # ── Pantalla: elegir cómo empezar ───────────────────────────────
            if train_phase == "choose_start":
                metodo_txt = "RL (gradiente)" if train_method == "rl" else "evolutivo"
                screen.blit(font_large.render(f"Entrenamiento — {metodo_txt}", True, (240, 242, 250)), (80, 100))

                # Toggle de método (evolutivo vs RL); el resaltado marca el activo.
                screen.blit(font_small.render("Método de entrenamiento:", True, (150, 158, 178)), (80, 146))
                _draw_button(screen, CS_BTN_EVO, "Evolutivo", font, mx, my, active=True,
                             highlight=(train_method == "evo"))
                _draw_button(screen, CS_BTN_RL, "RL (gradiente)", font, mx, my, active=True,
                             highlight=(train_method == "rl"))

                # Botón 1: siempre disponible
                _draw_button(screen, CS_BTN_NUEVO, "Nuevo entrenamiento (desde cero)", font, mx, my)

                # Botón 2: continuar desde latest.brain
                if ckpt_brain is not None:
                    gen = ckpt_meta.get("generation", "?")
                    fit = ckpt_meta.get("best_fitness_ever", 0.0)
                    label_cont = f"Continuar: latest.brain  ·  gen {gen}  ·  {fit:.2f}s"
                    _draw_button(screen, CS_BTN_CONT, label_cont, font, mx, my, active=ckpt_compatible)
                    if not ckpt_compatible:
                        screen.blit(
                            font_small.render("Arquitectura incompatible con params actuales", True, (220, 160, 80)),
                            (CS_BTN_CONT.x, CS_BTN_CONT.bottom + 6),
                        )
                else:
                    _draw_button(screen, CS_BTN_CONT, "Continuar: no hay checkpoint disponible", font, mx, my, active=False)

                # Separador visual y botón 3 en posición fija
                pygame.draw.line(screen, (55, 60, 75), (80, 392), (80 + CS_BTN_ELEGIR.width, 392))
                _draw_button(screen, CS_BTN_ELEGIR, "Elegir cerebro guardado…", font, mx, my)

                # Botón 4: limpiar cerebros (doble clic para confirmar — es irreversible)
                n_saved = _count_brains()
                if clear_confirm:
                    limpiar_label = "⚠ Clic de nuevo para BORRAR todos"
                else:
                    limpiar_label = f"Limpiar cerebros guardados ({n_saved})…"
                _draw_button(screen, CS_BTN_LIMPIAR, limpiar_label, font, mx, my, active=n_saved > 0)
                if clear_confirm:
                    screen.blit(
                        font_small.render("Esto borra TODOS los .brain de checkpoints/ (no se puede deshacer)", True, (235, 150, 90)),
                        (CS_BTN_LIMPIAR.x, CS_BTN_LIMPIAR.bottom + 6),
                    )

                # Feedback tras borrar
                if ckpt_msg_timer > 0:
                    ckpt_msg_timer -= 1
                    screen.blit(
                        font_small.render(f"✓  {ckpt_msg_name}", True, (110, 215, 140)),
                        (80, CS_BTN_LIMPIAR.bottom + 6),
                    )

                screen.blit(font_small.render("ESC  volver al menú", True, (100, 110, 130)), (80, h - 50))

            # ── Pantalla: lista de cerebros ──────────────────────────────────
            elif train_phase == "pick_brain":
                screen.blit(font_large.render("Elegir cerebro", True, (240, 242, 250)), (80, 80))
                screen.blit(font_small.render("Clic para cargar · ESC volver", True, (130, 140, 160)), (80, 114))

                entry_h = 46
                list_y0 = 150
                current_arch = neural_layer_sizes(GAME_CFG)

                if not pick_brain_list:
                    screen.blit(font.render("No hay archivos gen_*.brain en checkpoints/", True, (180, 140, 100)), (80, list_y0))
                else:
                    for i, (p, meta) in enumerate(pick_brain_list):
                        r = pygame.Rect(80, list_y0 + i * entry_h, w - 160, entry_h - 4)
                        compat = list(meta.get("layer_sizes", [])) == current_arch
                        hov    = r.collidepoint(mx, my) and compat
                        col_bg  = (50, 56, 72) if hov else (30, 33, 42) if not compat else (38, 42, 54)
                        col_brd = (80, 88, 110) if not compat else (100, 115, 145)
                        pygame.draw.rect(screen, col_bg,  r, border_radius=6)
                        pygame.draw.rect(screen, col_brd, r, width=1, border_radius=6)

                        gen  = meta.get("generation", "?")
                        fit  = meta.get("best_fitness_ever", None)
                        fit_s = f"{fit:.2f}s" if fit is not None else "—"
                        col_txt = (180, 185, 200) if compat else (90, 95, 110)
                        name_s  = p.name
                        info_s  = f"gen {gen}  ·  mejor fitness {fit_s}"
                        if not compat:
                            info_s += "  ·  arquitectura incompatible"
                        screen.blit(font.render(name_s, True, col_txt), (r.x + 12, r.y + 6))
                        screen.blit(font_small.render(info_s, True, col_txt), (r.x + 220, r.y + 10))

            # ── Entrenamiento activo ─────────────────────────────────────────
            else:
                # Cabecera
                screen.blit(
                    font.render("Entrenamiento · ESC menú · S checkpoint · TAB = capa de pesos en demo", True, (200, 205, 220)),
                    (lay.margin, 14),
                )
                draw_checkbox(screen, lay.cb_rect, show_training_view, "Ver demo tras cada generación", font)

                # Confirmación de checkpoint guardado
                if ckpt_msg_timer > 0:
                    ckpt_msg_timer -= 1
                    msg_surf = font_small.render(f"✓  {ckpt_msg_name}  guardado", True, (110, 215, 140))
                    screen.blit(msg_surf, (w - msg_surf.get_width() - lay.margin, 18))

                # Lógica de fases
                if train_phase == "init":
                    if train_method == "rl":
                        trainer = RLTrainer(GAME_CFG, RL_CFG, neural_layer_sizes(GAME_CFG))
                    else:
                        trainer = EvolutionTrainer(GAME_CFG, EVO_CFG, layer_sizes=neural_layer_sizes(GAME_CFG))
                    run_id = f"{train_method}_{time.strftime('%Y%m%d_%H%M%S')}"
                    if ckpt_brain is not None:
                        trainer.seed_population_from_brain(ckpt_brain)
                        trainer.best_fitness_ever = float(ckpt_meta.get("best_fitness_ever", -1.0))
                        trainer.generation        = int(ckpt_meta.get("generation", 0))
                        # Restaura el progreso del curriculum adaptativo si el checkpoint
                        # lo tiene; si no (checkpoint viejo), asume dificultad plena para
                        # no rebajar a un cerebro ya avanzado.
                        trainer.curriculum_progress = float(
                            ckpt_meta.get("curriculum_progress", 1.0)
                        )
                        # En evolución best_brain es un cerebro aparte; en RL es la propia
                        # política (ya cargada por seed_population_from_brain — no sobreescribir).
                        if train_method != "rl":
                            trainer.best_brain = ckpt_brain.copy()
                        ckpt_brain = None
                    train_phase = "evolve_start"
                    evo_result  = None
                    evo_error   = None

                if train_phase == "evolve_start" and evo_thread is None:
                    evo_error  = None
                    evo_result = None
                    train_thread_out.clear()

                    def run_generation_worker() -> None:
                        try:
                            assert trainer is not None
                            train_thread_out.append(("ok", trainer.step_generation()))
                        except BaseException as exc:
                            train_thread_out.append(("err", exc))

                    worker = threading.Thread(target=run_generation_worker, daemon=True)
                    worker.start()
                    evo_thread  = worker
                    train_phase = "evolving"

                if train_phase == "evolving" and evo_thread is not None:
                    if not evo_thread.is_alive():
                        item = train_thread_out[0] if train_thread_out else None
                        evo_thread = None
                        train_thread_out.clear()
                        if item and item[0] == "ok":
                            evo_result  = item[1]
                            train_phase = "evolve_done"
                        elif item and item[0] == "err":
                            evo_error   = item[1]
                            train_phase = "error"

                if train_phase == "error" and evo_error is not None:
                    screen.blit(
                        font.render(f"Error: {evo_error}", True, (255, 120, 120)),
                        (lay.train_arena.x + 12, lay.train_arena.y + 48),
                    )
                    screen.blit(
                        font_small.render("Pulsa ESC para volver al menú.", True, (180, 180, 200)),
                        (lay.train_arena.x + 12, lay.train_arena.y + 78),
                    )

                if train_phase == "evolve_done" and evo_result is not None and trainer is not None:
                    best_f, mean_f, best_brain_gen = evo_result
                    last_gen_stats = (best_f, mean_f)
                    evo_result = None
                    # Auto-guarda el historial de crecimiento cada generación (CSV pequeño),
                    # así queda registrado para comparar runs aunque no se pulse 'S'.
                    if run_id:
                        trainer.save_history_csv(CHECKPOINT_DIR / "history" / f"run_{run_id}.csv")
                    # Autoguardado periódico del campeón: sobrescribe latest.brain cada N gens
                    # para que un corte (apagón) no nos haga perder la población. La 'S' sigue
                    # disponible para snapshots con nombre (gen_NNNNN.brain).
                    if (AUTOSAVE_EVERY_GENS > 0 and best_brain_gen is not None
                            and trainer.generation % AUTOSAVE_EVERY_GENS == 0):
                        trainer.save_checkpoint(_method_latest_path(train_method), brain=best_brain_gen)
                    if show_training_view:
                        demo_brain  = best_brain_gen
                        # La demo usa la MISMA dificultad que la evaluación de esta gen
                        # (curriculum): sin esto, durante el warmup el cerebro se evaluaría
                        # sin enemigos pero se mostraría con dificultad plena (incoherente).
                        demo_state  = GameState.new_game(trainer.current_game_cfg(), seed=trainer.generation * 7777 + 1234)
                        demo_accum  = 0.0
                        weight_layer_tab = 0
                        train_phase = "demo"
                    else:
                        train_phase = "evolve_start"

                if train_phase == "demo" and demo_state is not None and demo_brain is not None:
                    demo_accum += dt_frame
                    sim_dt = EVO_CFG.fixed_dt
                    # Tope de tiempo del curriculum: sin enemigos el cerebro no muere nunca,
                    # así que cortamos la demo igual que se cortaría el episodio de evaluación.
                    demo_cap = trainer.curriculum_difficulty()[1] if trainer else EVO_CFG.max_episode_seconds
                    while demo_accum >= sim_dt and not demo_state.game_over and demo_state.time_alive < demo_cap:
                        demo_accum -= sim_dt
                        last_obs, last_logits, last_trace = run_demo_episode_step(demo_state, demo_brain, sim_dt)
                    draw_game_state(
                        screen, demo_state, lay.train_arena,
                        "Demo mejor individuo", font, subtitle=f"t={demo_state.time_alive:.1f}s",
                        show_vision=True,
                    )
                    draw_network_panel(
                        screen, lay.train_nn, demo_brain,
                        last_obs, last_trace, last_logits,
                        font_small, font_small, weight_layer_tab,
                    )
                    if demo_state.game_over or demo_state.time_alive >= demo_cap:
                        train_phase = "evolve_start"
                        demo_state  = None

                if train_phase == "evolving":
                    box = lay.train_arena
                    pygame.draw.rect(screen, (26, 30, 42), box, border_radius=8)
                    pygame.draw.rect(screen, (85, 95, 120), box, width=1, border_radius=8)
                    g   = trainer.generation if trainer else 0
                    _lbl = "update · jugando lote…" if isinstance(trainer, RLTrainer) else "gen · evaluando población…"
                    msg = font.render(f"{g} · {_lbl}", True, (185, 195, 220))
                    sub = font_small.render("(gráficos = historial hasta el último paso completado)", True, (130, 138, 155))
                    screen.blit(msg, (box.centerx - msg.get_width() // 2, box.centery - 16))
                    screen.blit(sub, (box.centerx - sub.get_width() // 2, box.centery + 14))

                # Stats y gráficos (solo cuando el trainer existe)
                if trainer is not None:
                    sx = lay.margin
                    y0 = lay.train_stats_y
                    is_rl = isinstance(trainer, RLTrainer)
                    unidad = "Update" if is_rl else "Generación"

                    # Dificultad actual del curriculum (común a ambos métodos)
                    cur_spawn, cur_cap = trainer.curriculum_difficulty()
                    prog_pct = 100.0 * trainer.curriculum_progress
                    if cur_spawn == float("inf"):
                        diff_str = f"Curriculum: WARMUP (sin enemigos · episodio {cur_cap:.0f}s)"
                    elif cur_spawn > GAME_CFG.spawn_interval:
                        diff_str = (f"Curriculum: rampa {prog_pct:.0f}% "
                                    f"(spawn {cur_spawn:.1f}s · episodio {cur_cap:.0f}s)")
                    else:
                        diff_str = f"Curriculum: pleno (spawn {cur_spawn:.1f}s · episodio {cur_cap:.0f}s)"

                    # Columna izquierda: progreso del entrenamiento
                    lines_left = [(f"{unidad}: {trainer.generation}", (170, 180, 195))]
                    if is_rl:
                        lines_left.append((f"Partidas jugadas: {trainer.episodes_total}", (170, 180, 195)))
                    else:
                        lines_left.append((f"Población:  {EVO_CFG.population_size}", (170, 180, 195)))
                    if trainer.best_fitness_history:
                        lines_left.append((f"Mejor supervivencia: {trainer.best_fitness_ever:.2f}s", (140, 220, 170)))
                    if last_gen_stats is not None:
                        bf, mf = last_gen_stats
                        etq = "superv." if is_rl else "mejor"
                        etq2 = "retorno" if is_rl else "media"
                        lines_left.append((f"Último {unidad.lower()} → {etq}: {bf:.2f}s  {etq2}: {mf:.2f}", (190, 200, 220)))
                    lines_left.append((diff_str, (220, 200, 140)))
                    if not is_rl and trainer.gens_since_improvement >= EVO_CFG.stagnation_patience:
                        lines_left.append((
                            f"⚠ Estancado {trainer.gens_since_improvement} gen → exploración elevada",
                            (235, 170, 120),
                        ))
                    for i, (line, col) in enumerate(lines_left):
                        screen.blit(font_small.render(line, True, col), (sx, y0 + i * 18))

                    # Columna derecha: resumen de hiperparámetros (según método)
                    ox = lay.train_stats_col2_x
                    hid_str = "→".join(str(x) for x in NN_HIDDEN)
                    in_d    = observation_dim(GAME_CFG)
                    if is_rl:
                        right_lines = [
                            f"Red: {in_d}→{hid_str}→{NUM_ACTIONS}  (política softmax)",
                            f"RL: REINFORCE  ·  lr {RL_CFG.lr}  ·  γ {RL_CFG.gamma}",
                            f"Lote: {RL_CFG.batch_episodes} partidas/update  ·  entropía {RL_CFG.entropy_coef}",
                            "Recompensa: +1/frame vivo  ·  baseline por lote",
                            "Ajustes: src/params.py (RL)",
                        ]
                    else:
                        mr_eff, ms_eff = trainer.mutation_schedule()
                        right_lines = [
                            f"Red: {in_d}→{hid_str}→{NUM_ACTIONS}",
                            f"Pool: {EVO_CFG.population_size}  elite: {EVO_CFG.elite_count}  torneo k: {EVO_CFG.tournament_k}",
                            f"Eval: {EVO_CFG.eval_episodes} epis/gen  fitness = media",
                            f"Mut: {100 * mr_eff:.1f}% pesos  σ={ms_eff:.3f}  (anneal {EVO_CFG.mutation_anneal_generations} gen)",
                            f"Cruce u.: {EVO_CFG.crossover_uniform_prob}  ·  Ajustes: src/params.py",
                        ]
                    for i, line in enumerate(right_lines):
                        screen.blit(font_small.render(line, True, (155, 162, 180)), (ox, y0 + i * 18))

                    draw_generation_history_charts(
                        screen, lay.train_charts,
                        trainer.best_fitness_history, trainer.mean_fitness_history,
                        font, font_small, max_visible=80,
                        event_history=trainer.curriculum_event_history,
                    )

        # --- Práctica ---
        elif mode == "practice" and human_state is not None:
            screen.blit(font.render("Práctica · flechas · ESC menú", True, (200, 205, 220)), (40, 16))
            keys = pygame.key.get_pressed()
            act  = ACTION_IDLE
            if keys[pygame.K_LEFT]:  act = ACTION_LEFT
            elif keys[pygame.K_RIGHT]: act = ACTION_RIGHT
            elif keys[pygame.K_UP]:    act = ACTION_UP
            elif keys[pygame.K_DOWN]:  act = ACTION_DOWN

            if not human_state.game_over:
                step_game(human_state, act, 1.0 / float(FPS))
            draw_game_state(screen, human_state, lay.practice_arena, "Humano", font, subtitle=f"Tiempo: {human_state.time_alive:.1f}s")
            if human_state.game_over:
                screen.blit(font.render("R reinicia", True, (200, 200, 120)), (w // 2 - 50, h - 80))
            if pygame.key.get_pressed()[pygame.K_r] and human_state.game_over:
                human_state = GameState.new_game(GAME_CFG, seed=np.random.randint(0, 10_000_000))

        # --- Versus ---
        elif mode == "versus" and ai_state is not None and human_state is not None and versus_brain is not None:
            screen.blit(
                font.render("Versus · IA izquierda · humano derecha · TAB capa pesos · ESC menú", True, (200, 205, 220)),
                (40, 10),
            )
            screen.blit(
                font_small.render(
                    f"Cerebro IA: {versus_loaded_label} · sim. IA = mismo dt que entrenamiento ({EVO_CFG.fixed_dt:.5f}s)",
                    True, (150, 175, 210),
                ),
                (40, 36),
            )
            keys  = pygame.key.get_pressed()
            act_h = ACTION_IDLE
            if keys[pygame.K_LEFT]:  act_h = ACTION_LEFT
            elif keys[pygame.K_RIGHT]: act_h = ACTION_RIGHT
            elif keys[pygame.K_UP]:    act_h = ACTION_UP
            elif keys[pygame.K_DOWN]:  act_h = ACTION_DOWN

            # IA y HUMANO avanzan en LOCKSTEP con el MISMO paso fijo (el de entrenamiento),
            # no con el dt variable del framerate. Así ambos juegan exactamente la misma
            # física y dificultad: mismo dt, mismas subdivisiones por frame y los enemigos
            # quedan sincronizados (las semillas ya coinciden al crear ambos estados).
            versus_ai_accum += dt_frame
            sim_dt = EVO_CFG.fixed_dt
            while versus_ai_accum >= sim_dt and not (ai_state.game_over and human_state.game_over):
                versus_ai_accum -= sim_dt
                if not ai_state.game_over:
                    last_obs, last_logits, last_trace = run_demo_episode_step(ai_state, versus_brain, sim_dt)
                if not human_state.game_over:
                    step_game(human_state, act_h, sim_dt)

            draw_game_state(screen, ai_state,    lay.versus_left,  "IA",     font, subtitle=f"{ai_state.time_alive:.1f}s")
            draw_game_state(screen, human_state, lay.versus_right, "Humano", font, subtitle=f"{human_state.time_alive:.1f}s")
            draw_network_panel(screen, lay.versus_nn, versus_brain, last_obs, last_trace, last_logits, font_small, font_small, weight_layer_tab)

            if ai_state.game_over and human_state.game_over:
                screen.blit(
                    font_small.render("Ambos eliminados. Vuelve al menú (ESC) y reentra para nueva partida.", True, (180, 180, 140)),
                    (40, h - 40),
                )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
