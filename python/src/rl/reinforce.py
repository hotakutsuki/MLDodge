"""
Entrenamiento por REINFORCE (policy gradient) — el "cerebro matemático", en paralelo al
evolutivo. Usa el MISMO MLP y el MISMO juego, pero en vez de mutación+selección aprende por
descenso del gradiente:

  1) La política es ESTOCÁSTICA: la red da probabilidades (softmax) y se samplea la acción.
  2) Se juegan episodios y se anota (estado, acción, recompensa). Recompensa = +1 por frame vivo.
  3) Retorno-a-futuro con DESCUENTO γ: cada acción se juzga por lo que pasó DESPUÉS (lo cercano
     a la muerte pesa más). Se resta un BASELINE (el promedio del lote) → "ventaja": ¿esta acción
     fue mejor o peor que lo típico?
  4) Gradiente `∇J = ventaja · ∇log π(a|s)` (backprop en el MLP) y paso de Adam.

Sin frameworks: todo NumPy, backprop en src/neural/mlp.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from dataclasses import replace
from pathlib import Path
import json

from src.game.engine import (GameConfig, GameState, NUM_ACTIONS,
                             encode_observation, step_game)
from src.neural.mlp import MLP, serialize_brain


@dataclass
class RLConfig:
    lr: float = 0.01                 # tasa de aprendizaje (paso de Adam)
    gamma: float = 0.99              # descuento: cuánto pesa el futuro lejano
    batch_episodes: int = 24         # partidas por actualización (lote)
    max_episode_seconds: float = 45.0
    fixed_dt: float = 1.0 / 128.0
    reward_per_step: float = 1.0     # +1 por cada frame sobrevivido (retorno ≈ tiempo vivo)
    entropy_coef: float = 0.01       # premia incertidumbre → explora, evita colapsar temprano
    entropy_coef_end: float = 0.0005 # la entropía se annealea con el curriculum para que la
                                     # política GREEDY (argmax) cristalice y sirva desplegada
    max_grad_norm: float = 1.0       # recorte de norma del gradiente (estabiliza; 0 = sin recorte)
    # --- Critic (actor-critic / A2C): función de valor para ventaja de baja varianza ---
    use_critic: bool = True          # V(s) aprendida como baseline (en vez del promedio del lote)
    value_lr: float = 0.01           # lr del critic (regresión, tolera más que la política)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    # --- Curriculum propio del RL (arrancar fácil e ir subiendo, como el evolutivo) ---
    curriculum_enabled: bool = True
    curriculum_warmup_updates: int = 15      # updates sin enemigos (aprender a no chocar paredes)
    curriculum_spawn_interval_start: float = 8.0
    curriculum_episode_seconds_start: float = 12.0
    curriculum_advance_threshold: float = 0.70   # sube si la supervivencia media ≥ 70% del cap
    curriculum_regress_threshold: float = 0.40   # baja si ≤ 40%
    curriculum_advance_step: float = 0.03
    curriculum_regress_step: float = 0.02


class RLTrainer:
    """Entrena UNA red por policy gradient. Análogo a EvolutionTrainer pero por gradiente."""

    def __init__(self, game_cfg: GameConfig, rl_cfg: RLConfig,
                 layer_sizes: List[int], rng: Optional[np.random.Generator] = None):
        self.game_cfg = game_cfg
        self.cfg = rl_cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        self.policy = MLP(layer_sizes, rng=self.rng)
        n = self.policy.num_parameters()
        self._m = np.zeros(n, dtype=np.float32)   # Adam: momento 1
        self._v = np.zeros(n, dtype=np.float32)   # Adam: momento 2
        self._t = 0
        # Critic: misma arquitectura pero salida 1 (valor del estado). Solo para entrenar
        # (no se guarda ni se despliega). Su propio optimizador Adam.
        self.value = None
        if rl_cfg.use_critic:
            value_sizes = list(layer_sizes[:-1]) + [1]
            self.value = MLP(value_sizes, rng=self.rng)
            nv = self.value.num_parameters()
            self._vm = np.zeros(nv, dtype=np.float32)
            self._vv = np.zeros(nv, dtype=np.float32)
            self._vt = 0
        # Métricas / historial (paralelo al del evolutivo)
        self.updates = 0
        self.episodes_total = 0          # partidas jugadas en total (moneda común de comparación)
        self.return_history: List[float] = []
        self.survival_history: List[float] = []
        self.last_champion = self.policy  # el mismo (una sola red), para la demo/versus
        # --- Interfaz compatible con EvolutionTrainer (para reusar el bucle de main.py) ---
        self.generation = 0               # aquí = nº de updates
        self.best_brain = self.policy
        self.best_fitness_ever = 0.0
        self.best_fitness_history: List[float] = []   # supervivencia por update (para gráficos)
        self.mean_fitness_history: List[float] = []   # retorno medio por update
        self.curriculum_event_history: List[str] = []
        self.curriculum_progress = 0.0
        self.gens_since_improvement = 0
        self.layer_sizes = list(layer_sizes)

    # --- Curriculum (mismo espíritu que el evolutivo, adaptativo por supervivencia) ---
    def curriculum_difficulty(self) -> Tuple[float, float]:
        """(spawn_interval, cap_episodio) efectivos. Warmup = sin enemigos; luego rampa por
        dominio (sube si sobrevive mucho, baja si poco)."""
        c = self.cfg
        spawn_final = self.game_cfg.spawn_interval
        ep_final = c.max_episode_seconds
        if not c.curriculum_enabled:
            return spawn_final, ep_final
        if self.generation < c.curriculum_warmup_updates:
            return float("inf"), float(c.curriculum_episode_seconds_start)
        t = min(1.0, max(0.0, self.curriculum_progress))
        spawn = c.curriculum_spawn_interval_start + (spawn_final - c.curriculum_spawn_interval_start) * t
        ep = c.curriculum_episode_seconds_start + (ep_final - c.curriculum_episode_seconds_start) * t
        return spawn, ep

    def current_game_cfg(self) -> GameConfig:
        spawn, _ = self.curriculum_difficulty()
        if spawn == self.game_cfg.spawn_interval:
            return self.game_cfg
        if spawn == float("inf"):
            return replace(self.game_cfg, spawn_interval=1e9)  # sin spawns efectivos
        return replace(self.game_cfg, spawn_interval=spawn)

    def _update_curriculum(self, mean_surv: float, cap: float) -> str:
        c = self.cfg
        if not c.curriculum_enabled or self.generation < c.curriculum_warmup_updates:
            return "warmup" if self.generation < c.curriculum_warmup_updates else "flat"
        frac = mean_surv / max(1e-6, cap)
        prev = self.curriculum_progress
        if frac >= c.curriculum_advance_threshold:
            self.curriculum_progress = min(1.0, prev + c.curriculum_advance_step)
        elif frac <= c.curriculum_regress_threshold:
            self.curriculum_progress = max(0.0, prev - c.curriculum_regress_step)
        if self.curriculum_progress > prev:
            return "up"
        if self.curriculum_progress < prev:
            return "down"
        return "flat"

    def mutation_schedule(self) -> Tuple[float, float]:
        """Compatibilidad con la interfaz del evolutivo (RL no muta)."""
        return 0.0, 0.0

    def seed_population_from_brain(self, brain: MLP) -> None:
        """Reanudar: carga los pesos del cerebro dado en la política y reinicia Adam."""
        self.policy.set_parameter_vector(brain.get_parameter_vector().copy())
        self.best_brain = self.policy
        self.last_champion = self.policy
        n = self.policy.num_parameters()
        self._m = np.zeros(n, dtype=np.float32)
        self._v = np.zeros(n, dtype=np.float32)
        self._t = 0

    def step_generation(self) -> Tuple[float, float, MLP]:
        """Una actualización de RL con la dificultad del curriculum. Firma compatible con el
        evolutivo: devuelve (supervivencia, retorno, policy)."""
        gcfg = self.current_game_cfg()
        _, cap = self.curriculum_difficulty()
        mean_surv, mean_ret = self.train_batch(gcfg, cap=cap)
        event = self._update_curriculum(mean_surv, cap)
        self.generation += 1
        if mean_surv > self.best_fitness_ever:
            self.best_fitness_ever = mean_surv
            self.gens_since_improvement = 0
        else:
            self.gens_since_improvement += 1
        self.best_fitness_history.append(mean_surv)
        self.mean_fitness_history.append(mean_ret)
        self.curriculum_event_history.append(event)
        self.last_champion = self.policy
        self.best_brain = self.policy
        return mean_surv, mean_ret, self.policy

    def save_checkpoint(self, path, extra=None, brain: Optional[MLP] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        net = (brain or self.policy)
        meta = {
            "generation": self.generation,
            "best_fitness_ever": self.best_fitness_ever,
            "curriculum_progress": self.curriculum_progress,
            "layer_sizes": self.layer_sizes,
            "method": "rl",
            "episodes_total": self.episodes_total,
            "game": {"arena_w": self.game_cfg.arena_w, "arena_h": self.game_cfg.arena_h,
                     "speed": self.game_cfg.speed, "spawn_interval": self.game_cfg.spawn_interval},
        }
        if extra:
            meta["extra"] = extra
        with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        with open(path, "wb") as f:
            f.write(serialize_brain(net))

    def save_history_csv(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.best_fitness_history)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("update,survival,return,event,episodes_total\n")
            for i in range(n):
                ev = self.curriculum_event_history[i] if i < len(self.curriculum_event_history) else ""
                ep = (i + 1) * self.cfg.batch_episodes
                f.write(f"{i},{self.best_fitness_history[i]:.4f},{self.mean_fitness_history[i]:.4f},{ev},{ep}\n")

    # --- Rollout de un episodio con la política estocástica ---
    def _rollout(self, game_cfg: GameConfig, seed: Optional[int], cap: Optional[float] = None):
        st = GameState.new_game(game_cfg, seed=seed)
        dt = self.cfg.fixed_dt
        max_steps = int((cap if cap is not None else self.cfg.max_episode_seconds) / dt) + 2
        obs: List[np.ndarray] = []
        acts: List[int] = []
        rews: List[float] = []
        for _ in range(max_steps):
            if st.game_over:
                break
            o = encode_observation(st)
            p = self.policy.action_probs(o)
            a = int(self.rng.choice(NUM_ACTIONS, p=p))
            step_game(st, a, dt)
            obs.append(o)
            acts.append(a)
            # +1 si sigue vivo tras el paso; 0 si esta acción lo mató (así se penaliza la fatal).
            rews.append(0.0 if st.game_over else self.cfg.reward_per_step)
        return obs, acts, rews, st.time_alive

    def _returns(self, rewards: List[float]) -> List[float]:
        """Retorno-a-futuro con descuento: G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ..."""
        g = 0.0
        out = [0.0] * len(rewards)
        for t in reversed(range(len(rewards))):
            g = rewards[t] + self.cfg.gamma * g
            out[t] = g
        return out

    def _adam(self, net, grad, m, v, t, lr):
        """Un paso de Adam (con gradient clipping) sobre `net`. Devuelve (m, v, t) nuevos.
        El clipping evita que un lote ruidoso pegue un paso que destruya la política."""
        c = self.cfg
        if c.max_grad_norm and c.max_grad_norm > 0.0:
            norm = float(np.linalg.norm(grad))
            if norm > c.max_grad_norm:
                grad = grad * (c.max_grad_norm / (norm + 1e-8))
        t += 1
        m = c.adam_beta1 * m + (1.0 - c.adam_beta1) * grad
        v = c.adam_beta2 * v + (1.0 - c.adam_beta2) * (grad * grad)
        mhat = m / (1.0 - c.adam_beta1 ** t)
        vhat = v / (1.0 - c.adam_beta2 ** t)
        theta = net.get_parameter_vector()
        theta = theta - lr * mhat / (np.sqrt(vhat) + c.adam_eps)
        net.set_parameter_vector(theta)
        return m, v, t

    def _current_entropy_coef(self) -> float:
        """Entropía annealada por progreso del curriculum: alta al principio (explora), baja
        al final para que la política greedy cristalice. Vale también al continuar un run."""
        c = self.cfg
        t = min(1.0, max(0.0, self.curriculum_progress))
        return float(c.entropy_coef + (c.entropy_coef_end - c.entropy_coef) * t)

    def _entropy_grad(self, X: np.ndarray, coef: float) -> np.ndarray:
        """Gradiente (de DESCENSO) de −entropía: empuja hacia distribuciones más inciertas.
        dH/dz_j = −p_j (log p_j + H). Restamos coef·dH del gradiente de pérdida (=ascender H)."""
        from src.neural.mlp import _softmax
        logits, _ = self.policy.forward_batch(X)
        p = _softmax(logits)
        logp = np.log(p + 1e-12)
        H = -(p * logp).sum(axis=1, keepdims=True)          # (N,1)
        dH_dz = -p * (logp + H)                              # (N, out)
        dZ = -coef * dH_dz / max(1, X.shape[0])
        return self.policy._backward_from_dlogits(X, dZ.astype(np.float32))

    def train_batch(self, game_cfg: Optional[GameConfig] = None,
                    cap: Optional[float] = None) -> Tuple[float, float]:
        """Una actualización: junta un lote de partidas, calcula ventajas y da un paso de Adam.
        Devuelve (supervivencia media en s, retorno medio)."""
        gcfg = game_cfg or self.game_cfg
        allX: List[np.ndarray] = []
        allA: List[int] = []
        allR: List[float] = []
        survs: List[float] = []
        for _ in range(self.cfg.batch_episodes):
            seed = int(self.rng.integers(0, 2**31 - 1))
            obs, acts, rews, tsurv = self._rollout(gcfg, seed, cap=cap)
            survs.append(tsurv)
            if not obs:
                continue
            allX.extend(obs)
            allA.extend(acts)
            allR.extend(self._returns(rews))
        self.updates += 1
        self.episodes_total += self.cfg.batch_episodes
        mean_surv = float(np.mean(survs)) if survs else 0.0
        if allX:
            X = np.asarray(allX, dtype=np.float32)
            A = np.asarray(allA, dtype=np.int64)
            R = np.asarray(allR, dtype=np.float32)
            # Baseline: con critic, ventaja = retorno − V(s) (específica del estado, baja
            # varianza). Sin critic, ventaja = retorno − promedio del lote.
            if self.value is not None:
                V = self.value.forward_batch(X)[0].reshape(-1)   # V(s) por paso
                adv = R - V
                # Entrenar el critic hacia el retorno (regresión MSE): dL/dV = (V − R).
                dZv = ((V - R) / max(1, len(R))).reshape(-1, 1).astype(np.float32)
                grad_v = self.value._backward_from_dlogits(X, dZv)
                self._vm, self._vv, self._vt = self._adam(
                    self.value, grad_v, self._vm, self._vv, self._vt, self.cfg.value_lr)
            else:
                adv = R - R.mean()
            std = float(adv.std())
            if std > 1e-6:
                adv = adv / std
            grad = self.policy.policy_gradients(X, A, adv)
            ecoef = self._current_entropy_coef()
            if ecoef > 0.0:
                grad = grad + self._entropy_grad(X, ecoef)
            self._m, self._v, self._t = self._adam(
                self.policy, grad, self._m, self._v, self._t, self.cfg.lr)
            self.return_history.append(float(R.mean()))
        self.survival_history.append(mean_surv)
        return mean_surv, (self.return_history[-1] if self.return_history else 0.0)
