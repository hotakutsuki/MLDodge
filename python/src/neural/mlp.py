"""
Perceptrón multicapa (MLP) con NumPy: solo matmul y activaciones.
Sin frameworks de deep learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LayerTrace:
    """Una capa densa: pre-activación (z) y salida tras activación (a)."""

    z: np.ndarray
    a: np.ndarray


@dataclass
class ForwardTrace:
    """Trazas de todas las capas (útil para visualizar la red en modo versus)."""

    layers: List[LayerTrace]
    input_vec: np.ndarray
    logits: np.ndarray


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(z: np.ndarray) -> np.ndarray:
    """Softmax estable (resta el máximo). Funciona con (out,) o (N, out)."""
    z = np.asarray(z, dtype=np.float32)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


class MLP:
    """
    Red fully-connected: capas [in -> h1 -> ... -> out].
    Última capa sin activación (logits); el resto ReLU.
    """

    def __init__(self, layer_sizes: Sequence[int], rng: Optional[np.random.Generator] = None):
        if len(layer_sizes) < 2:
            raise ValueError("Se necesitan al menos input_dim y output_dim")
        self.layer_sizes = list(int(x) for x in layer_sizes)
        self.rng = rng if rng is not None else np.random.default_rng()

        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        for i in range(len(self.layer_sizes) - 1):
            n_in, n_out = self.layer_sizes[i], self.layer_sizes[i + 1]
            # Inicialización He para ReLU (solo capas ocultas; la última también escala razonable)
            scale = np.sqrt(2.0 / max(1, n_in))
            w = self.rng.normal(0.0, scale, size=(n_out, n_in)).astype(np.float32)
            b = np.zeros(n_out, dtype=np.float32)
            self._weights.append(w)
            self._biases.append(b)

    @property
    def weight_matrices(self) -> Tuple[np.ndarray, ...]:
        return tuple(self._weights)

    @property
    def bias_vectors(self) -> Tuple[np.ndarray, ...]:
        return tuple(self._biases)

    def num_parameters(self) -> int:
        n = 0
        for w, b in zip(self._weights, self._biases):
            n += w.size + b.size
        return int(n)

    def get_parameter_vector(self) -> np.ndarray:
        """Vector 1D con todos los pesos y sesgos (orden: W0, b0, W1, b1, ...)."""
        parts = []
        for w, b in zip(self._weights, self._biases):
            parts.append(w.reshape(-1))
            parts.append(b.reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    def set_parameter_vector(self, vec: np.ndarray) -> None:
        """Restaura pesos desde un vector plano compatible con get_parameter_vector."""
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        idx = 0
        for i in range(len(self._weights)):
            w_shape = self._weights[i].shape
            n_w = self._weights[i].size
            n_b = self._biases[i].size
            self._weights[i] = vec[idx : idx + n_w].reshape(w_shape).copy()
            idx += n_w
            self._biases[i] = vec[idx : idx + n_b].copy()
            idx += n_b
        if idx != vec.size:
            raise ValueError("Tamaño del vector de parámetros incompatible con la arquitectura")

    def copy(self) -> MLP:
        """Clon con los mismos pesos."""
        other = MLP(self.layer_sizes, rng=self.rng)
        other.set_parameter_vector(self.get_parameter_vector())
        return other

    def forward(
        self, x: np.ndarray, return_trace: bool = False
    ) -> Tuple[np.ndarray, Optional[ForwardTrace]]:
        """
        x: vector columna o 1D de forma (input_dim,)
        Devuelve logits (out_dim,) y opcionalmente trazas por capa.
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.layer_sizes[0]:
            raise ValueError(f"Se esperaba entrada de tamaño {self.layer_sizes[0]}, got {x.shape[0]}")

        traces: List[LayerTrace] = []
        a = x
        num_layers = len(self._weights)

        for i in range(num_layers):
            z = self._weights[i] @ a + self._biases[i]
            # Última capa: sin activación (logits)
            if i == num_layers - 1:
                if return_trace:
                    traces.append(LayerTrace(z=z.copy(), a=z.copy()))
                logits = z
            else:
                a_next = _relu(z)
                if return_trace:
                    traces.append(LayerTrace(z=z.copy(), a=a_next.copy()))
                a = a_next

        logits = np.asarray(logits, dtype=np.float32)
        trace_obj = None
        if return_trace:
            trace_obj = ForwardTrace(layers=traces, input_vec=x.copy(), logits=logits.copy())
        return logits, trace_obj

    def decide_action(self, x: np.ndarray, return_trace: bool = False):
        """Devuelve índice de acción argmax(logits) en [0, out_dim)."""
        logits, trace = self.forward(x, return_trace=return_trace)
        action = int(np.argmax(logits))
        return action, logits, trace

    # --- Soporte para RL (policy gradient): softmax, forward batch y backprop ---

    def action_probs(self, x: np.ndarray) -> np.ndarray:
        """Distribución de probabilidad sobre acciones (softmax de los logits). Para RL la
        política es ESTOCÁSTICA: se samplea de aquí, no se toma el argmax."""
        logits, _ = self.forward(x)
        return _softmax(logits)

    def forward_batch(self, X: np.ndarray):
        """Forward de N ejemplos a la vez. X: (N, in_dim). Devuelve (logits (N,out), cache)
        donde cache guarda las entradas y pre-activaciones de cada capa para el backward."""
        A = np.asarray(X, dtype=np.float32)
        if A.ndim != 2 or A.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Se esperaba (N, {self.layer_sizes[0]}), got {A.shape}")
        inputs: List[np.ndarray] = []   # entrada a cada capa (a_{i-1})
        pre: List[np.ndarray] = []      # z de cada capa (para la máscara ReLU)
        num_layers = len(self._weights)
        for i in range(num_layers):
            inputs.append(A)
            Z = A @ self._weights[i].T + self._biases[i]     # (N, out_i)
            pre.append(Z)
            A = Z if i == num_layers - 1 else _relu(Z)
        return A, (inputs, pre)

    def policy_gradients(self, X: np.ndarray, actions: np.ndarray,
                         advantages: np.ndarray) -> np.ndarray:
        """Gradiente de la PÉRDIDA de policy gradient respecto a todos los pesos, en el mismo
        orden que get_parameter_vector (W0,b0,W1,b1,...). Para minimizar con el optimizador.

        Pérdida = − mean( ventaja · log π(a|s) ). Su derivada respecto a los logits es
        elegante: ventaja · (prob − onehot(a)). El resto es backprop lineal + ReLU estándar.
        """
        X = np.asarray(X, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        adv = np.asarray(advantages, dtype=np.float32).reshape(-1)
        n = X.shape[0]
        logits, cache = self.forward_batch(X)
        probs = _softmax(logits)
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), actions] = 1.0
        # dL/dlogits (para DESCENSO): ventaja·(prob − onehot), promediado sobre el batch.
        dZ = (adv[:, None] * (probs - onehot)) / max(1, n)   # (N, out)
        return self._backward(dZ.astype(np.float32), cache)

    def _backward_from_dlogits(self, X: np.ndarray, dZ: np.ndarray) -> np.ndarray:
        """Backprop dado un gradiente arbitrario sobre los logits (p. ej. el de entropía).
        Devuelve el gradiente plano en orden get_parameter_vector."""
        _, cache = self.forward_batch(np.asarray(X, dtype=np.float32))
        return self._backward(np.asarray(dZ, dtype=np.float32), cache)

    def _backward(self, dZ: np.ndarray, cache) -> np.ndarray:
        """Backprop lineal + ReLU dado dZ = dL/dlogits y el cache del forward_batch."""
        inputs, pre = cache
        num_layers = len(self._weights)
        grads: List[Optional[np.ndarray]] = [None] * (2 * num_layers)
        for i in reversed(range(num_layers)):
            A_prev = inputs[i]                     # (N, in_i)
            dW = dZ.T @ A_prev                     # (out_i, in_i)
            db = dZ.sum(axis=0)                    # (out_i,)
            grads[2 * i] = dW.astype(np.float32)
            grads[2 * i + 1] = db.astype(np.float32)
            if i > 0:
                dA = dZ @ self._weights[i]         # (N, in_i)
                dZ = dA * (pre[i - 1] > 0.0)       # atravesar la ReLU de la capa anterior
        return np.concatenate([g.reshape(-1) for g in grads]).astype(np.float32)


def serialize_brain(mlp: MLP) -> bytes:
    """Serializa arquitectura + parámetros en un bloque numpy comprimido."""
    payload = {
        "layer_sizes": np.array(mlp.layer_sizes, dtype=np.int32),
        "params": mlp.get_parameter_vector(),
    }
    buffer = __import__("io").BytesIO()
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


def deserialize_brain(data: bytes) -> MLP:
    buf = __import__("io").BytesIO(data)
    loaded = np.load(buf, allow_pickle=False)
    sizes = [int(x) for x in loaded["layer_sizes"]]
    net = MLP(sizes)
    net.set_parameter_vector(loaded["params"])
    return net


def save_brain_to_file(mlp: MLP, path: str) -> None:
    with open(path, "wb") as f:
        f.write(serialize_brain(mlp))


def load_brain_from_file(path: str) -> MLP:
    with open(path, "rb") as f:
        return deserialize_brain(f.read())
