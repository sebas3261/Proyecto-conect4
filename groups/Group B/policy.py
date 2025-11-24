import numpy as np
import json
import os
import tempfile
from connect4.policy import Policy
from typing import override


class UncertaintyWithEGreedy(Policy):

    def __init__(self):
        self.Q = {}
        self.memory = []
        self.epsilon = 0.01
        self.alpha = 0.2
        self.rng = np.random.default_rng()
        self._load_qvalues()

    @override
    def mount(self, time_out=None):
        self.memory.clear()

    @override
    def act(self, s: np.ndarray) -> int:
        """Devuelve la acción que el agente tomará, según la política epsilon-greedy."""

        b_real = s.copy()  # Copia del tablero real (sin normalizar)
        b = self._normalize(b_real)  # Normalización del tablero para el aprendizaje

        # Acciones posibles (columnas disponibles)
        available = np.flatnonzero(b_real[0] == 0)
        if available.size == 0:
            return -1  # Si no hay columnas disponibles, retornar -1

        state_key = b.tobytes().hex()  # Clave del estado en base al tablero normalizado

        # Inicializar Q-values para las nuevas acciones si no existen
        for c in available:
            key = f"{state_key}|{int(c)}"
            if key not in self.Q:
                self.Q[key] = 0.0

        # Selección de acción con epsilon-greedy
        if self.rng.random() < self.epsilon:
            action = int(self.rng.choice(available))  # Exploración: elige aleatoriamente una acción
        else:
            # Explotación: elige la acción con el valor Q más alto
            action = max(available, key=lambda c: self.Q[f"{state_key}|{c}"])

        # Guardar el estado y la acción en memoria para actualizar después
        self.memory.append((state_key, action))
        return action

    @override
    def final(self, reward: int):
        """Actualiza los Q-values según el premio recibido y limpia la memoria."""

        for s_key, a in self.memory:
            key = f"{s_key}|{a}"
            q = self.Q.get(key, 0.0)
            self.Q[key] = q + self.alpha * (reward - q)  # Fórmula de actualización de Q-value

        # Limpiar la memoria después de actualizar los Q-values
        self.memory.clear()

    # Utilidades para el manejo de archivos ------------

    def _normalize(self, board: np.ndarray) -> np.ndarray:
        """Normaliza el tablero para asegurarse de que el agente siempre juegue como el jugador 1."""
        b = board.copy()
        ones = np.count_nonzero(b == 1)
        negs = np.count_nonzero(b == -1)

        # Si hay más fichas -1, invertir todo el tablero para jugar como 1
        if negs > ones:
            b = -b

        return b

    def _state_key(self, s: np.ndarray) -> str:
        """Convierte el tablero en una cadena única para usar como clave de estado."""
        return ",".join(map(str, s.reshape(-1)))

    def _json_path(self) -> str:
        """Devuelve la ruta del archivo donde se guardarán los Q-values."""
        return os.path.join(os.path.dirname(__file__), "qvalues.json")

    def _load_qvalues(self):
        """Carga los Q-values desde el archivo json, si existe."""
        path = self._json_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                text = f.read().strip()
                self.Q = json.loads(text) if text else {}
        except Exception as e:
            print(f"Error al cargar los Q-values: {e}")
            self.Q = {}

    def _save_qvalues(self, path_override=None):
        """Guarda los Q-values de forma segura en un archivo temporal y luego renombra."""
        path = path_override or self._json_path()
        tmp = path + ".tmp"

        try:
            with open(tmp, "w") as f:
                json.dump(self.Q, f)

            # Intenta reemplazar el archivo original
            os.replace(tmp, path)
        except PermissionError:
            print("Error de permisos, intentando de nuevo...")
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp, path)
        except Exception as e:
            print(f"Error al guardar los Q-values: {e}")
