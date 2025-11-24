import numpy as np
import json
import os
import tempfile
from connect4.policy import Policy
from typing import override


class UncertaintyWithUCB1(Policy):

    def __init__(self, c=2):
        self.Q = {}  # Estimación de los Q-values
        self.N = {}  # Número de veces que cada acción ha sido seleccionada
        self.total_actions = 0  # Número total de acciones
        self.memory = []
        self.alpha = 0.2  # Learning rate
        self.c = c  # Parámetro de exploración
        self.rng = np.random.default_rng()
        self._load_qvalues()

    @override
    def mount(self, time_out=None):
        self.memory.clear()

    @override
    def act(self, s: np.ndarray) -> int:
        """Devuelve la acción que el agente tomará, según la política UCB1."""

        b_real = s.copy()  # Copia del tablero real (sin normalizar)
        b = self._normalize(b_real)  # Normalización del tablero para el aprendizaje

        # Acciones posibles (columnas disponibles)
        available = np.flatnonzero(b_real[0] == 0)
        if available.size == 0:
            return -1  # Si no hay columnas disponibles, retornar -1

        state_key = b.tobytes().hex()  # Clave del estado en base al tablero normalizado

        # Inicializar Q-values y N (número de veces seleccionada) para las nuevas acciones si no existen
        for c in available:
            key = f"{state_key}|{int(c)}"
            if key not in self.Q:
                self.Q[key] = 0.0  # Inicializa el Q-value
            if key not in self.N:
                self.N[key] = 0  # Inicializa el contador de veces que se seleccionó la acción

        # Selección de acción con UCB1
        ucb_values = {}
        for c in available:
            key = f"{state_key}|{int(c)}"
            # Calcular la cota superior de confianza (UCB1)
            ucb_values[c] = self.Q[key] + self.c * np.sqrt(np.log(self.total_actions + 1) / (self.N[key] + 1))

        # Seleccionar la acción con el mayor valor UCB
        action = max(ucb_values, key=ucb_values.get)

        # Guardar el estado y la acción en memoria para actualizar después
        self.memory.append((state_key, action))

        # Actualizar el número total de acciones
        self.total_actions += 1

        # Incrementar el contador de veces que se ha seleccionado la acción
        self.N[f"{state_key}|{action}"] += 1

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
