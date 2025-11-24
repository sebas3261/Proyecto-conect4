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
        self.epsilon = 0.0  # Sin exploración, solo explotación
        self.alpha = 0.2
        self.rng = np.random.default_rng()
        self._load_qvalues()  # Cargar los Q-values al iniciar

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

        # CORRECCIÓN: Usar el método correcto para generar la clave
        state_key = self._state_key_hex(b)

        # Inicializar Q-values para las nuevas acciones si no existen
        for c in available:
            key = f"{state_key}|{int(c)}"
            if key not in self.Q:
                self.Q[key] = 0.0

        # Selección de acción explotando los Q-values (sin exploración)
        action = max(available, key=lambda c: self.Q.get(f"{state_key}|{int(c)}", 0.0))
        print(f"Acción seleccionada (explotación): {action}")

        # Guardar el estado y la acción en memoria para actualizar después
        self.memory.append((state_key, action))
        return action

    @override
    def final(self, reward: int):
        """Actualiza los Q-values según el premio recibido y limpia la memoria."""
        for s_key, a in self.memory:
            key = f"{s_key}|{a}"
            q = self.Q.get(key, 0.0)
            self.Q[key] = q + self.alpha * (reward - q)

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

    def _state_key_hex(self, board: np.ndarray) -> str:
        """Convierte el tablero en una cadena hexadecimal compatible con el formato del JSON."""
        flat = board.reshape(-1)
        hex_parts = []
        for val in flat:
            if val == -1:
                hex_parts.append('ff')  # -1 se representa como 'ff'
            elif val == 0:
                hex_parts.append('00')  # 0 se representa como '00'
            elif val == 1:
                hex_parts.append('01')  # 1 se representa como '01'
            else:
                # Para valores inesperados
                hex_parts.append(f'{val & 0xFF:02x}')
        
        return ''.join(hex_parts)

    def _state_key(self, s: np.ndarray) -> str:
        """Convierte el tablero en una cadena única para usar como clave de estado (método alternativo)."""
        return ",".join(map(str, s.reshape(-1)))

    def _json_path(self) -> str:
        """Devuelve la ruta del archivo donde se guardarán los Q-values."""
        path = os.path.join(os.path.dirname(__file__), "qvalues.json")
        return path

    def _load_qvalues(self):
        """Carga los Q-values desde el archivo json, si existe."""
        path = self._json_path()
        if not os.path.exists(path):
            print("No se encontró el archivo de Q-values, inicializando vacío.")
            self.Q = {}
            return
        try:
            with open(path, "r") as f:
                text = f.read().strip()
                if text:
                    self.Q = json.loads(text)
                    print(f"Q-values cargados: {len(self.Q)} estados en memoria")
                else:
                    print("Archivo de Q-values vacío, inicializando vacío.")
                    self.Q = {}
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
