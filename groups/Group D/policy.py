import numpy as np
import random
import math
from connect4.policy import Policy


class MCTSFast(Policy):
    """
    Implementación de una política basada en Monte Carlo Tree Search (MCTS)
    para Conecta 4.

    Características:
    - No usa tabla de Q-values ni persistencia en disco.
    - Ajusta el número de simulaciones por jugada según un timeout aproximado.
    - Selecciona movimientos con UCB1 sobre el árbol de búsqueda.
    
    """

    def __init__(self):
        # Número base de simulaciones que se harán por movimiento.
        # Este valor luego se ajusta en mount() según el timeout.
        self.simulations_per_move = 200

        # Constante de exploración para UCB1:
        # valores mayores fomentan explorar nodos menos visitados.
        self.c_param = 1.2

        # Tiempo máximo (en segundos aprox.) permitido por jugada
        # (el framework puede pasarlo en mount()).
        self.time_out = None

    def mount(self, time_out=None):
        """
        Inicializa parámetros dependientes del tiempo disponible.

        Parameters
        ----------
        time_out : float or None
            Tiempo máximo (segundos aproximados) por jugada. Si es None,
            se asume un valor por defecto de 5 segundos.
        """
        # Ajusta el timeout interno
        self.time_out = time_out if time_out is not None else 5

        # Ajusta el número de simulaciones por jugada de forma simple:
        #   - No menos de 50
        #   - No más de 400
        #   - Proporcional al timeout (time_out * 40)
        self.simulations_per_move = max(
            50,
            min(400, int(self.time_out * 40))
        )

    def act(self, s: np.ndarray) -> int:
        """
        Elige una acción (columna) usando MCTS.

        Flujo general:
        1. Construye el nodo raíz a partir del estado actual.
        2. Ejecuta múltiples simulaciones:
           - Selección: recorre el árbol usando UCB1.
           - Expansión: crea nuevos hijos para acciones aún no probadas.
           - Rollout: simula la partida aleatoriamente desde ese punto.
           - Retropropagación: actualiza conteos y recompensas acumuladas.
        3. Devuelve la acción más visitada en la raíz.

        Parameters
        ----------
        s : np.ndarray
            Tablero de Conecta 4 (6x7) con valores:
            1  -> fichas del primer jugador
            -1 -> fichas del segundo jugador
            0  -> casilla vacía

        Returns
        -------
        int
            Índice de columna donde soltar la ficha (0 a 6).
        """
        # Convertimos a lista de listas para manipular fácilmente
        state = s.tolist()

        # Calculamos las jugadas legales (columnas no llenas)
        legal = self.get_legal_moves(state)
        if not legal:
            # Si no hay jugadas posibles, devolvemos cualquier valor válido (ej. 0)
            return 0

        # Determinar a quién le toca jugar:
        # - Si hay la misma cantidad de 1 y -1 → le toca al jugador 1
        # - Si hay una ficha más de 1 → le toca al jugador -1
        ones = sum(cell == 1 for row in state for cell in row)
        negs = sum(cell == -1 for row in state for cell in row)
        player = 1 if ones == negs else -1

        # Nodo raíz para el MCTS
        root = NodeFast(state, player)

        # Bucle principal de simulaciones de MCTS
        for _ in range(self.simulations_per_move):

            # Copias de trabajo del estado y jugador actual
            node = root
            sim_state = [row[:] for row in state]
            sim_player = player

            # 1. SELECCIÓN:
            # Descendemos por el árbol mientras:
            # - el nodo esté completamente expandido (no haya acciones nuevas)
            # - y el estado no sea terminal
            while node.is_fully_expanded and not self.is_terminal(sim_state):
                node = self.select_child(node)
                sim_state = self.apply_move(sim_state, node.action, sim_player)
                sim_player = -sim_player  # alternar jugador

            # 2. EXPANSIÓN:
            # Si aún no estamos en un estado terminal, intentamos expandir
            if not self.is_terminal(sim_state):
                action = node.pop_action()  # saca una acción no probada
                if action is not None:
                    # Aplicamos esa acción al estado de simulación
                    sim_state = self.apply_move(sim_state, action, sim_player)
                    # Creamos el nuevo nodo hijo resultante
                    child = NodeFast(
                        sim_state,
                        -sim_player,   # siguiente jugador
                        action=action,
                        parent=node
                    )
                    node.children[action] = child
                    node = child
                    sim_player = -sim_player

            # 3. ROLLOUT:
            # Simulamos de manera rápida y aleatoria el resto de la partida
            reward = self.fast_rollout(sim_state, sim_player, root.player)

            # 4. BACKPROPAGATION:
            # Propagamos la recompensa hacia arriba en el árbol
            self.backpropagate(node, reward)

        # En la raíz, devolvemos la acción más visitada (más robusta)
        return root.best_child()

    # ------------------------------------------------------------------
    # Utilidades para manejar el tablero y estados del juego
    # ------------------------------------------------------------------

    def get_legal_moves(self, state):
        """
        Devuelve las columnas que aún tienen espacio para jugar.

        Parameters
        ----------
        state : list[list[int]]
            Representación del tablero.

        Returns
        -------
        list[int]
            Lista de índices de columnas legales.
        """
        return [c for c in range(7) if state[0][c] == 0]

    def apply_move(self, state, col, player):
        """
        Aplica un movimiento (columna) al estado, soltando la ficha
        desde la parte superior hasta la primera casilla libre.

        Parameters
        ----------
        state : list[list[int]]
            Tablero actual.
        col : int
            Columna donde se suelta la ficha.
        player : int
            Jugador que mueve (1 o -1).

        Returns
        -------
        list[list[int]]
            Nuevo estado del tablero después de mover.
        """
        # Hacemos una copia profunda del tablero
        st = [row[:] for row in state]
        # Recorremos de abajo hacia arriba buscando la primera casilla vacía
        for r in range(5, -1, -1):
            if st[r][col] == 0:
                st[r][col] = player
                return st
        # Si la columna está llena (no debería pasar si se respeta get_legal_moves),
        # devolvemos el tablero sin cambios.
        return st

    def is_terminal(self, state):
        """
        Revisa si el estado es terminal:
        - Gana el jugador 1
        - Gana el jugador -1
        - O el tablero está lleno (empate)

        Parameters
        ----------
        state : list[list[int]]

        Returns
        -------
        bool
            True si el juego ha terminado, False en caso contrario.
        """
        return (
            self.check_win(state, 1)
            or self.check_win(state, -1)
            or not self.get_legal_moves(state)
        )

    def check_win(self, b, p):
        """
        Comprueba si el jugador p tiene 4 en línea en el tablero b.

        Parameters
        ----------
        b : list[list[int]]
            Tablero.
        p : int
            Jugador (1 o -1).

        Returns
        -------
        bool
            True si p tiene 4 en línea, False en caso contrario.
        """
        # Comprobación horizontal
        for r in range(6):
            br = b[r]
            for c in range(4):
                if br[c] == p and br[c+1] == p and br[c+2] == p and br[c+3] == p:
                    return True

        # Comprobación vertical
        for c in range(7):
            for r in range(3):
                if b[r][c] == p and b[r+1][c] == p and b[r+2][c] == p and b[r+3][c] == p:
                    return True

        # Diagonales hacia abajo a la derecha (↘)
        for r in range(3):
            for c in range(4):
                if b[r][c] == p and b[r+1][c+1] == p and b[r+2][c+2] == p and b[r+3][c+3] == p:
                    return True

        # Diagonales hacia abajo a la izquierda (↙)
        for r in range(3):
            for c in range(3, 7):
                if b[r][c] == p and b[r+1][c-1] == p and b[r+2][c-2] == p and b[r+3][c-3] == p:
                    return True

        return False

    # ------------------------------------------------------------------
    # Rollout rápido (simulación aleatoria limitada)
    # ------------------------------------------------------------------

    def fast_rollout(self, state, player, root_player):
        """
        Realiza una simulación rápida del juego a partir de un estado dado,
        eligiendo movimientos al azar durante un número limitado de turnos.

        Parameters
        ----------
        state : list[list[int]]
            Estado desde el que se simula.
        player : int
            Jugador que mueve primero en el rollout.
        root_player : int
            Jugador para el cual se calcula la recompensa (+1 victoria, -1 derrota).

        Returns
        -------
        float
            Recompensa desde la perspectiva de root_player:
            +1.0 si gana root_player,
            -1.0 si gana el oponente,
            0.0 si no hay resultado claro (empate o sin terminar).
        """
        sim = [row[:] for row in state]

        # Limitamos la profundidad del rollout a unas pocas jugadas (ej. 12 movimientos),
        # para mantenerlo rápido.
        for _ in range(12):
            if self.is_terminal(sim):
                break

            legal = self.get_legal_moves(sim)
            if not legal:
                break

            # Elegimos una acción aleatoria entre las legales
            a = random.choice(legal)
            sim = self.apply_move(sim, a, player)
            player = -player

        # Evaluamos el resultado desde el punto de vista de root_player
        if self.check_win(sim, root_player):
            return 1.0
        if self.check_win(sim, -root_player):
            return -1.0
        return 0.0

    # ------------------------------------------------------------------
    # Retropropagación de la recompensa en el árbol
    # ------------------------------------------------------------------

    def backpropagate(self, node, reward):
        """
        Propaga la recompensa hacia los ancestros del nodo visitado.

        Parameters
        ----------
        node : NodeFast
            Nodo desde el que se comienza a retropropagar.
        reward : float
            Recompensa obtenida en el rollout.
        """
        current = node
        while current:
            current.N += 1          # número de visitas
            current.W += reward     # suma de recompensas
            current = current.parent

    # ------------------------------------------------------------------
    # Selección de hijo con UCB1
    # ------------------------------------------------------------------

    def select_child(self, node):
        """
        Selecciona el mejor hijo de un nodo usando la fórmula UCB1.

        Parameters
        ----------
        node : NodeFast

        Returns
        -------
        NodeFast
            Hijo seleccionado para continuar la simulación.
        """
        best = None
        best_score = -1e9

        for child in node.children.values():
            if child.N == 0:
                # Siempre explorar primero nodos sin visitas
                score = float("inf")
            else:
                # UCB1: (media de recompensa) + c * sqrt(log(N_padre) / N_hijo)
                exploit = child.W / child.N
                explore = self.c_param * math.sqrt(
                    math.log(node.N + 1) / child.N
                )
                score = exploit + explore

            if score > best_score:
                best_score = score
                best = child

        return best


class NodeFast:
    """
    Nodo del árbol de MCTS para Conecta 4.

    Atributos principales:
    - state: estado del tablero.
    - player: jugador al que le tocaría mover en este nodo.
    - action: acción (columna) que llevó desde el padre a este nodo.
    - parent: referencia al nodo padre.
    - children: diccionario {acción: nodo_hijo}.
    - N: número de visitas a este nodo.
    - W: suma de recompensas acumuladas.
    - untried: lista de acciones aún no exploradas desde este nodo.
    - is_fully_expanded: True si ya no quedan acciones sin explorar.
    """

    def __init__(self, state, player, action=None, parent=None):
        self.state = state
        self.player = player
        self.action = action
        self.parent = parent

        # Hijos del nodo: {columna: NodeFast}
        self.children = {}

        # Contadores de MCTS
        self.N = 0          # visitas
        self.W = 0.0        # recompensa acumulada

        # Acciones aún no exploradas desde este estado
        self.untried = [c for c in range(7) if state[0][c] == 0]

        # Indica si ya se exploraron todas las acciones posibles
        self.is_fully_expanded = len(self.untried) == 0

    def pop_action(self):
        """
        Extrae y elimina una acción no probada de la lista 'untried'.

        Returns
        -------
        int or None
            Acción (columna) no probada, o None si ya no quedan.
        """
        if not self.untried:
            return None
        a = self.untried.pop()
        self.is_fully_expanded = (len(self.untried) == 0)
        return a

    def best_child(self):
        """
        Devuelve la acción asociada al hijo más visitado.

        Es la política final que se usa después de todas las simulaciones
        para decidir la jugada real en el entorno.

        Returns
        -------
        int
            Columna del hijo más visitado. Si no hay hijos (caso raro),
            devuelve una columna aleatoria válida (0 a 6).
        """
        best_a, best_n = None, -1
        for a, ch in self.children.items():
            if ch.N > best_n:
                best_n = ch.N
                best_a = a

        # Si por algún motivo no se expandió nada, jugamos una columna aleatoria
        return best_a if best_a is not None else random.randint(0, 6)
