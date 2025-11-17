from pydantic import BaseModel, ConfigDict, Field
from typing import List, Tuple, Union
import numpy as np

State = List[List[int]]
Action = int
Participant = Tuple[str, type]
Versus = List[Tuple[Union[Participant, None], Union[Participant, None]]]


class Game(BaseModel):
    """
    Representa UNA partida completa.
    Guarda TODO lo necesario:
    - quién fue +1
    - quién fue -1
    - historial de jugadas (estado, acción)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    player_plus1: str
    player_minus1: str
    history: List[Tuple[State, Action]]  # siempre listas, nunca numpy array


class Match(BaseModel):
    """
    Un match de (best of N) partidas.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    player_a: str
    player_b: str

    player_a_wins: int = 0
    player_b_wins: int = 0
    draws: int = 0

    games: List[Game] = Field(default_factory=list)
