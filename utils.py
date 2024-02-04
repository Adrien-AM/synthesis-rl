from typing import List, Tuple
from synth.syntax.program import Program
import pickle

def save_with_pickle(path: str, data: List[Tuple[Program, float]]):
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def load_with_pickle(path: str) -> List[Tuple[Program, float]]:
    with open(path, 'rb') as file:
        return pickle.load(file)
