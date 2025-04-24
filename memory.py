import random
from collections import defaultdict

from transition import Transition, State

class ExperienceReplay():
    def __init__(self, max_mem: int=10000) -> None:
        self.max_mem: int = max_mem
        self.memo: list[Transition] = []
    

    def append(self, replay: Transition) -> None:
        if len(self.memo) >= self.max_mem:
            self.memo.pop(0)
        self.memo.append(replay)
    

    # store: (s, a, r, s', d)
    def get_random_sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memo, min(batch_size, len(self.memo)))
    

    def __len__(self) -> int:
        return len(self.memo)


class ExplorationTracker():
    def __init__(self) -> None:
        # keep track of map number, x-coord, y-coord
        self.visited: dict[tuple[int, int, int], int] = defaultdict(int)


    
    def calculate_reward(self, state: State, weight: float) -> float:
        location: tuple[int, int, int] = (
            int(state['map number'] * 255 / 4),
            int(state['player x-position'] * 255 / 4),
            int(state['player y-position'] * 255 / 4)
        )

        self.visited[location] += 1

        return weight / self.visited[location]
    

    def clear_memory(self) -> None:
        self.visited = defaultdict(int)
