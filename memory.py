import random
from collections import defaultdict

from transition import Transition, State

class ExperienceReplay():
    def __init__(self, max_mem: int=10000) -> None:
        self.max_mem: int = max_mem
        self.memory: list[Transition] = []
    

    def append(self, replay: Transition) -> None:
        if len(self.memory) >= self.max_mem:
            self.memory.pop(0)
        self.memory.append(replay)
    

    # store: (s, a, r, s', d)
    def get_random_sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    

    def __len__(self) -> int:
        return len(self.memory)
    

    def __str__(self) -> str:
        return '\n'.join([str(transition) for transition in self.memory])


class ExplorationTracker():
    def __init__(self) -> None:
        # keep track of map number, x-coord, y-coord
        self.visited: dict[tuple[int, int, int], int] = defaultdict(int)
        self.stuck_steps: int = 0
        self.max_stuck: int = 8


    
    def calculate_reward(self, state: State, prev_state: State, weight: float, movement: bool) -> tuple[float, bool]:
        location: tuple[int, int, int] = (
            int(state['map number']),
            int(state['player x-position']),
            int(state['player y-position'])
        )
        prev_location: tuple[int, int, int] = (
            int(prev_state['map number']),
            int(prev_state['player x-position']),
            int(prev_state['player y-position'])
        )

        if movement and state['battle type'] == 0:
            if location != prev_location:
                self.visited[location] += 1
                self.stuck_steps = 0
            else:
                self.stuck_steps += 1

        if self.visited[location] > 10 or self.stuck_steps > self.max_stuck:
            return -1.0 * weight, True
        ######
        return weight / max(1, self.visited[location]), False
    

    def clear_memory(self) -> None:
        self.visited = defaultdict(int)
        self.stuck_steps = 0
