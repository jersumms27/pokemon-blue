import random

from transition import Transition

class ExperienceReplay():
    def __init__(self,
                 max_mem: int = 10000) -> None:
        self.max_mem: int = max_mem
        self.memory: list[Transition] = []
    

    def append(self,
               replay: Transition) -> None:
        if len(self.memory) >= self.max_mem:
            self.memory.pop(0)
        self.memory.append(replay)
    

    # (s, a, r, s', d)
    def get_random_sample(self,
                          batch_size: int) -> list[Transition]:
        return random.sample(self.memory,
                             min(batch_size, len(self.memory)))


    def __len__(self) -> int:
        return len(self.memory)


    def __str__(self) -> str:
        return '\n'.join([str(transition) for transition in self.memory])