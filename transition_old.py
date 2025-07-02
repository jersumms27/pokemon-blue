from typing import Iterator, Union
import math

class State:
    memory_map: dict[str, int] = {
        'player pokemon number': 0,
        'player pokemon hp 1': 1,
        'player pokemon hp 2': 2,
        'player pokemon max hp 1': 3,
        'player pokemon max hp 2': 4,
        'player pokemon level': 5,
        'player pokemon exp 1': 6,
        'player pokemon exp 2': 7,
        'player pokemon exp 3': 8,
        'player pokemon type 1': 9,
        'player pokemon type 2': 10,
        'player pokemon move 1': 11,
        'player pokemon move 2': 12,
        'player pokemon move 3': 13,
        'player pokemon move 4': 14,
        'player pokemon move 1 pp': 15,
        'player pokemon move 2 pp': 16,
        'player pokemon move 3 pp': 17,
        'player pokemon move 4 pp': 18,
        'player selected move': 19,
        'player pokemon attack mod': 20,
        'player pokemon defense mod': 21,
        'player pokemon speed mod': 22,
        'player pokemon special mod': 23,
        'player pokemon accuracy mod': 24,
        'player pokemon evasion mod': 25,
        'player pokemon status': 26,
        'enemy pokemon number': 27,
        'enemy pokemon hp 1': 28,
        'enemy pokemon hp 2': 29,
        'enemy pokemon max hp 1': 30,
        'enemy pokemon max hp 2': 21,
        'enemy selected move': 32,
        'enemy pokemon defense mod': 33,
        'enemy pokemon speed mod': 34,
        'enemy pokemon special mod': 35,
        'enemy pokemon accuracy mod': 36,
        'enemy pokemon evasion mod': 37,
        'enemy pokemon status': 38,
        'battle type': 39,
        'number of turns': 40,
        'number of pokemon in party': 41,
        'player pokemon 2 hp 1': 42,
        'player pokemon 2 hp 2': 43,
        'player pokemon 3 hp 1': 44,
        'player pokemon 3 hp 2': 45,
        'player pokemon 4 hp 1': 46,
        'player pokemon 4 hp 2': 47,
        'player pokemon 5 hp 1': 48,
        'player pokemon 5 hp 2': 49,
        'player pokemon 6 hp 1': 50,
        'player pokemon 6 hp 2': 51,
        'money 1': 52,
        'money 2': 53,
        'money 3': 54,
        'number of items': 55,
        'number of badges': 56,
        'map number': 57,
        'player x-position': 58,
        'player y-position': 59,
        'cursor x-position': 60,
        'cursor y-position': 61,
        'selected menu item': 62,
        'menu type': 63,
        'num actions': 64
    }

    def __init__(self, state: str, num_actions) -> None:
        parts: list[str] = state.strip().split(',')

        self.memory: tuple[float, ...] = tuple(
            [int(x) for x in parts[:-1]] + [float(num_actions)] # + [float(parts[-1])]
        )


    def __getitem__(self, key: str) -> float:
        return self.memory[State.memory_map[key]]
    

    def __str__(self) -> str:
         return ' '.join(str(key) + ':' + str(val) for key, val in zip(self.memory_map.keys(), self.memory))


class Transition:
    def __init__(self, state: State,
                 action: int,
                 next_state: State,) -> None:
        self.state: State = state
        self.action: int = action
        self.next_state: State = next_state

        self.reward: float = self.calculate_reward()
        self.terminal: bool = abs(self.reward) == 1.0
    

    def calculate_reward(self) -> float:
        def safe_div(numerator: float, denominator: float) -> float:
            return numerator / denominator if denominator != 0 else 0.0
        

        def get_diff(keys: list[str]) -> float:
            return (sum([self.next_state[keys[i]] * (256 ** i) for i in range(len(keys))]) -
                    sum([self.state[keys[i]] * (256 ** i) for i in range(len(keys))]))
        

        is_reset: bool = self.next_state['num actions'] < self.state['num actions']
        battle_started: bool = self.next_state['battle type'] != 0 and self.state['battle type'] == 0
        battle_won: bool = self.state['enemy pokemon hp 2'] + 256 * self.state['enemy pokemon hp 1'] == 0

        player_hp_diff: float
        if is_reset:
            player_hp_diff = 0.0
        else:
            player_hp_diff = safe_div(get_diff(['player pokemon hp 2', 'player pokemon hp 1']),
                                      self.next_state['player pokemon max hp 2'] + 256 * self.next_state['player pokemon max hp 1'])
        enemy_hp_diff: float = safe_div(get_diff(['enemy pokemon hp 2', 'enemy pokemon hp 1']),
                                        self.next_state['enemy pokemon max hp 2'] + 256 * self.next_state['enemy pokemon max hp 1'])
        level_diff: float = get_diff(['player pokemon level'])
        exp_diff: float = get_diff(['player pokemon exp 3', 'player pokemon exp 2', 'player pokemon exp 1']) / 10000
        party_diff: float = get_diff(['number of pokemon in party'])
        money_diff: float = get_diff(['money 3', 'money 2', 'money 1']) / 1000
        battle_diff: float = float(battle_started)

        # clamping enemy hp and exp because battle/emulator resets
        # also reset enemy hp and exp, resulting in harsh negative rewards

        # clamping player hp to only benefit healing, not penalize damage
        self.reward_components = {
            'player_hp': 0.25 * max(0.0, player_hp_diff),
            'enemy_hp': max(0.0, -0.25 * enemy_hp_diff),
            'level': 0.20 * level_diff,
            'exp': 0.20 * max(0.0, exp_diff),
            'party': 0.15 * party_diff,
            'money': 0.05 * money_diff,
            'battle': 0.10 * battle_diff
        }

        if self.next_state['number of badges'] == 1:
            return 1.0
        
        whited_out = (
            self.state['player pokemon hp 1'] + 256 * self.state['player pokemon hp 2'] == 0 and
            all(
                self.state[f'player pokemon {i} hp 1'] + 256 * self.state[f'player pokemon {i} hp 2'] == 0
                for i in range(2, 7)
            )
        )
        if whited_out:
            return -1.0

        reward = sum(self.reward_components.values())
        reward = max(-1.0 + 1e-6, min(1.0 - 1e-6, reward))
        if self.state['battle type'] == 0 and self.action < 2:
            reward -= 0.10

        return reward
    

    def __iter__(self) -> Iterator[Union[State, int, float, bool]]:
        yield self.state
        yield self.action
        yield self.reward
        yield self.next_state
        yield self.terminal


    def __str__(self) -> str:
        return str({
            'state': self.state.__str__(),
            'action': self.action,
            'reward': self.reward,
            'next state': self.next_state.__str__(),
            'terminal': self.terminal
        })