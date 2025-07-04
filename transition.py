from typing import Iterator, Union

class State:
    memory_list: list[str] = [
        # MAP AND PLAYER LOCATION
        'current_map_number',
        'player_x_position', 
        'player_y_position',
        'current_player_y_position_current_block',
        'current_player_x_position_current_block',
        
        # POKEMON #1 (ACTIVE POKEMON) - COMPLETE DATA
        'pokemon_1_number',
        'pokemon_1_current_hp_1',
        'pokemon_1_current_hp_2',
        'pokemon_1_max_hp_1',
        'pokemon_1_max_hp_2',
        'pokemon_1_level',
        'pokemon_1_exp_1',
        'pokemon_1_exp_2',
        'pokemon_1_exp_3',
        'pokemon_1_type_1',
        'pokemon_1_type_2',
        'pokemon_1_move_1',
        'pokemon_1_move_2',
        'pokemon_1_move_3',
        'pokemon_1_move_4',
        'pokemon_1_move_1_pp',
        'pokemon_1_move_2_pp',
        'pokemon_1_move_3_pp',
        'pokemon_1_move_4_pp',
        'pokemon_1_status',
        'pokemon_1_attack_1',
        'pokemon_1_attack_2',
        'pokemon_1_defense_1',
        'pokemon_1_defense_2',
        'pokemon_1_speed_1',
        'pokemon_1_speed_2',
        'pokemon_1_special_1',
        'pokemon_1_special_2',
        
        # POKEMON #2 - COMPLETE DATA
        'pokemon_2_number',
        'pokemon_2_current_hp_1',
        'pokemon_2_current_hp_2',
        'pokemon_2_max_hp_1',
        'pokemon_2_max_hp_2',
        'pokemon_2_level',
        'pokemon_2_exp_1',
        'pokemon_2_exp_2',
        'pokemon_2_exp_3',
        'pokemon_2_type_1',
        'pokemon_2_type_2',
        'pokemon_2_move_1',
        'pokemon_2_move_2',
        'pokemon_2_move_3',
        'pokemon_2_move_4',
        'pokemon_2_move_1_pp',
        'pokemon_2_move_2_pp',
        'pokemon_2_move_3_pp',
        'pokemon_2_move_4_pp',
        'pokemon_2_status',
        
        # POKEMON #3 - COMPLETE DATA
        'pokemon_3_number',
        'pokemon_3_current_hp_1',
        'pokemon_3_current_hp_2',
        'pokemon_3_max_hp_1',
        'pokemon_3_max_hp_2',
        'pokemon_3_level',
        'pokemon_3_exp_1',
        'pokemon_3_exp_2',
        'pokemon_3_exp_3',
        'pokemon_3_type_1',
        'pokemon_3_type_2',
        'pokemon_3_move_1',
        'pokemon_3_move_2',
        'pokemon_3_move_3',
        'pokemon_3_move_4',
        'pokemon_3_move_1_pp',
        'pokemon_3_move_2_pp',
        'pokemon_3_move_3_pp',
        'pokemon_3_move_4_pp',
        'pokemon_3_status',
        
        # POKEMON #4 - COMPLETE DATA
        'pokemon_4_number',
        'pokemon_4_current_hp_1',
        'pokemon_4_current_hp_2',
        'pokemon_4_max_hp_1',
        'pokemon_4_max_hp_2',
        'pokemon_4_level',
        'pokemon_4_exp_1',
        'pokemon_4_exp_2',
        'pokemon_4_exp_3',
        'pokemon_4_type_1',
        'pokemon_4_type_2',
        'pokemon_4_move_1',
        'pokemon_4_move_2',
        'pokemon_4_move_3',
        'pokemon_4_move_4',
        'pokemon_4_move_1_pp',
        'pokemon_4_move_2_pp',
        'pokemon_4_move_3_pp',
        'pokemon_4_move_4_pp',
        'pokemon_4_status',
        
        # POKEMON #5 - COMPLETE DATA
        'pokemon_5_number',
        'pokemon_5_current_hp_1',
        'pokemon_5_current_hp_2',
        'pokemon_5_max_hp_1',
        'pokemon_5_max_hp_2',
        'pokemon_5_level',
        'pokemon_5_exp_1',
        'pokemon_5_exp_2',
        'pokemon_5_exp_3',
        'pokemon_5_type_1',
        'pokemon_5_type_2',
        'pokemon_5_move_1',
        'pokemon_5_move_2',
        'pokemon_5_move_3',
        'pokemon_5_move_4',
        'pokemon_5_move_1_pp',
        'pokemon_5_move_2_pp',
        'pokemon_5_move_3_pp',
        'pokemon_5_move_4_pp',
        'pokemon_5_status',
        
        # POKEMON #6 - COMPLETE DATA
        'pokemon_6_number',
        'pokemon_6_current_hp_1',
        'pokemon_6_current_hp_2',
        'pokemon_6_max_hp_1',
        'pokemon_6_max_hp_2',
        'pokemon_6_level',
        'pokemon_6_exp_1',
        'pokemon_6_exp_2',
        'pokemon_6_exp_3',
        'pokemon_6_type_1',
        'pokemon_6_type_2',
        'pokemon_6_move_1',
        'pokemon_6_move_2',
        'pokemon_6_move_3',
        'pokemon_6_move_4',
        'pokemon_6_move_1_pp',
        'pokemon_6_move_2_pp',
        'pokemon_6_move_3_pp',
        'pokemon_6_move_4_pp',
        'pokemon_6_status',
        
        # PARTY INFORMATION
        'number_of_pokemon_in_party',
        
        # ENEMY POKEMON - COMPLETE DATA
        'enemy_pokemon_internal_id',
        'enemy_pokemon_current_hp_1',
        'enemy_pokemon_current_hp_2',
        'enemy_pokemon_max_hp_1',
        'enemy_pokemon_max_hp_2',
        'enemy_pokemon_level',
        'enemy_pokemon_type_1',
        'enemy_pokemon_type_2',
        'enemy_pokemon_move_1',
        'enemy_pokemon_move_2',
        'enemy_pokemon_move_3',
        'enemy_pokemon_move_4',
        'enemy_pokemon_pp_first_slot',
        'enemy_pokemon_pp_second_slot',
        'enemy_pokemon_pp_third_slot',
        'enemy_pokemon_pp_fourth_slot',
        'enemy_pokemon_status',
        'enemy_pokemon_attack_1',
        'enemy_pokemon_attack_2',
        'enemy_pokemon_defense_1',
        'enemy_pokemon_defense_2',
        'enemy_pokemon_speed_1',
        'enemy_pokemon_speed_2',
        'enemy_pokemon_special_1',
        'enemy_pokemon_special_2',
        
        # STAT MODIFIERS
        'player_pokemon_attack_modifier',
        'player_pokemon_defense_modifier',
        'player_pokemon_speed_modifier',
        'player_pokemon_special_modifier',
        'player_pokemon_accuracy_modifier',
        'player_pokemon_evasion_modifier',
        'enemy_pokemon_attack_modifier',
        'enemy_pokemon_defense_modifier',
        'enemy_pokemon_speed_modifier',
        'enemy_pokemon_special_modifier',
        'enemy_pokemon_accuracy_modifier',
        'enemy_pokemon_evasion_modifier',
        
        # BATTLE INFORMATION
        'type_of_battle',
        'battle_type',
        'number_of_turns_in_current_battle',
        'player_selected_move',
        'enemy_selected_move',
        'move_menu_type',
        'critical_hit_ohko_flag',
        'amount_of_damage_attack_is_about_to_do',
        
        # BATTLE STATUS FLAGS (PLAYER)
        'battle_status_player_byte_1',
        'battle_status_player_byte_2',
        'battle_status_player_byte_3',
        
        # BATTLE STATUS FLAGS (ENEMY)
        'battle_status_enemy_byte_1',
        'battle_status_enemy_byte_2',
        'battle_status_enemy_byte_3',
        
        # BATTLE COUNTERS
        'multi_hit_move_counter_player',
        'confusion_counter_player',
        'toxic_counter_player',
        'multi_hit_move_counter_enemy',
        'confusion_counter_enemy',
        'toxic_counter_enemy',
        
        # MONEY
        'money_1',
        'money_2',
        'money_3',
        
        # ITEMS
        'total_items',
        'item_1',
        'item_1_quantity',
        'item_2',
        'item_2_quantity',
        'item_3',
        'item_3_quantity',
        'item_4',
        'item_4_quantity',
        'item_5',
        'item_5_quantity',
        
        # BADGES
        'number_of_badges',
        
        # CURSOR AND MENU
        'cursor_x_position',
        'cursor_y_position',
        'selected_menu_item',
        'last_position_cursor_item_screen',
        'last_position_cursor_start_battle_menu',
        'item_highlighted_with_select',
        
        # FRAME COUNT (added by lua script)
        'total_frames'
    ]

    memory_map: dict[str, int]

    def __init__(self, state: str, num_actions: int, visited: set[tuple[int, int, int]] | None = None) -> None:
        parts: list[str] = state.strip().split(',')

        self.memory: tuple[float, ...] = tuple(
            [int(x) for x in parts[:-1]] + [float(num_actions)]
        )

        self.location: tuple[int, int, int] = (
            int(self.__getitem__('current_map_number')),
            int(self.__getitem__('player_x_position')) // 2,
            int(self.__getitem__('player_y_position')) // 2
        )

        if visited is None:
            self.visited: set[tuple[int, int, int]] = set([self.location])
        else:
            self.visited = visited.copy()
            self.visited.add(self.location)


    def __getitem__(self, key: str) -> float:
        return self.memory[State.memory_map[key]]
    

    def __str__(self) -> str:
        return ' '.join(str(key) + ':' + str(val) for key, val in zip(self.memory_map.keys(), self.memory))
    

State.memory_map = {State.memory_list[i]: i for i in range(len(State.memory_list))}
    

class Transition:
    def __init__(self,
                 state: State,
                 action: int,
                 next_state: State) -> None:
        self.state: State = state
        self.action: int = action
        self.next_state: State = next_state

        self.reward: float = self.calculate_reward()
        self.terminal: bool = abs(self.reward) == 1.0


    def calculate_reward(self) -> float:
        def safe_divide(numerator: float, denominator: float) -> float:
            return numerator / denominator if denominator != 0 else 0.0
        

        def get_diff(keys: list[str]) -> float:
            return (sum([self.next_state[keys[i]] * (256 ** i) for i in range(len(keys))]) -
                    sum([self.state[keys[i]] * (256 ** i) for i in range(len(keys))]))
        

        def normalize(value: float, max_diff: float) -> float:
            return max(-1.0, min(1.0, safe_divide(value, max_diff)))
        

        def is_dormant_phase() -> bool:
            return (
                self.state['player_x_position'] == self.next_state['player_x_position'] and
                self.state['player_y_position'] == self.next_state['player_y_position'] and
                self.state['cursor_x_position'] == self.next_state['cursor_x_position'] and
                self.state['cursor_y_position'] == self.next_state['cursor_y_position']
            )
        
        if is_dormant_phase():
            return 0.0
        

        whited_out: bool = all(self.next_state[f'pokemon_{i}_current_hp_1'] + 256 * self.next_state[f'pokemon_{i}_current_hp_2'] == 0 for i in range(1, 7))
        if whited_out:
            return -1.0
        gym_badge: bool = self.next_state['number_of_badges'] > 0
        if gym_badge:
            return 1.0

        battle_started: bool = self.next_state['battle_type'] != 0 and self.state['battle_type'] == 0
        battle_won: float = float(self.state['enemy_pokemon_current_hp_1'] + 256 * self.state['enemy_pokemon_current_hp_2'] > 0\
            and self.next_state['enemy_pokemon_current_hp_2'] + 256 * self.next_state['enemy_pokemon_current_hp_1'] == 0)
        
        revisited_chunk: float = float(self.state['battle_type'] == 0 and self.next_state.location in self.state.visited)

        player_hp_diff: float = sum([get_diff([f'pokemon_{i}_current_hp_2', f'pokemon_{i}_current_hp_1']) for i in range(1, 7)])
        enemy_hp_diff: float = get_diff(['enemy_pokemon_current_hp_2', 'enemy_pokemon_current_hp_1'])

        level_diff: float = get_diff(['pokemon_1_level'])
        exp_diff: float = get_diff(['pokemon_1_exp_3', 'pokemon_1_exp_2', 'pokemon_1_exp_1'])
        party_diff: float = get_diff(['number_of_pokemon_in_party'])
        money_diff: float = get_diff(['money_3', 'money_2', 'money_1'])
        battle_diff: float = float(battle_started)

        player_hp_norm: float = normalize(player_hp_diff, sum([self.next_state[f'pokemon_{i}_max_hp_2'] + 256 * self.next_state[f'pokemon_{i}_max_hp_1'] for i in range(1, int(self.next_state['number_of_pokemon_in_party'])+1)]))
        enemy_hp_norm: float = normalize(enemy_hp_diff, self.next_state['enemy_pokemon_max_hp_2'] + 256 * self.next_state['enemy_pokemon_max_hp_1'])

        level_norm: float = level_diff
        exp_norm: float = normalize(exp_diff, 5000.0)
        party_norm: float = party_diff
        money_norm: float = normalize(money_diff, 20000.0)
        battle_norm: float = battle_diff

        player_hp_weight: float = 0.20
        enemy_hp_weight: float = -0.10

        level_weight: float = 0.20
        exp_weight: float = 0.15
        party_weight: float = 0.10
        money_weight: float = 0.10
        battle_weight: float = 0.15

        revisit_weight: float = -0.10
        win_weight: float = 0.60

        reward: float = (
            player_hp_weight * player_hp_norm +
            enemy_hp_weight * enemy_hp_norm +
            level_weight * level_norm +
            exp_weight * exp_norm +
            party_weight * party_norm +
            money_weight * money_norm +
            battle_weight * battle_norm
        )
        reward += (
            revisit_weight * revisited_chunk +
            win_weight * battle_won
        )

        return max(-0.99, min(0.99, reward))
    

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