import time, os

from transition import State

action_path: str = 'C:/Users/jerem/Pokemon Blue DQN/communication/action.txt'
state_path: str = 'C:/Users/jerem/Pokemon Blue DQN/communication/state.txt'
reset_path: str = 'C:/Users/jerem/Pokemon Blue DQN/communication/reset.txt'
exit_path: str = 'C:/Users/jerem/Pokemon Blue DQN/communication/exit.txt'

action_ready_path: str = 'C:/Users/jerem/Pokemon Blue DQN/communication/action_ready.txt'
state_ready_path: str = 'C:/Users/jerem/Pokemon Blue DQN/communication/state_ready.txt'


def write_action(action: int) -> None:
    # wait to write action until Lua is ready to process
    while not os.path.exists(action_ready_path):
        time.sleep(0.001)
    
    if action >=2:
        action += 2
    
    # write action to action.txt
    with open(action_path, 'w') as f:
        f.write(str(action))
    
    # signal that Python has written action
    for _ in range(10):
        try:
            os.remove(action_ready_path)
            break
        except PermissionError:
            time.sleep(0.001)


def read_state(num_actions: int, prev_state: State | None = None) -> State:
    # wait until Lua has written state
    while not os.path.exists(state_ready_path):
        time.sleep(0.001)
    
    # open state.txt to read in state
    for _ in range(10):
        try:
            with open(state_path, 'r') as f:
                state: str = f.read()
            break
        except PermissionError:
            time.sleep(0.001)
    else:
        raise RuntimeError('Failed to read state.txt')
    
    # delete state-related files
    for path in [state_path, state_ready_path]:
        for _ in range(10):
            try:
                os.remove(path)
                break
            except PermissionError:
                time.sleep(0.001)
    if prev_state is None:
        return State(state, num_actions)
    return State(state, num_actions, prev_state.visited)


def reset_emulator() -> None:
    with open(reset_path, 'w') as f:
        f.write('reset')
    time.sleep(0.25)

    with open(action_ready_path, 'w') as f:
        f.write('1')


def terminate_emulator() -> None:
    reset_emulator()
    with open(exit_path, 'w') as f:
        f.write('exit')
    
    time.sleep(2)
    os.remove(action_ready_path)
    os.remove(exit_path)
    os.remove(reset_path)