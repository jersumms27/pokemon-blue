from dqn import DQN
from memory import ExperienceReplay, ExplorationTracker
import gameboy
from transition import Transition, State

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import Optimizer
import random
from matplotlib import pyplot as plt
from collections import defaultdict

state_size: int = 65
hidden_sizes: list[int] = [256, 256, 128] # sizes of hidden layers
num_actions: int = 6 # A, B, --SELECT, START,-- RIGHT, LEFT, UP, DOWN
max_epochs: int = 30 # number of iterations before training ends
max_actions_start: int = 500
max_actions_incr: int = 0

device: torch.device = torch.device('cuda')
max_mem: int = 10000 # max number of transitions stored

model_path: str = 'C:/Users/jerem/Pokemon Blue DQN/model/dqn_params.pth'
epsilon_path: str = 'C:/Users/jerem/Pokemon Blue DQN/model/epsilon.txt'
reward_path: str = 'C:/Users/jerem/Pokemon Blue DQN/data/reward_plot.png'
component_path: str = 'C:/Users/jerem/Pokemon Blue DQN/data/component_plot.png'

lr: float = 1e-4 # learning rate
start_training: int = 128
gamma: float = 0.85 # discount factor
batch_size: int = 128
epsilon: float = 1.0
epsilon_decay: float = 0.999
min_epsilon: float = 0.025
target_update: int = 1000 # how often to update target network to match q network
learn_freq: int = 5 # how often to train q network
explore_weight: float = 0.20


def train(epsilon: float, epsilon_decay: float) -> None:
    try:
        with open(epsilon_path, 'r') as f:
            epsilon = float(f.read())
    except:
        pass

    memory: ExperienceReplay = ExperienceReplay(max_mem)

    q_net: DQN = DQN(state_size, num_actions).to(device)
    try:
        q_net.load_state_dict(torch.load(model_path, weights_only=True))
    except:
        pass

    target_net: DQN = DQN(state_size, num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optim: Optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    explore_tracker: ExplorationTracker = ExplorationTracker()

    rewards: list[float] = []

    for epoch in range(max_epochs):
        component_totals: dict[str, float] = defaultdict(float)
        explore_tracker.clear_memory()
        total_reward: float = 0.0

        gameboy.reset_emulator()

        actions_taken: int = 0
        gameboy.write_action(1)
        current_state: State = gameboy.read_state(actions_taken)
        done: bool = False

        max_actions: int = max_actions_incr * epoch + max_actions_start
        while not done and actions_taken < max_actions:
            action: int
            action, epsilon = select_action(q_net, current_state, epsilon, epsilon_decay, update_epsilon=len(memory) >= start_training)

            gameboy.write_action(action)
            next_state: State = gameboy.read_state(actions_taken)
            transition: Transition = Transition(current_state, action, next_state)

            explore_reward: float
            explore_terminal: bool
            explore_reward, explore_terminal = explore_tracker.calculate_reward(next_state, current_state, explore_weight, action > 1)
            transition.reward += explore_reward
            transition.terminal = explore_terminal or transition.terminal

            for k, v in transition.reward_components.items():
                component_totals[k] += v

            done = transition.terminal
            total_reward += transition.reward
            memory.append(transition)

            if len(memory) >= start_training and actions_taken % learn_freq == 0:
                batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] = get_batch(memory, batch_size)
                update_dqn(batch, optim, q_net, target_net, gamma, num_actions, target_update)

            current_state = next_state
            actions_taken += 1
        
        rewards.append(total_reward)
        if epoch == 0:
            component_log = {k: [v] for k, v in component_totals.items()}
        else:
            for k in component_totals:
                component_log[k].append(component_totals[k])
        
        save_params(q_net, epsilon)

    plt.figure()
    for comp, vals in component_log.items():
        plt.plot(range(len(vals)), vals, label=comp)

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Component Reward")
    plt.title("Reward Components per Epoch")
    plt.grid()
    plt.tight_layout()
    plt.savefig(component_path)

    plt.figure()
    plt.plot(rewards, color='green')
    plt.title('Cumulative reward per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cumulative reward')
    plt.savefig(reward_path)


def select_action(net: DQN, state: State, epsilon: float, epsilon_decay: float, update_epsilon: bool) -> tuple[int, float]:
    if random.random() < epsilon:
        action: int = random.randint(0, num_actions - 1)
    else:
        state_tensor = torch.tensor([[x / 255.0 for x in state.memory]], dtype=torch.float32).to(device)
        with torch.no_grad():
            action = int(torch.argmax(net(state_tensor)).item())

    if update_epsilon:
        epsilon = max(epsilon * epsilon_decay, min_epsilon)
    return action, epsilon


def get_batch(memory: ExperienceReplay, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    state: list[State] = []
    action: list[int] = []
    next_state: list[State] = []
    reward: list[float] = []
    terminal: list[bool] = []

    batch: list[Transition] = memory.get_random_sample(batch_size)
    s: State
    a: int
    r: float
    ns: State
    t: bool

    for s, a, r, ns, t in batch:
        state.append(s)
        action.append(a)
        reward.append(r)
        next_state.append(ns)
        terminal.append(t)
    
    return (
        torch.stack([torch.tensor([x / 255.0 for x in s.memory], dtype=torch.float32) for s in state]),
        torch.tensor(action, dtype=torch.int64),
        torch.stack([torch.tensor([x / 255.0 for x in ns.memory], dtype=torch.float32) for ns in next_state]),
        torch.tensor(reward, dtype=torch.float32),
        torch.tensor(terminal, dtype=torch.bool)
    )


def update_dqn(batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
               optim: Optimizer,
               q_net: DQN,
               target_net: DQN,
               gamma: float,
               global_step: int,
               target_update: int) -> None:
    
    state, action, next_state, reward, terminal = batch
    state = state.to(device)
    action = action.to(device)
    next_state = next_state.to(device)
    reward = reward.to(device)
    terminal = terminal.to(device)

    with torch.no_grad():
        y: torch.Tensor = reward + gamma * torch.max(target_net(next_state), dim=1).values * (1.0 - terminal.float())
    
    y_hat: Tensor = q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
    loss: Tensor = F.mse_loss(y_hat, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if global_step % target_update == 0:
        target_net.load_state_dict(q_net.state_dict())


def save_params(net: DQN, epsilon: float) -> None:
    torch.save(net.state_dict(), model_path)

    with open(epsilon_path, 'w') as f:
        f.write(str(epsilon))


if __name__ == '__main__':
    train(epsilon, epsilon_decay)
    gameboy.terminate_emulator()