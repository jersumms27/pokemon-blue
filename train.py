from dataclasses import dataclass
import random

import torch.nn
from torch import Tensor
from torch.optim import Optimizer
import torch.nn.functional as F

from dqn import DQN
from memory import ExperienceReplay
from transition import State, Transition
import gameboy

@dataclass
class TrainConfig:
    state_size: int = 65
    hidden_sizes: list[int] = [256, 256, 128] # sizes of hidden layers
    num_actions: int = 6 # A, B, --SELECT, START,-- RIGHT, LEFT, UP, DOWN
    max_epochs: int = 30 # number of iterations before training ends
    max_actions_start: int = 500
    max_actions_incr: int = 0

    device: torch.device = torch.device('cuda')
    max_mem: int = 10000 # max number of transitions stored

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


def setup_training(cfg: TrainConfig) -> tuple[DQN, DQN, Optimizer, ExperienceReplay]:
    q_net: DQN = DQN(cfg.state_size, cfg.num_actions).to(cfg.device)
    target_net: DQN = DQN(cfg.state_size, cfg.num_actions).to(cfg.device)
    target_net.load_state_dict(q_net.state_dict())

    optim: Optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    memory: ExperienceReplay = ExperienceReplay(cfg.max_mem)

    return q_net, target_net, optim, memory


def run_epoch(cfg: TrainConfig,
              q_net: DQN,
              target_net: DQN,
              optim: Optimizer,
              memory: ExperienceReplay) -> None:
    for epoch in range(cfg.max_epochs):
        gameboy.reset_emulator()
        run_episode(cfg, q_net, target_net, optim, memory)


def run_episode(cfg: TrainConfig,
                q_net: DQN,
                target_net: DQN,
                optim: Optimizer,
                memory: ExperienceReplay) -> None:
    actions_taken: int = 0
    done: bool = False

    gameboy.write_action(1)
    current_state: State = gameboy.read_state(actions_taken)

    max_actions: int = 0
    while not done and actions_taken < max_actions:
        action: int
        action = select_action(cfg, q_net, current_state)

        gameboy.write_action(action)

        next_state: State = gameboy.read_state(actions_taken)
        transition: Transition = Transition(current_state, action, next_state)

        done = transition.terminal
        memory.append(transition)

        if len(memory) >= cfg.start_training and actions_taken % cfg.learn_freq == 0:
            batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] = get_batch(memory, cfg.batch_size)
            update_dqn(cfg, batch, q_net, target_net, optim, cfg.num_actions, cfg.target_update)

        current_state = next_state
        actions_taken += 1


def select_action(cfg: TrainConfig,
                  net: DQN,
                  state: State) -> int:
    if random.random() < cfg.epsilon:
        action: int = random.randint(0, cfg.num_actions-1)
    else:
        state_tensor = torch.tensor([[x / 255.0 for x in state.memory]], dtype=torch.float32).to(cfg.device)
        with torch.no_grad():
            action = int(torch.argmax(net(state_tensor)).item())
    
    cfg.epsilon = max(cfg.epsilon * cfg.epsilon_decay, cfg.min_epsilon)

    return action


def get_batch(memory: ExperienceReplay,
              batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    state: list[State] = []
    action: list[int] = []
    next_state: list[State] = []
    reward: list[float] = []
    terminal: list[bool] = []

    batch: list[Transition] = memory.get_random_sample(batch_size)
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


def update_dqn(cfg: TrainConfig,
               batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
               q_net: DQN,
               target_net: DQN,
               optim: Optimizer,
               global_step: int,
               target_update: int) -> None:
    state, action, next_state, reward, terminal = batch
    state, action, next_state, reward, terminal = state.to(cfg.device), action.to(cfg.device), next_state.to(cfg.device), reward.to(cfg.device), terminal.to(cfg.device)

    with torch.no_grad():
        y_true: Tensor = reward + cfg.gamma * torch.max(target_net(next_state), dim=1).values * (1.0 - terminal.float())
    
    y_hat: Tensor = q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
    loss: Tensor = F.mse_loss(y_hat, y_true)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if global_step % target_update == 0:
        target_net.load_state_dict(q_net.state_dict())