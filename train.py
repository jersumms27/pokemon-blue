from dataclasses import dataclass, field
import random

import torch.nn
from torch import Tensor
from torch.optim import Optimizer
import torch.nn.functional as F

from dqn import DQN
from memory import ExperienceReplay
from transition import State, Transition
from reward import RewardTracker
import gameboy

model_path: str = 'C:/Users/jerem/Pokemon Blue DQN/model/dqn_params.pth'
epsilon_path: str = 'C:/Users/jerem/Pokemon Blue DQN/model/epsilon.txt'
reward_path: str = 'C/:Users/jerem/Pokemon Blue DQN/analysis/reward.json'

@dataclass
class TrainConfig:
    state_size: int = 215
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256, 128]) # sizes of hidden layers
    num_actions: int = 6 # A, B, --SELECT, START,-- RIGHT, LEFT, UP, DOWN
    max_epochs: int = 2
    max_episodes: int = 30 # number of iterations before training ends
    max_actions_start: int = 2000
    max_actions_incr: int = 100

    device: torch.device = torch.device('cuda')
    max_mem: int = 10000 # max number of transitions stored

    lr: float = 2e-3 # learning rate
    start_training: int = 512
    gamma: float = 0.99 # discount factor
    batch_size: int = 256
    epsilon_init: float = 1.0
    epsilon_decay: float = 0.999999
    min_epsilon: float = 0.05
    target_update: int = 2000 # how often to update target network to match q network
    learn_freq: int = 4 # how often to train q network


def train(cfg: TrainConfig) -> None:
    q_net: DQN
    target_net: DQN
    optim: Optimizer
    memory: ExperienceReplay
    q_net, target_net, optim, memory = setup_training(cfg)

    reward_tracker: RewardTracker = RewardTracker()
    reward_tracker.load_from_file(reward_path)

    epsilon: float
    try:
        with open(epsilon_path, 'r') as f:
            epsilon = float(f.read())
    except:
        epsilon = cfg.epsilon_init
    
    for epoch in range(cfg.max_epochs):
        epsilon = run_epoch(cfg, q_net, target_net, optim, memory, epsilon, reward_tracker)

        reward_tracker.finish_epoch()

        save_params(q_net, epsilon)
        reward_tracker.save_to_file(reward_path)



def setup_training(cfg: TrainConfig) -> tuple[DQN, DQN, Optimizer, ExperienceReplay]:
    q_net: DQN = DQN(cfg.state_size, cfg.num_actions).to(cfg.device)
    try:
        q_net.load_state_dict(torch.load(model_path, weights_only=True))
    except:
        pass

    target_net: DQN = DQN(cfg.state_size, cfg.num_actions).to(cfg.device)
    target_net.load_state_dict(q_net.state_dict())

    optim: Optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    memory: ExperienceReplay = ExperienceReplay(cfg.max_mem)

    return q_net, target_net, optim, memory


def run_epoch(cfg: TrainConfig,
              q_net: DQN,
              target_net: DQN,
              optim: Optimizer,
              memory: ExperienceReplay,
              epsilon: float,
              reward_tracker: RewardTracker) -> float:
    global_step: int = 0

    for episode in range(cfg.max_episodes):
        gameboy.reset_emulator()
        global_step, epsilon = run_episode(cfg, q_net, target_net, optim, memory, episode, epsilon, global_step, reward_tracker)

        reward_tracker.finish_episode()

        if (episode + 1) % 10 == 0:
            save_params(q_net, epsilon)
    
    return epsilon


def run_episode(cfg: TrainConfig,
                q_net: DQN,
                target_net: DQN,
                optim: Optimizer,
                memory: ExperienceReplay,
                episode: int,
                epsilon: float,
                global_step: int,
                reward_tracker: RewardTracker) -> tuple[int, float]:
    local_step: int = 0
    done: bool = False

    gameboy.write_action(1)
    current_state: State = gameboy.read_state(local_step)

    max_actions: int = cfg.max_actions_start + episode * cfg.max_actions_incr
    while not done and local_step < max_actions:
        action: int = select_action(cfg, q_net, current_state, epsilon)

        gameboy.write_action(action)

        next_state: State = gameboy.read_state(local_step, current_state)
        transition: Transition = Transition(current_state, action, next_state)

        done = transition.terminal

        reward_tracker.add_reward(transition.reward)

        if not transition.is_dormant:
            memory.append(transition)

        if len(memory) >= cfg.start_training and local_step % cfg.learn_freq == 0:
            batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] = get_batch(memory, cfg.batch_size)
            update_dqn(cfg, batch, q_net, target_net, optim)
        
            if global_step % cfg.target_update == 0:
                update_target(q_net, target_net)
        
        epsilon = max(cfg.min_epsilon, epsilon * cfg.epsilon_decay)

        current_state = next_state
        local_step += 1
        global_step += 1

    return global_step, epsilon


def select_action(cfg: TrainConfig,
                  net: DQN,
                  state: State,
                  epsilon: float) -> int:
    if random.random() < epsilon:
        action: int = random.randint(0, cfg.num_actions-1)
    else:
        net.eval()
        state_tensor = torch.tensor([[x / 255.0 for x in state.memory]], dtype=torch.float32).to(cfg.device)
        with torch.no_grad():
            action = int(torch.argmax(net(state_tensor)).item())
        net.train()

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
               optim: Optimizer) -> None:
    state, action, next_state, reward, terminal = batch
    state, action, next_state, reward, terminal = state.to(cfg.device), action.to(cfg.device), next_state.to(cfg.device), reward.to(cfg.device), terminal.to(cfg.device)

    with torch.no_grad():
        y_true: Tensor = reward + cfg.gamma * torch.max(target_net(next_state), dim=1).values * (1.0 - terminal.float())
    
    y_hat: Tensor = q_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
    loss: Tensor = F.mse_loss(y_hat, y_true)

    optim.zero_grad()
    loss.backward()
    optim.step()


def update_target(q_net: DQN,
                  target_net: DQN) -> None:
    target_net.load_state_dict(q_net.state_dict())


def save_params(net: DQN,
                epsilon: float) -> None:
    torch.save(net.state_dict(), model_path)

    with open(epsilon_path, 'w') as f:
        f.write(str(epsilon))


if __name__ == '__main__':
    cfg: TrainConfig = TrainConfig()
    train(cfg)

    gameboy.terminate_emulator()