import json
from matplotlib import pyplot as plt
import numpy as np


class RewardTracker:
    def __init__(self,
                 path: str | None = None) -> None:
        loaded: bool = False

        if path is not None:
            try:
                self.load_from_file(path)
                loaded = True
            except (FileNotFoundError, json.JSONDecodeError):
                loaded = False

        if not loaded:
            self.episode_rewards: list[float] = []
            self.epoch_rewards: list[float] = []
        self.current_episode_reward: float = 0.0
        self.current_epoch_reward: float = 0.0
    

    def add_reward(self,
                   reward: float) -> None:
        self.current_episode_reward += reward
        self.current_epoch_reward += reward
    

    def finish_episode(self) -> float:
        episode_total: float = self.current_episode_reward
        self.episode_rewards.append(episode_total)
        self.current_episode_reward = 0.0

        return episode_total
    

    def finish_epoch(self) -> float:
        epoch_total: float = self.current_epoch_reward
        self.epoch_rewards.append(epoch_total)
        self.current_epoch_reward = 0.0

        return epoch_total
    

    def get_episode_avg(self,
                        n: int = 10) -> float:
        if not self.episode_rewards:
            return 0.0
        recent: list[float] = self.episode_rewards[-n:]

        return sum(recent) / len(recent)
    

    def save_to_file(self,
                     path: str) -> None:
        data = {
            'episode_rewards': self.episode_rewards,
            'epoch_rewards': self.epoch_rewards,
            'total_episodes': len(self.episode_rewards),
            'total_epochs': len(self.epoch_rewards)
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=3)
    

    def load_from_file(self,
                       path: str) -> None:
        with open(path, 'r') as f:
            data = json.load(f)
            self.episode_rewards = data.get('episode_rewards', [])
            self.epoch_rewards = data.get('epoch_rewards', [])
    

    def clear_data(self) -> None:
        self.episode_rewards.clear()
        self.epoch_rewards.clear()
        self.current_episode_reward = 0.0
        self.current_epoch_reward = 0.0


    def plot_episode_results(self, 
                           start: int | None = None,
                           end: int | None = None,
                           show_stats: bool = True,
                           show_trend: bool = True,
                           figsize: tuple[int, int] = (12, 8)) -> None:
        """Enhanced episode reward plotting with statistics and trend analysis"""
        
        start = start or 0
        end = end or len(self.episode_rewards)
        
        if start >= len(self.episode_rewards):
            print("Start index exceeds available data")
            return
            
        data = self.episode_rewards[start:end]
        x_vals = list(range(start, start + len(data)))
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Main plot
        ax1.plot(x_vals, data, color='#2E86AB', alpha=0.8, linewidth=1.5, label='Episode Rewards')
        
        # Add moving average if enough data
        if len(data) >= 10 and show_trend:
            window = min(20, len(data) // 4)
            moving_avg = []
            for i in range(window - 1, len(data)):
                avg = sum(data[i - window + 1:i + 1]) / window
                moving_avg.append(avg)
            
            ma_x = x_vals[window - 1:]
            ax1.plot(ma_x, moving_avg, color='#F18F01', linewidth=2.5, 
                    label=f'{window}-Episode Moving Average')
        
        # Add horizontal reference lines
        if data:
            mean_reward = np.mean(data)
            max_reward = max(data)
            min_reward = min(data)
            
            ax1.axhline(y=mean_reward, color='#C73E1D', linestyle='--', alpha=0.7, 
                       label=f'Mean: {mean_reward:.3f}')
            ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
            
            # Highlight extreme values
            ax1.axhline(y=max_reward, color='green', linestyle=':', alpha=0.6, 
                       label=f'Max: {max_reward:.3f}')
            ax1.axhline(y=min_reward, color='red', linestyle=':', alpha=0.6, 
                       label=f'Min: {min_reward:.3f}')
        
        # Styling
        ax1.set_title('Pokemon Blue DQN - Episode Rewards', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Cumulative Reward', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', framealpha=0.9)
        
        # Add statistics text box
        if show_stats and data:
            std_reward = np.std(data)
            recent_avg = np.mean(data[-10:]) if len(data) >= 10 else np.mean(data)
            
            stats_text = f"""Episode Range: {start} - {start + len(data) - 1}
            Episodes: {len(data)}
            Mean: {mean_reward:.3f}
            Std: {std_reward:.3f}
            Recent 10-ep avg: {recent_avg:.3f}"""
            
            ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Histogram of rewards (bottom subplot)
        if data:
            ax2.hist(data, bins=min(30, len(data) // 3), color='#2E86AB', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Reward Value', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Reward Distribution', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


    def plot_epoch_results(self,
                          start: int | None = None,
                          end: int | None = None,
                          show_progress: bool = True,
                          figsize: tuple[int, int] = (10, 6)) -> None:
        """Enhanced epoch reward plotting with progress indicators"""
        
        start = start or 0
        end = end or len(self.epoch_rewards)
        
        if start >= len(self.epoch_rewards):
            print("Start index exceeds available data")
            return
            
        data = self.epoch_rewards[start:end]
        x_vals = list(range(start + 1, start + len(data) + 1))  # Epochs start from 1
        
        plt.figure(figsize=figsize)
        
        # Bar plot for epoch rewards
        bars = plt.bar(x_vals, data, color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, data)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(data)*0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add trend line if enough data
        if len(data) >= 3 and show_progress:
            z = np.polyfit(x_vals, data, 1)
            p = np.poly1d(z)
            plt.plot(x_vals, p(x_vals), color='#F18F01', linewidth=3, 
                    label=f'Trend (slope: {z[0]:.3f})')
            
            # Color bars based on trend
            for i, bar in enumerate(bars):
                if i > 0:
                    if data[i] > data[i-1]:
                        bar.set_color('#2E8B57')  # Green for improvement
                    elif data[i] < data[i-1]:
                        bar.set_color('#CD5C5C')  # Red for decline
        
        # Add horizontal reference line at 0
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        
        # Statistics
        if data:
            mean_epoch = np.mean(data)
            plt.axhline(y=mean_epoch, color='red', linestyle='--', alpha=0.7, 
                       label=f'Mean: {mean_epoch:.3f}')
        
        # Styling
        plt.title('Pokemon Blue DQN - Epoch Rewards', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Total Epoch Reward', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        if show_progress and len(data) >= 3:
            plt.legend()
        
        # Add epoch numbers as x-tick labels
        plt.xticks(x_vals)
        
        plt.tight_layout()
        plt.show()


    def plot_with_moving_average(self, window: int = 10) -> None:
        if len(self.episode_rewards) < window:
            return
        
        moving_avg = [self.get_episode_avg(window) for i in range(window-1, len(self.episode_rewards))]
        
        plt.plot(self.episode_rewards, alpha=0.3, label='Raw')
        plt.plot(range(window-1, len(self.episode_rewards)), moving_avg, label=f'{window}-episode average')
        plt.legend()
        plt.show()


    def plot_training_summary(self, figsize: tuple[int, int] = (15, 10)) -> None:
        """Comprehensive training summary with multiple visualizations"""
        
        if not self.episode_rewards:
            print("No episode data to plot")
            return
            
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[2, 1, 1])
        
        # Main episode plot
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.episode_rewards, color='#2E86AB', alpha=0.6, linewidth=1)
        
        # Add moving averages
        if len(self.episode_rewards) >= 20:
            ma_20 = []
            for i in range(19, len(self.episode_rewards)):
                ma_20.append(np.mean(self.episode_rewards[i-19:i+1]))
            ax1.plot(range(19, len(self.episode_rewards)), ma_20, 
                    color='#F18F01', linewidth=2, label='20-Episode MA')
        
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax1.set_title('Episode Rewards Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Epoch summary
        if self.epoch_rewards:
            ax2 = fig.add_subplot(gs[0, 2])
            bars = ax2.bar(range(1, len(self.epoch_rewards) + 1), self.epoch_rewards, 
                          color='#A23B72', alpha=0.8)
            ax2.set_title('Epoch Totals', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Total Reward')
            ax2.grid(True, alpha=0.3)
        
        # Reward distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(self.episode_rewards, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax3.set_title('Reward Distribution', fontsize=12)
        ax3.set_xlabel('Reward Value')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Recent performance
        ax4 = fig.add_subplot(gs[1, 1])
        recent_data = self.episode_rewards[-20:] if len(self.episode_rewards) >= 20 else self.episode_rewards
        ax4.plot(recent_data, color='#C73E1D', linewidth=2, marker='o', markersize=4)
        ax4.set_title('Last 20 Episodes', fontsize=12)
        ax4.set_xlabel('Recent Episode')
        ax4.set_ylabel('Reward')
        ax4.grid(True, alpha=0.3)
        
        # Statistics
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        stats_text = f"""Training Statistics
        
        Total Episodes: {len(self.episode_rewards)}
        Total Epochs: {len(self.epoch_rewards)}

        Mean Reward: {np.mean(self.episode_rewards):.3f}
        Std Deviation: {np.std(self.episode_rewards):.3f}
        Max Reward: {max(self.episode_rewards):.3f}
        Min Reward: {min(self.episode_rewards):.3f}

        Recent 10-ep avg: {np.mean(self.episode_rewards[-10:]):.3f}"""
        
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()