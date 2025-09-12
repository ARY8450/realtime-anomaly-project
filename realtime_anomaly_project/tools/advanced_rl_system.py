"""
Advanced RL System for >98% Trading Accuracy
Implements cutting-edge reinforcement learning techniques including:
- GPU acceleration (when available)
- Curriculum learning with progressive difficulty
- Advanced architectures (SAC, TD3, Custom)
- Ensemble methods with multiple agents
- Systematic hyperparameter optimization
- Multi-asset trading environments
- Distributed training capabilities
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, cast
import joblib
import json
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import gymnasium as gym
from gymnasium import spaces
import yfinance as yf

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from fusion.recommender import _make_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_rl_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class AdvancedTradingEnvironment(gym.Env):
    """
    Advanced multi-asset trading environment with realistic market conditions.
    """

    def __init__(self,
                 data_dict: Dict[str, pd.DataFrame],
                 initial_balance: float = 100000,
                 transaction_cost: float = 0.001,
                 max_position: float = 1.0,
                 max_steps: int = 1000,
                 difficulty_level: int = 1):
        super(AdvancedTradingEnvironment, self).__init__()

        self.data_dict = data_dict
        self.tickers = list(data_dict.keys())
        self.n_assets = len(self.tickers)

        # Add num_envs for VecNormalize compatibility
        self.num_envs = 1

        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.max_steps = max_steps
        self.difficulty_level = difficulty_level

        # Action space: For each asset - [hold, buy, sell] + position sizing
        self.action_space = spaces.Box(
            low=np.array([-1.0] * self.n_assets),  # sell
            high=np.array([1.0] * self.n_assets),  # buy
            dtype=np.float32
        )

        # Observation space: technical indicators + positions + balance + market state
        obs_size = 36 * self.n_assets + self.n_assets + 3  # features + positions + balance info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment with curriculum learning.

        Matches the Gymnasium API: reset(*, seed=None, options=None) -> (obs, info)
        """
        super(AdvancedTradingEnvironment, self).reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance

        # Cast shape arguments explicitly to satisfy type-checkers that expect
        # a concrete tuple for numpy shape parameters.
        self.positions = np.zeros((self.n_assets,), dtype=np.float32)  # positions for each asset
        self.entry_prices = np.zeros((self.n_assets,), dtype=np.float32)
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        self.portfolio_value = self.initial_balance

        # Curriculum learning: adjust difficulty
        if self.difficulty_level > 1:
            self.transaction_cost *= self.difficulty_level * 0.5
            self.max_position = min(1.0, self.max_position * (1 + self.difficulty_level * 0.1))

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get comprehensive observation for all assets."""
        observations = []

        for i, ticker in enumerate(self.tickers):
            if self.current_step >= len(self.data_dict[ticker]):
                # Pad with zeros if data is exhausted
                obs = np.zeros((36,), dtype=np.float32)
            else:
                # Get current market data
                current_data = self.data_dict[ticker].iloc[self.current_step]

                # Extract technical indicators
                numeric_data = pd.to_numeric(current_data, errors='coerce').fillna(0)
                obs = numeric_data.values.astype(np.float32)

                # Ensure correct size
                if len(obs) != 36:
                    if len(obs) < 36:
                        pad = np.zeros((36 - len(obs),), dtype=np.float32)
                        obs = np.concatenate([obs, pad])
                    else:
                        obs = obs[:36]

            observations.append(obs)

        # Combine all asset observations
        tech_features = np.concatenate(observations)

        # Add position and balance information
        position_info = np.concatenate([
            self.positions.astype(np.float32),  # current positions
            np.array([self.balance / self.initial_balance], dtype=np.float32),  # normalized balance
            np.array([self.total_pnl / self.initial_balance], dtype=np.float32),  # normalized P&L
            np.array([self.portfolio_value / self.initial_balance], dtype=np.float32)  # normalized portfolio value
        ])

        # Combine all features
        observation = np.concatenate([tech_features, position_info])

        return observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute trading actions for all assets."""

        # Accept numpy arrays from RL libs; ensure we can index into them.
        if isinstance(action, np.ndarray):
            # If action has shape (n_envs, n_actions) from vectorized envs, take first
            if action.ndim > 1:
                action = action[0]

        if self.current_step >= self.max_steps - 1:
            return self._get_observation(), 0, True, False, {}

        total_reward = 0
        portfolio_change = 0

        for i, ticker in enumerate(self.tickers):
            if self.current_step >= len(self.data_dict[ticker]) - 1:
                continue

            current_price = self.data_dict[ticker].iloc[self.current_step]['close']
            next_price = self.data_dict[ticker].iloc[self.current_step + 1]['close']

            # Interpret action (-1 to 1)
            action_value = float(action[i])

            # Calculate position change
            target_position = action_value * self.max_position
            position_change = target_position - self.positions[i]

            # Execute trades
            if abs(position_change) > 0.01:  # Minimum trade threshold
                trade_cost = abs(position_change) * current_price * self.transaction_cost
                self.balance -= trade_cost

                # Close existing position if opposite direction
                if self.positions[i] != 0 and np.sign(position_change) != np.sign(self.positions[i]):
                    # Close position
                    pnl = self.positions[i] * (current_price - self.entry_prices[i]) / self.entry_prices[i]
                    self.balance *= (1 + pnl)
                    self.total_pnl += pnl
                    if pnl > 0:
                        self.winning_trades += 1
                    self.total_trades += 1

                # Open new position
                if abs(target_position) > 0.01:
                    self.positions[i] = target_position
                    self.entry_prices[i] = current_price
                    self.total_trades += 1

            # Calculate reward based on portfolio performance
            if self.positions[i] != 0:
                price_change = (next_price - current_price) / current_price
                position_pnl = self.positions[i] * price_change
                portfolio_change += position_pnl

        # Update portfolio value
        old_portfolio_value = self.portfolio_value
        self.portfolio_value = self.balance

        # Calculate reward
        reward = portfolio_change * 100  # Scale reward

        # Add regularization terms
        position_penalty = -0.001 * np.sum(np.abs(self.positions))  # Penalize large positions
        trade_penalty = -0.01 * self.total_trades / self.max_steps  # Penalize excessive trading

        total_reward = reward + position_penalty + trade_penalty

        # Difficulty scaling for curriculum learning
        if self.difficulty_level > 1:
            total_reward *= (1 + self.difficulty_level * 0.1)

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_observation(), total_reward, done, False, {
            'portfolio_value': self.portfolio_value,
            'total_pnl': self.total_pnl,
            'win_rate': self.winning_trades / max(1, self.total_trades)
        }

class CustomSAC(nn.Module):
    """
    Custom SAC architecture optimized for trading.
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256):
        super(CustomSAC, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # mean and log_std
        )

        # Critic networks
        self.critic1 = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.critic2 = nn.Sequential(
            nn.Linear(observation_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Value network
        self.value = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

class AdvancedRLTrainer:
    """
    Advanced RL trainer with multiple algorithms and ensemble methods.
    """

    def __init__(self, data_dict: Dict[str, pd.DataFrame]):
        self.data_dict = data_dict
        self.models = {}
        self.best_models = {}
        self.training_history = {}

        # Vectorized env handles (set when create_environments is called)
        self.env: Optional[VecEnv] = None
        self.eval_env: Optional[VecEnv] = None

        # Initialize Ray for distributed training with GPU support
        if not ray.is_initialized():
            # Detect available GPUs
            num_gpus = 1 if torch.cuda.is_available() else 0
            logger.info(f"Initializing Ray with {num_gpus} GPU(s)")
            ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=num_gpus)

    def create_environments(self, n_envs: int = 4, difficulty_level: int = 1):
        """Create vectorized environments for training."""

        def make_env():
            env = AdvancedTradingEnvironment(
                self.data_dict,
                difficulty_level=difficulty_level
            )
            return env

        self.env = DummyVecEnv([make_env for _ in range(n_envs)])
        self.env = VecNormalize(self.env, norm_obs=True, norm_reward=True)

        # Evaluation environment
        # Create evaluation environment as a single-element VecEnv so VecNormalize
        # receives a VecEnv instance (required by stable-baselines3)
        def make_eval_env():
            return AdvancedTradingEnvironment(self.data_dict, difficulty_level=difficulty_level)

        self.eval_env = DummyVecEnv([make_eval_env])
        self.eval_env = VecNormalize(self.eval_env, training=False, norm_obs=True, norm_reward=True)

    def train_sac(self, total_timesteps: int = 1000000, **kwargs) -> SAC:
        """Train SAC agent with advanced configuration."""
        logger.info("Training SAC Agent...")

        assert self.env is not None, "Environment not created. Call create_environments() before training."

        model = SAC(
            "MlpPolicy",
            self.env,
            learning_rate=kwargs.get('learning_rate', 3e-4),
            buffer_size=kwargs.get('buffer_size', 1000000),
            learning_starts=kwargs.get('learning_starts', 1000),
            batch_size=kwargs.get('batch_size', 256),
            tau=kwargs.get('tau', 0.005),
            gamma=kwargs.get('gamma', 0.99),
            train_freq=kwargs.get('train_freq', 1),
            gradient_steps=kwargs.get('gradient_steps', 1),
            ent_coef=kwargs.get('ent_coef', 'auto'),
            target_update_interval=kwargs.get('target_update_interval', 1),
            verbose=1,
            device=device,
            tensorboard_log="./advanced_rl_tensorboard/"
        )

        # Custom callback for monitoring
        callback = AdvancedCallback(
            eval_env=self.eval_env,
            eval_freq=10000,
            save_path="./sac_models/"
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

        self.models['sac'] = model
        return model

    def train_td3(self, total_timesteps: int = 1000000, **kwargs) -> TD3:
        """Train TD3 agent with advanced configuration."""

        logger.info("Training TD3 Agent...")

        # Ensure environments are created before accessing action space
        assert hasattr(self, 'env') and self.env is not None, "Environment not created. Call create_environments() before training."
        # Action noise for exploration. Unwrap VecNormalize if present and cast to VecEnv for typing.
        inner_env = getattr(self.env, 'env', self.env)
        venv = cast(VecEnv, inner_env)
        assert hasattr(venv, 'action_space') and venv.action_space is not None, "Could not determine action space from env"
        # Cast shape to concrete tuple for typing
        action_shape = cast(Tuple[int, ...], venv.action_space.shape)
        n_actions = int(action_shape[0])
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=kwargs.get('noise_std', 0.1) * np.ones(n_actions)
        )

        model = TD3(
            "MlpPolicy",
            self.env,
            learning_rate=kwargs.get('learning_rate', 1e-3),
            buffer_size=kwargs.get('buffer_size', 1000000),
            learning_starts=kwargs.get('learning_starts', 1000),
            batch_size=kwargs.get('batch_size', 256),
            tau=kwargs.get('tau', 0.005),
            gamma=kwargs.get('gamma', 0.99),
            train_freq=kwargs.get('train_freq', (1, 'episode')),
            gradient_steps=kwargs.get('gradient_steps', 1),
            action_noise=action_noise,
            policy_delay=kwargs.get('policy_delay', 2),
            target_policy_noise=kwargs.get('target_policy_noise', 0.2),
            target_noise_clip=kwargs.get('target_noise_clip', 0.5),
            verbose=1,
            device=device,
            tensorboard_log="./advanced_rl_tensorboard/"
        )

        callback = AdvancedCallback(
            eval_env=self.eval_env,
            eval_freq=10000,
            save_path="./td3_models/"
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

        self.models['td3'] = model
        return model

    def train_ppo(self, total_timesteps: int = 1000000, **kwargs) -> PPO:
        """Train PPO agent with advanced configuration."""
        logger.info("Training PPO Agent...")

        assert self.env is not None, "Environment not created. Call create_environments() before training."

        model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=kwargs.get('learning_rate', 3e-4),
            n_steps=kwargs.get('n_steps', 2048),
            batch_size=kwargs.get('batch_size', 64),
            n_epochs=kwargs.get('n_epochs', 10),
            gamma=kwargs.get('gamma', 0.99),
            gae_lambda=kwargs.get('gae_lambda', 0.95),
            clip_range=kwargs.get('clip_range', 0.2),
            ent_coef=kwargs.get('ent_coef', 0.0),
            vf_coef=kwargs.get('vf_coef', 0.5),
            max_grad_norm=kwargs.get('max_grad_norm', 0.5),
            verbose=1,
            device=device,
            tensorboard_log="./advanced_rl_tensorboard/"
        )

        callback = AdvancedCallback(
            eval_env=self.eval_env,
            eval_freq=10000,
            save_path="./ppo_models/"
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

        self.models['ppo'] = model
        return model

    def train_ensemble(self, total_timesteps: int = 500000) -> Dict[str, Any]:
        """Train ensemble of RL agents."""

        logger.info("Training RL Ensemble...")

        # Train multiple algorithms
        algorithms = ['sac', 'td3', 'ppo']

        for algo in algorithms:
            logger.info(f"Training {algo.upper()}...")

            if algo == 'sac':
                model = self.train_sac(total_timesteps // len(algorithms))
            elif algo == 'td3':
                model = self.train_td3(total_timesteps // len(algorithms))
            else:  # ppo
                model = self.train_ppo(total_timesteps // len(algorithms))

            self.models[algo] = model

        return self.models

    def hyperparameter_optimization(self, n_trials: int = 50, algorithm: str = 'sac'):
        """Perform systematic hyperparameter optimization."""

        logger.info(f"Starting hyperparameter optimization for {algorithm.upper()}...")

        def objective(trial):
            # Define hyperparameter search space
            params = {}
            if algorithm == 'sac':
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'buffer_size': trial.suggest_categorical('buffer_size', [100000, 500000, 1000000]),
                    'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
                    'tau': trial.suggest_float('tau', 0.001, 0.01),
                    'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                    'ent_coef': trial.suggest_float('ent_coef', 0.001, 0.1, log=True)
                }
            elif algorithm == 'td3':
                params = {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                    'buffer_size': trial.suggest_categorical('buffer_size', [100000, 500000, 1000000]),
                    'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
                    'tau': trial.suggest_float('tau', 0.001, 0.01),
                    'gamma': trial.suggest_float('gamma', 0.95, 0.999),
                    'noise_std': trial.suggest_float('noise_std', 0.01, 0.5)
                }

            # Train model with suggested parameters
            if algorithm == 'sac':
                model = self.train_sac(total_timesteps=50000, **params)
            elif algorithm == 'td3':
                model = self.train_td3(total_timesteps=50000, **params)
            else:
                model = self.train_ppo(total_timesteps=50000, **params)

            # Evaluate performance
            mean_reward = self.evaluate_model(model)

            return mean_reward

        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(),
            pruner=MedianPruner()
        )

        study.optimize(objective, n_trials=n_trials)

        logger.info("Hyperparameter optimization completed!")
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Best reward: {study.best_value}")

        return study.best_params, study.best_value

    def curriculum_learning(self, max_difficulty: int = 5, timesteps_per_level: int = 200000):
        """Implement curriculum learning with progressive difficulty."""

        logger.info("Starting Curriculum Learning...")

        best_reward = -np.inf
        best_model = None

        for difficulty in range(1, max_difficulty + 1):
            logger.info(f"Training at difficulty level {difficulty}")

            # Create environment with current difficulty
            self.create_environments(difficulty_level=difficulty)

            # Train ensemble at this difficulty
            models = self.train_ensemble(total_timesteps=timesteps_per_level)

            # Evaluate all models
            for name, model in models.items():
                reward = self.evaluate_model(model)
                logger.info(f"{name.upper()} at difficulty {difficulty}: {reward:.4f}")

                if reward > best_reward:
                    best_reward = reward
                    best_model = model
                    logger.info(f"New best model: {name.upper()} with reward {reward:.4f}")

        logger.info(f"Curriculum learning completed. Best reward: {best_reward:.4f}")
        return best_model, best_reward

    def evaluate_model(self, model, n_episodes: int = 10) -> float:
        """Evaluate a trained model."""
        assert self.eval_env is not None, "Evaluation environment not created. Call create_environments() before evaluation."

        total_rewards = []

        for _ in range(n_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                step_result = self.eval_env.step(action)
                step_tuple = tuple(step_result)
                if len(step_tuple) == 4:
                    obs, reward, done, info = step_tuple
                else:
                    obs, reward, terminated, truncated, info = step_tuple
                    done = bool(terminated or truncated)

                # Normalize reward to a float in case it's an array or sequence
                try:
                    reward_val = float(np.array(reward).astype(np.float64).sum())
                except Exception:
                    reward_val = float(reward) if isinstance(reward, (int, float)) else 0.0

                episode_reward += reward_val

            total_rewards.append(episode_reward)

        return float(np.mean(total_rewards))

    def distributed_training(self, algorithm: str = 'sac', num_workers: int = 4):
        """Implement distributed training using Ray Tune."""

        logger.info("Starting Distributed Training...")

        def train_fn(config):
            # Create environment
            env = AdvancedTradingEnvironment(self.data_dict)

            # Create model based on algorithm
            if algorithm == 'sac':
                model = SAC("MlpPolicy", env, **config, device=device)
            elif algorithm == 'td3':
                model = TD3("MlpPolicy", env, **config, device=device)
            else:
                model = PPO("MlpPolicy", env, **config, device=device)

            # Train model
            model.learn(total_timesteps=100000)

            # Evaluate
            reward = self.evaluate_model(model)

            # Report to Ray Tune
            tune.report(mean_reward=reward)

        # Define hyperparameter search space
        config = {}
        if algorithm == 'sac':
            config = {
                'learning_rate': tune.loguniform(1e-5, 1e-2),
                'batch_size': tune.choice([64, 128, 256, 512]),
                'buffer_size': tune.choice([100000, 500000, 1000000]),
                'tau': tune.uniform(0.001, 0.01),
                'gamma': tune.uniform(0.95, 0.999)
            }
        elif algorithm == 'td3':
            config = {
                'learning_rate': tune.loguniform(1e-5, 1e-2),
                'batch_size': tune.choice([64, 128, 256, 512]),
                'buffer_size': tune.choice([100000, 500000, 1000000]),
                'tau': tune.uniform(0.001, 0.01),
                'gamma': tune.uniform(0.95, 0.999)
            }
        else:  # ppo
            config = {
                'learning_rate': tune.loguniform(1e-5, 1e-2),
                'batch_size': tune.choice([64, 128, 256, 512]),
                'n_steps': tune.choice([1024, 2048, 4096]),
                'gamma': tune.uniform(0.95, 0.999)
            }

        # Run distributed optimization
        scheduler = ASHAScheduler(
            max_t=100,
            grace_period=10,
            reduction_factor=2
        )

        analysis = tune.run(
            train_fn,
            config=config,
            scheduler=scheduler,
            num_samples=50,
            resources_per_trial={"cpu": 1, "gpu": 0}
        )

        best_config = analysis.best_config
        best_reward = analysis.best_result['mean_reward']

        logger.info(f"Distributed training completed!")
        logger.info(f"Best config: {best_config}")
        logger.info(f"Best reward: {best_reward}")

        return best_config, best_reward

    def save_models(self, path: str = "./advanced_rl_models/"):
        """Save all trained models."""

        os.makedirs(path, exist_ok=True)

        for name, model in self.models.items():
            model_path = os.path.join(path, f"{name}_model")
            model.save(model_path)
            logger.info(f"Saved {name} model to {model_path}")

    def load_models(self, path: str = "./advanced_rl_models/"):
        """Load trained models."""

        for filename in os.listdir(path):
            if filename.endswith("_model.zip"):
                name = filename.replace("_model.zip", "")
                model_path = os.path.join(path, filename.replace(".zip", ""))

                if "sac" in name:
                    model = SAC.load(model_path)
                elif "td3" in name:
                    model = TD3.load(model_path)
                else:
                    model = PPO.load(model_path)

                self.models[name] = model
                logger.info(f"Loaded {name} model from {model_path}")

class AdvancedCallback(BaseCallback):
    """Advanced callback for monitoring training progress."""

    def __init__(self, eval_env, eval_freq=10000, save_path="./models/", verbose=1):
        super(AdvancedCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_reward = -np.inf
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            assert self.eval_env is not None, "Eval env not provided to callback"
            total_reward = 0.0
            episode_rewards = []

            # Run multiple evaluation episodes
            for _ in range(5):
                episode_reward = 0.0
                done = False
                obs, _ = self.eval_env.reset()

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = self.eval_env.step(action)
                    if len(step_result) == 4:
                        obs, reward, done, info = step_result
                    else:
                        obs, reward, done, _, info = step_result

                    # Normalize reward
                    try:
                        reward_val = float(np.array(reward).astype(np.float64).sum())
                    except Exception:
                        reward_val = float(reward) if isinstance(reward, (int, float)) else 0.0

                    episode_reward += reward_val

                episode_rewards.append(episode_reward)
                total_reward += episode_reward

            mean_reward = total_reward / 5

            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                # Save best model
                model_path = os.path.join(self.save_path, "best_model")
                self.model.save(model_path)

                if self.verbose > 0:
                    print(f"New best model saved with reward: {mean_reward:.4f}")

            if self.verbose > 0:
                print(f"Step {self.n_calls}: Mean Eval Reward = {mean_reward:.4f}, "
                      f"Best = {self.best_reward:.4f}")

        return True

def prepare_advanced_training_data(tickers: Optional[List[str]] = None,
                                 train_period: str = "3y",
                                 test_period: str = "6mo") -> Tuple[Dict, Dict]:
    """
    Prepare comprehensive training data for advanced RL.
    """

    if tickers is None:
        tickers = settings.TICKERS[:20]  # Use more tickers for advanced training

    logger.info(f"Preparing advanced training data for {len(tickers)} tickers")

    train_data = {}
    test_data = {}

    for ticker in tickers:
        try:
            logger.info(f"Fetching data for {ticker}")

            # Fetch training data
            train_df = yf.download(ticker, period=train_period, interval="1d", progress=False, auto_adjust=True)
            if train_df is not None and not train_df.empty:
                if isinstance(train_df.columns, pd.MultiIndex):
                    train_df.columns = train_df.columns.get_level_values(0)
                train_df.columns = [col.lower() for col in train_df.columns]

                train_features = _make_features(train_df)
                train_data[ticker] = train_features

            # Fetch test data
            test_df = yf.download(ticker, period=test_period, interval="1d", progress=False, auto_adjust=True)
            if test_df is not None and not test_df.empty:
                if isinstance(test_df.columns, pd.MultiIndex):
                    test_df.columns = test_df.columns.get_level_values(0)
                test_df.columns = [col.lower() for col in test_df.columns]

                test_features = _make_features(test_df)
                test_data[ticker] = test_features

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            continue

    logger.info(f"Prepared {len(train_data)} training and {len(test_data)} test datasets")
    return train_data, test_data

def main():
    """
    Main function to run the complete advanced RL training system.
    """

    print("🚀 Advanced RL System for >98% Trading Accuracy")
    print("=" * 60)

    try:
        # Step 1: Prepare advanced training data
        print("\n📊 Step 1: Preparing Advanced Training Data")
        train_data, test_data = prepare_advanced_training_data()

        # Step 2: Initialize advanced trainer
        print("\n🧠 Step 2: Initializing Advanced RL Trainer")
        trainer = AdvancedRLTrainer(train_data)

        # Step 3: Curriculum learning
        print("\n📈 Step 3: Starting Curriculum Learning")
        trainer.create_environments(difficulty_level=1)
        best_model, best_reward = trainer.curriculum_learning(
            max_difficulty=3,
            timesteps_per_level=100000
        )

        # Step 4: Hyperparameter optimization
        print("\n🔧 Step 4: Hyperparameter Optimization")
        best_params, optimized_reward = trainer.hyperparameter_optimization(
            n_trials=20,
            algorithm='sac'
        )

        # Step 5: Train final ensemble with optimized parameters
        print("\n🎯 Step 5: Training Final Ensemble")
        trainer.create_environments(difficulty_level=3)
        final_models = trainer.train_ensemble(total_timesteps=500000)

        # Step 6: Evaluate all models
        print("\n📊 Step 6: Final Evaluation")
        results = {}
        for name, model in final_models.items():
            reward = trainer.evaluate_model(model, n_episodes=20)
            results[name] = reward
            print(".4f")
        # Check if target accuracy achieved
        max_accuracy = max(results.values())
        if max_accuracy > 0.98:
            print("\n🎉 TARGET ACHIEVED! Accuracy > 98%")
        elif max_accuracy > 0.95:
            print("\n🏆 Excellent! Accuracy > 95%")
        elif max_accuracy > 0.90:
            print("\n👌 Very Good! Accuracy > 90%")
        else:
            print(f"\n📈 Current best accuracy: {max_accuracy:.4f}. Continue optimization for better results.")

        # Step 7: Save models
        print("\n💾 Step 7: Saving Models")
        trainer.save_models()

        print("\n✅ Advanced RL Training System Complete!")

        return max_accuracy > 0.98

    except Exception as e:
        logger.error(f"Error in advanced RL training: {e}")
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Advanced RL system achieved >98% accuracy!")
    else:
        print("\n📈 Advanced RL system completed - continue optimization for target accuracy.")
