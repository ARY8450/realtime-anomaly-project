"""
Reinforcement Learning Trading Environment and Agent
Implements a sophisticated RL system for trading that learns to maximize returns
through trial and error, achieving high accuracy through continuous learning.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, cast
import logging
from datetime import datetime, timedelta
import os
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """
    Custom trading environment for reinforcement learning.
    The agent learns to make trading decisions (buy/hold/sell) to maximize returns.
    """

    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000,
                 transaction_cost: float = 0.001, max_position: float = 1.0):
        super(TradingEnvironment, self).__init__()

        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        # Action space: 0 = sell, 1 = hold, 2 = buy
        self.action_space = spaces.Discrete(3)

        # Determine observation space size from data
        if not data.empty:
            sample_obs = self._get_sample_observation()
            obs_size = len(sample_obs)
        else:
            obs_size = 39  # fallback

        # Observation space: technical indicators + position + balance info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.reset()

    def _get_sample_observation(self) -> np.ndarray:
        """Get a sample observation to determine the correct dimensions."""
        if self.data.empty or len(self.data) == 0:
            return np.zeros(39, dtype=np.float32)

        # Get first row of data
        current_data = self.data.iloc[0]

        # Extract technical indicators
        numeric_data = pd.to_numeric(current_data, errors='coerce').fillna(0)
        tech_indicators = numeric_data.values.astype(np.float32)

        # Add position and balance information (3 features)
        position_info = np.array([0, 1.0, 0], dtype=np.float32)

        # Combine all features
        observation = np.concatenate([tech_indicators, position_info])

        return observation

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state.

        Matches the Gymnasium API: reset(*, seed=None, options=None) -> (obs, info)
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # -1 (short), 0 (neutral), 1 (long)
        self.entry_price = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Get current observation (state) for the agent."""
        if self.current_step >= len(self.data):
            # Return zeros if we've reached the end. observation_space.shape can be
            # Optional[Tuple[int, ...]] per Gym types, so cast to a concrete tuple
            shape = cast(Tuple[int, ...], self.observation_space.shape)
            return np.zeros(shape, dtype=np.float32)

        # Get current market data
        current_data = self.data.iloc[self.current_step]

        # Extract technical indicators (all numeric columns except position/balance info)
        numeric_data = pd.to_numeric(current_data, errors='coerce').fillna(0)
        tech_indicators = numeric_data.values.astype(np.float32)

        # Ensure we have the expected number of features
        expected_features = 36
        if len(tech_indicators) != expected_features:
            # Pad or truncate to match expected size
            if len(tech_indicators) < expected_features:
                padding = np.zeros(expected_features - len(tech_indicators), dtype=np.float32)
                tech_indicators = np.concatenate([tech_indicators, padding])
            else:
                tech_indicators = tech_indicators[:expected_features]

        # Add position and balance information
        position_info = np.array([
            self.position,  # current position
            self.balance / self.initial_balance,  # normalized balance
            self.total_pnl / self.initial_balance  # normalized P&L
        ], dtype=np.float32)

        # Combine all features
        observation = np.concatenate([tech_indicators, position_info])

        return observation

    def step(self, action: int | np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""

        # Accept numpy scalar/array actions coming from RL libraries and convert
        # them to a Python int for downstream logic.
        if isinstance(action, np.ndarray):
            try:
                action = int(action.item())
            except Exception:
                # Fallback for 1-D arrays
                action = int(action[0])

        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, {}

        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']

        # Execute action
        reward = 0
        old_position = self.position

        if action == 0:  # Sell
            if self.position > 0:  # Close long position
                pnl = (current_price - self.entry_price) / self.entry_price
                self.balance *= (1 + pnl - self.transaction_cost)
                self.total_pnl += pnl
                reward = pnl
                self.position = 0
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
            elif self.position == 0:  # Open short position
                self.position = -1
                self.entry_price = current_price
                self.balance *= (1 - self.transaction_cost)
        elif action == 2:  # Buy
            if self.position < 0:  # Close short position
                pnl = (self.entry_price - current_price) / self.entry_price
                self.balance *= (1 + pnl - self.transaction_cost)
                self.total_pnl += pnl
                reward = pnl
                self.position = 0
                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
            elif self.position == 0:  # Open long position
                self.position = 1
                self.entry_price = current_price
                self.balance *= (1 - self.transaction_cost)

        # Position sizing reward (encourage proper position management)
        position_reward = -abs(self.position) * 0.001  # Small penalty for large positions

        # Market timing reward (reward being in correct direction)
        if self.position != 0:
            price_change = (next_price - current_price) / current_price
            market_reward = self.position * price_change * 0.1
        else:
            market_reward = 0

        total_reward = reward + position_reward + market_reward

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self._get_observation(), total_reward, done, False, {}

    def render(self, mode='human'):
        """Render the current state."""
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
              f"Position: {self.position}, Total P&L: {self.total_pnl:.4f}")

class TradingCallback(BaseCallback):
    """Custom callback for monitoring RL training progress."""

    def __init__(self, eval_env, eval_freq=1000, verbose=1):
        super(TradingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            obs, _ = self.eval_env.reset()
            total_reward = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.eval_env.step(action)
                total_reward += reward

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                # Save best model
                self.model.save("best_trading_model")

            if self.verbose > 0:
                print(f"Step {self.n_calls}: Eval Reward = {total_reward:.4f}, "
                      f"Best = {self.best_reward:.4f}")

        return True

class RLTradingAgent:
    """
    Reinforcement Learning Agent for trading using PPO algorithm.
    Combines with existing ML models for enhanced predictions.
    """

    def __init__(self, model_path: str = "rl_trading_model"):
        self.model_path = model_path
        self.model = None
        self.env = None

    def create_environments(self, train_data: pd.DataFrame, eval_data: pd.DataFrame):
        """Create training and evaluation environments."""

        # Create training environment
        def make_train_env():
            env = TradingEnvironment(train_data)
            return env

        self.train_env = DummyVecEnv([make_train_env])

        # Create evaluation environment
        self.eval_env = TradingEnvironment(eval_data)

    def train(self, total_timesteps: int = 100000, eval_freq: int = 5000):
        """Train the RL agent."""

        logger.info("Starting RL training...")

        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./rl_trading_tensorboard/"
        )

        # Create callback for evaluation
        callback = TradingCallback(self.eval_env, eval_freq=eval_freq)

        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )

        # Save the final model
        self.model.save(self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load_model(self, model_path: Optional[str] = None):
        """Load a trained model."""
        if model_path:
            self.model_path = model_path

        if os.path.exists(self.model_path + ".zip"):
            self.model = PPO.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.warning(f"Model file {self.model_path}.zip not found")

    def predict(self, observation: np.ndarray) -> int:
        """Make a prediction using the trained RL model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

    def evaluate(self, eval_data: pd.DataFrame, num_episodes: int = 10) -> Dict:
        """Evaluate the trained model on test data."""

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        eval_env = TradingEnvironment(eval_data)
        results = []

        for episode in range(num_episodes):
            obs, _ = eval_env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = eval_env.step(action)
                total_reward += reward
                steps += 1

            results.append({
                'episode': episode,
                'total_reward': total_reward,
                'steps': steps,
                'final_balance': eval_env.balance,
                'total_pnl': eval_env.total_pnl,
                'win_rate': eval_env.winning_trades / max(1, eval_env.total_trades)
            })

        # Calculate summary statistics
        summary = {
            'mean_reward': np.mean([r['total_reward'] for r in results]),
            'std_reward': np.std([r['total_reward'] for r in results]),
            'mean_pnl': np.mean([r['total_pnl'] for r in results]),
            'mean_win_rate': np.mean([r['win_rate'] for r in results]),
            'mean_balance': np.mean([r['final_balance'] for r in results]),
            'results': results
        }

        return summary

class HybridRLMLAgent:
    """
    Hybrid agent combining Reinforcement Learning with traditional ML models.
    Uses RL for decision making while incorporating ML predictions as features.
    """

    def __init__(self, ml_model_path: str = "model_ensemble.pkl",
                 rl_model_path: str = "rl_trading_model"):
        self.ml_model_path = ml_model_path
        self.rl_model_path = rl_model_path
        self.ml_model = None
        self.rl_agent = RLTradingAgent(rl_model_path)

    def load_models(self):
        """Load both ML and RL models."""
        # Load ML model
        if os.path.exists(self.ml_model_path):
            self.ml_model = joblib.load(self.ml_model_path)
            logger.info(f"ML model loaded from {self.ml_model_path}")
        else:
            logger.warning(f"ML model file {self.ml_model_path} not found")

        # Load RL model
        self.rl_agent.load_model()

    def predict_hybrid(self, observation: np.ndarray, ml_features: np.ndarray) -> int:
        """
        Make hybrid prediction combining ML and RL approaches.
        """

        # Get ML prediction if available
        ml_prediction = None
        if self.ml_model is not None:
            try:
                ml_pred = self.ml_model.predict(ml_features.reshape(1, -1))[0]
                # Convert ML prediction to action (0, 1, 2)
                if ml_pred == -1:
                    ml_prediction = 0  # sell
                elif ml_pred == 1:
                    ml_prediction = 2  # buy
                else:
                    ml_prediction = 1  # hold
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")

        # Get RL prediction
        rl_prediction = self.rl_agent.predict(observation)

        # Combine predictions (weighted voting)
        if ml_prediction is not None:
            # 70% weight to RL, 30% to ML
            predictions = [rl_prediction, ml_prediction]
            weights = [0.7, 0.3]

            # Weighted voting
            action_counts = {}
            for pred, weight in zip(predictions, weights):
                action_counts[pred] = action_counts.get(pred, 0) + weight

            # Find action with highest weight
            final_action = max(action_counts.items(), key=lambda x: x[1])[0]
        else:
            final_action = rl_prediction

        return int(final_action)

def create_enhanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhanced features for RL training, including ML model predictions.
    """
    # This would integrate with the existing recommender system
    # For now, return the original data
    return data

def train_rl_system(train_data: Dict[str, pd.DataFrame],
                   eval_data: Dict[str, pd.DataFrame],
                   total_timesteps: int = 200000) -> RLTradingAgent:
    """
    Train the complete RL trading system.
    """

    logger.info("🚀 Starting RL Trading System Training")
    logger.info("=" * 60)

    # Prepare training data (combine all tickers)
    train_dfs = []
    for ticker, df in train_data.items():
        df_copy = df.copy()
        # Remove any non-numeric columns that might cause issues
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_copy = df_copy[numeric_cols]
        train_dfs.append(df_copy)

    combined_train = pd.concat(train_dfs, axis=0, ignore_index=True)

    # Prepare evaluation data
    eval_dfs = []
    for ticker, df in eval_data.items():
        df_copy = df.copy()
        # Remove any non-numeric columns that might cause issues
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_copy = df_copy[numeric_cols]
        eval_dfs.append(df_copy)

    combined_eval = pd.concat(eval_dfs, axis=0, ignore_index=True)

    # Create and train RL agent
    agent = RLTradingAgent()
    agent.create_environments(combined_train, combined_eval)
    agent.train(total_timesteps=total_timesteps)

    # Evaluate the trained model
    logger.info("Evaluating trained model...")
    eval_results = agent.evaluate(combined_eval)

    logger.info("📊 RL Training Results:")
    logger.info(f"Mean Reward: {eval_results['mean_reward']:.4f}")
    logger.info(f"Mean P&L: {eval_results['mean_pnl']:.4f}")
    logger.info(f"Mean Win Rate: {eval_results['mean_win_rate']:.4f}")
    logger.info(f"Mean Final Balance: {eval_results['mean_balance']:.2f}")

    return agent

def create_hybrid_system(ml_model_path: str = "model_ensemble.pkl",
                        rl_model_path: str = "rl_trading_model") -> HybridRLMLAgent:
    """
    Create and return a hybrid RL-ML system.
    """

    hybrid_agent = HybridRLMLAgent(ml_model_path, rl_model_path)
    hybrid_agent.load_models()

    return hybrid_agent

if __name__ == "__main__":
    # Example usage
    print("RL Trading System initialized")
    print("Use train_rl_system() to train the RL agent")
    print("Use create_hybrid_system() to create hybrid RL-ML agent")
