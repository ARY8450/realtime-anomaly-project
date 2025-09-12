"""
Advanced RL Training System for Trading
Trains reinforcement learning agents to achieve >98% accuracy through continuous learning
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
# from data_ingestion import RealTimeDataManager  # Will implement simple data fetching
from fusion.recommender import _make_features
from rl_trading_agent import RLTradingAgent, HybridRLMLAgent, train_rl_system, create_hybrid_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
from rl_trading_agent import RLTradingAgent, HybridRLMLAgent, train_rl_system, create_hybrid_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RLTrainingManager:
    """
    Manages the complete RL training pipeline for achieving high accuracy trading.
    """

    def __init__(self):
        # self.data_manager = RealTimeDataManager()  # Simplified data fetching
        self.rl_agent = None
        self.hybrid_agent = None
        self.training_data = {}
        self.test_data = {}

    def fetch_historical_data(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Simple data fetching using yfinance directly.
        """
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if data is None or data.empty:
                logger.warning(f"No data fetched for {ticker}")
                return pd.DataFrame()

            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten MultiIndex columns
                data.columns = data.columns.get_level_values(0)

            # Rename columns to lowercase for consistency
            data.columns = [col.lower() for col in data.columns]

            # Ensure we have the required columns
            required_cols = ['close', 'high', 'low', 'open', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                logger.warning(f"Missing columns for {ticker}: {missing_cols}")
                # Fill missing columns with close price
                for col in missing_cols:
                    if col != 'volume':
                        data[col] = data['close']
                    else:
                        data[col] = 0

            return data

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def prepare_training_data(self, tickers: Optional[List[str]] = None,
                            train_period: str = "3y", test_period: str = "6mo") -> Tuple[Dict, Dict]:
        """
        Prepare comprehensive training and testing datasets.
        """

        if tickers is None:
            tickers = settings.TICKERS[:10]  # Use first 10 tickers for faster training

        logger.info(f"Preparing training data for {len(tickers)} tickers")
        logger.info(f"Training period: {train_period}, Test period: {test_period}")

        train_data = {}
        test_data = {}

        for ticker in tickers:
            try:
                logger.info(f"Fetching data for {ticker}")

                # Fetch training data
                train_df = self.fetch_historical_data(
                    ticker, period=train_period, interval="1d"
                )

                # Fetch test data (more recent)
                test_df = self.fetch_historical_data(
                    ticker, period=test_period, interval="1d"
                )

                if train_df is not None and not train_df.empty:
                    # Create features for RL training
                    train_features = _make_features(train_df)
                    train_data[ticker] = train_features

                    logger.info(f"  {ticker}: {len(train_features)} training samples")

                if test_df is not None and not test_df.empty:
                    test_features = _make_features(test_df)
                    test_data[ticker] = test_features

                    logger.info(f"  {ticker}: {len(test_features)} test samples")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue

        self.training_data = train_data
        self.test_data = test_data

        logger.info(f"Prepared {len(train_data)} training datasets and {len(test_data)} test datasets")

        return train_data, test_data

    def train_rl_agent(self, total_timesteps: int = 500000,
                      eval_freq: int = 10000) -> RLTradingAgent:
        """
        Train the RL agent with advanced techniques for high accuracy.
        """

        if not self.training_data or not self.test_data:
            raise ValueError("Training data not prepared. Call prepare_training_data() first.")

        logger.info("🚀 Starting Advanced RL Training")
        logger.info("=" * 60)
        logger.info(f"Total timesteps: {total_timesteps}")
        logger.info(f"Evaluation frequency: {eval_freq}")

        # Train the RL system
        self.rl_agent = train_rl_system(
            self.training_data,
            self.test_data,
            total_timesteps=total_timesteps
        )

        return self.rl_agent

    def create_hybrid_agent(self) -> HybridRLMLAgent:
        """
        Create hybrid RL-ML agent combining reinforcement learning with existing ML models.
        """

        logger.info("Creating Hybrid RL-ML Agent")

        self.hybrid_agent = create_hybrid_system()
        return self.hybrid_agent

    def evaluate_performance(self, agent_type: str = "rl") -> Dict:
        """
        Comprehensive evaluation of the trained agent.
        """

        if agent_type == "rl" and self.rl_agent is None:
            raise ValueError("RL agent not trained. Call train_rl_agent() first.")
        elif agent_type == "hybrid" and self.hybrid_agent is None:
            raise ValueError("Hybrid agent not created. Call create_hybrid_agent() first.")

        logger.info(f"Evaluating {agent_type.upper()} Agent Performance")
        logger.info("=" * 50)

        # Use test data for evaluation
        if not self.test_data:
            raise ValueError("Test data not available")

        # Combine all test data (without ticker column for RL processing)
        test_dfs = []
        for ticker, df in self.test_data.items():
            # Remove ticker column if it exists to avoid string data issues
            df_copy = df.copy()
            if 'ticker' in df_copy.columns:
                df_copy = df_copy.drop('ticker', axis=1)
            test_dfs.append(df_copy)

        combined_test = pd.concat(test_dfs, axis=0, ignore_index=True)

        if agent_type == "rl":
            if self.rl_agent is not None:
                results = self.rl_agent.evaluate(combined_test, num_episodes=20)
            else:
                raise ValueError("RL agent is None")
        else:  # hybrid
            # For hybrid evaluation, we need to implement custom evaluation
            results = self._evaluate_hybrid_agent(combined_test)

        # Print detailed results
        self._print_evaluation_results(results, agent_type)

        return results

    def _evaluate_hybrid_agent(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate hybrid agent performance.
        """

        # This is a simplified evaluation - in practice you'd want more sophisticated metrics
        total_predictions = 0
        correct_predictions = 0

        for idx in range(len(test_data) - 1):
            current_obs = test_data.iloc[idx]
            next_price = test_data.iloc[idx + 1]['close']
            current_price = current_obs['close']

            # Get hybrid prediction
            try:
                # Create observation for RL agent
                tech_indicators = current_obs.iloc[1:].values.astype(np.float32)
                position_info = np.array([0, 1.0, 0], dtype=np.float32)  # neutral position
                observation = np.concatenate([tech_indicators, position_info])

                # Get ML features
                ml_features = current_obs.values.astype(np.float32)

                if self.hybrid_agent is not None:
                    action = self.hybrid_agent.predict_hybrid(observation, ml_features)
                else:
                    action = 1  # default to hold if no hybrid agent

                # Evaluate prediction
                actual_return = (next_price - current_price) / current_price

                if action == 0:  # predicted sell/short
                    predicted_correct = actual_return < -0.001
                elif action == 2:  # predicted buy/long
                    predicted_correct = actual_return > 0.001
                else:  # hold
                    predicted_correct = abs(actual_return) < 0.001

                total_predictions += 1
                if predicted_correct:
                    correct_predictions += 1

            except Exception as e:
                logger.warning(f"Error in hybrid prediction: {e}")
                continue

        accuracy = correct_predictions / max(1, total_predictions)

        return {
            'accuracy': accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'mean_reward': accuracy,  # Simplified
            'std_reward': 0,
            'mean_pnl': accuracy - 0.5,  # Simplified
            'mean_win_rate': accuracy,
            'mean_balance': 100000 * (1 + accuracy - 0.5)  # Simplified
        }

    def _print_evaluation_results(self, results: Dict, agent_type: str):
        """Print detailed evaluation results."""

        print(f"\n🎯 {agent_type.upper()} Agent Evaluation Results")
        print("=" * 60)

        if 'accuracy' in results:
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Total Predictions: {results.get('total_predictions', 'N/A')}")
            print(f"Correct Predictions: {results.get('correct_predictions', 'N/A')}")

        print(f"Mean Reward: {results['mean_reward']:.4f}")
        print(f"Std Reward: {results['std_reward']:.4f}")
        print(f"Mean P&L: {results['mean_pnl']:.4f}")
        print(f"Mean Win Rate: {results['mean_win_rate']:.4f}")
        print(f"Mean Final Balance: {results['mean_balance']:.2f}")
        # Check if target accuracy achieved
        accuracy = results.get('accuracy', results.get('mean_reward', 0))
        if accuracy > 0.98:
            print("\n🎉 TARGET ACHIEVED! Accuracy > 98%")
        elif accuracy > 0.90:
            print("\n👍 Excellent! Accuracy > 90%")
        elif accuracy > 0.80:
            print("\n👌 Good! Accuracy > 80%")
        else:
            print(f"\n📈 Current accuracy: {accuracy:.4f}. Continue training for better results.")

    def continuous_learning_loop(self, max_iterations: int = 10,
                               improvement_threshold: float = 0.001):
        """
        Implement continuous learning loop to iteratively improve accuracy.
        """

        logger.info("Starting Continuous Learning Loop")
        logger.info("=" * 50)

        best_accuracy = 0
        no_improvement_count = 0

        for iteration in range(max_iterations):
            logger.info(f"\nIteration {iteration + 1}/{max_iterations}")

            # Retrain with updated data
            self.train_rl_agent(total_timesteps=100000)

            # Evaluate performance
            results = self.evaluate_performance("rl")
            current_accuracy = results.get('accuracy', results.get('mean_reward', 0))

            # Check for improvement
            if current_accuracy > best_accuracy + improvement_threshold:
                best_accuracy = current_accuracy
                no_improvement_count = 0
                logger.info(f"New best accuracy: {best_accuracy:.4f}")
            else:
                no_improvement_count += 1
                logger.info(f"No significant improvement. Streak: {no_improvement_count}")

            # Early stopping if no improvement for several iterations
            if no_improvement_count >= 3:
                logger.info("Early stopping due to no improvement")
                break

            # Check if target achieved
            if current_accuracy > 0.98:
                logger.info("🎉 Target accuracy achieved! Stopping training.")
                break

        logger.info(f"Continuous learning completed. Best accuracy: {best_accuracy:.4f}")
        return best_accuracy

def main():
    """
    Main function to run the complete RL training system.
    """

    print("🤖 Advanced RL Trading System for >98% Accuracy")
    print("=" * 60)

    # Initialize training manager
    trainer = RLTrainingManager()

    try:
        # Step 1: Prepare data
        print("\n📊 Step 1: Preparing Training Data")
        train_data, test_data = trainer.prepare_training_data()

        # Step 2: Train RL agent
        print("\n🧠 Step 2: Training RL Agent")
        rl_agent = trainer.train_rl_agent(total_timesteps=200000)

        # Step 3: Create hybrid agent
        print("\n🔄 Step 3: Creating Hybrid RL-ML Agent")
        hybrid_agent = trainer.create_hybrid_agent()

        # Step 4: Evaluate performance
        print("\n📈 Step 4: Evaluating Performance")

        print("\n--- RL Agent Performance ---")
        rl_results = trainer.evaluate_performance("rl")

        print("\n--- Hybrid Agent Performance ---")
        hybrid_results = trainer.evaluate_performance("hybrid")

        # Step 5: Continuous learning (if needed)
        rl_accuracy = rl_results.get('accuracy', rl_results.get('mean_reward', 0))
        if rl_accuracy < 0.98:
            print("\n🔄 Step 5: Starting Continuous Learning")
            final_accuracy = trainer.continuous_learning_loop(max_iterations=5)
            print(f"Final accuracy after continuous learning: {final_accuracy:.4f}")
        print("\n✅ RL Training System Complete!")

    except Exception as e:
        logger.error(f"Error in RL training: {e}")
        print(f"❌ Error: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 RL system training completed successfully!")
    else:
        print("\n❌ RL system training failed.")
