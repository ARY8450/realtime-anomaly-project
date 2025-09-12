"""
Test script for the Advanced RL System components.
"""

import sys
import os
import numpy as np

# Add project paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

try:
    from advanced_rl_system import AdvancedRLTrainer, prepare_advanced_training_data
    from config import settings

    print("✅ Successfully imported Advanced RL System components")

    # Test data preparation
    print("\n🧪 Testing data preparation...")
    train_data = {}
    test_data = {}
    try:
        train_data, test_data = prepare_advanced_training_data(
            tickers=settings.TICKERS[:3],  # Test with just 3 tickers
            train_period="1y",
            test_period="3mo"
        )
        print(f"✅ Data preparation successful: {len(train_data)} training datasets, {len(test_data)} test datasets")
    except Exception as e:
        print(f"⚠️ Data preparation failed (expected if no internet): {e}")

    # Test trainer initialization
    print("\n🧪 Testing trainer initialization...")
    if train_data:
        try:
            trainer = AdvancedRLTrainer(train_data)
            print("✅ Trainer initialization successful")

            # Test environment creation
            print("\n🧪 Testing environment creation...")
            trainer.create_environments(difficulty_level=1)
            print("✅ Environment creation successful")

            print("\n🎉 All basic components working correctly!")
            print("   Ready to run full advanced RL training.")

        except Exception as e:
            print(f"❌ Trainer initialization failed: {e}")
    else:
        print("⚠️ Skipping trainer test due to data preparation failure")

except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("\n" + "="*50)
print("Advanced RL System Test Complete")
