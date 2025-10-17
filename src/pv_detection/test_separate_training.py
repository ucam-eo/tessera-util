#!/usr/bin/env python3
"""
Test script for the separate CNN training functionality.
Tests training for a single year to verify the implementation works correctly.
"""

import sys
import os
import subprocess
import time
import logging

# Add the current directory to Python path
sys.path.append('/maps/zf281/btfm4rs/src/pv_detection')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_separate_training():
    """Test the separate training script with a single year."""
    logger.info("=== Testing Separate CNN Training ===")
    
    # Test configuration - use a single year for quick testing
    test_year = "2019"  # Use 2019 as test year
    
    # Test parameters
    test_args = [
        "/maps/zf281/miniconda3/envs/detectree-env/bin/python",
        "train_separate_cnn.py",
        "--years", test_year,
        "--num_epochs", "2",  # Very few epochs for testing
        "--batch_size", "512",  # Smaller batch size for testing
        "--use_cache",  # Use cache for faster loading
        "--quick_eval",  # Use quick evaluation
        "--output_dir", "/maps/zf281/btfm4rs/src/pv_detection/test_models",
        "--data_dir", "/maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1",
        "--cache_dir", "/maps/zf281/btfm4rs/src/pv_detection/cnn_cache"
    ]
    
    logger.info(f"Testing with year: {test_year}")
    logger.info(f"Command: {' '.join(test_args)}")
    
    try:
        # Change to the correct directory
        os.chdir('/maps/zf281/btfm4rs/src/pv_detection')
        
        # Run the training script
        start_time = time.time()
        result = subprocess.run(
            test_args,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        end_time = time.time()
        
        logger.info(f"Training completed in {end_time - start_time:.1f} seconds")
        logger.info(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            logger.info("✅ Training completed successfully!")
            
            # Check if model files were created
            expected_model_dir = f"/maps/zf281/btfm4rs/src/pv_detection/test_models/year_{test_year}"
            expected_model_file = f"{expected_model_dir}/cnn_model_{test_year}.pth"
            expected_history_file = f"{expected_model_dir}/training_history_{test_year}.json"
            expected_config_file = f"{expected_model_dir}/config_{test_year}.json"
            
            files_exist = []
            for file_path in [expected_model_file, expected_history_file, expected_config_file]:
                if os.path.exists(file_path):
                    files_exist.append(f"✅ {file_path}")
                else:
                    files_exist.append(f"❌ {file_path}")
            
            logger.info("Expected output files:")
            for file_status in files_exist:
                logger.info(f"  {file_status}")
            
            # Print last few lines of stdout
            if result.stdout:
                stdout_lines = result.stdout.strip().split('\n')
                logger.info("Last few lines of output:")
                for line in stdout_lines[-10:]:
                    logger.info(f"  {line}")
            
            return True
            
        else:
            logger.error("❌ Training failed!")
            logger.error(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Training timed out!")
        return False
    except Exception as e:
        logger.error(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_help_message():
    """Test that the help message works."""
    logger.info("=== Testing Help Message ===")
    
    try:
        os.chdir('/maps/zf281/btfm4rs/src/pv_detection')
        result = subprocess.run([
            "/maps/zf281/miniconda3/envs/detectree-env/bin/python",
            "train_separate_cnn.py",
            "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            logger.info("✅ Help message works correctly")
            logger.info("Help message preview:")
            help_lines = result.stdout.split('\n')[:10]
            for line in help_lines:
                logger.info(f"  {line}")
            return True
        else:
            logger.error("❌ Help message failed")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing help message: {e}")
        return False

def test_import():
    """Test that all imports work correctly."""
    logger.info("=== Testing Imports ===")
    
    try:
        # Test importing the main components
        from train_separate_cnn import SeparateCNNTrainer, train_single_year
        logger.info("✅ Successfully imported SeparateCNNTrainer and train_single_year")
        
        # Test that we can create a basic config
        config = {
            'data_dir': '/maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1',
            'model_type': 'deep',
            'patch_size': 3,
            'input_channels': 128,
            'num_classes': 2,
            'dropout_rate': 0.2,
            'batch_size': 512,
            'num_epochs': 2,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'momentum': 0.999,
            'scheduler': 'cosine',
            'step_size': 30,
            'gamma': 0.1,
            'patience': 10,
            'val_ratio': 0.1,
            'use_balanced_sampling': False,
            'max_neg_pos_ratio': 9.0,
            'pos_neg_ratio': 0.2,
            'use_class_weights': False,
            'normalize_patches': False,
            'num_workers': 4,
            'early_stopping': False,
            'early_stopping_patience': 20,
            'random_seed': 42
        }
        
        # Test creating trainer instance (without loading data)
        trainer = SeparateCNNTrainer(config, year=2019)
        logger.info("✅ Successfully created SeparateCNNTrainer instance")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting separate training tests...\n")
    
    # Run tests
    test_results = []
    
    # Test 1: Import test
    test_results.append(("Import Test", test_import()))
    
    # Test 2: Help message test
    test_results.append(("Help Message Test", test_help_message()))
    
    # Test 3: Actual training test (commented out for now to avoid long runtime)
    # Uncomment the line below to run the full training test
    test_results.append(("Training Test", test_separate_training()))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1)