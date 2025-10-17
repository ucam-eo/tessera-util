#!/usr/bin/env python3
"""
Test script for the disable_main_training_data parameter in train_cnn.py
"""

import subprocess
import sys
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_help_message():
    """Test that the help message includes the new parameter."""
    logger.info("=== Testing Help Message ===")
    try:
        result = subprocess.run([
            '/maps/zf281/miniconda3/envs/detectree-env/bin/python', 
            'train_cnn.py', 
            '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if 'disable_main_training_data' in result.stdout:
            logger.info("✅ Help message contains disable_main_training_data parameter")
            return True
        else:
            logger.error("❌ Help message does not contain disable_main_training_data parameter")
            return False
            
    except Exception as e:
        logger.error(f"❌ Help message test failed: {e}")
        return False

def test_parameter_validation():
    """Test parameter validation logic."""
    logger.info("=== Testing Parameter Validation ===")
    
    # Test 1: disable_main_training_data without use_extra_training_data should fail
    logger.info("Testing disable_main_training_data without extra data...")
    try:
        result = subprocess.run([
            '/maps/zf281/miniconda3/envs/detectree-env/bin/python', 
            'train_cnn.py',
            '--years', '2019',
            '--num_epochs', '1',
            '--batch_size', '32',
            '--data_dir', '/maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1',
            '--disable_main_training_data'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0 and 'Cannot disable main training data without enabling extra training data' in result.stderr:
            logger.info("✅ Correctly rejected disable_main_training_data without extra data")
            validation_test_1 = True
        else:
            logger.error("❌ Should have rejected disable_main_training_data without extra data")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            validation_test_1 = False
            
    except Exception as e:
        logger.error(f"❌ Validation test 1 failed: {e}")
        validation_test_1 = False
    
    return validation_test_1

def test_extra_data_only_training():
    """Test training with only extra data."""
    logger.info("=== Testing Extra Data Only Training ===")
    
    # Create a small test output directory
    test_output_dir = "/maps/zf281/btfm4rs/src/pv_detection/test_disable_main_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        result = subprocess.run([
            '/maps/zf281/miniconda3/envs/detectree-env/bin/python', 
            'train_cnn.py',
            '--years', '2019',
            '--num_epochs', '1',
            '--batch_size', '32',
            '--data_dir', '/maps/zf281/btfm4rs/data/downstream/pv_detection/roi_1',
            '--use_extra_training_data',
            '--extra_data_dir', '/maps/zf281/btfm4rs/data/downstream/pv_detection/extra_training_data',
            '--uk_data_dir', '/maps/zf281/btfm4rs/data/downstream/pv_detection/uk',
            '--disable_main_training_data',
            '--output_dir', test_output_dir,
            '--quick_eval'
        ], capture_output=True, text=True, timeout=300)
        
        logger.info(f"Training completed with return code: {result.returncode}")
        
        # Check if training completed successfully
        if result.returncode == 0:
            logger.info("✅ Extra data only training completed successfully")
            
            # Check if expected log messages are present
            if 'Main training data disabled - skipping main data loading' in result.stdout:
                logger.info("✅ Correctly skipped main data loading")
            else:
                logger.warning("⚠️ Expected log message about skipping main data not found")
                
            if 'Using only extra training data' in result.stdout:
                logger.info("✅ Correctly used only extra training data")
            else:
                logger.warning("⚠️ Expected log message about using only extra data not found")
                
            return True
        else:
            logger.error("❌ Extra data only training failed")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Extra data only training test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting tests for disable_main_training_data parameter...")
    
    tests = [
        ("Help Message Test", test_help_message),
        ("Parameter Validation Test", test_parameter_validation),
        ("Extra Data Only Training Test", test_extra_data_only_training)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())