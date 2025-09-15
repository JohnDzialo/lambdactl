#!/usr/bin/env python3
"""
Test script for the separated GPU load test architecture

This script validates that:
1. Pure GPU load test script runs without monitoring overhead
2. Resource monitoring script runs independently 
3. Both scripts can be coordinated properly
4. Results are collected from both processes
"""

import subprocess
import time
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pure_gpu_test(quick: bool = True) -> Tuple[bool, Optional[str], Optional[dict]]:
    """Test the pure GPU load test script"""
    logger.info("Testing pure GPU load test script...")
    
    script_path = Path(__file__).parent / "load_tests" / "gpu_loadtest.py"
    if not script_path.exists():
        logger.error(f"GPU test script not found: {script_path}")
        return False, "Script not found", None
    
    # Run the GPU test
    cmd = ["python3", str(script_path)]
    if quick:
        cmd.append("--quick")
    
    output_file = Path(f"test_gpu_results_{int(time.time())}.json")
    cmd.extend(["--output", str(output_file)])
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        start_time = time.time()
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        duration = time.time() - start_time
        logger.info(f"GPU test completed in {duration:.1f}s")
        
        if result.returncode == 0:
            if output_file.exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                logger.info("GPU test successful, results parsed")
                output_file.unlink()  # Cleanup
                return True, None, results
            else:
                logger.warning("GPU test successful but no output file found")
                return True, None, None
        else:
            logger.error(f"GPU test failed with return code {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return False, result.stderr, None
            
    except subprocess.TimeoutExpired:
        logger.error("GPU test timed out")
        return False, "Test timed out", None
    except Exception as e:
        logger.error(f"GPU test failed with exception: {e}")
        return False, str(e), None


def run_resource_monitor_test(duration: int = 30) -> Tuple[bool, Optional[str], Optional[dict]]:
    """Test the resource monitoring script"""
    logger.info(f"Testing resource monitoring script for {duration}s...")
    
    script_path = Path(__file__).parent / "load_tests" / "resource_monitor.py"
    if not script_path.exists():
        logger.error(f"Resource monitor script not found: {script_path}")
        return False, "Script not found", None
    
    output_file = Path(f"test_monitor_results_{int(time.time())}.json")
    
    # Run the monitoring script
    cmd = [
        "python3", str(script_path),
        "--interval", "5",  # 5 second intervals for quick test
        "--duration", str(duration),
        "--output", str(output_file),
        "--quiet"
    ]
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        start_time = time.time()
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 30)
        
        actual_duration = time.time() - start_time
        logger.info(f"Resource monitoring completed in {actual_duration:.1f}s")
        
        if result.returncode == 0:
            if output_file.exists():
                with open(output_file, 'r') as f:
                    results = json.load(f)
                logger.info("Resource monitoring successful, results parsed")
                logger.info(f"Collected {len(results.get('snapshots', []))} snapshots")
                output_file.unlink()  # Cleanup
                return True, None, results
            else:
                logger.warning("Resource monitoring successful but no output file found")
                return True, None, None
        else:
            logger.error(f"Resource monitoring failed with return code {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return False, result.stderr, None
            
    except subprocess.TimeoutExpired:
        logger.error("Resource monitoring timed out")
        return False, "Monitoring timed out", None
    except Exception as e:
        logger.error(f"Resource monitoring failed with exception: {e}")
        return False, str(e), None


def run_coordinated_test(gpu_duration: int = 60, monitor_duration: int = 90) -> Tuple[bool, str]:
    """Test coordinated execution of GPU test and monitoring"""
    logger.info("Testing coordinated GPU test + monitoring...")
    
    gpu_script = Path(__file__).parent / "load_tests" / "gpu_loadtest.py"
    monitor_script = Path(__file__).parent / "load_tests" / "resource_monitor.py"
    
    if not gpu_script.exists() or not monitor_script.exists():
        return False, "One or both scripts not found"
    
    timestamp = int(time.time())
    gpu_output = Path(f"test_coordinated_gpu_{timestamp}.json")
    monitor_output = Path(f"test_coordinated_monitor_{timestamp}.json")
    
    try:
        # Start monitoring first
        monitor_cmd = [
            "python3", str(monitor_script),
            "--interval", "10",
            "--duration", str(monitor_duration),
            "--output", str(monitor_output),
            "--quiet"
        ]
        
        logger.info("Starting resource monitoring...")
        monitor_process = subprocess.Popen(monitor_cmd)
        
        # Brief delay to let monitoring start
        time.sleep(2)
        
        # Start GPU test
        gpu_cmd = [
            "python3", str(gpu_script),
            "--quick",
            "--output", str(gpu_output)
        ]
        
        logger.info("Starting GPU test...")
        start_time = time.time()
        
        gpu_result = subprocess.run(gpu_cmd, capture_output=True, text=True, timeout=gpu_duration + 30)
        
        gpu_duration_actual = time.time() - start_time
        logger.info(f"GPU test completed in {gpu_duration_actual:.1f}s")
        
        # Wait for monitoring to complete
        logger.info("Waiting for monitoring to complete...")
        monitor_result = monitor_process.wait(timeout=30)
        
        # Check results
        gpu_success = gpu_result.returncode == 0 and gpu_output.exists()
        monitor_success = monitor_result == 0 and monitor_output.exists()
        
        if gpu_success and monitor_success:
            # Parse results to validate
            with open(gpu_output, 'r') as f:
                gpu_data = json.load(f)
            with open(monitor_output, 'r') as f:
                monitor_data = json.load(f)
                
            logger.info("Coordinated test successful!")
            logger.info(f"GPU tests: {len(gpu_data.get('results', []))}")
            logger.info(f"Monitoring snapshots: {len(monitor_data.get('snapshots', []))}")
            
            # Cleanup
            gpu_output.unlink()
            monitor_output.unlink()
            
            return True, "Coordinated test successful"
        else:
            error_msgs = []
            if not gpu_success:
                error_msgs.append(f"GPU test failed: {gpu_result.stderr}")
            if not monitor_success:
                error_msgs.append(f"Monitoring failed: return code {monitor_result}")
            return False, "; ".join(error_msgs)
            
    except Exception as e:
        # Cleanup processes
        try:
            monitor_process.kill()
        except:
            pass
        
        return False, f"Coordinated test failed: {e}"


def main():
    """Run all tests for the separated architecture"""
    print("Testing Separated GPU Load Test Architecture")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Pure GPU test
    print("\nTest 1: Pure GPU Load Test")
    print("-" * 30)
    gpu_success, gpu_error, gpu_results = run_pure_gpu_test(quick=True)
    if gpu_success:
        print("✅ Pure GPU test PASSED")
        if gpu_results:
            successful_tests = len([r for r in gpu_results.get('results', []) if r.get('success')])
            total_tests = len(gpu_results.get('results', []))
            print(f"   {successful_tests}/{total_tests} GPU tests successful")
    else:
        print(f"❌ Pure GPU test FAILED: {gpu_error}")
        all_passed = False
    
    # Test 2: Resource monitoring
    print("\nTest 2: Resource Monitoring")
    print("-" * 30)
    monitor_success, monitor_error, monitor_results = run_resource_monitor_test(duration=30)
    if monitor_success:
        print("✅ Resource monitoring PASSED")
        if monitor_results:
            snapshots = len(monitor_results.get('snapshots', []))
            duration = monitor_results.get('duration_seconds', 0)
            print(f"   {snapshots} snapshots collected over {duration:.1f}s")
    else:
        print(f"❌ Resource monitoring FAILED: {monitor_error}")
        all_passed = False
    
    # Test 3: Coordinated execution
    print("\nTest 3: Coordinated Execution")
    print("-" * 30)
    coord_success, coord_message = run_coordinated_test(gpu_duration=60, monitor_duration=90)
    if coord_success:
        print("✅ Coordinated execution PASSED")
        print(f"   {coord_message}")
    else:
        print(f"❌ Coordinated execution FAILED: {coord_message}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Separated architecture is working!")
        print("\nThe new architecture provides:")
        print("• Pure GPU compute without monitoring overhead")
        print("• Independent resource monitoring with configurable intervals")
        print("• Coordinated execution of both processes")
        print("• Separate result collection from both systems")
    else:
        print("❌ SOME TESTS FAILED - Check the logs above")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())