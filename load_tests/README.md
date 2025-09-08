# GPU Load Testing Suite

A comprehensive PyTorch-based GPU benchmarking tool designed to evaluate GPU performance across training and inference workloads on remote Lambda Labs instances.

## Overview

The `gpu_loadtest.py` script performs systematic GPU performance evaluation using neural network models of varying complexity. It measures compute throughput, memory utilization, and resource efficiency to provide detailed performance insights.

## Features

- **Comprehensive Benchmarking**: Both training and inference workloads
- **Resource Monitoring**: Real-time CPU, GPU, and memory tracking with min/max/avg statistics
- **Configurable Parameters**: Epoch and batch-based testing with flexible configuration
- **Detailed Logging**: File and console logging with configurable verbosity
- **JSON Results**: Structured output with performance metrics and system information
- **Error Handling**: Graceful degradation and comprehensive error reporting

## Test Suite

### Training Benchmarks

Training tests evaluate GPU performance under compute-intensive workloads including forward passes, backpropagation, and gradient updates.

#### Simple CNN Training
- **Model**: 3-layer Convolutional Neural Network with pooling and fully connected layers
- **Operations Tested**: Convolution, pooling, ReLU activations, backpropagation
- **Batch Sizes**: 16, 32, 64
- **Test Descriptions**:
  - **Batch 16**: Small batch CNN training - Tests GPU efficiency with limited parallelism and memory usage
  - **Batch 32**: Medium batch CNN training - Balanced test of GPU compute and memory bandwidth  
  - **Batch 64**: Large batch CNN training - Tests GPU memory capacity and high-throughput compute

#### ResNet Training
- **Model**: Residual Network with skip connections, batch normalization, and residual blocks
- **Operations Tested**: Complex convolutions, batch normalization, residual connections, gradient flow
- **Batch Sizes**: 8, 16, 32
- **Test Descriptions**:
  - **Batch 8**: Small batch ResNet training - Tests GPU with complex operations and moderate memory usage
  - **Batch 16**: Medium batch ResNet training - Balanced complex compute with memory efficiency
  - **Batch 32**: Large batch ResNet training - Stress test for GPU memory and complex operations

### Inference Benchmarks

Inference tests evaluate GPU performance for forward-pass only operations, optimized for throughput and latency scenarios.

#### Simple CNN Inference
- **Model**: Same CNN architecture as training, evaluation mode
- **Operations Tested**: Forward pass efficiency without gradient computation
- **Batch Sizes**: 16, 32, 64
- **Test Descriptions**:
  - **Batch 16**: Small batch CNN inference - Tests low-latency inference scenarios
  - **Batch 32**: Medium batch CNN inference - Balanced inference throughput testing
  - **Batch 64**: Large batch CNN inference - High-throughput inference performance test

#### ResNet Inference
- **Model**: Same ResNet architecture as training, evaluation mode
- **Operations Tested**: Complex forward passes with skip connections and normalization
- **Batch Sizes**: 8, 16, 32
- **Test Descriptions**:
  - **Batch 8**: Small batch ResNet inference - Complex model low-latency inference test
  - **Batch 16**: Medium batch ResNet inference - Balanced complex inference workload
  - **Batch 32**: Large batch ResNet inference - High-throughput complex model inference

## Metrics Collected

### Performance Metrics
- **Throughput**: Samples processed per second
- **Latency**: Average iteration time in milliseconds
- **Duration**: Total test execution time
- **Iterations**: Number of batches processed

### Resource Metrics
- **GPU Utilization**: Peak GPU compute usage percentage
- **GPU Memory**: Peak GPU memory allocation in MB
- **CPU Usage**: Peak CPU utilization percentage
- **System Memory**: Peak system memory usage

### System Information
- **Hardware**: GPU names, CPU count, total memory
- **Software**: PyTorch version, CUDA version
- **Environment**: Hostname, timestamp

## Usage

### Standalone Execution
```bash
# Basic usage with default parameters (3 epochs, 200 batches)
python gpu_loadtest.py

# Custom epochs and batches with output file
python gpu_loadtest.py --num-epochs 5 --num-batches 500 --output results.json

# Quick test mode (reduced iterations)
python gpu_loadtest.py --quick

# With detailed file logging and custom parameters
python gpu_loadtest.py --log-file gpu_test.log --num-epochs 2 --num-batches 100 --num-workers 4
```

### Remote Deployment
The script is designed to be deployed and executed on remote GPU instances via the Lambda Labs CLI:

```bash
python cli.py test load --deployment-file cluster.json --ssh-key ~/.ssh/key.pem --num-epochs 3 --num-batches 200
```

## Command Line Arguments

- `--output, -o`: Output file for results (default: timestamped filename)
- `--device`: Device to use - 'cuda' or 'cpu' (default: auto-detect)
- `--num-epochs`: Number of epochs for training tests (default: 3)
- `--num-batches`: Number of batches per test (default: 200)
- `--num-workers`: Number of worker threads for data loading (default: 0)
- `--quick`: Run quick test with reduced iterations
- `--log-file`: Path for detailed logging output
- `--metrics-sample-rate`: Resource monitoring sample rate in seconds (default: 5.0)

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "system_info": {
    "hostname": "gpu-instance-1",
    "gpu_names": ["NVIDIA A100-SXM4-40GB"],
    "cuda_version": "11.8",
    "pytorch_version": "2.0.0"
  },
  "test_summary": {
    "requested_duration_minutes": 5,
    "actual_duration_seconds": 302.1,
    "total_tests": 12,
    "successful_tests": 12,
    "failed_tests": 0
  },
  "results": [
    {
      "test_name": "train_simple_cnn_bs32",
      "test_type": "training",
      "description": "Medium batch CNN training - Balanced test of GPU compute and memory bandwidth...",
      "model_name": "simple_cnn",
      "batch_size": 32,
      "throughput_samples_per_second": 1248.3,
      "avg_iteration_time_ms": 25.6,
      "peak_gpu_memory_mb": 2048,
      "peak_gpu_utilization_percent": 95.2,
      "success": true
    }
  ]
}
```

## Dependencies

### Required
- `torch` - PyTorch deep learning framework
- `psutil` - System resource monitoring
- `numpy` - Numerical computations

### Optional
- `GPUtil` - Enhanced GPU monitoring (provides additional GPU metrics)

## Performance Insights

### What Each Test Reveals

**Training Tests** evaluate:
- Memory bandwidth under gradient storage
- Compute efficiency with forward/backward passes
- Memory management for activations and gradients
- Optimization algorithm overhead

**Inference Tests** evaluate:
- Pure computational throughput
- Memory efficiency without gradient storage
- Latency characteristics for different batch sizes
- Model complexity impact on performance

### Batch Size Impact
- **Small batches** (8-16): Test low-latency scenarios and memory efficiency
- **Medium batches** (16-32): Balance between throughput and resource usage
- **Large batches** (32-64): Stress test memory capacity and maximum throughput

## Integration

This script integrates with the broader Lambda Labs testing infrastructure:

1. **Remote Deployment**: Automatically deployed to instances via SSH
2. **Result Collection**: Results collected and aggregated across multiple instances
3. **Performance Analysis**: Enables comparison across different GPU types and configurations
4. **Automated Reporting**: Integrated into CLI workflow for seamless testing

The GPU load testing suite provides comprehensive performance evaluation capabilities essential for GPU selection, performance validation, and infrastructure optimization in machine learning workflows.