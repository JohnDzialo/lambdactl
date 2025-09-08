# Lambda Labs GPU Load Testing Suite

A command-line tool for deploying GPU clusters to Lambda Labs and running comprehensive PyTorch-based load tests with remote execution capabilities.

## Features

- **GPU Cluster Deployment**: Deploy multiple GPU instances in parallel
- **PyTorch Load Testing**: Run comprehensive GPU benchmarks including training and inference workloads
- **Remote Execution**: SSH-based remote test execution with automatic setup
- **Performance Monitoring**: Real-time resource monitoring with min/max/avg statistics
- **Result Aggregation**: Collect and aggregate results from multiple instances
- **Easy Cleanup**: Terminate instances individually or by deployment

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Configure API Key

```bash
python cli.py configure
# Enter your Lambda Labs API key when prompted
```

### 2. List Available Instance Types

```bash
python cli.py list-types --available
```

### 3. Deploy a Cluster

```bash
# Deploy 2 A100 instances
python cli.py deploy cluster --type gpu_1x_a100 --count 2 --name-prefix my-cluster --region us-west-1 --ssh-key my-key

# Deploy with additional options
python cli.py deploy cluster --type gpu_1x_h100 --count 4 --ssh-key my-key --region us-east-1 --wait --timeout 20
```

### 4. Run GPU Load Tests

```bash
# Run load tests using deployment file
python cli.py test load --deployment-file ~/.lambda-deploy/metrics/deployment_1234567890.json --ssh-key ~/.ssh/my-key.pem

# Run load tests on specific instances
python cli.py test load --instances "192.168.1.100,192.168.1.101" --ssh-key ~/.ssh/my-key.pem --num-epochs 5 --num-batches 500

# Quick test mode
python cli.py test load --deployment-file deployment.json --ssh-key key.pem --quick
```

### 5. Cleanup

```bash
# Terminate instances from deployment
python cli.py cleanup terminate --deployment-file ~/.lambda-deploy/metrics/deployment_1234567890.json

# Terminate specific instances
python cli.py cleanup terminate --instances "i-123,i-456"
```

## Commands Reference

### Configuration
- `configure` - Set up Lambda Labs API credentials

### Instance Management
- `list-types` - Show available GPU instance types and pricing
  - `--available` - Show only instance types with available capacity
- `list-instances` - Show all current instances

### Deployment
- `deploy cluster` - Deploy a cluster of GPU instances
  - `--type` - Instance type (default: gpu_1x_a100)
  - `--count` - Number of instances (default: 2)
  - `--name-prefix` - Instance name prefix (default: cluster)
  - `--region` - Region to deploy in (required)
  - `--ssh-key` - SSH key name to use (required)
  - `--file-systems` - Comma-separated list of file system names
  - `--wait/--no-wait` - Wait for instances to be ready (default: True)
  - `--timeout` - Timeout in minutes (default: 15)

### GPU Load Testing
- `test load` - Run PyTorch GPU load tests on instances
  - `--instances` - Comma-separated instance IPs
  - `--deployment-file` - Use instances from deployment file
  - `--ssh-key` - SSH private key file path (required)
  - `--username` - SSH username (default: ubuntu)
  - `--quick` - Run quick test with reduced iterations
  - `--num-epochs` - Number of epochs for training tests (default: 3)
  - `--num-batches` - Number of batches per test (default: 200)
  - `--num-workers` - Number of worker threads for data loading (default: 0)
  - `--metrics-sample-rate` - Resource monitoring sample rate in seconds (default: 5.0)
  - `--max-workers` - Maximum parallel workers (default: 5)
  - `--output` - Output file for aggregated results

### Cleanup
- `cleanup terminate` - Terminate instances
  - `--instances` - Comma-separated instance IDs
  - `--deployment-file` - Terminate instances from deployment file

## GPU Load Testing Details

The GPU load testing suite performs comprehensive PyTorch benchmarks:

### Test Types
- **Training Benchmarks**: CNN and ResNet training with forward/backward passes
- **Inference Benchmarks**: Forward-pass only performance testing
- **Multiple Batch Sizes**: Tests with different batch sizes (8, 16, 32, 64)
- **Resource Monitoring**: CPU, GPU, and memory usage tracking

### Models Tested
- **Simple CNN**: 3-layer convolutional network
- **ResNet**: Residual network with skip connections and batch normalization

### Metrics Collected
- Throughput (samples/sec)
- Average iteration time
- Peak GPU memory usage
- Peak GPU utilization
- CPU and system memory usage
- Min/Max/Average resource statistics

## Example Workflow

```bash
# 1. Configure API credentials
python cli.py configure

# 2. Check available GPU types
python cli.py list-types --available

# 3. Deploy GPU cluster
python cli.py deploy cluster --type gpu_1x_a100 --count 2 --name-prefix benchmark --region us-west-1 --ssh-key my-lambda-key

# 4. Run comprehensive GPU load tests
python cli.py test load --deployment-file ~/.lambda-deploy/metrics/deployment_*.json --ssh-key ~/.ssh/lambda_key.pem --num-epochs 5 --num-batches 300

# 5. Cleanup
python cli.py cleanup terminate --deployment-file ~/.lambda-deploy/metrics/deployment_*.json
```

## File Locations

- **Configuration**: `~/.lambda-deploy/config.json`
- **Logs**: `~/.lambda-deploy/logs/lambda-deploy.log`
- **Metrics & Deployments**: `~/.lambda-deploy/metrics/`
- **GPU Test Results**: JSON files with detailed performance metrics

## GPU Test Results Format

Results are saved as JSON with comprehensive metrics:

```json
{
  "summary": {
    "total_instances": 2,
    "successful_instances": 2,
    "failed_instances": 0,
    "success_rate": 1.0
  },
  "performance_summary": {
    "total_tests": 24,
    "training": {
      "avg_throughput": 1248.3,
      "max_throughput": 2156.7
    },
    "inference": {
      "avg_throughput": 3421.5,
      "max_throughput": 4789.2
    }
  },
  "successful_results": [...]
}
```

## Dependencies

- **PyTorch**: Deep learning framework for GPU benchmarks
- **Paramiko**: SSH client for remote execution
- **Pydantic**: Data validation and serialization
- **Click**: Command-line interface framework
- **Requests**: HTTP client for Lambda Labs API
- **psutil**: System resource monitoring
- **GPUtil**: GPU monitoring (optional)

## Architecture

- **CLI Module**: Click-based command interface
- **Lambda API**: HTTP client for Lambda Labs API
- **Remote Executor**: SSH-based remote test execution
- **GPU Load Tester**: PyTorch benchmarking suite with resource monitoring
- **Test Suite**: Pytest-based testing framework

## Notes

- **Remote Setup**: Automatically installs PyTorch and dependencies on remote instances
- **SSH Key Required**: Must provide SSH private key file for remote access
- **CUDA Support**: Tests automatically detect and use available GPUs
- **Resource Monitoring**: Background monitoring during tests with configurable sample rates
- **Error Handling**: Comprehensive error handling with detailed logging
- **Parallel Execution**: Tests run in parallel across multiple instances