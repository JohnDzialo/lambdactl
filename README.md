# Lambda Labs GPU Load Testing Suite

A comprehensive command-line tool for deploying GPU clusters to Lambda Labs and running high-performance PyTorch-based load tests with separated GPU testing and resource monitoring for optimal performance.

## Features

- **GPU Cluster Deployment**: Deploy multiple GPU instances in parallel with advanced configuration options
- **Separated Architecture**: Independent GPU load testing and resource monitoring processes to avoid CPU bottlenecks
- **PyTorch Load Testing**: Comprehensive benchmarks including CNNs, Transformers, and memory stress tests
- **Mixed Precision Support**: FP32 and FP16 testing for optimal performance analysis
- **Remote Execution**: SSH-based remote test execution with automatic dependency setup
- **Resource Monitoring**: Standalone monitoring with configurable intervals (CPU, Memory, GPU, Network, Disk I/O)
- **Result Aggregation**: Intelligent aggregation of results from multiple instances
- **Easy Cleanup**: Terminate instances individually or by deployment batch

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

# Deploy H100 instances with advanced options
python cli.py deploy cluster --type gpu_1x_h100_pcie --count 4 --ssh-key my-key --region us-east-1 --wait --timeout 20

# Deploy with custom configurations
python cli.py deploy cluster --type gpu_8x_h100 --count 1 --name-prefix benchmark \
  --region us-west-2 --ssh-key my-key \
  --tags "project=ml,team=research" \
  --firewall-rules rule-id-1,rule-id-2 \
  --hostname custom-hostname \
  --image custom-image-id
```

### 4. Run GPU Load Tests

```bash
# Run load tests using deployment file with monitoring
python cli.py test load --deployment-file ~/.lambda-deploy/metrics/deployment_1234567890.json \
  --ssh-key ~/.ssh/my-key.pem \
  --monitoring --monitoring-interval 60

# Run load tests on specific instances without monitoring
python cli.py test load --instances "instance-id-1,instance-id-2" \
  --ssh-key ~/.ssh/my-key.pem \
  --no-monitoring

# Quick test mode with custom monitoring interval
python cli.py test load --deployment-file deployment.json \
  --ssh-key key.pem \
  --quick \
  --monitoring-interval 30 \
  --max-workers 10

# Custom output file
python cli.py test load --deployment-file deployment.json \
  --ssh-key key.pem \
  --output custom_results.json
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
  - `--file-systems` - Comma-separated list of file system names (optional)
  - `--hostname` - Custom hostname for instances (optional)
  - `--image` - Custom image ID (optional)
  - `--user-data` - User data script to run on instance startup (optional)
  - `--tags` - Tags for instances, format: key=value,key=value (optional)
  - `--firewall-rules` - Firewall ruleset IDs, comma-separated (optional)
  - `--wait/--no-wait` - Wait for instances to be ready (default: True)
  - `--timeout` - Timeout in minutes (default: 15)

### GPU Load Testing
- `test load` - Run PyTorch GPU load tests on instances with separated monitoring
  - `--instances` - Comma-separated instance IDs to test
  - `--deployment-file` - Use instances from deployment file
  - `--ssh-key` - SSH private key file path (required)
  - `--username` - SSH username (default: ubuntu)
  - `--quick` - Run quick test with reduced iterations
  - `--monitoring/--no-monitoring` - Enable/disable resource monitoring (default: enabled)
  - `--monitoring-interval` - Resource monitoring interval in seconds (default: 60.0)
  - `--max-workers` - Maximum parallel workers (default: 5)
  - `--output` - Output file for aggregated results

### Cleanup
- `cleanup terminate` - Terminate instances
  - `--instances` - Comma-separated instance IDs
  - `--deployment-file` - Terminate instances from deployment file

## GPU Load Testing Details

The GPU load testing suite performs comprehensive PyTorch benchmarks with separated architecture for optimal performance:

### Separated Architecture
- **GPU Load Testing Process**: Runs pure GPU workloads without CPU monitoring overhead
- **Resource Monitoring Process**: Independent monitoring process with configurable intervals
- **No CPU Bottlenecks**: Separation ensures accurate GPU performance measurements

### Test Types
- **Training Benchmarks**: Forward and backward passes with gradient computation
- **Inference Benchmarks**: Forward-pass only performance testing
- **Mixed Precision**: Both FP32 and FP16 testing for performance comparison
- **Memory Scaling Tests**: Find maximum batch sizes for models

### Models Tested
- **Simple CNN**: 4-layer convolutional network (~8.9M parameters)
- **Large Transformer (1B)**: 12-layer transformer with 1024 hidden dimensions (~256M parameters)
- **Large Transformer (3B)**: 16-layer transformer with 1536 hidden dimensions
- **GPU Memory Stressor**: Dense network designed to stress GPU memory (~536M parameters)

### Metrics Collected
- Throughput (samples/sec)
- Average iteration time (ms)
- Peak GPU memory usage (MB)
- GPU utilization percentage
- CPU and system memory usage
- Network I/O (MB sent/received)
- Disk I/O (MB read/written)
- Load average (Unix systems)

## Example Workflow

```bash
# 1. Configure API credentials
python cli.py configure

# 2. Check available GPU types
python cli.py list-types --available

# 3. Deploy GPU cluster
python cli.py deploy cluster --type gpu_1x_h100_pcie --count 2 --name-prefix benchmark --region us-west-1 --ssh-key my-lambda-key

# 4. Run comprehensive GPU load tests with monitoring
python cli.py test load --deployment-file ~/.lambda-deploy/metrics/deployment_*.json --ssh-key ~/.ssh/lambda_key.pem --monitoring-interval 30

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
    "success_rate": 1.0,
    "monitoring_enabled": true,
    "monitoring_interval_seconds": 60.0
  },
  "performance_summary": {
    "total_tests": 48,
    "training_tests": 24,
    "inference_tests": 24,
    "training": {
      "avg_throughput": 1248.3,
      "max_throughput": 2156.7,
      "min_throughput": 892.1
    },
    "inference": {
      "avg_throughput": 3421.5,
      "max_throughput": 4789.2,
      "min_throughput": 2103.8
    }
  },
  "successful_results": [
    {
      "hostname": "instance-1",
      "success": true,
      "execution_time": 1823.5,
      "test_results": {...},
      "monitoring_results": {
        "duration_seconds": 1950.2,
        "snapshots": [...],
        "system_info": {...}
      }
    }
  ]
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

### Separated Testing Architecture
The system uses a separated architecture to ensure optimal performance:

1. **GPU Load Testing Process** (`gpu_loadtest.py`)
   - Pure GPU workload execution without monitoring overhead
   - Direct GPU memory allocation for synthetic datasets
   - Mixed precision training and inference benchmarks
   - Memory scaling tests to find maximum batch sizes

2. **Resource Monitoring Process** (`resource_monitor.py`)
   - Standalone monitoring with configurable intervals
   - Tracks CPU, Memory, GPU, Network, and Disk I/O
   - Runs independently to avoid affecting GPU performance
   - Graceful shutdown with signal handling

3. **Remote Executor** (`remote_executor.py`)
   - Coordinates both processes on remote instances
   - SSH-based deployment and execution
   - Parallel execution across multiple instances
   - Automatic dependency installation

### Core Modules
- **CLI Module** (`cli.py`): Click-based command interface
- **Lambda API** (`lambda_api.py`): HTTP client for Lambda Labs API
- **Remote Executor**: SSH-based remote test execution with Paramiko
- **GPU Load Tester**: PyTorch benchmarking suite optimized for H100/H200
- **Resource Monitor**: Standalone monitoring with psutil and GPUtil

## Performance Optimizations

- **GPU-Direct Data**: Synthetic datasets created directly on GPU memory
- **Mixed Precision**: FP16 support for improved throughput
- **Zero Data Workers**: Eliminates CPU bottlenecks in data loading
- **Separated Monitoring**: Independent process prevents monitoring overhead
- **Batch Processing**: Efficient parallel execution across instances

## Notes

- **H100/H200 Optimized**: Designed for high-end GPU instances with large batch sizes
- **Automatic Setup**: Installs PyTorch with CUDA support on remote instances
- **SSH Key Required**: Must provide SSH private key file for remote access
- **CUDA Detection**: Automatically detects and utilizes available GPUs
- **Configurable Monitoring**: Adjust monitoring interval based on test duration
- **Comprehensive Logging**: Detailed logs for debugging and analysis
- **Error Recovery**: Robust error handling with retry mechanisms