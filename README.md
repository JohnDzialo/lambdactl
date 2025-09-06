# Lambda Labs Cluster Manager

A command-line tool for deploying GPU clusters to Lambda Labs, running load tests, and collecting performance metrics.

## Features

- **1-Click Cluster Deployment**: Deploy multiple GPU instances in parallel
- **Load Testing**: Run stress tests, benchmarks, or custom commands on deployed instances
- **Metrics Collection**: Collect and save performance metrics over time
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
python cli.py list-types
```

### 3. Deploy a Cluster

```bash
# Deploy 2 A100 instances
python cli.py deploy cluster --type gpu_1x_a100 --count 2 --name-prefix my-cluster

# Deploy with SSH key
python cli.py deploy cluster --type gpu_1x_h100 --count 4 --ssh-key my-key
```

### 4. Run Load Tests

```bash
# Stress test using deployment file
python cli.py test load --deployment-file ~/.lambda-deploy/metrics/deployment_1234567890.json --duration 15

# Stress test specific instances
python cli.py test load --instances "i-123,i-456" --test-type stress --duration 10

# Benchmark test
python cli.py test load --instances "i-123" --test-type benchmark --duration 5
```

### 5. Collect Metrics

```bash
# Collect metrics for 10 minutes, every 30 seconds
python cli.py metrics collect --deployment-file ~/.lambda-deploy/metrics/deployment_1234567890.json --duration 10 --interval 30
```

### 6. Cleanup

```bash
# Terminate instances from deployment
python cli.py cleanup terminate --deployment-file ~/.lambda-deploy/metrics/deployment_1234567890.json

# Terminate specific instances
python cli.py cleanup terminate --instances "i-123,i-456"

# Terminate ALL instances (careful!)
python cli.py cleanup terminate --all
```

## Commands Reference

### Configuration
- `configure` - Set up Lambda Labs API credentials

### Instance Management
- `list-types` - Show available GPU instance types and pricing
- `list-instances` - Show all current instances

### Deployment
- `deploy cluster` - Deploy a cluster of GPU instances
  - `--type` - Instance type (default: gpu_1x_a100)
  - `--count` - Number of instances (default: 2)
  - `--name-prefix` - Instance name prefix (default: cluster)
  - `--ssh-key` - SSH key name to use
  - `--wait/--no-wait` - Wait for instances to be ready (default: True)
  - `--timeout` - Timeout in minutes (default: 15)

### Load Testing
- `test load` - Run load tests on instances
  - `--instances` - Comma-separated instance IDs
  - `--deployment-file` - Use instances from deployment file
  - `--duration` - Test duration in minutes (default: 10)
  - `--test-type` - Test type: stress, benchmark, custom (default: stress)
  - `--custom-command` - Custom command for custom test type

### Metrics
- `metrics collect` - Collect performance metrics
  - `--instances` - Comma-separated instance IDs
  - `--deployment-file` - Use instances from deployment file
  - `--duration` - Collection duration in minutes (default: 5)
  - `--interval` - Collection interval in seconds (default: 10)

### Cleanup
- `cleanup terminate` - Terminate instances
  - `--instances` - Comma-separated instance IDs
  - `--deployment-file` - Terminate instances from deployment file
  - `--all` - Terminate ALL instances (requires confirmation)

## File Locations

- Configuration: `~/.lambda-deploy/config.json`
- Logs: `~/.lambda-deploy/logs/lambda-deploy.log`
- Metrics & Deployments: `~/.lambda-deploy/metrics/`

## Example Workflow

```bash
# 1. Configure
python cli.py configure

# 2. Check available types
python cli.py list-types

# 3. Deploy cluster
python cli.py deploy cluster --type gpu_1x_a100 --count 4 --name-prefix loadtest

# 4. Run load test (will use the deployment file from step 3)
python cli.py test load --deployment-file ~/.lambda-deploy/metrics/deployment_*.json --duration 15

# 5. Collect metrics during another test
python cli.py metrics collect --deployment-file ~/.lambda-deploy/metrics/deployment_*.json --duration 20 --interval 15

# 6. Cleanup
python cli.py cleanup terminate --deployment-file ~/.lambda-deploy/metrics/deployment_*.json
```

## API Integration

The tool uses Lambda Labs' Cloud API endpoints:

- Authentication via Bearer token
- Instance management (create, list, terminate, restart)
- SSH key management
- File system management
- Instance status monitoring

## Extending the Tool

The CLI is built with Click and can be easily extended:

1. **Add new commands**: Create new command groups or individual commands
2. **Custom load tests**: Implement new test types in the load testing module
3. **Enhanced metrics**: Add more sophisticated metrics collection via SSH
4. **Visualization**: Add plotting capabilities for collected metrics

## Notes

- Instance deployment uses parallel processing for faster cluster creation
- Metrics collection currently uses simulated data (extend with actual SSH-based collection)
- All operations are logged to both console and log files
- Configuration and data files are stored in `~/.lambda-deploy/`