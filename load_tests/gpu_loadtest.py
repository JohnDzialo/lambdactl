#!/usr/bin/env python3
"""
GPU Load Testing Script

This script performs comprehensive GPU benchmarking including:
- Training workloads with various neural network models
- Inference workloads with different batch sizes
- Memory usage testing
- Resource monitoring and timing

Designed to be deployed and run on remote GPU instances to evaluate performance.
"""

import json
import time
import sys
import logging
import psutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not available. GPU monitoring will be limited.")


@dataclass
class SystemInfo:
    """System information snapshot"""

    hostname: str
    cpu_count: int
    memory_total_gb: float
    gpu_count: int
    gpu_names: List[str]
    cuda_version: str
    pytorch_version: str
    timestamp: float


@dataclass
class ResourceMetrics:
    """Resource usage metrics at a point in time"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_metrics: List[Dict[str, Any]]


@dataclass
class ResourceStats:
    """Resource usage statistics over time"""

    min_value: float
    max_value: float
    avg_value: float
    sample_count: int


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""

    test_name: str
    test_type: str  # 'training' or 'inference'
    model_name: str
    batch_size: int
    duration_seconds: float
    iterations: int
    throughput_samples_per_second: float
    avg_iteration_time_ms: float
    peak_gpu_memory_mb: float
    peak_gpu_utilization_percent: float
    success: bool
    description: Optional[str] = None
    error_message: Optional[str] = None
    # Enhanced resource statistics
    cpu_usage_stats: Optional[ResourceStats] = None
    memory_usage_stats: Optional[ResourceStats] = None
    gpu_utilization_stats: Optional[ResourceStats] = None
    gpu_memory_stats: Optional[ResourceStats] = None
    monitoring_sample_rate: Optional[float] = None
    total_samples_collected: Optional[int] = None


class ResourceMonitor:
    """Background resource monitoring"""

    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self.metrics: List[ResourceMetrics] = []
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None

    def start_monitoring(self):
        """Start background resource monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.metrics = []
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop_monitoring(self):
        """Stop background resource monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=5.0)

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()

                # GPU metrics
                gpu_metrics = []
                if GPUTIL_AVAILABLE and torch.cuda.is_available():
                    try:
                        gpus = GPUtil.getGPUs()
                        for i, gpu in enumerate(gpus):
                            gpu_metrics.append(
                                {
                                    "gpu_id": i,
                                    "name": gpu.name,
                                    "utilization_percent": gpu.load * 100,
                                    "memory_used_mb": gpu.memoryUsed,
                                    "memory_total_mb": gpu.memoryTotal,
                                    "memory_percent": (gpu.memoryUsed / gpu.memoryTotal)
                                    * 100,
                                    "temperature_c": gpu.temperature,
                                }
                            )
                    except Exception as e:
                        print(f"Warning: GPU monitoring failed: {e}")

                # Add PyTorch GPU memory if available
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        torch_memory_mb = torch.cuda.memory_allocated(i) / (1024 * 1024)
                        torch_memory_reserved_mb = torch.cuda.memory_reserved(i) / (
                            1024 * 1024
                        )

                        if i < len(gpu_metrics):
                            gpu_metrics[i][
                                "torch_memory_allocated_mb"
                            ] = torch_memory_mb
                            gpu_metrics[i][
                                "torch_memory_reserved_mb"
                            ] = torch_memory_reserved_mb

                metric = ResourceMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_gb=memory.used / (1024**3),
                    gpu_metrics=gpu_metrics,
                )

                self.metrics.append(metric)

            except Exception as e:
                print(f"Warning: Resource monitoring error: {e}")

            time.sleep(self.interval)

    def get_peak_metrics(self) -> Dict[str, float]:
        """Get peak resource usage during monitoring period"""
        if not self.metrics:
            return {}

        peak_cpu = max(m.cpu_percent for m in self.metrics)
        peak_memory = max(m.memory_percent for m in self.metrics)

        peak_gpu_util = 0.0
        peak_gpu_memory = 0.0

        for metric in self.metrics:
            for gpu in metric.gpu_metrics:
                peak_gpu_util = max(peak_gpu_util, gpu.get("utilization_percent", 0))
                peak_gpu_memory = max(peak_gpu_memory, gpu.get("memory_used_mb", 0))

        return {
            "peak_cpu_percent": peak_cpu,
            "peak_memory_percent": peak_memory,
            "peak_gpu_utilization_percent": peak_gpu_util,
            "peak_gpu_memory_mb": peak_gpu_memory,
        }

    def get_resource_stats(self) -> Dict[str, ResourceStats]:
        """Calculate comprehensive resource statistics (min/max/average)"""
        if not self.metrics:
            return {}

        # Collect all values for each resource type
        cpu_values = [m.cpu_percent for m in self.metrics]
        memory_values = [m.memory_percent for m in self.metrics]

        # GPU metrics aggregation
        gpu_util_values = []
        gpu_memory_values = []

        for metric in self.metrics:
            # Use first GPU if available (can be extended for multi-GPU)
            if metric.gpu_metrics:
                gpu = metric.gpu_metrics[0]
                gpu_util_values.append(gpu.get("utilization_percent", 0))
                gpu_memory_values.append(gpu.get("memory_used_mb", 0))

        stats = {}

        # CPU statistics
        if cpu_values:
            stats["cpu_usage"] = ResourceStats(
                min_value=min(cpu_values),
                max_value=max(cpu_values),
                avg_value=sum(cpu_values) / len(cpu_values),
                sample_count=len(cpu_values),
            )

        # Memory statistics
        if memory_values:
            stats["memory_usage"] = ResourceStats(
                min_value=min(memory_values),
                max_value=max(memory_values),
                avg_value=sum(memory_values) / len(memory_values),
                sample_count=len(memory_values),
            )

        # GPU utilization statistics
        if gpu_util_values:
            stats["gpu_utilization"] = ResourceStats(
                min_value=min(gpu_util_values),
                max_value=max(gpu_util_values),
                avg_value=sum(gpu_util_values) / len(gpu_util_values),
                sample_count=len(gpu_util_values),
            )

        # GPU memory statistics
        if gpu_memory_values:
            stats["gpu_memory"] = ResourceStats(
                min_value=min(gpu_memory_values),
                max_value=max(gpu_memory_values),
                avg_value=sum(gpu_memory_values) / len(gpu_memory_values),
                sample_count=len(gpu_memory_values),
            )

        return stats


class SimpleConvNet(nn.Module):
    """Simple CNN for training benchmarks"""

    def __init__(self, num_classes=10):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class ResNetBlock(nn.Module):
    """Simple ResNet block for heavier training"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SimpleResNet(nn.Module):
    """Simple ResNet for heavy training workloads"""

    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class GPULoadTester:
    """Main GPU load testing class"""

    def __init__(self, device=None, metrics_sample_rate: float = 5.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")

        self.metrics_sample_rate = metrics_sample_rate
        self.monitor = ResourceMonitor(interval_seconds=metrics_sample_rate)
        self.results: List[BenchmarkResult] = []

        # Test descriptions for different model types and configurations
        self.test_descriptions = {
            # Training descriptions
            "train_simple_cnn": {
                "base": "Convolutional Neural Network training - Tests basic GPU compute with convolution, pooling, and backpropagation operations",
                "batch_specific": {
                    16: "Small batch CNN training - Tests GPU efficiency with limited parallelism and memory usage",
                    32: "Medium batch CNN training - Balanced test of GPU compute and memory bandwidth",
                    64: "Large batch CNN training - Tests GPU memory capacity and high-throughput compute",
                },
            },
            "train_resnet": {
                "base": "ResNet training with skip connections - Tests complex GPU workloads with residual blocks and batch normalization",
                "batch_specific": {
                    8: "Small batch ResNet training - Tests GPU with complex operations and moderate memory usage",
                    16: "Medium batch ResNet training - Balanced complex compute with memory efficiency",
                    32: "Large batch ResNet training - Stress test for GPU memory and complex operations",
                },
            },
            # Inference descriptions
            "inference_simple_cnn": {
                "base": "CNN inference throughput - Tests GPU inference speed and efficiency without gradient computation",
                "batch_specific": {
                    16: "Small batch CNN inference - Tests low-latency inference scenarios",
                    32: "Medium batch CNN inference - Balanced inference throughput testing",
                    64: "Large batch CNN inference - High-throughput inference performance test",
                },
            },
            "inference_resnet": {
                "base": "ResNet inference throughput - Tests complex model inference with skip connections and normalization",
                "batch_specific": {
                    8: "Small batch ResNet inference - Complex model low-latency inference test",
                    16: "Medium batch ResNet inference - Balanced complex inference workload",
                    32: "Large batch ResNet inference - High-throughput complex model inference",
                },
            },
        }

    def get_test_description(
        self, test_name: str, model_name: str, batch_size: int, test_type: str
    ) -> str:
        """Generate description for a specific test configuration"""
        # Create lookup key
        key = f"{test_type}_{model_name}"

        if key in self.test_descriptions:
            desc_config = self.test_descriptions[key]
            base_desc = desc_config["base"]

            # Check for batch-specific description
            if batch_size in desc_config.get("batch_specific", {}):
                specific_desc = desc_config["batch_specific"][batch_size]
                return f"{specific_desc}. {base_desc}"
            else:
                return f"Batch size {batch_size} {test_type} - {base_desc}"
        else:
            # Fallback description
            operation = (
                "forward/backward pass"
                if test_type == "training"
                else "forward pass only"
            )
            return f"{model_name} {test_type} with batch size {batch_size} - GPU {operation} performance test"

    def get_system_info(self) -> SystemInfo:
        """Collect system information"""
        import socket

        gpu_names = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_names.append(torch.cuda.get_device_name(i))

        return SystemInfo(
            hostname=socket.gethostname(),
            cpu_count=psutil.cpu_count(),
            memory_total_gb=psutil.virtual_memory().total / (1024**3),
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            gpu_names=gpu_names,
            cuda_version=(
                torch.version.cuda if torch.cuda.is_available() else "Not available"
            ),
            pytorch_version=torch.__version__,
            timestamp=time.time(),
        )

    def create_synthetic_dataset(
        self,
        batch_size: int,
        num_batches: int,
        num_workers: int,
        input_size=(3, 32, 32),
        num_classes=10,
    ):
        """Create synthetic dataset for training/inference"""
        total_samples = batch_size * num_batches

        # Generate random data
        X = torch.randn(total_samples, *input_size)
        y = torch.randint(0, num_classes, (total_samples,))

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        return dataloader

    def run_training_benchmark(
        self,
        model_name: str,
        model: nn.Module,
        batch_size: int,
        num_epochs: int = 1,
        num_batches: int = 100,
        num_workers: int = 0,
    ) -> BenchmarkResult:
        """Run a training benchmark"""
        logger = logging.getLogger("gpu_loadtest")
        print(f"Running training benchmark: {model_name}, batch_size={batch_size}")
        logger.info(
            f"Starting training benchmark: {model_name}, batch_size={batch_size}, epochs={num_epochs}, batches={num_batches}"
        )

        try:
            model = model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # Create synthetic dataset
            dataloader = self.create_synthetic_dataset(
                batch_size, num_batches, num_workers
            )

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                logger.debug("Cleared GPU memory and reset peak memory stats")

            self.monitor.start_monitoring()
            start_time = time.time()
            iteration_count = 0

            model.train()
            logger.debug(f"Starting training loop: {num_epochs} epochs")
            for epoch in range(num_epochs):
                for batch_idx, (data, target) in enumerate(dataloader):
                    if batch_idx >= num_batches:
                        break

                    data, target = data.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    iteration_count += 1

                    # Log progress every 10 iterations
                    if iteration_count % 10 == 0:
                        logger.debug(
                            f"Training iteration {iteration_count}, loss: {loss.item():.4f}"
                        )

            end_time = time.time()
            self.monitor.stop_monitoring()

            duration = end_time - start_time
            throughput = (iteration_count * batch_size) / duration
            avg_iteration_time = (duration * 1000) / iteration_count

            peak_metrics = self.monitor.get_peak_metrics()
            resource_stats = self.monitor.get_resource_stats()

            # Get peak PyTorch memory if available
            peak_memory_mb = 0
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            test_name = f"train_{model_name}_bs{batch_size}"
            description = self.get_test_description(
                test_name, model_name, batch_size, "training"
            )

            result = BenchmarkResult(
                test_name=test_name,
                test_type="training",
                model_name=model_name,
                batch_size=batch_size,
                duration_seconds=duration,
                iterations=iteration_count,
                throughput_samples_per_second=throughput,
                avg_iteration_time_ms=avg_iteration_time,
                peak_gpu_memory_mb=max(
                    peak_memory_mb, peak_metrics.get("peak_gpu_memory_mb", 0)
                ),
                peak_gpu_utilization_percent=peak_metrics.get(
                    "peak_gpu_utilization_percent", 0
                ),
                success=True,
                description=description,
                # Enhanced resource statistics
                cpu_usage_stats=resource_stats.get("cpu_usage"),
                memory_usage_stats=resource_stats.get("memory_usage"),
                gpu_utilization_stats=resource_stats.get("gpu_utilization"),
                gpu_memory_stats=resource_stats.get("gpu_memory"),
                monitoring_sample_rate=self.metrics_sample_rate,
                total_samples_collected=len(self.monitor.metrics),
            )

            self.results.append(result)
            print(
                f"  Completed: {throughput:.1f} samples/sec, {avg_iteration_time:.1f}ms/iter"
            )
            logger.info(
                f"Training benchmark completed: {model_name}, throughput={throughput:.1f} samples/sec, duration={duration:.2f}s"
            )
            return result

        except Exception as e:
            self.monitor.stop_monitoring()
            logger.error(
                f"Training benchmark failed: {model_name}, batch_size={batch_size}, error: {e}",
                exc_info=True,
            )

            test_name = f"train_{model_name}_bs{batch_size}"
            description = self.get_test_description(
                test_name, model_name, batch_size, "training"
            )

            error_result = BenchmarkResult(
                test_name=test_name,
                test_type="training",
                model_name=model_name,
                batch_size=batch_size,
                duration_seconds=0,
                iterations=0,
                throughput_samples_per_second=0,
                avg_iteration_time_ms=0,
                peak_gpu_memory_mb=0,
                peak_gpu_utilization_percent=0,
                success=False,
                description=description,
                error_message=str(e),
                monitoring_sample_rate=self.metrics_sample_rate,
                total_samples_collected=len(self.monitor.metrics),
            )
            self.results.append(error_result)
            print(f"  Failed: {e}")
            return error_result

    def run_inference_benchmark(
        self,
        model_name: str,
        model: nn.Module,
        batch_size: int,
        num_batches: int = 1000,
        num_workers: int = 1,
    ) -> BenchmarkResult:
        """Run an inference benchmark"""
        logger = logging.getLogger("gpu_loadtest")
        print(f"Running inference benchmark: {model_name}, batch_size={batch_size}")
        logger.info(
            f"Starting inference benchmark: {model_name}, batch_size={batch_size}, batches={num_batches}"
        )

        try:
            model = model.to(self.device)
            model.eval()

            # Create synthetic dataset
            dataloader = self.create_synthetic_dataset(
                batch_size, num_batches, num_workers
            )

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                logger.debug("Cleared GPU memory and reset peak memory stats")

            self.monitor.start_monitoring()
            start_time = time.time()
            iteration_count = 0

            logger.debug(f"Starting inference loop: {num_batches} batches")
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    if batch_idx >= num_batches:
                        break

                    data = data.to(self.device)
                    output = model(data)
                    iteration_count += 1

                    # Log progress every 50 iterations
                    if iteration_count % 50 == 0:
                        logger.debug(f"Inference iteration {iteration_count}")

            end_time = time.time()
            self.monitor.stop_monitoring()

            duration = end_time - start_time
            throughput = (iteration_count * batch_size) / duration
            avg_iteration_time = (duration * 1000) / iteration_count

            peak_metrics = self.monitor.get_peak_metrics()
            resource_stats = self.monitor.get_resource_stats()

            # Get peak PyTorch memory if available
            peak_memory_mb = 0
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            test_name = f"inference_{model_name}_bs{batch_size}"
            description = self.get_test_description(
                test_name, model_name, batch_size, "inference"
            )

            result = BenchmarkResult(
                test_name=test_name,
                test_type="inference",
                model_name=model_name,
                batch_size=batch_size,
                duration_seconds=duration,
                iterations=iteration_count,
                throughput_samples_per_second=throughput,
                avg_iteration_time_ms=avg_iteration_time,
                peak_gpu_memory_mb=max(
                    peak_memory_mb, peak_metrics.get("peak_gpu_memory_mb", 0)
                ),
                peak_gpu_utilization_percent=peak_metrics.get(
                    "peak_gpu_utilization_percent", 0
                ),
                success=True,
                description=description,
                # Enhanced resource statistics
                cpu_usage_stats=resource_stats.get("cpu_usage"),
                memory_usage_stats=resource_stats.get("memory_usage"),
                gpu_utilization_stats=resource_stats.get("gpu_utilization"),
                gpu_memory_stats=resource_stats.get("gpu_memory"),
                monitoring_sample_rate=self.metrics_sample_rate,
                total_samples_collected=len(self.monitor.metrics),
            )

            self.results.append(result)
            print(
                f"  Completed: {throughput:.1f} samples/sec, {avg_iteration_time:.2f}ms/iter"
            )
            logger.info(
                f"Inference benchmark completed: {model_name}, throughput={throughput:.1f} samples/sec, duration={duration:.2f}s"
            )
            return result

        except Exception as e:
            self.monitor.stop_monitoring()
            logger.error(
                f"Inference benchmark failed: {model_name}, batch_size={batch_size}, error: {e}",
                exc_info=True,
            )

            test_name = f"inference_{model_name}_bs{batch_size}"
            description = self.get_test_description(
                test_name, model_name, batch_size, "inference"
            )

            error_result = BenchmarkResult(
                test_name=test_name,
                test_type="inference",
                model_name=model_name,
                batch_size=batch_size,
                duration_seconds=0,
                iterations=0,
                throughput_samples_per_second=0,
                avg_iteration_time_ms=0,
                peak_gpu_memory_mb=0,
                peak_gpu_utilization_percent=0,
                success=False,
                description=description,
                error_message=str(e),
                monitoring_sample_rate=self.metrics_sample_rate,
                total_samples_collected=len(self.monitor.metrics),
            )
            self.results.append(error_result)
            print(f"  Failed: {e}")
            return error_result

    def run_full_benchmark_suite(
        self, num_epochs: int = 3, num_batches: int = 200, num_workers: int = 0
    ) -> Dict[str, Any]:
        """Run complete benchmark suite with specified epochs and batches"""
        logger = logging.getLogger("gpu_loadtest")

        print("Starting GPU Load Test Suite")
        print("=" * 50)

        start_time = time.time()
        system_info = self.get_system_info()

        logger.info(
            f"Starting benchmark suite with {num_epochs} epochs and {num_batches} batches"
        )
        logger.info(f"System: {system_info.hostname}")
        logger.info(
            f"GPUs: {', '.join(system_info.gpu_names) if system_info.gpu_names else 'None'}"
        )

        print(f"System: {system_info.hostname}")
        print(
            f"GPUs: {', '.join(system_info.gpu_names) if system_info.gpu_names else 'None'}"
        )
        print(
            f"PyTorch: {system_info.pytorch_version}, CUDA: {system_info.cuda_version}"
        )
        print(f"Training epochs: {num_epochs}, Batches per test: {num_batches}")
        print()

        # Define test configurations
        models_and_configs = [
            ("simple_cnn", SimpleConvNet(), [16, 32, 64]),
            ("resnet", SimpleResNet(), [8, 16, 32]),
        ]

        logger.info(
            f"Using {num_epochs} training epochs and {num_batches} batches per test"
        )

        test_count = 0

        # Run training benchmarks
        print("Training Benchmarks:")
        print("-" * 30)
        logger.info("Starting training benchmarks")

        for model_name, model_class, batch_sizes in models_and_configs:
            for batch_size in batch_sizes:
                logger.info(
                    f"Running training benchmark: {model_name}, batch_size={batch_size}"
                )
                self.run_training_benchmark(
                    model_name,
                    model_class,
                    batch_size,
                    num_epochs=num_epochs,
                    num_batches=num_batches,
                    num_workers=num_workers,
                )
                test_count += 1

                # Brief pause between tests
                time.sleep(1)

        print()

        # Run inference benchmarks
        print("Inference Benchmarks:")
        print("-" * 30)
        logger.info("Starting inference benchmarks")

        for model_name, model_class, batch_sizes in models_and_configs:
            for batch_size in batch_sizes:
                logger.info(
                    f"Running inference benchmark: {model_name}, batch_size={batch_size}"
                )
                self.run_inference_benchmark(
                    model_name, model_class, batch_size, num_batches=num_batches
                )
                test_count += 1

                # Brief pause between tests
                time.sleep(1)

        actual_end_time = time.time()
        total_duration = actual_end_time - start_time

        # Compile results
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]

        logger.info(f"Benchmark suite completed in {total_duration:.1f}s")
        logger.info(f"Executed {test_count} total tests")
        logger.info(f"Successful tests: {len(successful_tests)}")
        logger.info(f"Failed tests: {len(failed_tests)}")

        summary = {
            "system_info": asdict(system_info),
            "test_summary": {
                "num_epochs": num_epochs,
                "num_batches": num_batches,
                "actual_duration_seconds": total_duration,
                "total_tests": len(self.results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "tests_executed": test_count,
                "timestamp": datetime.now().isoformat(),
            },
            "results": [asdict(r) for r in self.results],
        }

        print(f"\nSummary:")
        print(f"Configuration: {num_epochs} epochs, {num_batches} batches per test")
        print(
            f"Execution Time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)"
        )
        print(f"Tests: {len(successful_tests)}/{len(self.results)} successful")

        if successful_tests:
            training_results = [
                r for r in successful_tests if r.test_type == "training"
            ]
            inference_results = [
                r for r in successful_tests if r.test_type == "inference"
            ]

            if training_results:
                avg_training_throughput = np.mean(
                    [r.throughput_samples_per_second for r in training_results]
                )
                print(
                    f"Avg Training Throughput: {avg_training_throughput:.1f} samples/sec"
                )
                logger.info(
                    f"Training tests completed: {len(training_results)}, avg throughput: {avg_training_throughput:.1f} samples/sec"
                )

            if inference_results:
                avg_inference_throughput = np.mean(
                    [r.throughput_samples_per_second for r in inference_results]
                )
                print(
                    f"Avg Inference Throughput: {avg_inference_throughput:.1f} samples/sec"
                )
                logger.info(
                    f"Inference tests completed: {len(inference_results)}, avg throughput: {avg_inference_throughput:.1f} samples/sec"
                )

        if failed_tests:
            logger.warning(f"Failed tests: {len(failed_tests)}")
            for failed_test in failed_tests:
                logger.error(
                    f"Failed test {failed_test.test_name}: {failed_test.error_message}"
                )

        return summary

    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to JSON file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"gpu_loadtest_results_{timestamp}.json"

        # Only save existing results - don't run tests again
        summary = {
            "system_info": asdict(self.get_system_info()),
            "results": [asdict(r) for r in self.results],
            "test_summary": {
                "total_tests": len(self.results),
                "successful_tests": len([r for r in self.results if r.success]),
                "failed_tests": len([r for r in self.results if not r.success]),
                "timestamp": datetime.now().isoformat(),
            },
        }

        with open(filename, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to: {filename}")
        return filename


def setup_logging(log_file: str = None) -> logging.Logger:
    """Setup logging configuration for GPU load testing"""
    logger = logging.getLogger("gpu_loadtest")
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")

    return logger


def main():
    """Main entry point for standalone execution"""
    import argparse

    parser = argparse.ArgumentParser(description="GPU Load Testing Script")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick test with reduced iterations"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of epochs for training tests (default: 3)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=200,
        help="Number of batches per test (default: 200)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker threads for data loading (default: 0)",
    )
    parser.add_argument("--log-file", help="Log file path for detailed logging")
    parser.add_argument(
        "--metrics-sample-rate",
        type=float,
        default=5.0,
        help="Resource monitoring sample rate in seconds (default: 5.0)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting GPU Load Test")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Initialize tester
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        logger.info(f"Metrics sample rate: {args.metrics_sample_rate} seconds")
        tester = GPULoadTester(
            device=device, metrics_sample_rate=args.metrics_sample_rate
        )

        # Run benchmarks with specified parameters
        if args.quick:
            logger.info("Running quick test mode...")
            print("Running quick test mode...")
            # Run a subset of tests for quick verification
            tester.run_training_benchmark(
                "simple_cnn", SimpleConvNet(), 16, num_epochs=1, num_batches=10
            )
            tester.run_inference_benchmark(
                "simple_cnn", SimpleConvNet(), 32, num_batches=50
            )
        else:
            logger.info(
                f"Running full benchmark suite with {args.num_epochs} epochs and {args.num_batches} batches"
            )
            # Run full benchmark suite with specified parameters
            tester.run_full_benchmark_suite(
                num_epochs=args.num_epochs,
                num_batches=args.num_batches,
                num_workers=args.num_workers,
            )

        # Save results
        output_file = args.output or f"gpu_loadtest_results_{int(time.time())}.json"
        result_file = tester.save_results(output_file)
        logger.info(f"Results saved to: {result_file}")

    except Exception as e:
        logger.error(f"GPU load test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
