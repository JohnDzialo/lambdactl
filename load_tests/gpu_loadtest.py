#!/usr/bin/env python3
"""
Pure GPU Load Testing Script

Focused on GPU compute workloads without CPU-intensive monitoring.
Designed for H100/H200 instances with large batch sizes and models.
"""

import json
import time
import sys
import logging
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
    success: bool
    description: Optional[str] = None
    error_message: Optional[str] = None
    precision: str = "fp32"  # fp32, fp16, bf16
    sequence_length: Optional[int] = None
    model_parameters: Optional[int] = None


class SimpleConvNet(nn.Module):
    """Simple CNN for training benchmarks"""
    def __init__(self, num_classes=1000):  # Increased classes for more compute
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 2 * 2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 512 * 2 * 2)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class LargeTransformer(nn.Module):
    """Large Transformer model for H100/H200 testing"""
    def __init__(self, vocab_size=50000, d_model=2048, nhead=32, num_layers=24, seq_len=2048):
        super(LargeTransformer, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Calculate approximate parameter count
        self.param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        seq_len = x.size(1)
        pos_emb = self.pos_embedding[:seq_len].unsqueeze(0)
        x = self.embedding(x) + pos_emb
        x = self.transformer(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x


class GPUMemoryStressor(nn.Module):
    """Model designed to stress GPU memory and compute"""
    def __init__(self, hidden_dim=8192, num_layers=8):
        super(GPUMemoryStressor, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        self.layers = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, 1000)
        
        self.param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        x = self.layers(x)
        return self.final(x)


class PureGPULoadTester:
    """Pure GPU load testing class without CPU monitoring overhead"""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("CUDA not available, using CPU")

        self.results: List[BenchmarkResult] = []

    def get_system_info(self) -> SystemInfo:
        """Collect basic system information"""
        import socket
        import psutil

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
            cuda_version=torch.version.cuda if torch.cuda.is_available() else "Not available",
            pytorch_version=torch.__version__,
            timestamp=time.time(),
        )

    def create_gpu_dataset(self, batch_size: int, num_batches: int, input_shape, num_classes=1000, device=None):
        """Create synthetic dataset directly on GPU to avoid CPU bottleneck"""
        device = device or self.device
        total_samples = batch_size * num_batches
        
        # Generate data directly on GPU
        if isinstance(input_shape, tuple) and len(input_shape) == 3:  # Image data
            X = torch.randn(total_samples, *input_shape, device=device)
            y = torch.randint(0, num_classes, (total_samples,), device=device)
        else:  # Sequence data
            seq_len = input_shape
            X = torch.randint(0, 50000, (total_samples, seq_len), device=device)  # Token IDs
            y = torch.randint(0, num_classes, (total_samples,), device=device)

        dataset = TensorDataset(X, y)
        # Use minimal workers to avoid CUDA multi-process issues
        # Since data is already on GPU, don't use pin_memory
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  # No subprocesses to avoid CUDA conflicts
            pin_memory=False  # Data is already on GPU
        )
        
        return dataloader

    def run_training_benchmark(
        self,
        model_name: str,
        model: nn.Module,
        batch_size: int,
        num_epochs: int = 1,
        num_batches: int = 100,
        use_mixed_precision: bool = False,
        input_shape=(3, 224, 224),
    ) -> BenchmarkResult:
        """Run training benchmark with optional mixed precision"""
        print(f"Training: {model_name}, batch_size={batch_size}, mixed_precision={use_mixed_precision}")
        
        try:
            model = model.to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Mixed precision setup
            scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
            precision = "fp16" if use_mixed_precision else "fp32"
            
            # Get model parameter count
            param_count = sum(p.numel() for p in model.parameters())
            
            # Create dataset on GPU
            dataloader = self.create_gpu_dataset(batch_size, num_batches, input_shape)

            # Clear GPU memory and reset stats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()
            iteration_count = 0

            model.train()
            for epoch in range(num_epochs):
                for batch_idx, (data, target) in enumerate(dataloader):
                    if batch_idx >= num_batches:
                        break

                    # Data is already on GPU, no need to move it
                    # data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                    optimizer.zero_grad()
                    
                    if use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            output = model(data)
                            loss = criterion(output, target)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                    iteration_count += 1

            end_time = time.time()
            duration = end_time - start_time
            throughput = (iteration_count * batch_size) / duration
            avg_iteration_time = (duration * 1000) / iteration_count

            # Get peak GPU memory
            peak_memory_mb = 0
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            test_name = f"train_{model_name}_bs{batch_size}_{precision}"
            
            result = BenchmarkResult(
                test_name=test_name,
                test_type="training",
                model_name=model_name,
                batch_size=batch_size,
                duration_seconds=duration,
                iterations=iteration_count,
                throughput_samples_per_second=throughput,
                avg_iteration_time_ms=avg_iteration_time,
                peak_gpu_memory_mb=peak_memory_mb,
                success=True,
                precision=precision,
                model_parameters=param_count,
                sequence_length=input_shape if isinstance(input_shape, int) else None,
                description=f"{model_name} training with {precision}, {param_count:,} parameters"
            )

            self.results.append(result)
            print(f"  ✓ {throughput:.1f} samples/sec, {avg_iteration_time:.1f}ms/iter, {peak_memory_mb:.0f}MB peak")
            return result

        except Exception as e:
            error_result = BenchmarkResult(
                test_name=f"train_{model_name}_bs{batch_size}_{precision}",
                test_type="training",
                model_name=model_name,
                batch_size=batch_size,
                duration_seconds=0,
                iterations=0,
                throughput_samples_per_second=0,
                avg_iteration_time_ms=0,
                peak_gpu_memory_mb=0,
                success=False,
                error_message=str(e),
                precision=precision if 'precision' in locals() else "fp32"
            )
            self.results.append(error_result)
            print(f"  ✗ Failed: {e}")
            return error_result

    def run_inference_benchmark(
        self,
        model_name: str,
        model: nn.Module,
        batch_size: int,
        num_batches: int = 500,
        use_mixed_precision: bool = False,
        input_shape=(3, 224, 224),
    ) -> BenchmarkResult:
        """Run inference benchmark with optional mixed precision"""
        print(f"Inference: {model_name}, batch_size={batch_size}, mixed_precision={use_mixed_precision}")
        
        try:
            model = model.to(self.device)
            model.eval()
            
            precision = "fp16" if use_mixed_precision else "fp32"
            param_count = sum(p.numel() for p in model.parameters())
            
            # Create dataset on GPU
            dataloader = self.create_gpu_dataset(batch_size, num_batches, input_shape)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()
            iteration_count = 0

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    if batch_idx >= num_batches:
                        break

                    # Data is already on GPU, no need to move it
                    # data = data.to(self.device, non_blocking=True)
                    
                    if use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            output = model(data)
                    else:
                        output = model(data)
                    
                    # Force GPU sync to get accurate timing
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    iteration_count += 1

            end_time = time.time()
            duration = end_time - start_time
            throughput = (iteration_count * batch_size) / duration
            avg_iteration_time = (duration * 1000) / iteration_count

            peak_memory_mb = 0
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            test_name = f"inference_{model_name}_bs{batch_size}_{precision}"
            
            result = BenchmarkResult(
                test_name=test_name,
                test_type="inference",
                model_name=model_name,
                batch_size=batch_size,
                duration_seconds=duration,
                iterations=iteration_count,
                throughput_samples_per_second=throughput,
                avg_iteration_time_ms=avg_iteration_time,
                peak_gpu_memory_mb=peak_memory_mb,
                success=True,
                precision=precision,
                model_parameters=param_count,
                sequence_length=input_shape if isinstance(input_shape, int) else None,
                description=f"{model_name} inference with {precision}, {param_count:,} parameters"
            )

            self.results.append(result)
            print(f"  ✓ {throughput:.1f} samples/sec, {avg_iteration_time:.2f}ms/iter, {peak_memory_mb:.0f}MB peak")
            return result

        except Exception as e:
            error_result = BenchmarkResult(
                test_name=f"inference_{model_name}_bs{batch_size}_{precision}",
                test_type="inference",
                model_name=model_name,
                batch_size=batch_size,
                duration_seconds=0,
                iterations=0,
                throughput_samples_per_second=0,
                avg_iteration_time_ms=0,
                peak_gpu_memory_mb=0,
                success=False,
                error_message=str(e),
                precision=precision if 'precision' in locals() else "fp32"
            )
            self.results.append(error_result)
            print(f"  ✗ Failed: {e}")
            return error_result

    def run_memory_scaling_test(self, model_name: str, model_class, base_batch_size: int = 32):
        """Find maximum batch size for a model"""
        print(f"\nMemory Scaling Test: {model_name}")
        print("-" * 40)
        
        batch_size = base_batch_size
        max_successful_batch = 0
        
        while batch_size <= 4096:  # Reasonable upper limit
            try:
                print(f"Testing batch size: {batch_size}")
                model = model_class()
                result = self.run_training_benchmark(
                    model_name, model, batch_size, num_epochs=1, num_batches=5
                )
                if result.success:
                    max_successful_batch = batch_size
                    print(f"  ✓ Success at batch size {batch_size}")
                    batch_size = int(batch_size * 1.5)  # Increase by 50%
                else:
                    break
                    
                del model
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  ✗ OOM at batch size {batch_size}")
                    break
                else:
                    raise e
        
        print(f"Maximum batch size for {model_name}: {max_successful_batch}")
        return max_successful_batch

    def run_comprehensive_benchmark(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run comprehensive GPU benchmarks optimized for H100/H200"""
        print("Starting Comprehensive GPU Load Test")
        print("=" * 50)
        
        start_time = time.time()
        system_info = self.get_system_info()
        
        print(f"System: {system_info.hostname}")
        print(f"GPUs: {', '.join(system_info.gpu_names) if system_info.gpu_names else 'None'}")
        print(f"PyTorch: {system_info.pytorch_version}, CUDA: {system_info.cuda_version}")
        print()

        if quick_mode:
            # Quick test configurations
            test_configs = [
                ("simple_cnn", SimpleConvNet, [64, 128], (3, 224, 224)),
                ("transformer_1b", lambda: LargeTransformer(d_model=1024, num_layers=12), [16, 32], 1024),
            ]
            num_batches = 20
        else:
            # Full test configurations for H100/H200
            test_configs = [
                ("simple_cnn", SimpleConvNet, [128, 256, 512], (3, 224, 224)),
                ("transformer_1b", lambda: LargeTransformer(d_model=1024, num_layers=12), [32, 64, 96], 1024),
                ("transformer_3b", lambda: LargeTransformer(d_model=1536, num_layers=16), [16, 32, 48], 2048),
                ("memory_stressor", lambda: GPUMemoryStressor(hidden_dim=8192), [64, 128, 256], 8192),
            ]
            num_batches = 100

        # Training benchmarks
        print("Training Benchmarks:")
        print("-" * 30)
        
        for model_name, model_class, batch_sizes, input_shape in test_configs:
            for batch_size in batch_sizes:
                # FP32 training
                model = model_class()
                self.run_training_benchmark(
                    model_name, model, batch_size, 
                    num_epochs=1, num_batches=num_batches, 
                    input_shape=input_shape
                )
                del model
                
                # FP16 training (if supported)
                if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
                    model = model_class()
                    self.run_training_benchmark(
                        model_name, model, batch_size, 
                        num_epochs=1, num_batches=num_batches,
                        use_mixed_precision=True, input_shape=input_shape
                    )
                    del model
                
                torch.cuda.empty_cache()
                time.sleep(1)

        print()

        # Inference benchmarks
        print("Inference Benchmarks:")
        print("-" * 30)
        
        for model_name, model_class, batch_sizes, input_shape in test_configs:
            for batch_size in batch_sizes:
                # FP32 inference
                model = model_class()
                self.run_inference_benchmark(
                    model_name, model, batch_size, 
                    num_batches=num_batches * 2, input_shape=input_shape
                )
                del model
                
                # FP16 inference
                if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
                    model = model_class()
                    self.run_inference_benchmark(
                        model_name, model, batch_size, 
                        num_batches=num_batches * 2,
                        use_mixed_precision=True, input_shape=input_shape
                    )
                    del model
                
                torch.cuda.empty_cache()
                time.sleep(1)

        # Memory scaling tests (if not quick mode)
        if not quick_mode:
            print("\nMemory Scaling Tests:")
            print("-" * 30)
            for model_name, model_class, _, input_shape in test_configs[:2]:  # Just test first 2 models
                self.run_memory_scaling_test(model_name, model_class)

        end_time = time.time()
        total_duration = end_time - start_time

        # Compile results
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]

        summary = {
            "system_info": asdict(system_info),
            "test_summary": {
                "total_duration_seconds": total_duration,
                "total_tests": len(self.results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "quick_mode": quick_mode,
                "timestamp": datetime.now().isoformat(),
            },
            "results": [asdict(r) for r in self.results],
        }

        print(f"\nSummary:")
        print(f"Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        print(f"Tests: {len(successful_tests)}/{len(self.results)} successful")

        if successful_tests:
            training_results = [r for r in successful_tests if r.test_type == "training"]
            inference_results = [r for r in successful_tests if r.test_type == "inference"]

            if training_results:
                avg_training_throughput = np.mean([r.throughput_samples_per_second for r in training_results])
                print(f"Avg Training Throughput: {avg_training_throughput:.1f} samples/sec")

            if inference_results:
                avg_inference_throughput = np.mean([r.throughput_samples_per_second for r in inference_results])
                print(f"Avg Inference Throughput: {avg_inference_throughput:.1f} samples/sec")

        return summary

    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to JSON file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"gpu_loadtest_results_{timestamp}.json"

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
    """Setup logging configuration"""
    logger = logging.getLogger("gpu_loadtest_pure")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Pure GPU Load Testing Script")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--device", help="Device to use (cuda/cpu)")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--log-file", help="Log file path")

    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    logger.info("Starting Pure GPU Load Test")

    try:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tester = PureGPULoadTester(device=device)

        # Run benchmarks
        tester.run_comprehensive_benchmark(quick_mode=args.quick)

        # Save results
        output_file = args.output or f"gpu_loadtest_results_{int(time.time())}.json"
        tester.save_results(output_file)

    except Exception as e:
        logger.error(f"GPU load test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()