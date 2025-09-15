#!/usr/bin/env python3
"""
Standalone Resource Monitoring Script

Collects system resource usage (CPU, memory, GPU) at configurable intervals.
Runs independently from GPU load tests to avoid CPU bottlenecks.
"""

import json
import time
import sys
import logging
import signal
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import psutil

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not available. GPU monitoring will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some GPU metrics unavailable.")


@dataclass
class ResourceSnapshot:
    """Single resource usage snapshot"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_metrics: List[Dict[str, Any]]
    load_average: Optional[List[float]] = None  # Linux/macOS only


@dataclass
class MonitoringSession:
    """Complete monitoring session data"""
    start_time: float
    end_time: float
    duration_seconds: float
    interval_seconds: float
    hostname: str
    snapshots: List[ResourceSnapshot]
    system_info: Dict[str, Any]


class ResourceMonitor:
    """Standalone resource monitoring with configurable intervals"""

    def __init__(self, interval_seconds: float = 60.0, output_file: str = None):
        self.interval = interval_seconds
        self.output_file = output_file or f"resource_monitor_{int(time.time())}.json"
        self.monitoring = False
        self.snapshots: List[ResourceSnapshot] = []
        self.start_time: Optional[float] = None
        self.baseline_disk_io = None
        self.baseline_network_io = None
        self.logger = self._setup_logging()
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the monitor"""
        logger = logging.getLogger("resource_monitor")
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

        return logger

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop_monitoring()
        sys.exit(0)

    def get_system_info(self) -> Dict[str, Any]:
        """Collect static system information"""
        import socket
        import platform
        
        info = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "boot_time": psutil.boot_time(),
        }
        
        # CPU info
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_lines = f.readlines()
                model_name = next((line.split(': ')[1].strip() for line in cpu_lines 
                                 if line.startswith('model name')), "Unknown")
                info["cpu_model"] = model_name
        except:
            info["cpu_model"] = "Unknown"
        
        # GPU info
        gpu_info = []
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_mb": gpu_props.total_memory / (1024 * 1024),
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                    "multiprocessor_count": gpu_props.multi_processor_count
                })
        
        info["gpu_info"] = gpu_info
        info["gpu_count"] = len(gpu_info)
        
        return info

    def collect_resource_snapshot(self) -> ResourceSnapshot:
        """Collect current resource usage snapshot"""
        current_time = time.time()
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # Disk I/O (delta from baseline)
        disk_io = psutil.disk_io_counters()
        if self.baseline_disk_io is None:
            self.baseline_disk_io = disk_io
            disk_read_mb = 0.0
            disk_write_mb = 0.0
        else:
            disk_read_mb = (disk_io.read_bytes - self.baseline_disk_io.read_bytes) / (1024 * 1024)
            disk_write_mb = (disk_io.write_bytes - self.baseline_disk_io.write_bytes) / (1024 * 1024)
        
        # Network I/O (delta from baseline)
        net_io = psutil.net_io_counters()
        if self.baseline_network_io is None:
            self.baseline_network_io = net_io
            net_sent_mb = 0.0
            net_recv_mb = 0.0
        else:
            net_sent_mb = (net_io.bytes_sent - self.baseline_network_io.bytes_sent) / (1024 * 1024)
            net_recv_mb = (net_io.bytes_recv - self.baseline_network_io.bytes_recv) / (1024 * 1024)
        
        # Load average (Unix-like systems)
        load_avg = None
        try:
            load_avg = list(psutil.getloadavg())
        except AttributeError:
            pass  # Not available on Windows
        
        # GPU metrics
        gpu_metrics = []
        
        # GPUtil metrics
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_data = {
                        "gpu_id": i,
                        "name": gpu.name,
                        "utilization_percent": gpu.load * 100,
                        "memory_used_mb": gpu.memoryUsed,
                        "memory_total_mb": gpu.memoryTotal,
                        "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        "temperature_c": gpu.temperature,
                    }
                    gpu_metrics.append(gpu_data)
            except Exception as e:
                self.logger.warning(f"GPUtil monitoring failed: {e}")
        
        # PyTorch GPU metrics
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    torch_memory_allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                    torch_memory_reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                    torch_memory_max = torch.cuda.max_memory_allocated(i) / (1024 * 1024)
                    
                    # Add to existing GPU data or create new entry
                    if i < len(gpu_metrics):
                        gpu_metrics[i].update({
                            "torch_memory_allocated_mb": torch_memory_allocated,
                            "torch_memory_reserved_mb": torch_memory_reserved,
                            "torch_memory_max_mb": torch_memory_max,
                        })
                    else:
                        # Create new entry if GPUtil not available
                        gpu_metrics.append({
                            "gpu_id": i,
                            "name": torch.cuda.get_device_name(i),
                            "torch_memory_allocated_mb": torch_memory_allocated,
                            "torch_memory_reserved_mb": torch_memory_reserved,
                            "torch_memory_max_mb": torch_memory_max,
                        })
            except Exception as e:
                self.logger.warning(f"PyTorch GPU monitoring failed: {e}")
        
        return ResourceSnapshot(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb,
            gpu_metrics=gpu_metrics,
            load_average=load_avg
        )

    def start_monitoring(self, duration_seconds: Optional[float] = None):
        """Start resource monitoring"""
        self.logger.info(f"Starting resource monitoring (interval: {self.interval}s)")
        if duration_seconds:
            self.logger.info(f"Monitoring duration: {duration_seconds}s")
        
        self.monitoring = True
        self.start_time = time.time()
        self.snapshots = []
        
        # Reset baselines
        self.baseline_disk_io = None
        self.baseline_network_io = None
        
        end_time = self.start_time + duration_seconds if duration_seconds else None
        
        try:
            while self.monitoring:
                # Collect snapshot
                snapshot = self.collect_resource_snapshot()
                self.snapshots.append(snapshot)
                
                # Log summary every 10 snapshots
                if len(self.snapshots) % 10 == 0:
                    self.logger.info(f"Collected {len(self.snapshots)} snapshots. "
                                   f"CPU: {snapshot.cpu_percent:.1f}%, "
                                   f"Memory: {snapshot.memory_percent:.1f}%, "
                                   f"GPUs: {len(snapshot.gpu_metrics)}")
                
                # Check if we should stop (duration limit)
                if end_time and time.time() >= end_time:
                    self.logger.info("Monitoring duration reached, stopping...")
                    break
                
                # Sleep until next interval
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring interrupted by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}", exc_info=True)
        finally:
            # Always save results when monitoring stops
            self._save_results_and_cleanup()

    def _save_results_and_cleanup(self):
        """Save results and cleanup, called from finally block"""
        if not self.snapshots:
            self.logger.warning("No snapshots collected")
            self.monitoring = False
            return
        
        end_time = time.time()
        duration = end_time - self.start_time
        self.logger.info(f"Monitoring stopped. Collected {len(self.snapshots)} snapshots over {duration:.1f}s")
        
        # Create monitoring session
        session = MonitoringSession(
            start_time=self.start_time,
            end_time=end_time,
            duration_seconds=duration,
            interval_seconds=self.interval,
            hostname=self.get_system_info()["hostname"],
            snapshots=self.snapshots,
            system_info=self.get_system_info()
        )
        
        # Save to file
        self.save_session(session)
        
        # Print summary
        self.print_summary(session)
        
        self.monitoring = False

    def stop_monitoring(self):
        """Stop monitoring and save results"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        end_time = time.time()
        
        if not self.snapshots:
            self.logger.warning("No snapshots collected")
            return
        
        duration = end_time - self.start_time
        self.logger.info(f"Monitoring stopped. Collected {len(self.snapshots)} snapshots over {duration:.1f}s")
        
        # Create monitoring session
        session = MonitoringSession(
            start_time=self.start_time,
            end_time=end_time,
            duration_seconds=duration,
            interval_seconds=self.interval,
            hostname=self.get_system_info()["hostname"],
            snapshots=self.snapshots,
            system_info=self.get_system_info()
        )
        
        # Save to file
        self.save_session(session)
        
        # Print summary
        self.print_summary(session)

    def save_session(self, session: MonitoringSession):
        """Save monitoring session to JSON file"""
        try:
            data = asdict(session)
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Resource monitoring data saved to: {self.output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save monitoring data: {e}")

    def print_summary(self, session: MonitoringSession):
        """Print monitoring session summary"""
        if not session.snapshots:
            return
        
        print("\nResource Monitoring Summary")
        print("=" * 50)
        print(f"Duration: {session.duration_seconds:.1f}s")
        print(f"Samples: {len(session.snapshots)}")
        print(f"Interval: {session.interval_seconds}s")
        print(f"Hostname: {session.hostname}")
        
        # CPU statistics
        cpu_values = [s.cpu_percent for s in session.snapshots]
        print(f"\nCPU Usage:")
        print(f"  Min: {min(cpu_values):.1f}%")
        print(f"  Max: {max(cpu_values):.1f}%")
        print(f"  Avg: {sum(cpu_values)/len(cpu_values):.1f}%")
        
        # Memory statistics
        mem_values = [s.memory_percent for s in session.snapshots]
        print(f"\nMemory Usage:")
        print(f"  Min: {min(mem_values):.1f}%")
        print(f"  Max: {max(mem_values):.1f}%")
        print(f"  Avg: {sum(mem_values)/len(mem_values):.1f}%")
        
        # GPU statistics
        if session.snapshots[0].gpu_metrics:
            gpu_count = len(session.snapshots[0].gpu_metrics)
            print(f"\nGPU Usage ({gpu_count} GPUs):")
            
            for gpu_id in range(gpu_count):
                gpu_utils = []
                gpu_memory = []
                gpu_name = "Unknown"
                
                for snapshot in session.snapshots:
                    if gpu_id < len(snapshot.gpu_metrics):
                        gpu = snapshot.gpu_metrics[gpu_id]
                        gpu_name = gpu.get("name", f"GPU {gpu_id}")
                        if "utilization_percent" in gpu:
                            gpu_utils.append(gpu["utilization_percent"])
                        if "memory_percent" in gpu:
                            gpu_memory.append(gpu["memory_percent"])
                
                print(f"  GPU {gpu_id} ({gpu_name}):")
                if gpu_utils:
                    print(f"    Utilization - Min: {min(gpu_utils):.1f}%, Max: {max(gpu_utils):.1f}%, Avg: {sum(gpu_utils)/len(gpu_utils):.1f}%")
                if gpu_memory:
                    print(f"    Memory - Min: {min(gpu_memory):.1f}%, Max: {max(gpu_memory):.1f}%, Avg: {sum(gpu_memory)/len(gpu_memory):.1f}%")
        
        # I/O statistics
        total_disk_read = session.snapshots[-1].disk_io_read_mb if session.snapshots else 0
        total_disk_write = session.snapshots[-1].disk_io_write_mb if session.snapshots else 0
        total_net_sent = session.snapshots[-1].network_sent_mb if session.snapshots else 0
        total_net_recv = session.snapshots[-1].network_recv_mb if session.snapshots else 0
        
        print(f"\nI/O Summary:")
        print(f"  Disk Read: {total_disk_read:.1f} MB")
        print(f"  Disk Write: {total_disk_write:.1f} MB")
        print(f"  Network Sent: {total_net_sent:.1f} MB")
        print(f"  Network Received: {total_net_recv:.1f} MB")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone Resource Monitor")
    parser.add_argument("--interval", "-i", type=float, default=60.0,
                       help="Monitoring interval in seconds (default: 60.0)")
    parser.add_argument("--duration", "-d", type=float,
                       help="Monitoring duration in seconds (infinite if not specified)")
    parser.add_argument("--output", "-o", 
                       help="Output file for monitoring data (default: auto-generated)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress periodic status messages")
    
    args = parser.parse_args()
    
    # Setup monitor
    output_file = args.output or f"resource_monitor_{int(time.time())}.json"
    monitor = ResourceMonitor(interval_seconds=args.interval, output_file=output_file)
    
    if args.quiet:
        monitor.logger.setLevel(logging.WARNING)
    
    try:
        # Start monitoring
        monitor.start_monitoring(duration_seconds=args.duration)
        
    except Exception as e:
        monitor.logger.error(f"Resource monitoring failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure we stop and save data
        monitor.stop_monitoring()


if __name__ == "__main__":
    main()