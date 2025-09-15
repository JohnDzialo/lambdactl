#!/usr/bin/env python3
"""
Remote Executor Module v2 - Separated Architecture

This module provides SSH-based remote execution capabilities for deploying and running
GPU load tests with separate resource monitoring on Lambda Labs instances. It handles:
- SSH connection management
- File transfer (SCP) for both GPU test and monitoring scripts
- Coordinated execution of GPU tests and background monitoring
- Result collection from both processes
- Error handling and retries

Uses paramiko for SSH operations with pydantic models.
Features separated GPU testing and resource monitoring for optimal performance.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import paramiko
from paramiko import SSHClient
from pydantic import BaseModel


class ConnectionConfig(BaseModel):
    """SSH connection configuration"""

    hostname: str
    port: int = 22
    username: str = "ubuntu"
    key_file: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 5.0


class MonitoringConfig(BaseModel):
    """Configuration for resource monitoring"""
    
    enabled: bool = True
    interval_seconds: float = 60.0
    duration_buffer_seconds: float = 300.0  # Extra time to monitor after test completes
    metrics: List[str] = ["cpu", "memory", "gpu", "network", "disk"]


class RemoteExecutionResult(BaseModel):
    """Result of a remote command execution"""

    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    error_message: Optional[str] = None


class FileTransferResult(BaseModel):
    """Result of a file transfer operation"""

    success: bool
    local_path: str
    remote_path: str
    file_size: int
    transfer_time: float
    error_message: Optional[str] = None


class LoadTestResult(BaseModel):
    """Result of a complete load test execution"""

    hostname: str
    success: bool
    execution_time: float
    local_results_file: Optional[str] = None
    local_monitoring_file: Optional[str] = None
    error: Optional[str] = None
    stdout: Optional[str] = None
    stage: Optional[str] = None
    total_workflow_time: Optional[float] = None
    test_results: Optional[Dict[str, Any]] = None
    monitoring_results: Optional[Dict[str, Any]] = None


class AggregatedResults(BaseModel):
    """Aggregated results from multiple instances"""

    summary: Dict[str, Any]
    successful_results: List[LoadTestResult]
    failed_results: List[LoadTestResult]
    performance_summary: Optional[Dict[str, Any]] = None


class SSHConnection:
    """Manages SSH connection with automatic retry and cleanup"""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.client: Optional[SSHClient] = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self) -> bool:
        """Establish SSH connection with retries"""
        for attempt in range(self.config.retry_attempts):
            try:
                self.client = SSHClient()
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                # Connect with key or password
                if self.config.key_file:
                    self.client.connect(
                        hostname=self.config.hostname,
                        port=self.config.port,
                        username=self.config.username,
                        key_filename=self.config.key_file,
                        timeout=self.config.timeout,
                    )
                elif self.config.password:
                    self.client.connect(
                        hostname=self.config.hostname,
                        port=self.config.port,
                        username=self.config.username,
                        password=self.config.password,
                        timeout=self.config.timeout,
                    )
                else:
                    raise ValueError("Either key_file or password must be provided")

                # Test connection
                self.execute_command("echo 'Connection test'", timeout=10)
                self.logger.info(f"Connected to {self.config.hostname}")
                return True

            except Exception as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if self.client:
                    try:
                        self.client.close()
                    except:
                        pass
                    self.client = None

                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)

        self.logger.error(
            f"Failed to connect to {self.config.hostname} after {self.config.retry_attempts} attempts"
        )
        return False

    def disconnect(self):
        """Close SSH connection"""
        try:
            if self.client:
                self.client.close()
                self.client = None
        except Exception as e:
            self.logger.warning(f"Error during disconnect: {e}")

    def execute_command(
        self, command: str, timeout: int = 300
    ) -> RemoteExecutionResult:
        """Execute command on remote host"""
        if not self.client:
            raise RuntimeError("Not connected to remote host")

        start_time = time.time()

        try:
            stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)

            # Read output
            stdout_data = stdout.read().decode("utf-8")
            stderr_data = stderr.read().decode("utf-8")
            return_code = stdout.channel.recv_exit_status()

            execution_time = time.time() - start_time

            return RemoteExecutionResult(
                success=return_code == 0,
                return_code=return_code,
                stdout=stdout_data,
                stderr=stderr_data,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return RemoteExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr="",
                execution_time=execution_time,
                error_message=str(e),
            )

    def upload_file_via_ssh(
        self, local_path: Union[str, Path], remote_path: str
    ) -> FileTransferResult:
        """Upload file using SSH commands and cat"""
        if not self.client:
            raise RuntimeError("Not connected to remote host")

        local_path = Path(local_path)
        if not local_path.exists():
            return FileTransferResult(
                success=False,
                local_path=str(local_path),
                remote_path=remote_path,
                file_size=0,
                transfer_time=0,
                error_message="Local file does not exist",
            )

        start_time = time.time()

        try:
            # Read local file content
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Create remote directory if needed
            remote_dir = str(Path(remote_path).parent)
            self.execute_command(f"mkdir -p {remote_dir}", timeout=30)

            # Upload content using cat and heredoc
            cmd = f"cat > {remote_path} << 'EOF'\n{content}\nEOF"
            result = self.execute_command(cmd, timeout=120)

            if not result.success:
                raise Exception(f"Failed to write file: {result.stderr}")

            file_size = local_path.stat().st_size
            transfer_time = time.time() - start_time

            return FileTransferResult(
                success=True,
                local_path=str(local_path),
                remote_path=remote_path,
                file_size=file_size,
                transfer_time=transfer_time,
            )

        except Exception as e:
            transfer_time = time.time() - start_time
            return FileTransferResult(
                success=False,
                local_path=str(local_path),
                remote_path=remote_path,
                file_size=local_path.stat().st_size if local_path.exists() else 0,
                transfer_time=transfer_time,
                error_message=str(e),
            )

    def download_file_via_ssh(
        self, remote_path: str, local_path: Union[str, Path]
    ) -> FileTransferResult:
        """Download file using SSH commands and cat"""
        if not self.client:
            raise RuntimeError("Not connected to remote host")

        local_path = Path(local_path)
        start_time = time.time()

        try:
            # Create local directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Get file content using cat
            result = self.execute_command(f"cat {remote_path}", timeout=120)

            if not result.success:
                raise Exception(f"Failed to read remote file: {result.stderr}")

            # Write content to local file
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(result.stdout)

            file_size = local_path.stat().st_size
            transfer_time = time.time() - start_time

            return FileTransferResult(
                success=True,
                local_path=str(local_path),
                remote_path=remote_path,
                file_size=file_size,
                transfer_time=transfer_time,
            )

        except Exception as e:
            transfer_time = time.time() - start_time
            return FileTransferResult(
                success=False,
                local_path=str(local_path),
                remote_path=remote_path,
                file_size=0,
                transfer_time=transfer_time,
                error_message=str(e),
            )


class RemoteLoadTestExecutor:
    """High-level executor for running GPU load tests with separate resource monitoring"""

    def __init__(self, ssh_key_file: str, monitoring_config: MonitoringConfig = None):
        self.ssh_key_file = ssh_key_file
        self.monitoring_config = monitoring_config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)

    def setup_instance(self, hostname: str, username: str = "ubuntu") -> bool:
        """Setup instance with required dependencies"""
        config = ConnectionConfig(
            hostname=hostname, username=username, key_file=self.ssh_key_file
        )

        with SSHConnection(config) as conn:
            if not conn.client:
                return False

            self.logger.info(f"Setting up instance {hostname}")

            # Update system
            result = conn.execute_command("sudo apt-get update", timeout=300)
            if not result.success:
                self.logger.error(f"Failed to update system: {result.stderr}")
                return False

            # Install Python packages
            setup_commands = [
                "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
                "pip3 install psutil GPUtil numpy",
            ]

            for cmd in setup_commands:
                result = conn.execute_command(cmd, timeout=600)
                if not result.success:
                    self.logger.warning(
                        f"Command failed (continuing): {cmd} - {result.stderr}"
                    )

            # Verify installation
            result = conn.execute_command(
                "python3 -c 'import torch; print(torch.cuda.is_available())'",
                timeout=30,
            )
            if result.success and "True" in result.stdout:
                self.logger.info(f"Instance {hostname} setup complete - CUDA available")
                return True
            else:
                self.logger.warning(
                    f"Instance {hostname} setup complete - CUDA may not be available"
                )
                return True  # Continue anyway for CPU testing

    def deploy_scripts(self, hostname: str, username: str = "ubuntu") -> bool:
        """Deploy both GPU load test and resource monitoring scripts to remote instance"""
        config = ConnectionConfig(
            hostname=hostname, username=username, key_file=self.ssh_key_file
        )

        with SSHConnection(config) as conn:
            if not conn.client:
                return False

            self.logger.info(f"Deploying scripts to {hostname}")

            # Upload the pure GPU load test script
            local_gpu_script = Path(__file__).parent / "load_tests" / "gpu_loadtest.py"
            remote_gpu_script = "/tmp/gpu_loadtest.py"

            result = conn.upload_file_via_ssh(local_gpu_script, remote_gpu_script)
            if not result.success:
                self.logger.error(f"Failed to upload GPU test script: {result.error_message}")
                return False

            # Upload the resource monitoring script
            local_monitor_script = Path(__file__).parent / "load_tests" / "resource_monitor.py"
            remote_monitor_script = "/tmp/resource_monitor.py"

            result = conn.upload_file_via_ssh(local_monitor_script, remote_monitor_script)
            if not result.success:
                self.logger.error(f"Failed to upload monitoring script: {result.error_message}")
                return False

            # Make scripts executable
            for script in [remote_gpu_script, remote_monitor_script]:
                result = conn.execute_command(f"chmod +x {script}", timeout=30)
                if not result.success:
                    self.logger.warning(f"Failed to make {script} executable: {result.stderr}")

            return True

    def run_load_test(
        self,
        hostname: str,
        output_file: str = None,
        username: str = "ubuntu",
        quick: bool = False,
    ) -> LoadTestResult:
        """Run load test and monitoring on remote instance and collect results"""
        config = ConnectionConfig(
            hostname=hostname, username=username, key_file=self.ssh_key_file
        )

        with SSHConnection(config) as conn:
            if not conn.client:
                return LoadTestResult(
                    hostname=hostname,
                    success=False,
                    execution_time=0,
                    error="Failed to connect to instance",
                )

            self.logger.info(f"Running load test and monitoring on {hostname}")

            # Prepare file paths
            timestamp = int(time.time())
            remote_gpu_script = "/tmp/gpu_loadtest.py"
            remote_monitor_script = "/tmp/resource_monitor.py"
            remote_gpu_output = output_file or f"/tmp/gpu_loadtest_results_{timestamp}.json"
            remote_monitor_output = f"/tmp/resource_monitor_{timestamp}.json"
            remote_gpu_log = f"/tmp/gpu_loadtest_log_{timestamp}.log"

            # Prepare GPU test command
            gpu_cmd_args = ["python3", remote_gpu_script, "--output", remote_gpu_output]
            if quick:
                gpu_cmd_args.append("--quick")
            gpu_cmd_args.extend(["--log-file", remote_gpu_log])
            gpu_command = " ".join(gpu_cmd_args)

            # Start resource monitoring if enabled
            monitor_pid = None
            if self.monitoring_config.enabled:
                self.logger.info(f"Starting resource monitoring (interval: {self.monitoring_config.interval_seconds}s)")
                
                # Estimate test duration for monitoring
                estimated_test_duration = 1800 if not quick else 300  # 30 min full, 5 min quick
                monitor_duration = estimated_test_duration + self.monitoring_config.duration_buffer_seconds
                
                monitor_cmd_args = [
                    "python3", remote_monitor_script,
                    "--interval", str(self.monitoring_config.interval_seconds),
                    "--duration", str(monitor_duration),
                    "--output", remote_monitor_output,
                    "--quiet"
                ]
                monitor_command = " ".join(monitor_cmd_args)
                
                # Start monitoring in background
                bg_monitor_command = f"nohup {monitor_command} > /tmp/monitor_{timestamp}.out 2>&1 & echo $!"
                monitor_result = conn.execute_command(bg_monitor_command, timeout=30)
                
                if monitor_result.success and monitor_result.stdout.strip():
                    monitor_pid = monitor_result.stdout.strip()
                    self.logger.info(f"Resource monitoring started with PID: {monitor_pid}")
                    time.sleep(2)  # Brief delay to let monitoring start
                else:
                    self.logger.warning(f"Failed to start resource monitoring: {monitor_result.stderr}")

            # Execute GPU load test
            self.logger.info(f"Starting GPU load test: {gpu_command}")
            result = conn.execute_command(gpu_command, timeout=3600)  # 1 hour timeout

            # Stop monitoring if it was started
            if monitor_pid and self.monitoring_config.enabled:
                self.logger.info("Stopping resource monitoring")
                stop_cmd = f"kill {monitor_pid} 2>/dev/null || true"
                conn.execute_command(stop_cmd, timeout=30)
                time.sleep(2)  # Give monitoring time to save results

            if not result.success:
                self.logger.error(f"Load test failed on {hostname}: {result.stderr}")
                return LoadTestResult(
                    hostname=hostname,
                    success=False,
                    execution_time=result.execution_time,
                    error=result.stderr,
                    stdout=result.stdout,
                    stage="execute",
                )

            # Download GPU test results
            local_gpu_output = Path(f"results_{hostname}_{timestamp}.json")
            gpu_download_result = conn.download_file_via_ssh(remote_gpu_output, local_gpu_output)

            if not gpu_download_result.success:
                self.logger.error(
                    f"Failed to download GPU test results from {hostname}: {gpu_download_result.error_message}"
                )
                return LoadTestResult(
                    hostname=hostname,
                    success=False,
                    execution_time=result.execution_time,
                    error=f"Failed to download GPU test results: {gpu_download_result.error_message}",
                    stage="download",
                )

            # Download monitoring results if enabled
            local_monitor_output = None
            monitoring_results = None
            if self.monitoring_config.enabled:
                local_monitor_output = Path(f"monitoring_{hostname}_{timestamp}.json")
                monitor_download_result = conn.download_file_via_ssh(remote_monitor_output, local_monitor_output)
                
                if monitor_download_result.success:
                    try:
                        with open(local_monitor_output, "r") as f:
                            monitoring_results = json.load(f)
                        self.logger.info(f"Downloaded monitoring results from {hostname}")
                    except Exception as e:
                        self.logger.warning(f"Failed to parse monitoring results: {e}")
                        local_monitor_output = None
                else:
                    self.logger.warning(f"Failed to download monitoring results: {monitor_download_result.error_message}")
                    local_monitor_output = None

            # Parse GPU test results
            try:
                with open(local_gpu_output, "r") as f:
                    test_results = json.load(f)

                self.logger.info(f"Load test completed successfully on {hostname}")
                return LoadTestResult(
                    hostname=hostname,
                    success=True,
                    execution_time=result.execution_time,
                    local_results_file=str(local_gpu_output),
                    local_monitoring_file=str(local_monitor_output) if local_monitor_output else None,
                    test_results=test_results,
                    monitoring_results=monitoring_results,
                )

            except Exception as e:
                self.logger.error(f"Failed to parse GPU test results from {hostname}: {e}")
                return LoadTestResult(
                    hostname=hostname,
                    success=False,
                    execution_time=result.execution_time,
                    error=f"Failed to parse GPU test results: {e}",
                    stage="parse",
                )

    def run_parallel_load_tests(
        self,
        hostnames: List[str],
        username: str = "ubuntu",
        quick: bool = False,
        max_workers: int = 5,
    ) -> List[LoadTestResult]:
        """Run load tests on multiple instances in parallel"""
        self.logger.info(f"Running parallel load tests on {len(hostnames)} instances")

        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_hostname = {}
            for hostname in hostnames:
                future = executor.submit(
                    self.run_complete_test,
                    hostname,
                    username,
                    quick,
                )
                future_to_hostname[future] = hostname

            # Collect results
            for future in as_completed(future_to_hostname):
                hostname = future_to_hostname[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Exception running test on {hostname}: {e}")
                    results.append(
                        LoadTestResult(
                            hostname=hostname,
                            success=False,
                            execution_time=0,
                            error=str(e),
                            stage="exception",
                        )
                    )

        return results

    def run_complete_test(
        self,
        hostname: str,
        username: str = "ubuntu",
        quick: bool = False,
    ) -> LoadTestResult:
        """Run complete test workflow: setup, deploy, run, collect"""
        start_time = time.time()

        try:
            # Setup instance
            setup_success = self.setup_instance(hostname, username)
            if not setup_success:
                return LoadTestResult(
                    hostname=hostname,
                    success=False,
                    execution_time=0,
                    error="Failed to setup instance",
                    stage="setup",
                )

            # Deploy scripts
            deploy_success = self.deploy_scripts(hostname, username)
            if not deploy_success:
                return LoadTestResult(
                    hostname=hostname,
                    success=False,
                    execution_time=0,
                    error="Failed to deploy scripts",
                    stage="deploy",
                )

            # Run load test
            result = self.run_load_test(
                hostname,
                username=username,
                quick=quick,
            )

            total_time = time.time() - start_time
            result.total_workflow_time = total_time

            return result

        except Exception as e:
            total_time = time.time() - start_time
            return LoadTestResult(
                hostname=hostname,
                success=False,
                execution_time=0,
                error=str(e),
                total_workflow_time=total_time,
                stage="exception",
            )

    def aggregate_results(self, results: List[LoadTestResult]) -> AggregatedResults:
        """Aggregate results from multiple instances"""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        summary = {
            "total_instances": len(results),
            "successful_instances": len(successful_results),
            "failed_instances": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "monitoring_enabled": self.monitoring_config.enabled,
            "monitoring_interval_seconds": self.monitoring_config.interval_seconds,
            "timestamp": time.time(),
        }

        aggregated = AggregatedResults(
            summary=summary,
            successful_results=successful_results,
            failed_results=failed_results,
        )

        if successful_results:
            # Aggregate performance metrics
            all_test_results = []
            for result in successful_results:
                if result.test_results and "results" in result.test_results:
                    for test_result in result.test_results["results"]:
                        test_result["source_hostname"] = result.hostname
                        all_test_results.append(test_result)

            if all_test_results:
                training_results = [
                    r for r in all_test_results if r.get("test_type") == "training"
                ]
                inference_results = [
                    r for r in all_test_results if r.get("test_type") == "inference"
                ]

                performance_summary = {
                    "total_tests": len(all_test_results),
                    "training_tests": len(training_results),
                    "inference_tests": len(inference_results),
                }

                if training_results:
                    training_throughputs = [
                        r["throughput_samples_per_second"]
                        for r in training_results
                        if r.get("success")
                    ]
                    if training_throughputs:
                        performance_summary["training"] = {
                            "avg_throughput": sum(training_throughputs)
                            / len(training_throughputs),
                            "max_throughput": max(training_throughputs),
                            "min_throughput": min(training_throughputs),
                        }

                if inference_results:
                    inference_throughputs = [
                        r["throughput_samples_per_second"]
                        for r in inference_results
                        if r.get("success")
                    ]
                    if inference_throughputs:
                        performance_summary["inference"] = {
                            "avg_throughput": sum(inference_throughputs)
                            / len(inference_throughputs),
                            "max_throughput": max(inference_throughputs),
                            "min_throughput": min(inference_throughputs),
                        }

                aggregated.performance_summary = performance_summary

        return aggregated