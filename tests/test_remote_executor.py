"""
Tests for Remote Executor module - focused on existing functionality
"""

import pytest
from unittest.mock import Mock, patch
import json
import tempfile
import os

from remote_executor import (
    ConnectionConfig,
    RemoteExecutionResult,
    LoadTestResult,
    AggregatedResults,
    RemoteLoadTestExecutor,
)


class TestDataModels:
    """Test core Pydantic models"""

    def test_connection_config(self):
        """Test ConnectionConfig creation"""
        config = ConnectionConfig(hostname="test-host", key_file="key.pem")
        assert config.hostname == "test-host"
        assert config.username == "ubuntu"  # default
        assert config.port == 22  # default
        assert config.key_file == "key.pem"

    def test_remote_execution_result(self):
        """Test RemoteExecutionResult model"""
        result = RemoteExecutionResult(
            success=True,
            return_code=0,
            stdout="Command output",
            stderr="",
            execution_time=2.5,
        )

        assert result.success is True
        assert result.stdout == "Command output"
        assert result.return_code == 0
        assert result.execution_time == 2.5

    def test_load_test_result(self):
        """Test LoadTestResult model"""
        result = LoadTestResult(
            hostname="test-host", success=True, execution_time=120.5
        )

        assert result.hostname == "test-host"
        assert result.success is True
        assert result.execution_time == 120.5

    def test_aggregated_results(self):
        """Test AggregatedResults model"""
        test_results = [
            LoadTestResult(hostname="host1", success=True, execution_time=100.0),
            LoadTestResult(
                hostname="host2", success=False, execution_time=50.0, error="Failed"
            ),
        ]

        successful_results = [r for r in test_results if r.success]
        failed_results = [r for r in test_results if not r.success]

        aggregated = AggregatedResults(
            summary={"total": 2, "successful": 1, "failed": 1},
            successful_results=successful_results,
            failed_results=failed_results,
        )

        assert len(aggregated.successful_results) == 1
        assert len(aggregated.failed_results) == 1
        assert aggregated.summary["total"] == 2


class TestRemoteLoadTestExecutor:
    """Test RemoteLoadTestExecutor - core functionality only"""

    def setup_method(self):
        """Set up test fixtures"""
        self.executor = RemoteLoadTestExecutor("test_key.pem")

    def test_remote_executor_initialization(self):
        """Test RemoteLoadTestExecutor initialization"""
        assert self.executor.ssh_key_file == "test_key.pem"
        assert self.executor.logger is not None

    def test_aggregate_results(self):
        """Test result aggregation"""
        test_results = [
            LoadTestResult(hostname="host1", success=True, execution_time=100.0),
            LoadTestResult(
                hostname="host2", success=False, execution_time=30.0, error="Failed"
            ),
        ]

        aggregated = self.executor.aggregate_results(test_results)

        assert len(aggregated.successful_results) == 1
        assert len(aggregated.failed_results) == 1
        assert aggregated.summary is not None


if __name__ == "__main__":
    pytest.main([__file__])
