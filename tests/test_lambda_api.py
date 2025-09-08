"""
Tests for Lambda Labs API module - focused on core functionality
"""

import pytest
from unittest.mock import Mock, patch
from pydantic import ValidationError

from lambda_api import Instance, InstanceRequest, LambdaLabsAPI


class TestDataModels:
    """Test Pydantic data models"""

    def test_instance_creation(self):
        """Test Instance model creation with required fields"""
        instance_data = {
            "id": "i-123456",
            "name": "test-instance",
            "instance_type": "gpu_1x_a100",
            "status": "running",
        }

        instance = Instance(**instance_data)
        assert instance.id == "i-123456"
        assert instance.name == "test-instance"
        assert instance.instance_type == "gpu_1x_a100"
        assert instance.status == "running"

    def test_instance_request_creation(self):
        """Test InstanceRequest model creation"""
        request_data = {
            "instance_type_name": "gpu_1x_a100",
            "name": "test-instance",
            "region_name": "us-west-1",
            "ssh_key_names": ["my-key"],
        }

        request = InstanceRequest(**request_data)
        assert request.instance_type_name == "gpu_1x_a100"
        assert request.name == "test-instance"
        assert request.region_name == "us-west-1"
        assert request.ssh_key_names == ["my-key"]

    def test_instance_request_image_validation(self):
        """Test InstanceRequest image validation"""
        base_data = {
            "instance_type_name": "gpu_1x_a100",
            "name": "test-instance",
            "region_name": "us-west-1",
            "ssh_key_names": ["my-key"],
        }

        # Valid image with id
        valid_data = base_data.copy()
        valid_data["image"] = {"id": "ubuntu-20-04"}
        request = InstanceRequest(**valid_data)
        assert request.image["id"] == "ubuntu-20-04"

        # Invalid image without id should raise validation error
        invalid_data = base_data.copy()
        invalid_data["image"] = {"name": "Ubuntu 20.04"}  # Missing 'id'

        with pytest.raises(ValidationError):
            InstanceRequest(**invalid_data)


class TestLambdaLabsAPI:
    """Test LambdaLabsAPI class - core methods only"""

    def setup_method(self):
        """Set up test fixtures"""
        self.api = LambdaLabsAPI("test-api-key")

    def test_api_initialization(self):
        """Test API initialization"""
        assert self.api.api_key == "test-api-key"
        assert self.api.base_url == "https://cloud.lambdalabs.com/api/v1"
        assert self.api.session is not None
        assert "Authorization" in self.api.session.headers
        assert self.api.session.headers["Authorization"] == "Bearer test-api-key"

    @patch("lambda_api.requests.Session")
    def test_list_instances(self, mock_session_class):
        """Test listing instances"""
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "i-123", "name": "instance1", "status": "running"},
                {"id": "i-456", "name": "instance2", "status": "stopped"},
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_session.request.return_value = mock_response

        # Create new API instance to use mocked session
        api = LambdaLabsAPI("test-key")
        instances = api.list_instances()

        assert len(instances) == 2
        assert instances[0]["id"] == "i-123"
        assert instances[1]["id"] == "i-456"


if __name__ == "__main__":
    pytest.main([__file__])
