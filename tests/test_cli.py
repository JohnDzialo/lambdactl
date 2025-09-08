"""
Tests for CLI module - focused on core functionality
"""

import pytest
from unittest.mock import Mock, patch

from cli import load_config


class TestConfigManagement:
    """Test configuration loading"""

    @patch("cli.CONFIG_FILE")
    def test_load_config_empty_when_no_file(self, mock_config_file):
        """Test loading configuration returns empty dict when file doesn't exist"""
        mock_config_file.exists.return_value = False
        result = load_config()
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__])
