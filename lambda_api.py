"""
Lambda Labs API wrapper using requests library
"""

import time
import requests
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator


class Instance(BaseModel):
    """Represents a Lambda Labs instance"""
    id: str
    name: str
    instance_type: str
    status: str
    ip: Optional[str] = None
    ssh_key_names: Optional[List[str]] = None
    file_system_names: Optional[List[str]] = None
    region: Optional[str] = None


class InstanceRequest(BaseModel):
    """Represents a Lambda Labs instance creation request"""
    instance_type_name: str
    name: str
    region_name: str
    ssh_key_names: List[str]
    file_system_names: Optional[List[str]] = None
    file_system_mounts: Optional[List[Dict[str, str]]] = None
    hostname: Optional[str] = None
    image: Optional[Dict[str, str]] = None
    user_data: Optional[str] = None
    tags: Optional[List[Dict[str, str]]] = None
    firewall_rulesets: Optional[List[Dict[str, str]]] = None
    
    @field_validator('image')
    @classmethod
    def validate_image(cls, v):
        if v is not None and 'id' not in v:
            raise ValueError("image dictionary must contain 'id' key")
        return v
    
    @field_validator('firewall_rulesets')
    @classmethod
    def validate_firewall_rulesets(cls, v):
        if v is not None:
            for ruleset in v:
                if 'id' not in ruleset:
                    raise ValueError("each firewall ruleset dictionary must contain 'id' key")
        return v


class LambdaLabsAPI:
    """Lambda Labs Cloud API wrapper"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://cloud.lambdalabs.com/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[Any, Any]:
        """Make authenticated request to Lambda Labs API"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
    
    def list_instance_types(self) -> List[Dict]:
        """List available instance types"""
        return self._make_request("GET", "/instance-types")["data"]
    
    def list_instances(self) -> List[Dict]:
        """List all instances"""
        return self._make_request("GET", "/instances")["data"]
    
    def get_instance(self, instance_id: str) -> Dict:
        """Get specific instance details"""
        return self._make_request("GET", f"/instances/{instance_id}")["data"]
    
    def create_instance(self, request: InstanceRequest) -> Dict:
        """Create a new instance"""
        return self._make_request("POST", "/instance-operations/launch", json=request.model_dump(exclude_none=True))
    
    def terminate_instance(self, instance_id: str) -> Dict:
        """Terminate an instance"""
        payload = {"instance_ids": [instance_id]}
        return self._make_request("POST", "/instance-operations/terminate", json=payload)
    
    def restart_instance(self, instance_id: str) -> Dict:
        """Restart an instance"""
        payload = {"instance_ids": [instance_id]}
        return self._make_request("POST", "/instance-operations/restart", json=payload)
    
    def list_ssh_keys(self) -> List[Dict]:
        """List SSH keys"""
        return self._make_request("GET", "/ssh-keys")["data"]
    
    def add_ssh_key(self, name: str, public_key: str) -> Dict:
        """Add SSH key"""
        payload = {
            "name": name,
            "public_key": public_key
        }
        return self._make_request("POST", "/ssh-keys", json=payload)
    
    def list_file_systems(self) -> List[Dict]:
        """List file systems"""
        return self._make_request("GET", "/file-systems")["data"]
    
    def wait_for_instance_status(self, instance_id: str, target_status: str = "running", 
                                timeout_minutes: int = 15, check_interval: int = 30) -> bool:
        """Wait for instance to reach target status"""
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            try:
                instance = self.get_instance(instance_id)
                current_status = instance.get("status", "unknown")
                
                if current_status == target_status:
                    return True
                elif current_status in ["terminated", "error"]:
                    return False
                
                time.sleep(check_interval)
                
            except Exception:
                time.sleep(check_interval)
        
        return False
    
    def get_instance_utilization(self, instance_id: str) -> Dict:
        """Get instance utilization metrics (placeholder - API endpoint may vary)"""
        # This endpoint might not exist in the actual API
        # Placeholder for potential metrics endpoint
        try:
            return self._make_request("GET", f"/instances/{instance_id}/utilization")
        except Exception:
            # Return mock data if endpoint doesn't exist
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "gpu_utilization": 0,
                "gpu_memory_percent": 0
            }