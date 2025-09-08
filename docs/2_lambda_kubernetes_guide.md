# Document 2: Kubernetes for AI Infrastructure

## Learning Objectives

By completing this document, you will:
- ✅ Understand Kubernetes architecture and why it's ideal for AI workloads
- ✅ Explain control plane vs data plane components
- ✅ Describe how the NVIDIA GPU Operator works
- ✅ Understand GPU resource management and scheduling
- ✅ Explain container runtime requirements for GPU workloads
- ✅ Troubleshoot common Kubernetes GPU issues

---

## 1. Why Kubernetes for AI?

### Traditional vs Container-Based AI Infrastructure

#### Traditional Approach (Problems)
```
Physical/VM Infrastructure:
┌─────────────────────────────────────────────────────────────┐
│ Server 1: PyTorch 1.9, CUDA 11.2, Python 3.8             │
│ Server 2: TensorFlow 2.4, CUDA 11.0, Python 3.7          │
│ Server 3: Mixed versions, dependency conflicts             │
└─────────────────────────────────────────────────────────────┘

Problems:
❌ Dependency conflicts between frameworks
❌ Manual server provisioning and configuration
❌ Poor resource utilization (often <30%)
❌ Difficult scaling and load balancing
❌ No isolation between different projects
```

#### Kubernetes Approach (Solutions)
```
Container-Based Infrastructure:
┌─────────────────────────────────────────────────────────────┐
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│ │ PyTorch Pod │ │TensorFlow   │ │ JAX Pod     │           │
│ │ + CUDA      │ │ Pod + CUDA  │ │ + CUDA      │           │
│ │ Isolated    │ │ Isolated    │ │ Isolated    │           │
│ └─────────────┘ └─────────────┘ └─────────────┘           │
│                    Same Physical Server                     │
└─────────────────────────────────────────────────────────────┘

Benefits:
✅ Complete dependency isolation
✅ Automatic resource scheduling
✅ High utilization (often >80%)
✅ Auto-scaling based on demand
✅ Multi-tenant security
```

### Kubernetes Advantages for AI Workloads

🔄 **Dynamic Resource Allocation**: Automatically assign GPUs to workloads based on availability  
📦 **Environment Consistency**: Same container runs anywhere (dev, staging, production)  
🔧 **Framework Flexibility**: Support multiple AI frameworks simultaneously  
📈 **Auto-Scaling**: Scale training jobs up/down based on queue demand  
🛡️ **Security**: Isolated environments for different teams/projects  
⚡ **Efficiency**: Pack multiple small jobs on powerful GPU nodes  

---

## 2. Kubernetes Architecture Overview

### High-Level Architecture

```
Kubernetes Cluster Architecture:

                Control Plane
        ┌─────────────────────────────┐
        │  ┌─────────┐ ┌─────────────┐│
        │  │   API   │ │  Scheduler  ││
        │  │ Server  │ │             ││
        │  └─────────┘ └─────────────┘│
        │  ┌─────────┐ ┌─────────────┐│
        │  │ etcd    │ │ Controller  ││
        │  │Database │ │  Manager    ││
        │  └─────────┘ └─────────────┘│
        └─────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ Worker  │    │ Worker  │    │ Worker  │
   │ Node 1  │    │ Node 2  │    │ Node 3  │
   │         │    │         │    │         │
   │8x H100  │    │8x H100  │    │8x H100  │
   │512GB RAM│    │512GB RAM│    │512GB RAM│
   └─────────┘    └─────────┘    └─────────┘
```

### Key Concepts

**Cluster**: Complete Kubernetes environment (control plane + worker nodes)  
**Node**: Physical or virtual machine running Kubernetes  
**Pod**: Smallest deployable unit (usually 1 container + resources)  
**Service**: Network abstraction for accessing pods  
**Deployment**: Manages pods and ensures desired state  

---

## 3. Control Plane Components

The control plane makes global decisions about the cluster and manages cluster state.

### 3.1 API Server

**What it does**: Central communication hub for all cluster operations

**Key Functions**:
- Exposes Kubernetes REST API
- Authenticates and authorizes requests
- Validates resource configurations
- Stores data in etcd

**AI-Specific Considerations**:
- Handles GPU resource requests and scheduling
- Manages large-scale training job submissions
- Coordinates distributed training workflows

**Example API Request**:
```yaml
# Request 4 GPUs for a training job
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-training
spec:
  containers:
  - name: trainer
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
    resources:
      limits:
        nvidia.com/gpu: 4
        memory: 32Gi
        cpu: 16
```

### 3.2 etcd Database

**What it does**: Stores all cluster configuration and state data

**Contains**:
- Node information and resource capacity
- Pod specifications and current state
- GPU allocation and usage data
- Service definitions and endpoints

**High Availability**: Usually runs on 3 or 5 nodes for fault tolerance

**Performance**: Critical for large clusters with many AI workloads

### 3.3 Scheduler

**What it does**: Decides which node should run each pod

**GPU Scheduling Process**:
1. **Filtering**: Find nodes with available GPUs
2. **Scoring**: Rank nodes based on resource utilization
3. **Binding**: Assign pod to best node

**AI-Specific Scheduling Factors**:
- GPU memory requirements
- GPU type compatibility (H100 vs A100)
- Network bandwidth for distributed training
- Data locality (where training data is stored)

**Scheduling Example**:
```
Job Request: 8x GPU PyTorch training

Available Nodes:
Node-1: 8x H100 (all free) ← Best choice
Node-2: 8x H100 (4 used, 4 free)
Node-3: 8x A100 (all free)

Scheduler picks Node-1: Complete resource isolation
```

### 3.4 Controller Manager

**What it does**: Ensures cluster reaches and maintains desired state

**Key Controllers for AI**:
- **Node Controller**: Monitors GPU node health
- **ReplicaSet Controller**: Manages training job replicas
- **Job Controller**: Handles batch processing workloads
- **GPU Device Controller**: Manages GPU resource allocation

---

## 4. Data Plane Components (Worker Nodes)

Worker nodes run the actual AI workloads and containers.

### 4.1 kubelet

**What it does**: Primary node agent that manages pods and containers

**GPU-Specific Functions**:
- Communicates with GPU device plugins
- Mounts GPU devices into containers
- Reports GPU resource usage to API server
- Restarts failed GPU workloads

**Health Monitoring**:
- Monitors GPU temperature and utilization
- Detects GPU memory errors
- Reports node capacity changes

### 4.2 Container Runtime

**What it does**: Actually runs containers with GPU access

**Lambda Labs Uses**: containerd with NVIDIA Container Runtime

**GPU Integration Process**:
```
Container Start Process:
1. kubelet receives pod spec with GPU request
2. Container runtime calls NVIDIA runtime hooks
3. NVIDIA runtime exposes GPU devices to container
4. Container starts with CUDA libraries available
5. Application can use GPUs directly
```

**Runtime Configuration**:
```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

### 4.3 kube-proxy

**What it does**: Handles network routing for services

**AI Workload Networking**:
- Routes traffic to distributed training pods
- Load balances inference requests
- Enables communication between training replicas

---

## 5. NVIDIA GPU Operator Deep Dive

The GPU Operator automates GPU software stack management in Kubernetes.

### 5.1 What Problems Does GPU Operator Solve?

#### Before GPU Operator (Manual Setup)
```
Manual GPU Configuration Per Node:
1. Install NVIDIA drivers (version matching)
2. Install CUDA toolkit
3. Configure container runtime
4. Install device plugin
5. Set up monitoring
6. Configure security policies
7. Test GPU access

Problems:
❌ Time-consuming manual process
❌ Version compatibility issues  
❌ Difficult to maintain consistency
❌ Hard to update/upgrade
```

#### With GPU Operator (Automated)
```
Automated GPU Stack Management:
1. Deploy GPU Operator once
2. Operator detects GPU nodes
3. Automatically installs all components
4. Manages upgrades and updates
5. Provides monitoring and health checks

Benefits:
✅ One-time deployment
✅ Automatic version management
✅ Consistent across all nodes
✅ Easy updates and maintenance
```

### 5.2 GPU Operator Components

```
GPU Operator Stack:

┌─────────────────────────────────────────┐
│            GPU Operator                 │
│         (Controller)                    │
└─────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Driver  │   │ Device  │   │Container│
│DaemonSet│   │ Plugin  │   │ Toolkit │
│         │   │         │   │         │
│• NVIDIA │   │• GPU    │   │• Runtime│
│  drivers│   │  discovery│   │  config │
│• CUDA   │   │• Resource│   │• Hook   │
│  libs   │   │  allocation│ │  setup  │
└─────────┘   └─────────┘   └─────────┘

┌─────────┐   ┌─────────┐   ┌─────────┐
│   DCGM  │   │   MIG   │   │   NFD   │
│Exporter │   │Manager  │   │ (Node   │
│         │   │         │   │Feature  │
│• GPU    │   │• GPU    │   │Discovery│
│  metrics│   │  partition│  │)        │
│• Health │   │• Instance│   │• Hardware│
│  monitor│   │  mgmt   │   │  detection│
└─────────┘   └─────────┘   └─────────┘
```

### 5.3 Key GPU Operator Features

#### Automatic Driver Management
- Detects GPU hardware automatically
- Installs correct driver version
- Handles driver updates without downtime
- Supports multiple GPU generations

#### Device Plugin
- Advertises GPU resources to Kubernetes
- Manages GPU allocation to pods
- Supports GPU sharing and MIG partitioning
- Provides resource isolation

#### Container Toolkit
- Configures container runtime for GPU access
- Provides OCI hooks for GPU containers
- Manages CUDA library access
- Handles device mounting

### 5.4 GPU Resource Types

#### Standard GPU Resources
```yaml
# Request whole GPUs
resources:
  limits:
    nvidia.com/gpu: 2  # 2 complete GPUs
```

#### MIG (Multi-Instance GPU) Resources
```yaml
# Request MIG instances (H100 partitions)
resources:
  limits:
    nvidia.com/mig-1g.10gb: 1  # 1/7 of H100 with 10GB memory
    nvidia.com/mig-3g.40gb: 1  # 3/7 of H100 with 40GB memory
```

#### GPU Memory Sharing
```yaml
# Request specific GPU memory amount (Volcano scheduler)
resources:
  limits:
    volcano.sh/gpu-memory: 8192  # 8GB GPU memory
```

---

## 6. Container Runtime and GPU Integration

### 6.1 Container Lifecycle with GPUs

```
GPU Container Startup Process:

1. Pod Creation
   ├── API Server receives pod spec
   ├── Scheduler assigns to GPU node
   └── kubelet starts container creation

2. Runtime Preparation  
   ├── containerd calls NVIDIA runtime
   ├── NVIDIA runtime checks GPU availability
   └── Prepares GPU device access

3. Container Start
   ├── Mounts GPU devices (/dev/nvidia*)
   ├── Sets up CUDA library paths
   ├── Configures GPU environment variables
   └── Starts application container

4. GPU Access
   ├── Application detects available GPUs
   ├── CUDA initializes GPU connection
   └── Training/inference begins
```

### 6.2 GPU Environment Variables

When containers start with GPU access, these environment variables are automatically set:

```bash
# GPU visibility and configuration
NVIDIA_VISIBLE_DEVICES=0,1    # Which GPUs are available
CUDA_VISIBLE_DEVICES=0,1      # CUDA-specific GPU visibility

# Driver and library paths
NVIDIA_DRIVER_CAPABILITIES=compute,utility
LD_LIBRARY_PATH=/usr/local/nvidia/lib64

# Container runtime info
NVIDIA_REQUIRE_CUDA=cuda>=11.8
```

### 6.3 Common GPU Container Issues and Solutions

#### Issue 1: "CUDA device not found"

**Symptoms**: Container starts but can't detect GPUs
**Cause**: Missing NVIDIA container runtime
**Solution**: Ensure GPU Operator is properly installed

```bash
# Check GPU Operator status
kubectl get pods -n gpu-operator

# Verify node has GPU resources
kubectl describe node <gpu-node-name>
```

#### Issue 2: "Out of memory" errors

**Symptoms**: Training crashes with CUDA OOM
**Cause**: Multiple containers sharing same GPU
**Solution**: Use proper resource limits

```yaml
# Ensure exclusive GPU access
resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    nvidia.com/gpu: 1  # Important: request = limit
```

#### Issue 3: "Driver version mismatch"

**Symptoms**: CUDA runtime errors
**Cause**: Container CUDA version incompatible with driver
**Solution**: Use compatible container images

```yaml
# Use Lambda Stack images (pre-tested combinations)
image: lambdalabs/pytorch:2.1.0-cuda12.1-cudnn8-devel
```

---

## 7. GPU Resource Management and Scheduling

### 7.1 Resource Allocation Strategies

#### Exclusive GPU Access (Recommended for Training)
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: exclusive-training
spec:
  containers:
  - name: pytorch-trainer
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
    resources:
      limits:
        nvidia.com/gpu: 1    # Entire GPU dedicated
        memory: 32Gi
        cpu: 8
      requests:
        nvidia.com/gpu: 1    # Must match limits
        memory: 32Gi
        cpu: 8
```

#### GPU Sharing (For Inference or Small Models)
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: shared-inference
spec:
  containers:
  - name: inference-server
    image: tensorflow/serving:latest-gpu
    resources:
      limits:
        volcano.sh/gpu-memory: 4096  # 4GB of GPU memory
        memory: 8Gi
        cpu: 4
```

### 7.2 Node Affinity for GPU Types

Ensure workloads run on appropriate GPU hardware:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: h100-training
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: gpu.nvidia.com/model
            operator: In
            values:
            - H100
            - H200
  containers:
  - name: training
    image: pytorch/pytorch:latest
    resources:
      limits:
        nvidia.com/gpu: 8
```

### 7.3 Priority Classes for Workload Management

```yaml
# High priority for production training
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority-training
value: 1000
globalDefault: false
description: "High priority for production ML training"

---
# Low priority for experimentation
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: low-priority-experiments  
value: 100
globalDefault: false
description: "Low priority for research experiments"
```

Use in pod specifications:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: production-training
spec:
  priorityClassName: high-priority-training
  containers:
  - name: trainer
    image: pytorch/pytorch:latest
    resources:
      limits:
        nvidia.com/gpu: 4
```

---

## 8. Networking for AI Workloads

### 8.1 Service Types for AI Applications

#### ClusterIP (Internal Communication)
```yaml
# For distributed training communication
apiVersion: v1
kind: Service
metadata:
  name: pytorch-master
spec:
  type: ClusterIP
  selector:
    app: pytorch-training
    role: master
  ports:
  - port: 23456
    targetPort: 23456
```

#### LoadBalancer (External Access)
```yaml
# For model serving endpoints
apiVersion: v1
kind: Service
metadata:
  name: model-serving
spec:
  type: LoadBalancer
  selector:
    app: inference-server
  ports:
  - port: 80
    targetPort: 8080
```

### 8.2 Network Plugins for AI

**Recommended for Lambda Labs**:
- **Calico**: High performance with network policies
- **Cilium**: eBPF-based with advanced features
- **Flannel**: Simple and reliable

**Considerations**:
- Low latency for distributed training
- High bandwidth for data transfer
- Network policies for multi-tenancy

---

## 9. Hands-On Exercises

### Exercise 1: Deploy GPU Workload

**Objective**: Deploy a simple GPU workload and verify GPU access

**Task**: Create and deploy this pod:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  containers:
  - name: cuda-test
    image: nvidia/cuda:12.1-devel-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
  restartPolicy: Never
```

**Questions**:
1. What GPU information does `nvidia-smi` show?
2. How much GPU memory is available?
3. What CUDA version is installed?

### Exercise 2: Troubleshoot GPU Access

**Scenario**: A customer reports their PyTorch container can't access GPUs

**Given Pod Spec**:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: broken-pytorch
spec:
  containers:
  - name: pytorch
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
    command: ["python", "-c", "import torch; print(torch.cuda.is_available())"]
    resources:
      limits:
        memory: 4Gi
        cpu: 2
  restartPolicy: Never
```

**Your Task**: Identify the problem and provide corrected YAML

### Exercise 3: Resource Planning

**Scenario**: Customer wants to run 10 concurrent training jobs, each needing 2 GPUs

**Requirements**:
- Each job needs 2x H100 GPUs
- 32GB RAM per job
- Jobs should not interfere with each other

**Your Task**:
1. Calculate minimum cluster requirements
2. Design pod resource specifications
3. Consider node affinity and anti-affinity rules

---

## 10. Best Practices for AI Workloads

### 10.1 Resource Management

✅ **Always set GPU requests = limits** (avoid overcommitment)  
✅ **Use appropriate memory ratios** (8-16GB RAM per GPU)  
✅ **Set CPU limits** to prevent noisy neighbor issues  
✅ **Use priority classes** for workload prioritization  

### 10.2 Container Images

✅ **Use Lambda Stack images** (pre-optimized and tested)  
✅ **Pin specific versions** for reproducibility  
✅ **Keep images lightweight** (remove unnecessary packages)  
✅ **Use multi-stage builds** for production images  

### 10.3 Monitoring and Observability

✅ **Monitor GPU utilization** with DCGM metrics  
✅ **Track memory usage** to prevent OOM errors  
✅ **Set up alerts** for failed pods and resource exhaustion  
✅ **Use logging** for debugging distributed training  

### 10.4 Security

✅ **Use namespace isolation** for different teams  
✅ **Apply network policies** for traffic control  
✅ **Limit container privileges** (avoid privileged containers)  
✅ **Use service accounts** with minimal permissions  

---

## 11. Assessment Quiz

### Technical Knowledge Questions

**Question 1**: What is the primary purpose of the NVIDIA GPU Operator?
a) Schedule AI workloads efficiently
b) Automate GPU software stack management
c) Monitor GPU performance
d) Provide GPU sharing capabilities

**Question 2**: Which Kubernetes component decides which node should run a GPU pod?
a) kubelet
b) API Server
c) Scheduler
d) Controller Manager

**Question 3**: What happens if you set GPU requests ≠ GPU limits?
a) Better resource utilization
b) Pod scheduling may fail
c) Automatic GPU sharing
d) Improved performance

**Question 4**: Which container runtime component provides GPU access?
a) containerd
b) NVIDIA Container Runtime
c) Docker Engine
d) CRI-O

### Practical Application Questions

**Question 5**: A customer has 4 nodes with 8 H100 GPUs each. They want to run training jobs that need 4 GPUs per job. What's the maximum number of concurrent jobs?

**Question 6**: Explain why you should use `requests: nvidia.com/gpu: 1` even when `limits: nvidia.com/gpu: 1` is already specified.

**Question 7**: A pod is stuck in "Pending" state with the message "Insufficient nvidia.com/gpu". What are three possible causes and solutions?

---

## 12. Troubleshooting Guide

### Common Issues and Solutions

#### Pod Stuck in Pending State

**Symptoms**: Pod remains in Pending status
**Check**: `kubectl describe pod <pod-name>`

**Possible Causes**:
1. **Insufficient GPU resources**: No nodes have requested GPU count
2. **Node affinity mismatch**: Pod requires specific node labels
3. **Resource conflicts**: GPU already allocated to other pods

**Solutions**:
```bash
# Check available GPU resources
kubectl describe nodes | grep nvidia.com/gpu

# Check GPU Operator status
kubectl get pods -n gpu-operator

# Verify node labels
kubectl get nodes --show-labels | grep gpu
```

#### CUDA Runtime Errors

**Symptoms**: "CUDA driver version is insufficient" or "CUDA device not found"

**Solutions**:
1. Verify GPU Operator installation
2. Check container image CUDA compatibility
3. Ensure proper resource requests

#### GPU Memory Issues

**Symptoms**: "RuntimeError: CUDA out of memory"

**Solutions**:
1. Reduce batch size
2. Use gradient accumulation
3. Enable memory optimization flags
4. Use GPU memory monitoring

---

## 13. Customer Scenarios

### Scenario 1: Research Lab Migration

**Customer**: University AI research lab currently using individual workstations

**Challenge**: Researchers wait for GPU access, difficult collaboration, inconsistent environments

**Kubernetes Solution**:
- Multi-tenant cluster with namespace isolation
- Jupyter notebook deployments with GPU access
- Shared datasets through persistent volumes
- Resource quotas for fair sharing

**Value Proposition**: 3x better GPU utilization, easier collaboration, consistent environments

### Scenario 2: AI Startup Scaling

**Customer**: Computer vision startup growing from 5 to 50 engineers

**Challenge**: Manual infrastructure management, inconsistent development environments, difficult scaling

**Kubernetes Solution**:
- Automated scaling based on job queue length
- CI/CD integration for model training
- Development/staging/production namespace separation
- Cost optimization through efficient resource usage

**Value Proposition**: 60% infrastructure cost reduction, 10x faster deployment, better developer productivity

### Scenario 3: Enterprise AI Platform

**Customer**: Fortune 500 company building internal AI platform

**Challenge**: Multiple teams, compliance requirements, integration with existing systems

**Kubernetes Solution**:
- Private cloud with enterprise security
- RBAC integration with corporate identity systems
- Compliance auditing and logging
- Multi-cluster management for different environments

**Value Proposition**: Enterprise-grade security, governance, scalability for hundreds of data scientists

---

## 14. Key Takeaways and Next Steps

### Essential Knowledge Summary

🏗️ **Kubernetes Architecture**: Control plane manages cluster, data plane runs workloads

🎯 **GPU Operator**: Automates GPU software stack management and eliminates manual configuration

📦 **Container Runtime**: NVIDIA Container Runtime provides GPU access to containers

⚙️ **Resource Management**: Proper GPU resource allocation critical for performance and isolation

🔧 **Best Practices**: Monitor resources, use appropriate images, implement security policies

### Customer Value Propositions

💰 **Cost Efficiency**: 80%+ GPU utilization vs 30% with traditional infrastructure  
⚡ **Faster Deployment**: Minutes instead of hours to deploy GPU workloads  
🔒 **Better Security**: Namespace isolation and container security  
📈 **Easy Scaling**: Automatic scaling based on demand  
🛠️ **Reduced Ops Overhead**: Automated management reduces manual work  

### Next Learning Steps

1. **Hands-On Practice**: Deploy actual GPU workloads on Kubernetes cluster
2. **Advanced Topics**: Study job schedulers (Volcano, Kubeflow) in Document 4
3. **Customer Preparation**: Practice explaining Kubernetes benefits in business terms

### Prepare for Document 3

**Next Topic**: High-Performance Computing and Networking
- InfiniBand networking fundamentals
- RDMA programming concepts
- Performance optimization techniques
- Multi-node training considerations

---

## Additional Resources

### Documentation
- [Kubernetes Official Documentation](https://kubernetes.io/docs/)
- [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)
- [Lambda Labs Kubernetes Examples](https://github.com/LambdaLabsML/examples)

### Tools and Utilities
- `kubectl` command reference
- GPU monitoring with `nvidia-smi` and DCGM
- Container debugging techniques

### Training Materials
- Kubernetes certification courses (CKA, CKAD)
- NVIDIA Deep Learning Institute Kubernetes courses
- Cloud Native Computing Foundation (CNCF) training

---

**Document 2 Complete! 🎉**

You now understand how Kubernetes provides the foundation for scalable, efficient AI infrastructure. This knowledge enables you to explain technical benefits to customers and help them understand why container orchestration is essential for modern AI workloads.

**Next**: Document 3 - High-Performance Computing and Networking