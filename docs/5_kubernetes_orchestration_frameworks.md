# 8. Kubernetes Orchestration with Volcano, Ray, and Kubeflow

## 8.1 Overview of Kubernetes AI Orchestration

Kubernetes has become the de facto standard for container orchestration, and its capabilities have been extended to support AI and ML workloads through specialized frameworks. Each framework addresses different aspects of the AI lifecycle:

- **Volcano**: Kubernetes-native batch scheduler optimized for HPC and AI workloads
- **Ray/KubeRay**: Distributed computing framework for scaling Python applications
- **Kubeflow**: Comprehensive ML platform covering the entire AI lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Kubernetes AI Orchestration Landscape                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                             AI Lifecycle                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │    Data     │  │   Model     │  │   Model     │  │   Model     │       │
│  │ Preparation │  │  Training   │  │Optimization │  │  Serving    │       │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘       │
│         │               │               │               │                 │
├─────────▼───────────────▼───────────────▼───────────────▼─────────────────┤
│                         Framework Capabilities                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           Kubeflow                                     │ │
│  │  Full AI Lifecycle • Pipelines • Training • Serving • Model Registry  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                             Ray                                        │ │
│  │  Distributed Computing • Scaling • Serving • Hyperparameter Tuning    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           Volcano                                      │ │
│  │  Batch Scheduling • Gang Scheduling • GPU Management • Queue Control   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                        Kubernetes Foundation                               │
│  Resource Management • Container Orchestration • Service Discovery         │
│  Networking • Storage • Security • Monitoring • Auto-scaling               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 8.2 Volcano: Kubernetes-Native Batch Scheduler

Volcano is a Kubernetes-native batch scheduling system designed for high-performance workloads, including AI/ML, Big Data, and HPC applications.

### 8.2.1 Volcano Installation and Setup

```bash
# Install Volcano using Helm
helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace

# Alternative: Install using kubectl
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml

# Verify installation
kubectl get pods -n volcano-system
```

### 8.2.2 Volcano Queue Configuration

```yaml
# Volcano Queue Configuration
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: ai-training-queue
spec:
  weight: 100
  capability:
    cpu: "400"
    memory: "800Gi" 
    nvidia.com/gpu: "32"
  reclaimable: true
  guarantee:
    cpu: "200"
    memory: "400Gi"
    nvidia.com/gpu: "16"
---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: inference-queue
spec:
  weight: 50
  capability:
    cpu: "200"
    memory: "400Gi"
    nvidia.com/gpu: "16"
  reclaimable: false
```

### 8.2.3 PyTorch Training with Volcano

```yaml
# Distributed PyTorch Training Job with Volcano
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: pytorch-distributed-training
spec:
  minAvailable: 3  # Gang scheduling - all pods must be scheduled together
  schedulerName: volcano
  queue: ai-training-queue
  
  plugins:
    env: []
    svc: []
  
  tasks:
  - replicas: 1
    name: master
    template:
      spec:
        restartPolicy: OnFailure
        containers:
        - name: pytorch-master
          image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
          imagePullPolicy: IfNotPresent
          command:
          - sh
          - -c
          - |
            pip install accelerate transformers datasets
            python -c "
            import torch
            import torch.distributed as dist
            import torch.nn as nn
            import torch.optim as optim
            from torch.nn.parallel import DistributedDataParallel as DDP
            import os
            
            # Initialize distributed training
            os.environ['MASTER_ADDR'] = 'pytorch-distributed-training-master-0.pytorch-distributed-training'
            os.environ['MASTER_PORT'] = '23456'
            os.environ['WORLD_SIZE'] = '3'
            os.environ['RANK'] = '0'
            
            dist.init_process_group('nccl')
            
            # Simple model for demonstration
            model = nn.Linear(1000, 10).cuda()
            model = DDP(model)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            for epoch in range(100):
                data = torch.randn(64, 1000).cuda()
                target = torch.randint(0, 10, (64,)).cuda()
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    print(f'Rank 0, Epoch {epoch}, Loss: {loss.item():.4f}')
            
            print('Training completed on master')
            "
          resources:
            requests:
              cpu: "4"
              memory: "8Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "8"
              memory: "16Gi"
              nvidia.com/gpu: "1"
          env:
          - name: NVIDIA_VISIBLE_DEVICES
            value: "all"
          
  - replicas: 2
    name: worker
    template:
      spec:
        restartPolicy: OnFailure
        containers:
        - name: pytorch-worker
          image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
          imagePullPolicy: IfNotPresent
          command:
          - sh
          - -c
          - |
            pip install accelerate transformers datasets
            python -c "
            import torch
            import torch.distributed as dist
            import torch.nn as nn
            import torch.optim as optim
            from torch.nn.parallel import DistributedDataParallel as DDP
            import os
            import time
            
            # Wait for master to be ready
            time.sleep(10)
            
            # Initialize distributed training
            os.environ['MASTER_ADDR'] = 'pytorch-distributed-training-master-0.pytorch-distributed-training'
            os.environ['MASTER_PORT'] = '23456'
            os.environ['WORLD_SIZE'] = '3'
            
            # Determine rank based on hostname
            hostname = os.environ['HOSTNAME']
            if 'worker-0' in hostname:
                os.environ['RANK'] = '1'
            else:
                os.environ['RANK'] = '2'
            
            dist.init_process_group('nccl')
            
            # Simple model for demonstration
            model = nn.Linear(1000, 10).cuda()
            model = DDP(model)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            for epoch in range(100):
                data = torch.randn(64, 1000).cuda()
                target = torch.randint(0, 10, (64,)).cuda()
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    rank = dist.get_rank()
                    print(f'Rank {rank}, Epoch {epoch}, Loss: {loss.item():.4f}')
            
            rank = dist.get_rank()
            print(f'Training completed on worker rank {rank}')
            "
          resources:
            requests:
              cpu: "4"
              memory: "8Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "8"
              memory: "16Gi"
              nvidia.com/gpu: "1"
          env:
          - name: NVIDIA_VISIBLE_DEVICES
            value: "all"
```

## 8.3 Ray and KubeRay: Distributed Computing

Ray is a distributed computing framework that makes it easy to scale Python applications. KubeRay provides Kubernetes-native Ray cluster management.

### 8.3.1 KubeRay Installation

```bash
# Install KubeRay operator
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.0.0

# Verify installation
kubectl get pods -l app.kubernetes.io/name=kuberay-operator
```

### 8.3.2 RayCluster Configuration

```yaml
# RayCluster for distributed ML workloads
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ml-training-cluster
spec:
  rayVersion: '2.8.0'
  enableInTreeAutoscaling: true
  autoscalerOptions:
    upscalingMode: Default
    idleTimeoutSeconds: 60
    imagePullPolicy: Always
    
  headGroupSpec:
    serviceType: ClusterIP
    rayStartParams:
      dashboard-host: '0.0.0.0'
      port: '6379'
      object-store-memory: '100000000'
      num-cpus: '4'
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.8.0-gpu
          imagePullPolicy: Always
          ports:
          - containerPort: 6379
            name: gcs-server
          - containerPort: 8265
            name: dashboard
          - containerPort: 10001
            name: client
          resources:
            requests:
              cpu: "4"
              memory: "16Gi"
            limits:
              cpu: "8"
              memory: "32Gi"
          env:
          - name: RAY_DISABLE_DOCKER_CPU_WARNING
            value: "1"
          - name: TYPE
            value: "head"
          volumeMounts:
          - name: shared-storage
            mountPath: /shared
        volumes:
        - name: shared-storage
          persistentVolumeClaim:
            claimName: ray-shared-storage
            
  workerGroupSpecs:
  - replicas: 4
    minReplicas: 2
    maxReplicas: 10
    groupName: gpu-workers
    rayStartParams:
      num-cpus: '8'
      num-gpus: '1'
      object-store-memory: '100000000'
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.8.0-gpu
          imagePullPolicy: Always
          resources:
            requests:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "16"
              memory: "64Gi"
              nvidia.com/gpu: "1"
          env:
          - name: RAY_DISABLE_DOCKER_CPU_WARNING
            value: "1"
          - name: TYPE
            value: "worker"
          volumeMounts:
          - name: shared-storage
            mountPath: /shared
        volumes:
        - name: shared-storage
          persistentVolumeClaim:
            claimName: ray-shared-storage
```

### 8.3.3 RayJob for Distributed Training

```yaml
# RayJob for distributed PyTorch training
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: pytorch-distributed-job
spec:
  rayClusterSpec:
    rayVersion: '2.8.0'
    headGroupSpec:
      rayStartParams:
        dashboard-host: '0.0.0.0'
        port: '6379'
        num-cpus: '4'
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray:2.8.0-gpu
            resources:
              requests:
                cpu: "4"
                memory: "16Gi"
              limits:
                cpu: "8"
                memory: "32Gi"
    workerGroupSpecs:
    - replicas: 3
      minReplicas: 3
      maxReplicas: 3
      groupName: gpu-workers
      rayStartParams:
        num-cpus: '8'
        num-gpus: '1'
      template:
        spec:
          containers:
          - name: ray-worker
            image: rayproject/ray:2.8.0-gpu
            resources:
              requests:
                cpu: "8"
                memory: "32Gi"
                nvidia.com/gpu: "1"
              limits:
                cpu: "16"
                memory: "64Gi"
                nvidia.com/gpu: "1"
  
  entrypoint: |
    import ray
    import torch
    import torch.nn as nn
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig, RunConfig
    import numpy as np
    
    ray.init()
    
    def train_func():
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        import ray.train.torch
        
        # Simple model
        model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        # Prepare model for distributed training
        model = ray.train.torch.prepare_model(model)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Synthetic dataset
        X = torch.randn(1000, 784)
        y = torch.randint(0, 10, (1000,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Prepare dataloader
        dataloader = ray.train.torch.prepare_data_loader(dataloader)
        
        # Training loop
        for epoch in range(10):
            total_loss = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            ray.train.report({"loss": avg_loss, "epoch": epoch})
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    # Configure distributed training
    scaling_config = ScalingConfig(
        num_workers=3,
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 8}
    )
    
    run_config = RunConfig(
        name="pytorch-distributed-training"
    )
    
    # Create and run trainer
    trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config
    )
    
    result = trainer.fit()
    print("Training completed successfully!")
    print(f"Final metrics: {result.metrics}")
  
  runtimeEnvYAML: |
    pip:
      - torch==2.1.0
      - torchvision==0.16.0
      - numpy==1.24.3
  
  submitterPodTemplate:
    spec:
      containers:
      - name: ray-job-submitter
        image: rayproject/ray:2.8.0-gpu
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
```

## 8.4 Kubeflow: Comprehensive ML Platform

Kubeflow provides a comprehensive ML platform covering the entire AI lifecycle from data preparation to model serving.

### 8.4.1 Kubeflow Installation

```bash
# Install Kubeflow using manifests
git clone https://github.com/kubeflow/manifests.git
cd manifests

# Install individual components or full platform
kubectl apply -k apps/pipeline/upstream/env/cert-manager/platform-agnostic-multi-user
kubectl apply -k apps/katib/upstream/installs/katib-with-kubeflow
kubectl apply -k apps/training-operator/upstream/overlays/kubeflow
kubectl apply -k apps/kserve/upstream/overlays/kubeflow

# Verify installation
kubectl get pods -n kubeflow
```

### 8.4.2 Kubeflow Training Job

```yaml
# PyTorchJob with Kubeflow Training Operator
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: distributed-pytorch-training
  namespace: kubeflow
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
            command:
            - python
            - -m
            - torch.distributed.launch
            - --nproc_per_node=1
            - --nnodes=3
            - --node_rank=0
            - --master_addr=distributed-pytorch-training-master-0
            - --master_port=23456
            - /workspace/train.py
            args:
            - --epochs=10
            - --batch-size=64
            - --learning-rate=0.001
            resources:
              requests:
                cpu: "4"
                memory: "16Gi"
                nvidia.com/gpu: "1"
              limits:
                cpu: "8"
                memory: "32Gi"
                nvidia.com/gpu: "1"
            volumeMounts:
            - name: training-code
              mountPath: /workspace
            - name: training-data
              mountPath: /data
            - name: model-output
              mountPath: /output
          volumes:
          - name: training-code
            configMap:
              name: training-script
          - name: training-data
            persistentVolumeClaim:
              claimName: training-data-pvc
          - name: model-output
            persistentVolumeClaim:
              claimName: model-output-pvc
              
    Worker:
      replicas: 2
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
            command:
            - python
            - -m
            - torch.distributed.launch
            - --nproc_per_node=1
            - --nnodes=3
            - --master_addr=distributed-pytorch-training-master-0
            - --master_port=23456
            - /workspace/train.py
            args:
            - --epochs=10
            - --batch-size=64
            - --learning-rate=0.001
            resources:
              requests:
                cpu: "4"
                memory: "16Gi"
                nvidia.com/gpu: "1"
              limits:
                cpu: "8"
                memory: "32Gi"
                nvidia.com/gpu: "1"
            volumeMounts:
            - name: training-code
              mountPath: /workspace
            - name: training-data
              mountPath: /data
            - name: model-output
              mountPath: /output
          volumes:
          - name: training-code
            configMap:
              name: training-script
          - name: training-data
            persistentVolumeClaim:
              claimName: training-data-pvc
          - name: model-output
            persistentVolumeClaim:
              claimName: model-output-pvc
```

### 8.4.3 Katib Hyperparameter Tuning

```yaml
# Katib Experiment for hyperparameter optimization
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: pytorch-hyperparameter-tuning
  namespace: kubeflow
spec:
  algorithm:
    algorithmName: random
    algorithmSettings:
    - name: random_state
      value: "10"
  
  objective:
    type: maximize
    goal: 0.99
    objectiveMetricName: Validation-Accuracy
    additionalMetricNames:
    - Train-Accuracy
    - Training-Loss
  
  parameters:
  - name: learning_rate
    parameterType: double
    feasibleSpace:
      min: "0.0001"
      max: "0.1"
  - name: batch_size
    parameterType: int
    feasibleSpace:
      min: "16"
      max: "128"
      step: "16"
  - name: num_layers
    parameterType: int
    feasibleSpace:
      min: "2"
      max: "5"
  - name: hidden_size
    parameterType: int
    feasibleSpace:
      min: "64"
      max: "512"
      step: "64"
  
  parallelTrialCount: 4
  maxTrialCount: 20
  maxFailedTrialCount: 3
  
  trialTemplate:
    primaryContainerName: training-container
    trialParameters:
    - name: learningRate
      description: Learning rate for training
      reference: learning_rate
    - name: batchSize
      description: Batch size for training
      reference: batch_size
    - name: numLayers
      description: Number of hidden layers
      reference: num_layers
    - name: hiddenSize
      description: Size of hidden layers
      reference: hidden_size
    
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          spec:
            restartPolicy: Never
            containers:
            - name: training-container
              image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
              command:
              - python
              - -c
              - |
                import torch
                import torch.nn as nn
                import torch.optim as optim
                from torch.utils.data import DataLoader, TensorDataset
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score
                import argparse
                import numpy as np
                
                # Parse hyperparameters
                parser = argparse.ArgumentParser()
                parser.add_argument('--learning-rate', type=float, default=0.01)
                parser.add_argument('--batch-size', type=int, default=32)
                parser.add_argument('--num-layers', type=int, default=3)
                parser.add_argument('--hidden-size', type=int, default=128)
                args = parser.parse_args()
                
                # Generate synthetic data
                np.random.seed(42)
                X = np.random.randn(5000, 20)
                y = (X.sum(axis=1) > 0).astype(int)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Convert to tensors
                X_train = torch.FloatTensor(X_train)
                X_test = torch.FloatTensor(X_test)
                y_train = torch.LongTensor(y_train)
                y_test = torch.LongTensor(y_test)
                
                # Create model
                layers = []
                input_size = 20
                
                for i in range(args.num_layers):
                    layers.append(nn.Linear(input_size, args.hidden_size))
                    layers.append(nn.ReLU())
                    input_size = args.hidden_size
                
                layers.append(nn.Linear(input_size, 2))
                model = nn.Sequential(*layers)
                
                # Training setup
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
                
                # Data loaders
                train_dataset = TensorDataset(X_train, y_train)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                
                # Training loop
                model.train()
                for epoch in range(50):
                    total_loss = 0
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    avg_loss = total_loss / len(train_loader)
                    
                    # Validation
                    model.eval()
                    with torch.no_grad():
                        train_outputs = model(X_train)
                        _, train_predicted = torch.max(train_outputs.data, 1)
                        train_accuracy = accuracy_score(y_train.numpy(), train_predicted.numpy())
                        
                        test_outputs = model(X_test)
                        _, test_predicted = torch.max(test_outputs.data, 1)
                        test_accuracy = accuracy_score(y_test.numpy(), test_predicted.numpy())
                    
                    model.train()
                    
                    # Log metrics (Katib format)
                    print(f"epoch={epoch}")
                    print(f"Training-Loss={avg_loss:.6f}")
                    print(f"Train-Accuracy={train_accuracy:.6f}")
                    print(f"Validation-Accuracy={test_accuracy:.6f}")
                
                print(f"Final Validation-Accuracy={test_accuracy:.6f}")
              
              args:
              - --learning-rate=${trialParameters.learningRate}
              - --batch-size=${trialParameters.batchSize}
              - --num-layers=${trialParameters.numLayers}
              - --hidden-size=${trialParameters.hiddenSize}
              
              resources:
                requests:
                  cpu: "2"
                  memory: "4Gi"
                  nvidia.com/gpu: "1"
                limits:
                  cpu: "4"
                  memory: "8Gi"
                  nvidia.com/gpu: "1"
```

This comprehensive section provides practical examples for deploying and managing AI/ML workloads using Volcano, Ray/KubeRay, and Kubeflow on Kubernetes, covering distributed training, hyperparameter tuning, and model serving scenarios.