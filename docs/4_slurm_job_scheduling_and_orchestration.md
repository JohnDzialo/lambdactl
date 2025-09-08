# 7. SLURM Job Scheduling and Orchestration
## 7.1 SLURM Architecture Overview

SLURM is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters. As a cluster workload manager, SLURM has three key functions: it allocates exclusive and/or non-exclusive access to resources to users, provides a framework for starting and monitoring work on allocated nodes, and arbitrates contention for resources by managing a queue of pending work.

### 7.1.1 Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SLURM Architecture Components                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Management Layer                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   slurmctld     │  │   slurmdbd      │  │   slurmrestd    │             │
│  │  (Controller)   │  │  (Database)     │  │  (REST API)     │             │
│  │                 │  │                 │  │                 │             │
│  │• Job Scheduling │  │• Accounting     │  │• HTTP Interface │             │
│  │• Resource Mgmt  │  │• Job History    │  │• JSON API       │             │
│  │• Node Monitoring│  │• User Tracking  │  │• Authentication │             │
│  │• Policy Control │  │• Multi-cluster  │  │• Job Submission │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
├─────────────────────────────────────────────────────────────────────────────┤
│                            Compute Layer                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          Compute Nodes                                 │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │   slurmd    │  │   slurmd    │  │   slurmd    │  │   slurmd    │   │ │
│  │  │   Node 1    │  │   Node 2    │  │   Node 3    │  │   Node N    │   │ │
│  │  │             │  │             │  │             │  │             │   │ │
│  │  │• Job Exec   │  │• Job Exec   │  │• Job Exec   │  │• Job Exec   │   │ │
│  │  │• Resource   │  │• Resource   │  │• Resource   │  │• Resource   │   │ │
│  │  │  Monitoring │  │  Monitoring │  │  Monitoring │  │  Monitoring │   │ │
│  │  │• Log Mgmt   │  │• Log Mgmt   │  │• Log Mgmt   │  │• Log Mgmt   │   │ │
│  │  │• Health     │  │• Health     │  │• Health     │  │• Health     │   │ │
│  │  │  Reporting  │  │  Reporting  │  │  Reporting  │  │  Reporting  │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                             User Interface                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         Client Commands                                │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │    srun     │  │   sbatch    │  │   squeue    │  │   scancel   │   │ │
│  │  │(Interactive)│  │ (Batch Job) │  │(Job Status) │  │(Cancel Job) │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │    sinfo    │  │   sacct     │  │   salloc    │  │  scontrol   │   │ │
│  │  │(Node Info)  │  │(Accounting) │  │(Allocation) │  │ (Control)   │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.1.2 Key Entities

The entities managed by SLURM daemons include nodes (the compute resource), partitions (which group nodes into logical sets), jobs (allocations of resources assigned to a user for a specified amount of time), and job steps (sets of parallel tasks within a job).

## 7.2 SLURM Configuration and Setup

### 7.2.1 Basic Configuration File

```bash
# /etc/slurm/slurm.conf - Basic Lambda Labs Configuration
ClusterName=lambda-cluster
SlurmUser=slurm
SlurmdUser=root
SlurmctldHost=lambda-head-node
MpiDefault=none
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup

# Scheduling Configuration
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory

# GPU Configuration
GresTypes=gpu
NodeName=DEFAULT CPUs=64 Sockets=2 CoresPerSocket=32 ThreadsPerCore=1 RealMemory=515000
NodeName=lambda-gpu-[001-016] Gres=gpu:h100:8 Feature=h100
NodeName=lambda-gpu-[017-032] Gres=gpu:h200:8 Feature=h200

# Partition Configuration
PartitionName=gpu Default=YES Nodes=lambda-gpu-[001-032] MaxTime=7-00:00:00 State=UP
PartitionName=interactive Nodes=lambda-gpu-[001-004] MaxTime=02:00:00 State=UP
PartitionName=batch Nodes=lambda-gpu-[005-032] MaxTime=7-00:00:00 State=UP

# Resource Limits
MaxJobCount=10000
MaxArraySize=10000
MaxStepCount=40000
MaxTasksPerNode=64

# Accounting
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=lambda-head-node
JobAcctGatherType=jobacct_gather/linux
JobAcctGatherFrequency=30

# Communication Settings
SlurmctldPort=6817
SlurmdPort=6818
SlurmctldTimeout=120
SlurmdTimeout=300
InactiveLimit=0
KillWait=30
MinJobAge=300
```

### 7.2.2 GPU Resource Configuration

```bash
# /etc/slurm/gres.conf - GPU Resource Configuration
AutoDetect=nvml
Name=gpu Type=h100 File=/dev/nvidia[0-7] Cores=0-31
Name=gpu Type=h200 File=/dev/nvidia[0-7] Cores=32-63

# Alternative explicit configuration
# NodeName=lambda-gpu-001 Name=gpu Type=h100 File=/dev/nvidia0 CPUs=0-7
# NodeName=lambda-gpu-001 Name=gpu Type=h100 File=/dev/nvidia1 CPUs=8-15
# NodeName=lambda-gpu-001 Name=gpu Type=h100 File=/dev/nvidia2 CPUs=16-23
# NodeName=lambda-gpu-001 Name=gpu Type=h100 File=/dev/nvidia3 CPUs=24-31
# NodeName=lambda-gpu-001 Name=gpu Type=h100 File=/dev/nvidia4 CPUs=32-39
# NodeName=lambda-gpu-001 Name=gpu Type=h100 File=/dev/nvidia5 CPUs=40-47
# NodeName=lambda-gpu-001 Name=gpu Type=h100 File=/dev/nvidia6 CPUs=48-55
# NodeName=lambda-gpu-001 Name=gpu Type=h100 File=/dev/nvidia7 CPUs=56-63
```

## 7.3 SLURM Job Submission and Management

### 7.3.1 Essential Commands

**Job Submission Commands:**
```bash
# Interactive job submission
srun --nodes=1 --ntasks=8 --gres=gpu:2 --time=01:00:00 --pty bash

# Batch job submission
sbatch my_training_script.slurm

# Resource allocation (creates allocation, spawns shell)
salloc --nodes=2 --ntasks=16 --gres=gpu:4 --time=02:00:00
```

**Job Monitoring Commands:**
```bash
# View job queue
squeue -u $USER                    # Your jobs only
squeue -p gpu                      # Jobs in GPU partition
squeue --format="%.10i %.15j %.8u %.2t %.10M %.6D %R"

# View node information
sinfo -N                           # All nodes
sinfo -p gpu                       # GPU partition nodes
sinfo --format="%.15N %.6T %.14C %.10m %.25f %25G"

# Job accounting information
sacct -j <job_id>                  # Specific job
sacct --starttime=2024-01-01 --format=JobID,JobName,User,State,ExitCode,MaxRSS

# Cancel jobs
scancel <job_id>                   # Cancel specific job
scancel -u $USER                   # Cancel all your jobs
scancel -p gpu                     # Cancel jobs in GPU partition
```

### 7.3.2 Batch Job Script Examples

#### 7.3.2.1 Single-GPU PyTorch Training

```bash
#!/bin/bash
#SBATCH --job-name=pytorch-single-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=pytorch_single_%j.out
#SBATCH --error=pytorch_single_%j.err

# Environment setup
module load cuda/12.1
module load python/3.11
source /shared/environments/pytorch-env/bin/activate

# Job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Run training script
python train_model.py \
    --dataset /shared/datasets/imagenet \
    --model resnet50 \
    --batch-size 128 \
    --epochs 10 \
    --output-dir /shared/results/$SLURM_JOB_ID

echo "End time: $(date)"
```

#### 7.3.2.2 Multi-GPU Distributed Training

```bash
#!/bin/bash
#SBATCH --job-name=pytorch-multi-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=pytorch_multi_%j.out
#SBATCH --error=pytorch_multi_%j.err

# Environment setup
module load cuda/12.1
module load python/3.11
module load openmpi/4.1.4
source /shared/environments/pytorch-env/bin/activate

# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID

echo "Master node: $MASTER_ADDR"
echo "World size: $WORLD_SIZE"
echo "Node: $SLURMD_NODENAME"
echo "Local rank: $SLURM_LOCALID"

# Run distributed training
srun python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_distributed.py \
        --dataset /shared/datasets/c4 \
        --model gpt2-large \
        --batch-size 32 \
        --epochs 5 \
        --output-dir /shared/results/$SLURM_JOB_ID
```

### 7.3.3 Job Arrays for Parameter Sweeps

```bash
#!/bin/bash
#SBATCH --job-name=hyperparameter-sweep
#SBATCH --partition=gpu
#SBATCH --array=1-20%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=sweep_%A_%a.out
#SBATCH --error=sweep_%A_%a.err

# Environment setup
module load cuda/12.1
source /shared/environments/pytorch-env/bin/activate

# Parameter arrays
learning_rates=(0.001 0.01 0.1 0.0001 0.00001)
batch_sizes=(16 32 64 128)
optimizers=("adam" "sgd" "adamw" "rmsprop")

# Calculate parameter combinations
lr_index=$(( ($SLURM_ARRAY_TASK_ID - 1) % ${#learning_rates[@]} ))
bs_index=$(( (($SLURM_ARRAY_TASK_ID - 1) / ${#learning_rates[@]}) % ${#batch_sizes[@]} ))
opt_index=$(( (($SLURM_ARRAY_TASK_ID - 1) / (${#learning_rates[@]} * ${#batch_sizes[@]})) % ${#optimizers[@]} ))

LR=${learning_rates[$lr_index]}
BS=${batch_sizes[$bs_index]}
OPT=${optimizers[$opt_index]}

echo "Task $SLURM_ARRAY_TASK_ID: LR=$LR, BS=$BS, OPT=$OPT"

# Run training with specific parameters
python train_model.py \
    --learning-rate $LR \
    --batch-size $BS \
    --optimizer $OPT \
    --dataset /shared/datasets/cifar10 \
    --output-dir /shared/results/sweep_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
```

## 7.4 Advanced SLURM Features

### 7.4.1 Resource Reservations

```bash
# Create advance reservation for specific users/groups
scontrol create reservation \
    ReservationName=ai-team-training \
    StartTime=2024-12-01T09:00:00 \
    Duration=7-00:00:00 \
    Nodes=lambda-gpu-[001-008] \
    Users=alice,bob,charlie \
    Features=h100

# Create maintenance reservation
scontrol create reservation \
    ReservationName=maintenance \
    StartTime=2024-12-15T02:00:00 \
    EndTime=2024-12-15T06:00:00 \
    Nodes=ALL \
    Flags=MAINT,IGNORE_JOBS
```

### 7.4.2 Quality of Service (QoS) Configuration

```bash
# Create QoS levels for different user priorities
sacctmgr add qos name=high priority=1000 MaxWall=7-00:00:00 MaxJobsPerUser=10
sacctmgr add qos name=normal priority=500 MaxWall=3-00:00:00 MaxJobsPerUser=5
sacctmgr add qos name=low priority=100 MaxWall=1-00:00:00 MaxJobsPerUser=2

# Associate QoS with accounts
sacctmgr modify account ai-research set qos=high
sacctmgr modify account students set qos=normal
```

### 7.4.3 Job Dependencies and Workflows

```bash
# Submit job with dependency on previous job completion
job1=$(sbatch --parsable single_gpu_job.slurm)
job2=$(sbatch --parsable --dependency=afterok:$job1 post_process_job.slurm)
job3=$(sbatch --parsable --dependency=afterok:$job2 final_analysis.slurm)

# Complex dependency example - wait for multiple jobs
sbatch --dependency=afterok:123:124:125 final_job.slurm
```

### 7.4.4 Integration with Lambda Labs Infrastructure

```bash
# Lambda-specific SLURM configuration additions
# Enable GPU health checking
HealthCheckProgram=/opt/lambda/bin/gpu_health_check
HealthCheckInterval=300

# Custom job submit plugin for Lambda Cloud integration
JobSubmitPlugins=lambda_cloud_billing

# Integration with Lambda monitoring
JobCompHost=lambda-monitoring-01
JobCompPort=6819
JobCompType=jobcomp/lambda_metrics

# Burst to Lambda Cloud on-demand instances
ResumeProgram=/opt/lambda/bin/burst_to_cloud
SuspendProgram=/opt/lambda/bin/suspend_cloud_nodes
ResumeTimeout=600
SuspendTimeout=30
```

This SLURM configuration provides a robust foundation for managing AI/ML workloads on Lambda Labs infrastructure, with support for GPU scheduling, distributed training, and efficient resource utilization.