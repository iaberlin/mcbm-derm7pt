MCBM REPLICATION on MPI3D

replication target:
original: https://github.com/antonioalmudevar/minimal_cbm
target: mcbm on mpi3d
original config: mpi3d-mcbm-1.yaml
original results: TBD


original setup:
dataset: mpi3d real
model: mcbm with resnet-20 backbone
n_epochs: 50
gamma: 1

###---------------------------------------------------------------------------------

replica experiment:
config: mpi3d-mcbm-replica.yaml (with gamma=1)
cluster: bwUniCluster 3.0
GPU: A100
Python 3.9.23
PyTorch 1.12.1+cu113
requirements adjustet: requirements-cluster.txt

dataset:
mpi3d_real.nz: https://github.com/rr-learning/disentanglement_dataset

| Change                         | Reason                                       | Expected scientific impact     |
| ------------------------------ | -------------------------------------------- | ------------------------------ |
| MPI3D path changed             | Different filesystem                         | None                           |
| MPI3D loader adapted           | Dataset representation differed              | Should preserve data semantics |
| NHWC → NCHW                    | PyTorch encoder requirement                  | None                           |
| W&B disabled/offline           | Logging not required                         | None                           |
| Hard-coded W&B API key removed | Security                                     | None                           |
| Dependency versions adjusted   | Original environment incompatible on cluster | To be assessed                 |


Pre-replication validation:
H100 compatibility test: FAILED
Reason: PyTorch 1.12.1+cu113 does not support H100 (sm_90).
Decision: Use NVIDIA A100.

MPI3D MCBM Smoke Test
Job ID: 6241242
Seed: 42
Epochs: 1
GPU: NVIDIA A100-PCIE-40GB

Result: COMPLETED (exit code 0)
Wall time: 00:20:20
Peak memory: 18.70 GB

###---------------------------------------------------------------------------------

FULL REPLICATION RUN:
config: configs/mpi3d/mpi3d-mcbm-replica.yaml
slurm script: jobs/mpi3d_mcbm_replica.sh
seed: 42
epochs: 50
job ID: 6243140

| Metric                 | Original MCBM | Replication | Difference |
| ---------------------- | ------------: | ----------: | ---------: |
| Task accuracy          |             … |           … |          … |
| Concept accuracy       |             … |           … |          … |
| ECE                    |             … |           … |          … |
| Brier                  |             … |           … |          … |


task-related nuisance URR 10.7+- 0.1
nuisance value: 0.0
avg. concept accuracy: 100
task accuracy: 92.7 +- 1.1





###---------------------------------------------------------------------------------

Experiment Log:

2026-08-10
H100 test failed due to unsupported sm_90 architecture.
Decision: use A100.

2026-08-10
MPI3D loader failed because dataset contained only `images`.
Investigated expected dataset representation.

2026-08-10
Input shape mismatch discovered: NHWC instead of NCHW.
Loader adapted.

2026-08-10
Smoke test 6241242 completed successfully.

2026-08-11
Full 50-epoch replication submitted as job 6243140.

2026-08-14
Job still pending due to Priority.
Slurm currently predicts start on 2026-08-19 20:20.

2026-08-18
Job COMPLETED
Job Wall-clock time: 01:08:53
Memory Utilized: 14.87 GB
Memory Efficiency: 61.97% of 24.00 GB (24.00 GB/node)


###------------------------------------------------------------------------

2026-08-22
started replication on different seeds

JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
6454555 gpu_a100_ mpi3d-mc ma_iadam PD       0:00      1 (Priority)
6454641 gpu_a100_ mpi3d-mc ma_iadam PD       0:00      1 (Priority)
6454640 gpu_a100_ mpi3d-mc ma_iadam PD       0:00      1 (Priority)




