# contextual-isaacgym

## Installation
Install Python and required packages including
- Python 3.7.12
- torch==1.11.0
- gym==0.23.1

Install IsaacGym_Preview_4_Package and create an Anaconda environment following the IsaacGym docs.
```
bash create_conda_env_rlgpu.sh
```

Clone the IsaacGymEnvs repository and install.
```
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs.git
cd IsaacGymEnvs
pip install -e .
```

## Usage
### Default Environments
#### Training
```
python -m src.train_ppo --context-encoder-type {...} --env-id {...}
```
#### Evaluation
```
python -m src.test_ppo --checkpoint-path {.../checkpoint_111111.pt}
```
### Randomized Environments
#### Training
```
python -m src.train_ppo --context-encoder-type {...} --env-id {Contextual...} --randomization-type {...}
```
#### Evaluation
```
python -m src.test_ppo --checkpoint-path {.../checkpoint_111111.pt} --test-env-id {Contextual...} --test-randomization-type {...}
```

### Argument options
context-encoder-type: [none, vanilla_stacked, vanilla_rnn, ...] (refer to context_encoder in [base_config.yaml](https://github.com/CheongWoong/contextual-isaacgym/blob/main/configs/base_config.yaml))

env-id (task name): [Ant, Anymal, ...] (refer to [IsaacGym tasks](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/tree/main/isaacgymenvs/tasks) for default environments, [Contextual-IsaacGym tasks](https://github.com/CheongWoong/contextual-isaacgym/blob/main/src/envs/contextual_isaacgymenvs/tasks/__init__.py) for randomized environments.

randomization-type: [default, ...] varies across tasks (refer to context_encoder in [configs/randomization/~.yaml](https://github.com/CheongWoong/contextual-isaacgym/tree/main/configs/randomization))

For additional options, refer to [arguments.py](https://github.com/CheongWoong/contextual-isaacgym/blob/main/src/utils/arguments.py).

## References
```
CleanRL
Stable Baseline 3
IsaacGym
PPO
```
