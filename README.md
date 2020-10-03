# Modular-HER 

Modular-HER is revised from OpenAI baselines and supports many improvement for HER as modules.

## Functions
- [x] DDPG (https://arxiv.org/abs/1509.02971);
- [x] HER (https://arxiv.org/abs/1707.01495);
- [ ] SHER (https://arxiv.org/abs/2002.02089);
- [ ] Prioritized HER;
- [ ] Energe-based Prioritized HER(https://www.researchgate.net/publication/341776498_Energy-Based_Hindsight_Experience_Prioritization);
- [ ] Curriculum-guided Hindsight Experience Replay (http://papers.nips.cc/paper/9425-curriculum-guided-hindsight-experience-replay);
- [x] Our methods: Multi-step HER ($\lambda$) and Model-based Multi-step HER;
- [ ] more to be continued...


## Prerequisites 
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows
### Ubuntu 
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```
    
### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

## Tensorflow versions
The master branch supports Tensorflow from version 1.4 to 1.14. For Tensorflow 2.0 support, please use tf2 branch.

## Installation
- Clone the repo and cd into it:
    ```bash
    git clone http
    cd mher
- Install baselines package
    ```bash
    pip install -e .
    ```


## Usage

Training and saving models,saving logs.
```bash
python -m mher.run --alg=her --env=FetchPush-v1 --num_epoch 100 --num_env 12 --network=mlp --log_path=~/logs/FetchSlide_push_cpu12_n_step_3/ --save_path=~/policies/her/fetchreach5k
```


## Update
* 6.11 first update of n-step her, add support of num_epoch;
* 7.02 update action threshold method for correction;
* 7.12 update taylor correction;
* 8.2 update lambda multi-step HER
* 8.23 update model-based multi-step HER
* 9.27 update readme

