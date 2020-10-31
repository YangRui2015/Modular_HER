# Modular-HER 
Modular-HER is revised from OpenAI baselines and supports many improvements for Hindsight Experience Replay (HER) as modules. We aim to provide a more **modular**, **readable** and **consice** package for Multi-goal Reinforcement Learning.


## Functions
- [x] DDPG (https://arxiv.org/abs/1509.02971);
- [x] HER (future, episode, final, random) (https://arxiv.org/abs/1707.01495);
- [x] Cut HER (incrementally increase the future sample length);
- [x] SHER (https://arxiv.org/abs/2002.02089);
- [x] Prioritized HER (same as PHER in https://arxiv.org/abs/1905.08786);
- [ ] Energe-based Prioritized HER(https://www.researchgate.net/publication/341776498_Energy-Based_Hindsight_Experience_Prioritization);
- [ ] Curriculum-guided Hindsight Experience Replay (http://papers.nips.cc/paper/9425-curriculum-guided-hindsight-experience-replay);
- [x] nstep DDPG and nstep HER;
- [ ] more to be continued...


## Prerequisites 
Require python3 (>=3.5), tensorflow (>=1.4,<=1.14) and system packages CMake, OpenMPI and zlib. Those can be installed as follows

#### Ubuntu :
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### Mac OS X  :
With [Homebrew](https://brew.sh) installed, run the following:
```bash
brew install cmake openmpi
```

## Installation
```bash
git clone https://github.com/YangRui2015/Modular_HER.git
cd Modular_HER
pip install -e .
```


## Usage
Trainging DDPG and save logs and models.
```bash
python -m mher.run --env=FetchReach-v1 --num_epoch 30 --num_env 1 --sampler random --play_episodes 5 --log_path=~/logs/fetchreach/ --save_path=~/logs/models/fetchreach_ddpg/
```

Trainging HER + DDPG with different sampler ('her_future', 'her_random', 'her_last', 'her_episode' are supported).
```bash
python -m mher.run --env=FetchReach-v1 --num_epoch 30 --num_env 1 --sampler her_future --play_episodes 5 --log_path=~/logs/fetchreach/ --save_path=~/logs/models/fetchreach_herfuture/
```

Training SAC + HER.
```bash
python -m mher.run  --env=FetchReach-v1 --num_epoch 50  --algo sac --sac_alpha 0.05 --sampler her_episode 
```

All support sampler flags.
| Group | Samplers | 
| ------ | ------ | 
| Random sampler | random | 
| HER | her_future, her_episode, her_last, her_random |
| Nstep| nstep, nstep_her_future, nstep_her_epsisode, nstep_her_last, nstep_her_random|
| Priority| priority, priority_her_future, priority_her_episode, priority_her_random, priority_her_last|


## Results

We use a group of test parameters in DEFAULT_ENV_PARAMS for performance comparison in FetchReach-v1 environment. 

1. Performance of HER of different goal sample methods (future, random, episode, last).

<div  align="center"> <img src="./data/mher_all.png" width=500;  /></div>    

2. Performance of Nstep HER and Nstep DDPG.

<div  align="center"><img src="./data/mher_all_step.png" width=500;" /></div>

3. Performance of SHER (Not good enough in FetchReach environment, I will test more envs to report). 

<div  align="center"><img src="./data/mher_sac.png" width=500;" /></div>


## Update

* 9.27 V0.0: update readme;
* 10.3 V0.5: revised code framework hugely, supported DDPG and HER(future, last, final, random);
* 10.4 V0.6: update code framework, add rollouts and samplers packages;
* 10.6 add nstep sampler and nstep her sampler;
* 10.7 fix bug of nstep her sampler;
* 10.16 add priority experience replay and cut her;
* 10.31 V1.0: add SHER support;
