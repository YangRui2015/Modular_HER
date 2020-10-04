import json
import os
import time

import click
import numpy as np
from mpi4py import MPI

import mher.config as config
from mher.common import logger
from mher.common.mpi_moments import mpi_moments
from mher.rollouts.rollout import RolloutWorker


def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]


def train(*, policy, rollout_worker, evaluator, n_epochs, n_test_rollouts, n_cycles, 
            n_batches, policy_save_interval, save_path, random_init, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()
    if save_path:
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')

    # random_init buffer and o/g/u stat 
    if random_init:
        logger.info('Random initializing ...')
        rollout_worker.clear_history()
        for epi in range(int(random_init) // rollout_worker.rollout_batch_size): 
            episode = rollout_worker.generate_rollouts(random_ac=True)
            policy.store_episode(episode)
        if policy.use_dynamic_nstep and policy.n_step > 1:
            policy.update_dynamic_model(init=True)

    best_success_rate = -1
    logger.info('Start training...')
    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    for epoch in range(n_epochs):
        time_start = time.time()
        # train
        rollout_worker.clear_history()
        for i in range(n_cycles):
            policy.dynamic_batch = False
            episode = rollout_worker.generate_rollouts()
            policy.store_episode(episode)
            for j in range(n_batches):   
                policy.train()
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        time_end = time.time()
        logger.record_tabular('epoch', epoch)
        logger.record_tabular('epoch time(min)', (time_end - time_start)/60)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs_stats():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate > best_success_rate and save_path:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            policy.save(best_policy_path)
            policy.save(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            policy.save(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]
    
    if rank == 0 and save_path:
        policy_path = periodic_policy_path.format(epoch)
        logger.info('Saving final policy to {} ...'.format(policy_path))
        policy.save(policy_path)

    return policy

