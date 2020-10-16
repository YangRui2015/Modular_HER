import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from mher.algos.actor_critic import ActorCritic
from mher.algos.util import (convert_episode_to_batch_major, dims_to_shapes,
                             flatten_grads, get_var, import_function,
                             store_args, transitions_in_episode_batch)
from mher.common import logger, tf_util
from mher.common.mpi_adam import MpiAdam
from mher.common.normalizer import Normalizer
from tensorflow.contrib.staging import StagingArea


class DDPG(object):
    @store_args
    def __init__(self, buffer, input_dims, hidden, layers, polyak, Q_lr, pi_lr, 
                norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, subtract_goals, 
                relative_goals, clip_pos_returns, clip_return, gamma, vloss_type='normal', 
                priority=False, reuse=False, **kwargs):
        """
        buffer (object): buffer to save transitions
        input_dims (dict of ints): dimensions for the observation (o), the goal (g), 
            and the actions (u)
        hidden (int): number of units in the hidden layers
        layers (int): number of hidden layers
        polyak (float): coefficient for Polyak-averaging of the target network
        Q_lr (float): learning rate for the Q (critic) network
        pi_lr (float): learning rate for the pi (actor) network
        norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
        norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
        max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
        action_l2 (float): coefficient for L2 penalty on the actions
        clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
        scope (str): the scope used for the TensorFlow graph
        subtract_goals (function): function that subtracts goals from each other
        relative_goals (boolean): whether or not relative goals should be fed into the network
        clip_pos_returns (boolean): whether or not positive returns should be clipped
        clip_return (float): clip returns to be in [-clip_return, clip_return]
        gamma (float): gamma used for Q learning updates
        vloss_type (str): value loss type, 'normal', 'tf_gamma', 'target'
        priority(boolean): use priority or not
        reuse (boolean): whether or not the networks should be reused
        """
        if self.clip_return is None:
            self.clip_return = np.inf
        self.dimo, self.dimg, self.dimu = self.input_dims['o'], self.input_dims['g'], self.input_dims['u']
        self.stage_shapes = self.get_stage_shapes()

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                                            shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
            self._create_network(reuse=reuse)
    
    def get_stage_shapes(self):
        # Prepare staging area for feeding data to the model. save data for HER
        input_shapes = dims_to_shapes(self.input_dims)
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        if self.vloss_type == 'tf_gamma':
            stage_shapes['gamma'] = (None,)
        if self.priority:
            stage_shapes['w'] = (None,)
        return stage_shapes
    
    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()
        # normalizer for input
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('u_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.u_stats = Normalizer(self.dimu, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])
        if self.priority:
            batch_tf['w'] = tf.reshape(batch_tf['w'], [-1,1])

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = ActorCritic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = ActorCritic(target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(get_var(self.scope + "/main")) == len(get_var(self.scope + '/target'))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        logger.log('value loss type: {}'.format(self.vloss_type))
        if self.vloss_type == 'tf_gamma':
            target_tf = tf.clip_by_value(batch_tf['r'] + batch_tf['gamma'] * target_Q_pi_tf, * clip_range)
        elif self.vloss_type == 'target':
            target_tf = tf.clip_by_value(batch_tf['r'], * clip_range)
        else:
            target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, * clip_range)
        
        self.abs_td_error_tf = tf.abs(tf.stop_gradient(target_tf) - self.main.Q_tf)
        self.Q_loss = tf.square(self.abs_td_error_tf)
        if self.priority:
            self.Q_loss_tf = tf.reduce_mean(batch_tf['w'] * self.Q_loss)
        else:
            self.Q_loss_tf = tf.reduce_mean(self.Q_loss)
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))

        # varibles
        self.main_Q_var = get_var(self.scope + '/main/Q')
        self.main_pi_var = get_var(self.scope + '/main/pi')
        self.target_Q_var = get_var(self.scope + '/target/Q')
        self.target_pi_var = get_var(self.scope + '/target/pi')

        Q_grads_tf = tf.gradients(self.Q_loss_tf, self.main_Q_var)
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self.main_pi_var)
        assert len(self.main_Q_var) == len(Q_grads_tf)
        assert len(self.main_pi_var) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self.main_Q_var)
        self.pi_grads_vars_tf = zip(pi_grads_tf, self.main_pi_var)
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self.main_Q_var)
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self.main_pi_var)

        # optimizers
        self.Q_adam = MpiAdam(self.main_Q_var, scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self.main_pi_var, scale_grad_by_procs=False)
        self.main_vars = self.main_Q_var + self.main_pi_var
        self.target_vars = self.target_Q_var+ self.target_pi_var
        self.init_target_net_op = list(map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), 
                                        zip(self.target_vars, self.main_vars)))

        # initialize all variables
        self.global_vars = get_var(self.scope, key='global')
        tf.variables_initializer(self.global_vars).run()
        self._sync_optimizers()
        self._init_target_net()

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, g, ag=None):
        if self.relative_goals and ag:
            g_shape = g.shape
            g, ag = g.reshape(-1, self.dimg), ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def step(self, obs):  # act without noise
        actions = self.get_actions(obs['observation'], obs['achieved_goal'], obs['desired_goal'])
        return actions, None, None, None

    def simple_get_action(self, o, g, use_target_net=False):
        o,g = self._preprocess_og(o=o,g=g)
        policy = self.target if use_target_net else self.main  # in n-step self.target performs better
        action = self.sess.run(policy.pi_tf, feed_dict={
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg)
        })
        return action

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
        o, g = self._preprocess_og(o=o, g=g, ag=ag)
        policy = self.target if use_target_net else self.main
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
        } 
        ret = self.sess.run(vals, feed_dict=feed)
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u = np.clip(u + noise, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0] 
        ret[0] = u.copy()
        if len(ret) == 1:
            return ret[0]
        else:
            return ret
    
    def get_Q_fun(self, o, g, u=None, Q_pi=True):
        o, g = self._preprocess_og(o, g)
        policy = self.target
        if Q_pi or (u is None): 
            return policy.get_Q_pi(o,g)
        else:
            return policy.get_Q(o, g, u)

    def store_episode(self, episode_batch, update_stats=True): 
        """episode_batch: array of batch_size x (T or T+1) x dim_key, 'o' is of size T+1, others are of size T"""
        self.buffer.store_episode(episode_batch)
        if update_stats:   # episode doesn't has key o_2
            os, gs, ags = episode_batch['o'].copy(), episode_batch['g'].copy(), episode_batch['ag'].copy()
            os, gs = self._preprocess_og(o=os, g=gs, ag=ags)
            # update normalizer online 
            self.o_stats.update_all(os)
            self.g_stats.update_all(gs)

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self): # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad, abs_td_error = self.sess.run([  
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf,  
            self.abs_td_error_tf
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad, abs_td_error

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)
        
    def stage_batch(self, batch=None):
        if batch is None:
            if self.priority:
                transitions, idxes = self.buffer.sample()
            else:
                transitions = self.buffer.sample()
            o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
            ag, ag_2 = transitions['ag'], transitions['ag_2']
            transitions['o'], transitions['g'] = self._preprocess_og(o=o, g=g, ag=ag)
            transitions['o_2'], transitions['g_2'] = self._preprocess_og(o=o_2, g=g, ag=ag_2)
        
            batch = [transitions[key] for key in self.stage_shapes.keys()]
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))
        if self.priority:
            return idxes

    def train(self, stage=True):
        if stage:
            idxes = self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad, abs_td_error = self._grads()
        self._update(Q_grad, pi_grad)
        if self.priority:
            self.buffer.update_priorities(idxes, abs_td_error)
        return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def logs_stats(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]
        logs += [('stats_u/mean', np.mean(self.sess.run([self.u_stats.mean])))]
        logs += [('stats_u/std', np.mean(self.sess.run([self.u_stats.std])))]
        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def save(self, save_path):
        tf_util.save_variables(save_path)
