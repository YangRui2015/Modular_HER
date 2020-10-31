from collections import OrderedDict

import numpy as np
import tensorflow as tf
from mher.algos.util import dims_to_shapes, store_args, get_var
from mher.common import logger, tf_util
from mher.common.normalizer import Normalizer
from tensorflow.contrib.staging import StagingArea


class Algorithm(object):
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
        self.init_target_net_op = None
        self.update_target_net_op = None

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                                            shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)
            self._create_network(reuse=reuse)
        
        logger.log('value loss type: {}'.format(self.vloss_type))

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

    def _create_normalizer(self, reuse):
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
    
    def _get_batch_tf(self):
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i]) for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])
        if self.priority:
            batch_tf['w'] = tf.reshape(batch_tf['w'], [-1,1])
        return batch_tf
    
    def _create_target_main(self, AC_class, reuse, batch_tf):
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = AC_class(batch_tf, self.dimo, self.dimg, self.dimu, self.max_u, self.o_stats, self.g_stats, self.hidden, self.layers, self.sess)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = AC_class(target_batch_tf, self.dimo, self.dimg, self.dimu, self.max_u, self.o_stats, self.g_stats, self.hidden, self.layers, self.sess)
            vs.reuse_variables()
        assert len(get_var(self.scope + "/main")) == len(get_var(self.scope + '/target'))

    def _clip_target(self, batch_tf, clip_range, target_V_tf):
        if self.vloss_type == 'tf_gamma':
            target_tf = tf.clip_by_value(batch_tf['r'] + batch_tf['gamma'] * target_V_tf, * clip_range)
        elif self.vloss_type == 'target':
            target_tf = tf.clip_by_value(batch_tf['r'], * clip_range)
        else:
            target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_V_tf, * clip_range)
        return target_tf
        
    def _create_network(self, reuse=False):
        raise NotImplementedError

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
        u = self.simple_get_action(o, g, use_target_net)
        if compute_Q:
            Q_pi = self.get_Q_fun(o, g)

        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u = np.clip(u + noise, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0] 

        if compute_Q:
            return [u, Q_pi]
        else:
            return u
    
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
        raise NotImplementedError

    def _grads(self): # Avoid feed_dict here for performance!
        raise NotImplementedError

    def _update(self, Q_grad, pi_grad):
        raise NotImplementedError
        
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
        critic_loss, actor_loss, Value_grad, pi_grad, abs_td_error = self._grads()
        self._update(Value_grad, pi_grad)
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
