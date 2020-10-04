import mher.common.tf_util as U
import numpy as np
import tensorflow as tf
from mher.algos.normalizer import Normalizer, NormalizerNumpy
from mher.algos.util import store_args
from mher.common import logger
from mher.common.mpi_adam import MpiAdam


def nn(input, layers_sizes, reuse=None, flatten=False, use_layer_norm=False, name=""):
    """Creates a simple neural network
    """
    for i, size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        norm = tf.contrib.layers.layer_norm if i < len(layers_sizes) - 1 else None
        input = tf.layers.dense(inputs=input,
                                units=size,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),

                                reuse=reuse,
                                name=name + '_' + str(i))
        if use_layer_norm and norm:
            input = norm(input, reuse=reuse, scope=name + '_layer_norm_' + str(i))
        if activation:
            input = activation(input)
    if flatten:
        assert layers_sizes[-1] == 1
        input = tf.reshape(input, [-1])
    return input


def _vars(scope):
    res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    assert len(res) > 0
    return res


class ForwardDynamics:
    @store_args
    def __init__(self, dimo, dimu,o_stats, u_stats, clip_norm=5, norm_eps=1e-4, hidden=400, layers=4, learning_rate=1e-3):
        self.sess = U.get_session()
        with tf.variable_scope('forward_dynamics'):
            self.obs0 = tf.placeholder(tf.float32, shape=(None,self.dimo) , name='obs0')
            self.obs1 = tf.placeholder(tf.float32, shape=(None,self.dimo) , name='obs1')
            self.actions = tf.placeholder(tf.float32, shape=(None,self.dimu) , name='actions')

            self.dynamics_scope = tf.get_variable_scope().name
            obs0_norm = self.o_stats.normalize(self.obs0)
            obs1_norm = self.o_stats.normalize(self.obs1)
            actions_norm = self.u_stats.normalize(self.actions)
            input = tf.concat(values=[obs0_norm, actions_norm], axis=-1)
            self.next_state_diff_tf = nn(input, [hidden] * layers + [self.dimo])
            self.next_state_denorm = self.o_stats.denormalize(self.next_state_diff_tf + obs0_norm)

            # no normalize 
            # input = tf.concat(values=[self.obs0, self.actions], axis=-1)
            # self.next_state_diff_tf = nn(input,[hidden] * layers+ [self.dimo])
            # self.next_state_tf = self.next_state_diff_tf + self.obs0
            # self.next_state_denorm = self.next_state_tf

        # loss functions
        self.per_sample_loss_tf = tf.reduce_mean(tf.abs(self.next_state_diff_tf - obs1_norm + obs0_norm), axis=1)
        # self.per_sample_loss_tf = tf.reduce_mean(tf.abs(self.next_state_tf - self.obs1), axis=1)
        self.mean_loss_tf = tf.reduce_mean(self.per_sample_loss_tf)
        self.test_loss_tf = tf.reduce_mean(tf.abs(self.next_state_denorm - self.obs1))
        # self.test_loss_tf = tf.reduce_mean(tf.abs(self.next_state_tf - self.obs1))

        self.dynamics_grads = U.flatgrad(self.mean_loss_tf, _vars(self.dynamics_scope), clip_norm=clip_norm)

        # optimizers
        self.dynamics_adam = MpiAdam(_vars(self.dynamics_scope), scale_grad_by_procs=False)
        # initial
        tf.variables_initializer(_vars(self.dynamics_scope)).run()
        self.dynamics_adam.sync()
    
    def predict_next_state(self, obs0, actions):
        obs1 = self.sess.run(self.next_state_denorm, feed_dict={
            self.obs0: obs0,
            self.actions:actions
        })
        return obs1

    def _get_intrinsic_rewards(self, obs0, actions, obs1):
        intrinsic_rewards = self.sess.run(self.per_sample_loss_tf, feed_dict={
            self.obs0: obs0,
            self.actions: actions,
            self.obs1: obs1
        })
        return intrinsic_rewards
    
    def update(self, obs0, actions, obs1):
        dynamics_grads, dynamics_loss, dynamics_per_sample_loss, test_loss = self.sess.run(
                [self.dynamics_grads, self.mean_loss_tf, self.per_sample_loss_tf, self.test_loss_tf],
                feed_dict={
                    self.obs0: obs0,
                    self.actions: actions,
                    self.obs1: obs1
                })
        self.dynamics_adam.update(dynamics_grads, stepsize=self.learning_rate)
        return dynamics_loss, test_loss

    def get_intrinsic_rewards(self, obs0, actions, obs1, update=True):
        if update:
            return self.update(obs0, actions, obs1)
        else:
            return self._get_intrinsic_rewards(obs0, actions, obs1)

# numpy forward dynamics
class ForwardDynamicsNumpy:
    @store_args
    def __init__(self, dimo, dimu, clip_norm=5, norm_eps=1e-4, hidden=256, layers=8, learning_rate=1e-3):
        self.obs_normalizer = NormalizerNumpy(size=dimo, eps=norm_eps)
        self.action_normalizer = NormalizerNumpy(size=dimu, eps=norm_eps)
        self.sess = U.get_session()

        with tf.variable_scope('forward_dynamics_numpy'):
            self.obs0_norm = tf.placeholder(tf.float32, shape=(None,self.dimo) , name='obs0')
            self.obs1_norm = tf.placeholder(tf.float32, shape=(None,self.dimo) , name='obs1')
            self.actions_norm = tf.placeholder(tf.float32, shape=(None,self.dimu) , name='actions')

            self.dynamics_scope = tf.get_variable_scope().name
            input = tf.concat(values=[self.obs0_norm, self.actions_norm], axis=-1)
            self.next_state_diff_tf = nn(input, [hidden] * layers + [self.dimo])
            self.next_state_norm_tf = self.next_state_diff_tf + self.obs0_norm

        # loss functions
        self.per_sample_loss_tf = tf.reduce_mean(tf.abs(self.next_state_diff_tf - self.obs1_norm + self.obs0_norm), axis=1)
        self.mean_loss_tf = tf.reduce_mean(self.per_sample_loss_tf)
        self.dynamics_grads = U.flatgrad(self.mean_loss_tf, _vars(self.dynamics_scope), clip_norm=clip_norm)

        # optimizers
        self.dynamics_adam = MpiAdam(_vars(self.dynamics_scope), scale_grad_by_procs=False)
        # initial
        tf.variables_initializer(_vars(self.dynamics_scope)).run()
        self.dynamics_adam.sync()
    
    def predict_next_state(self, obs0, actions):
        obs0_norm = self.obs_normalizer.normalize(obs0)
        action_norm = self.action_normalizer.normalize(actions)
        obs1 = self.sess.run(self.next_state_norm_tf, feed_dict={
            self.obs0_norm: obs0_norm,
            self.actions_norm:action_norm
        })
        obs1_norm = self.obs_normalizer.denormalize(obs1)
        return obs1_norm
    
    def clip_gauss_noise(self, size):
        clip_range = 0.002
        std = 0.001
        return np.clip(np.random.normal(0, std, size), -clip_range, clip_range)
        # return 0
    
    def update(self, obs0, actions, obs1, times=1):
        self.obs_normalizer.update(obs0)
        self.obs_normalizer.update(obs1)
        self.action_normalizer.update(actions)

        for _ in range(times):
            obs0_norm = self.obs_normalizer.normalize(obs0) + self.clip_gauss_noise(size=self.dimo)
            action_norm = self.action_normalizer.normalize(actions) + self.clip_gauss_noise(size=self.dimu)
            obs1_norm = self.obs_normalizer.normalize(obs1) #+ self.clip_gauss_noise(size=self.dimo)
            
            dynamics_grads, dynamics_loss, dynamics_per_sample_loss = self.sess.run(
                    [self.dynamics_grads, self.mean_loss_tf, self.per_sample_loss_tf],
                    feed_dict={
                        self.obs0_norm: obs0_norm,
                        self.actions_norm: action_norm,
                        self.obs1_norm: obs1_norm
                    })
            self.dynamics_adam.update(dynamics_grads, stepsize=self.learning_rate)
        return dynamics_loss



class RandomNetworkDistillation:
    def __init__(self, obs0, action, obs1, clip_norm, hidden, layers):
        logger.info("Using Random Network Distillation")
        rep_size = hidden

        with tf.variable_scope('random_network_distillation'):
            self.rnd_scope = tf.get_variable_scope().name
            # Random Target Network

            with tf.variable_scope('target_network'):
                xr = nn(obs1, [hidden] * layers + [rep_size])

            with tf.variable_scope('predictor_network'):
                self.predictor_scope = tf.get_variable_scope().name
                xr_hat = nn(obs1, [hidden] * layers + [rep_size])

        total_parameters = 0
        for variable in _vars(self.predictor_scope):
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        logger.info("params in target rnd network: {}".format(total_parameters))

        self.feat_var = tf.reduce_mean(tf.nn.moments(xr, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(xr))
        # loss functions
        self.per_sample_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(xr) - xr_hat), axis=-1, keepdims=True)
        self.mean_loss_tf = tf.reduce_mean(self.per_sample_loss_tf)

        self.dynamics_grads = U.flatgrad(self.mean_loss_tf, _vars(self.predictor_scope), clip_norm=clip_norm)

        # optimizers
        self.dynamics_adam = MpiAdam(_vars(self.predictor_scope), scale_grad_by_procs=False)
