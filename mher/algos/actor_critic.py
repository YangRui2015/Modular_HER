import numpy as np
import tensorflow as tf
from mher.algos.sac_utils import apply_squashing_func, mlp_gaussian_policy
from mher.algos.util import nn, store_args


class ActorCritic: 
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, sess):
        """The actor-critic network and related training code.
        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled accordingly
            o_stats (mher.algos.Normalizer): normalizer for observations
            g_stats (mher.algos.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u'] 

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor
        self._network(input_pi, o, g)

    
    def _network(self, input_pi, o, g):
        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(input_pi, [self.hidden] * self.layers + [self.dimu]))

        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True) 

    def get_Q(self, o, g, u):
        feed = {
            self.o_tf: o.reshape(-1, self.dimo),
            self.g_tf: g.reshape(-1, self.dimg),
            self.u_tf: u.reshape(-1, self.dimu)
        }
        return self.sess.run(self.Q_tf, feed_dict=feed)

    def get_Q_pi(self, o, g):
        feed = {
            self.o_tf: o.reshape(-1, self.dimo),
            self.g_tf:g.reshape(-1, self.dimg)
        }
        return self.sess.run(self.Q_pi_tf, feed_dict=feed)


class SAC_ActorCritic(ActorCritic):
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, sess):
        super(SAC_ActorCritic, self).__init__(**self.__dict__)


    def _network(self, input_pi, o, g):
        with tf.variable_scope('pi'):
            self.mu_tf, self.pi_tf, self.logp_pi_tf, self.log_std = mlp_gaussian_policy(input_pi, self.dimu,
                                                                          hidden_sizes=[self.hidden] * self.layers,
                                                                          activation=tf.nn.relu,
                                                                          output_activation=None)
            self.mu_tf, self.pi_tf, self.logp_pi_tf = apply_squashing_func(self.mu_tf, self.pi_tf, self.logp_pi_tf)

        with tf.variable_scope('q1'):
            self.q1_pi_tf = nn(tf.concat(axis=1, values=[o, g, self.pi_tf]),
                                layers_sizes=[self.hidden] * self.layers + [1])
            self.q1_tf = nn(tf.concat(axis=1, values=[o, g, self.u_tf]),
                             layers_sizes=[self.hidden] * self.layers + [1], reuse=True)
        with tf.variable_scope('q2'):
            self.q2_pi_tf = nn(tf.concat(axis=1, values=[o, g, self.pi_tf]),
                                layers_sizes=[self.hidden] * self.layers + [1])
            self.q2_tf = nn(tf.concat(axis=1, values=[o, g, self.u_tf]),
                             layers_sizes=[self.hidden] * self.layers + [1], reuse=True)
        with tf.variable_scope('min'):
            self.min_q_pi_tf = tf.minimum(self.q1_pi_tf, self.q2_pi_tf)
            self.min_q_tf = tf.minimum(self.q1_tf, self.q2_tf)
        with tf.variable_scope('v'):
            self.v_tf = nn(input_pi,layers_sizes=[self.hidden] * self.layers + [1])

    def get_Q(self, o, g, u):
        feed = {
            self.o_tf: o.reshape(-1, self.dimo),
            self.g_tf: g.reshape(-1, self.dimg),
            self.u_tf: u.reshape(-1, self.dimu)
        }
        return self.sess.run(self.min_q_tf, feed_dict=feed)

    def get_Q_pi(self, o, g):
        feed = {
            self.o_tf: o.reshape(-1, self.dimo),
            self.g_tf: g.reshape(-1, self.dimg)
        }
        return self.sess.run(self.min_q_pi_tf, feed_dict=feed)
    
    def get_V(self, o, g):
        feed = {
            self.o_tf: o.reshape(-1, self.dimo),
            self.g_tf: g.reshape(-1, self.dimg)
        }
        return self.sess.run(self.v_tf, feed_dict=feed)


    
            


    



