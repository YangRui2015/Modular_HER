import numpy as np
import tensorflow as tf
from mher.algos.actor_critic import ActorCritic
from mher.algos.algorithm import Algorithm
from mher.algos.util import flatten_grads, get_var, store_args
from mher.common import logger, tf_util
from mher.common.mpi_adam import MpiAdam


class DDPG(Algorithm):
    @store_args
    def __init__(self, buffer, input_dims, hidden, layers, polyak, Q_lr, pi_lr, 
                norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, subtract_goals, 
                relative_goals, clip_pos_returns, clip_return, gamma, vloss_type='normal', 
                priority=False, reuse=False, **kwargs):
        """
        see algorithm
        """
        super(DDPG, self).__init__(**self.__dict__)
    
    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()
        # normalizer for input
        self._create_normalizer(reuse)
        batch_tf = self._get_batch_tf()

        # networks
        self._create_target_main(ActorCritic, reuse, batch_tf)

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = self._clip_target(batch_tf, clip_range, target_Q_pi_tf)

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


