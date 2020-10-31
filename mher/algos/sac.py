import numpy as np
import tensorflow as tf
from mher.algos.actor_critic import SAC_ActorCritic
from mher.algos.algorithm import Algorithm
from mher.algos.util import flatten_grads, get_var, store_args
from mher.common import logger, tf_util
from mher.common.mpi_adam import MpiAdam
from mher.common import logger


class SAC(Algorithm):
    @store_args
    def __init__(self, buffer, input_dims, hidden, layers, polyak, Q_lr, pi_lr,
                 norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, subtract_goals, 
                 relative_goals, clip_pos_returns, clip_return, gamma, vloss_type='normal',
                 priority=False, sac_alpha=0.03, reuse=False, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).
        Args:
            sac_alpha: hyperparameter in SAC
        """
        super(SAC, self).__init__(**self.__dict__)

    def _name_variable(self, name, main=True):
        if main:
            return self.scope + '/main/' + name
        else:
            return self.scope + '/target/' + name
    
    def _create_network(self, reuse=False):
        logger.info("Creating a SAC agent with action space %d x %s..." % (self.dimu, self.max_u))
        self.sess = tf_util.get_session()
        self._create_normalizer(reuse)
        batch_tf = self._get_batch_tf()

        # networks
        self._create_target_main(SAC_ActorCritic, reuse, batch_tf)

        # loss functions
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = self._clip_target(batch_tf, clip_range, self.target.v_tf)
        q_backup_tf = tf.stop_gradient(target_tf)
        v_backup_tf = tf.stop_gradient(self.main.min_q_pi_tf - self.sac_alpha * self.main.logp_pi_tf)

        q1_loss_tf = 0.5 * tf.reduce_mean((q_backup_tf - self.main.q1_tf) ** 2)
        q2_loss_tf = 0.5 * tf.reduce_mean((q_backup_tf - self.main.q2_tf) ** 2)
        v_loss_tf = 0.5 * tf.reduce_mean((v_backup_tf - self.main.v_tf) ** 2)
        self.abs_tf_error_tf = tf.reduce_mean(tf.abs(q_backup_tf - self.main.q1_tf) + tf.abs(q_backup_tf -self.main.q2_tf))

        self.value_loss_tf = q1_loss_tf + q2_loss_tf + v_loss_tf
        self.pi_loss_tf = tf.reduce_mean(self.sac_alpha * self.main.logp_pi_tf - self.main.q1_pi_tf)
        
        # virables
        value_params = get_var(self._name_variable('q')) + get_var(self._name_variable('v'))
        pi_params = get_var(self._name_variable('pi'))
        # gradients
        V_grads_tf = tf.gradients(self.value_loss_tf, value_params)
        pi_grads_tf = tf.gradients(self.pi_loss_tf, pi_params)
        self.V_grad_tf = flatten_grads(grads=V_grads_tf, var_list=value_params)
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=pi_params)

        # optimizers
        self.V_adam = MpiAdam(value_params, scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(pi_params, scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = get_var(self._name_variable('pi')) + get_var(self._name_variable('q1')) + get_var(self._name_variable('q2')) + get_var(self._name_variable('v'))
        self.target_vars = get_var(self._name_variable('pi', main=False)) + get_var(self._name_variable('q1', main=False)) + get_var(self._name_variable('q2', main=False)) + get_var(self._name_variable('v', main=False))

        self.init_target_net_op = list(map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), \
                                        zip(self.target_vars, self.main_vars)))

        # initialize all variables
        self.global_vars = get_var(self.scope, key='global')
        tf.variables_initializer(self.global_vars).run()
        self._sync_optimizers()
        self._init_target_net()


    def _sync_optimizers(self):
        self.V_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        critic_loss, actor_loss, V_grad, pi_grad, abs_td_error = self.sess.run([
            self.value_loss_tf,
            self.pi_loss_tf,
            self.V_grad_tf,
            self.pi_grad_tf,
            self.abs_tf_error_tf
        ])
        return critic_loss, actor_loss, V_grad, pi_grad, abs_td_error

    def _update(self, V_grad, pi_grad):
        self.V_adam.update(V_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)
    
    # sac doesn't need noise
    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
        o, g = self._preprocess_og(o=o, g=g, ag=ag)
        if not noise_eps and not random_eps:
            u = self.simple_get_action(o, g, use_target_net, deterministic=True)
        else:
            u = self.simple_get_action(o, g, use_target_net, deterministic=False)

        if compute_Q:
            Q_pi = self.get_Q_fun(o, g)

        u = np.clip(u, -self.max_u, self.max_u)
        if u.shape[0] == 1:
            u = u[0] 

        if compute_Q:
            return [u, Q_pi]
        else:
            return u

    def simple_get_action(self, o, g, use_target_net=False, deterministic=False):
        o,g = self._preprocess_og(o=o,g=g)
        policy = self.target if use_target_net else self.main  # in n-step self.target performs better
        act_tf = policy.mu_tf if deterministic else policy.pi_tf
        action, logp_pi, min_q_pi, q1_pi, q2_pi,log_std  = self.sess.run( \
            [act_tf, policy.logp_pi_tf, policy.min_q_pi_tf, policy.q1_pi_tf, policy.q2_pi_tf, policy.log_std], \
            feed_dict={
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg)
        })
        return action
