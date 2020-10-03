import tensorflow as tf
from mher.algos.util import store_args, flatten_grads, nn
from mher.common.mpi_adam import MpiAdam
import mher.common.tf_util as U
import numpy as np 


class Max_Return_Net:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers):
        self.sess = U.get_session()
        self.lr = 1e-3
        self.scope = 'max_return'
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        # tf.get_variable_scope()._name
        with tf.variable_scope('max_return'):
            # for max return prediction
            self.scope = tf.get_variable_scope().name
            self.future_return = tf.placeholder(tf.float32, shape=(None,1) , name='future_return')
            self.zero_return = tf.placeholder(tf.float32, shape=(None,1) , name='zero_return')

            inputs = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            with tf.variable_scope('Q'):
                self.max_return_tf = nn(inputs, [self.hidden] * self.layers + [1], reuse=False)
        
        self.future_return_loss = tf.reduce_mean(tf.square(self.max_return_tf - self.future_return))
        self.max_return_loss = tf.reduce_mean(tf.square(tf.maximum(self.zero_return, self.future_return - self.max_return_tf)))
        max_return_grads_tf = tf.gradients(self.max_return_loss, self._vars(self.scope))
        future_return_grads_tf = tf.gradients(self.future_return_loss, self._vars(self.scope))

        # self.fr_grads_vars_tf = zip(future_return_grads_tf, self._vars('max_return'))
        # self.mr_grads_vars_tf = zip(max_return_grads_tf, self._vars('max_return'))
        self.max_return_grad_tf = flatten_grads(grads=max_return_grads_tf, var_list=self._vars(self.scope))
        self.future_return_grad_tf = flatten_grads(grads=future_return_grads_tf, var_list=self._vars(self.scope))

        # optimizers
        self.max_return_adam = MpiAdam(self._vars(self.scope), scale_grad_by_procs=False)
        tf.variables_initializer(self._vars(self.scope)).run()
        
    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope) # self.scope + '/' 
        assert len(res) > 0
        return res
        
    def update(self, inputs, init=False):
        if init:
            return_grad, return_loss, max_tf = self.sess.run(
                                        [self.future_return_grad_tf, self.future_return_loss,
                                        self.max_return_tf],
                                        feed_dict = {
                                            self.future_return:inputs['return'],
                                            self.o_tf:inputs['o'],
                                            self.g_tf:inputs['g'],
                                            self.u_tf:inputs['u']
                                            }
                                        )
        else:
            zero_return = np.zeros_like(inputs['return'])
            return_grad, return_loss, max_tf = self.sess.run(
                                        [self.max_return_grad_tf, self.max_return_loss,self.max_return_tf],
                                        feed_dict = {
                                            self.zero_return:zero_return,
                                            self.future_return:inputs['return'],
                                            self.o_tf:inputs['o'],
                                            self.g_tf:inputs['g'],
                                            self.u_tf:inputs['u']
                                            }
                                        )
        self.max_return_adam.update(return_grad, self.lr)
        return return_loss, max_tf