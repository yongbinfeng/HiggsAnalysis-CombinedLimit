from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.opt import ExternalOptimizerInterface

from scipy.optimize import SR1


__all__ = ['ScipyHessOptimizerInterface']

class SR1Mod(SR1):
  def __init__(self, min_denominator=1e-8, init_scale='auto', initialhess=None):
        self.initialhess = initialhess
        super(SR1Mod, self).__init__(min_denominator, init_scale)


  def initialize(self, n, approx_type):
      """Initialize internal matrix.
      Allocate internal memory for storing and updating
      the Hessian or its inverse.
      Parameters
      ----------
      n : int
          Problem dimension.
      approx_type : {'hess', 'inv_hess'}
          Selects either the Hessian or the inverse Hessian.
          When set to 'hess' the Hessian will be stored and updated.
          When set to 'inv_hess' its inverse will be used instead.
      """
      
      #approx_type = 'inv_hess'
      
      print("SR1Mod.initialize")
      self.first_iteration = True
      self.n = n
      self.approx_type = approx_type
      if approx_type not in ('hess', 'inv_hess'):
          raise ValueError("`approx_type` must be 'hess' or 'inv_hess'.")
      # Create matrix
      if self.approx_type == 'hess':
          self.B = np.eye(n, dtype=np.float64)
      else:
          self.H = np.eye(n, dtype=np.float64)
      
      if type(self.initialhess) is np.ndarray:
        print("hessian initialize")
        self.init_scale = 1.
        if self.approx_type == 'hess':
          print(self.initialhess.dtype)
          self.B = self.initialhess
          print("B initialize")
          print(self.B)
        else:
          self.H = np.linalg.inv(self.initialhess)
          print("H initialize")
          print(self.H)
          


class ScipyHessOptimizerInterface(ExternalOptimizerInterface):
  """Wrapper allowing `scipy.optimize.minimize` to operate a `tf.Session`.
  Example:
  ```python
  vector = tf.Variable([7., 7.], 'vector')
  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))
  optimizer = ScipyOptimizerInterface(loss, options={'maxiter': 100})
  with tf.Session() as session:
    optimizer.minimize(session)
  # The value of vector should now be [0., 0.].
  ```
  Example with simple bound constraints:
  ```python
  vector = tf.Variable([7., 7.], 'vector')
  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))
  optimizer = ScipyOptimizerInterface(
      loss, var_to_bounds={vector: ([1, 2], np.infty)})
  with tf.Session() as session:
    optimizer.minimize(session)
  # The value of vector should now be [1., 2.].
  ```
  Example with more complicated constraints:
  ```python
  vector = tf.Variable([7., 7.], 'vector')
  # Make vector norm as small as possible.
  loss = tf.reduce_sum(tf.square(vector))
  # Ensure the vector's y component is = 1.
  equalities = [vector[1] - 1.]
  # Ensure the vector's x component is >= 1.
  inequalities = [vector[0] - 1.]
  # Our default SciPy optimization algorithm, L-BFGS-B, does not support
  # general constraints. Thus we use SLSQP instead.
  optimizer = ScipyOptimizerInterface(
      loss, equalities=equalities, inequalities=inequalities, method='SLSQP')
  with tf.Session() as session:
    optimizer.minimize(session)
  # The value of vector should now be [1., 1.].
  ```
  """

  _DEFAULT_METHOD = 'L-BFGS-B'

  def minimize(self,
               session=None,
               feed_dict=None,
               fetches=None,
               step_callback=None,
               loss_callback=None,
               **run_kwargs):
    """Minimize a scalar `Tensor`.
    Variables subject to optimization are updated in-place at the end of
    optimization.
    Note that this method does *not* just return a minimization `Op`, unlike
    `Optimizer.minimize()`; instead it actually performs minimization by
    executing commands to control a `Session`.
    Args:
      session: A `Session` instance.
      feed_dict: A feed dict to be passed to calls to `session.run`.
      fetches: A list of `Tensor`s to fetch and supply to `loss_callback`
        as positional arguments.
      step_callback: A function to be called at each optimization step;
        arguments are the current values of all optimization variables
        flattened into a single vector.
      loss_callback: A function to be called every time the loss and gradients
        are computed, with evaluated fetches supplied as positional arguments.
      **run_kwargs: kwargs to pass to `session.run`.
    """
    session = session or ops.get_default_session()
    feed_dict = feed_dict or {}
    fetches = fetches or []

    loss_callback = loss_callback or (lambda *fetches: None)
    step_callback = step_callback or (lambda xk: None)

    # Construct loss function and associated gradient.
    loss_grad_func = self._make_eval_func([self._loss,
                                           self._packed_loss_grad], session,
                                          feed_dict, fetches, loss_callback)

    hess = tf.hessians(self._loss,self._vars)[0]
    hess_func = self._make_eval_func([hess], session,
                                          feed_dict, [], lambda *fetches: None)

    # Construct equality constraint functions and associated gradients.
    equality_funcs = self._make_eval_funcs(self._equalities, session, feed_dict,
                                           fetches)
    equality_grad_funcs = self._make_eval_funcs(self._packed_equality_grads,
                                                session, feed_dict, fetches)

    # Construct inequality constraint functions and associated gradients.
    inequality_funcs = self._make_eval_funcs(self._inequalities, session,
                                             feed_dict, fetches)
    inequality_grad_funcs = self._make_eval_funcs(self._packed_inequality_grads,
                                                  session, feed_dict, fetches)

    # Get initial value from TF session.
    initial_packed_var_val = session.run(self._packed_var)

    # Perform minimization.
    packed_var_val = self._minimize(
        initial_val=initial_packed_var_val,
        loss_grad_func=loss_grad_func,
        hess_func=hess_func,
        equality_funcs=equality_funcs,
        equality_grad_funcs=equality_grad_funcs,
        inequality_funcs=inequality_funcs,
        inequality_grad_funcs=inequality_grad_funcs,
        packed_bounds=self._packed_bounds,
        step_callback=step_callback,
        optimizer_kwargs=self.optimizer_kwargs)
    var_vals = [
        packed_var_val[packing_slice] for packing_slice in self._packing_slices
    ]

    # Set optimization variables to their new values.
    session.run(
        self._var_updates,
        feed_dict=dict(zip(self._update_placeholders, var_vals)),
        **run_kwargs)


  def _minimize(self, initial_val, loss_grad_func, hess_func, equality_funcs,
                equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
                packed_bounds, step_callback, optimizer_kwargs):

    def loss_grad_func_wrapper(x):
      # SciPy's L-BFGS-B Fortran implementation requires gradients as doubles.
      loss, gradient = loss_grad_func(x)
      return loss, gradient.astype('float64')
    
    def hess_func_wrapper(x):
      return hess_func(x)[0]
    
    def grad_func_wrapper(x):
      return loss_grad_func_wrapper(x)[1]

    optimizer_kwargs = dict(optimizer_kwargs.items())
    method = optimizer_kwargs.pop('method', self._DEFAULT_METHOD)

    constraints = []
    for func, grad_func in zip(equality_funcs, equality_grad_funcs):
      constraints.append({'type': 'eq', 'fun': func, 'jac': grad_func})
    for func, grad_func in zip(inequality_funcs, inequality_grad_funcs):
      constraints.append({'type': 'ineq', 'fun': func, 'jac': grad_func})

    import scipy.optimize  # pylint: disable=g-import-not-at-top

    minimize_args = [loss_grad_func_wrapper, initial_val]
    minimize_kwargs = {
        'jac': True,
        'hess' : SR1Mod(initialhess=hess_func_wrapper(initial_val)),
        #'hess' : SR1Mod(),
        #'hess' : SR1Mod(),
        #'hess' : scipy.optimize.SR1(),
        #'hess' : hess_func_wrapper,
        #'callback': step_callback,
        'callback': None,
        'method': method,
        'constraints': constraints,
        'bounds': packed_bounds,
    }

    for kwarg in minimize_kwargs:
      if kwarg in optimizer_kwargs:
        if kwarg == 'bounds':
          # Special handling for 'bounds' kwarg since ability to specify bounds
          # was added after this module was already publicly released.
          raise ValueError(
              'Bounds must be set using the var_to_bounds argument')
        raise ValueError(
            'Optimizer keyword arg \'{}\' is set '
            'automatically and cannot be injected manually'.format(kwarg))

    minimize_kwargs.update(optimizer_kwargs)

    
    result = scipy.optimize.minimize(*minimize_args, **minimize_kwargs)

    message_lines = [
        'Optimization terminated with:',
        '  Message: %s',
        '  Objective function value: %f',
    ]
    message_args = [result.message, result.fun]
    if hasattr(result, 'nit'):
      # Some optimization methods might not provide information such as nit and
      # nfev in the return. Logs only available information.
      message_lines.append('  Number of iterations: %d')
      message_args.append(result.nit)
    if hasattr(result, 'nfev'):
      message_lines.append('  Number of functions evaluations: %d')
      message_args.append(result.nfev)
    logging.info('\n'.join(message_lines), *message_args)

    return result['x']
