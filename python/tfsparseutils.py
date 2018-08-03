import tensorflow as tf

from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.framework import ops


class SimpleSparseTensor:
  def __init__(self, indices, values, dense_shape):
    self.indices = indices
    self.values = values
    self.dense_shape = dense_shape

def simple_sparse_tensor_dense_matmul(sp_a,
                               b,
                               adjoint_a=False,
                               adjoint_b=False,
                               name=None):
  
  #sp_a = _convert_to_sparse_tensor(sp_a)
  with ops.name_scope(name, "SparseTensorDenseMatMul",
                      [sp_a.indices, sp_a.values, b]) as name:
    b = ops.convert_to_tensor(b, name="b")
    #to be compatible with different tensorflow versions where the name of the op changed
    try:
      dense_mat_mul = gen_sparse_ops.sparse_tensor_dense_mat_mul
    except:
      dense_mat_mul = gen_sparse_ops._sparse_tensor_dense_mat_mul
      
    return dense_mat_mul(
        a_indices=sp_a.indices,
        a_values=sp_a.values,
        a_shape=sp_a.dense_shape,
        b=b,
        adjoint_a=adjoint_a,
        adjoint_b=adjoint_b)

def flatten_indices(indices,shape):
  if indices.dtype != tf.int32:
    indices = tf.cast(indices,tf.int32)
  if len(shape) == 1:
    indicesflat = tf.squeeze(indices,-1)
    return (indicesflat, int(shape[0]))
  flat_shape = 1
  for s in shape:
    flat_shape *= int(s)
  flat_shape = (flat_shape,)
  idxmodifier = tf.cumprod(shape, exclusive=True, reverse=True)
  idxmodifier = tf.expand_dims(idxmodifier,-1)
  indicesflat = tf.matmul(indices,idxmodifier)
  indicesflat = tf.squeeze(indicesflat,-1)
  return (indicesflat, flat_shape[0])

def sparse_reduce_sum_0(in_sparse, ndims=1, reduced_shape=None):
  reduced_shape = in_sparse.dense_shape[:-ndims]
  print(reduced_shape)
  indicespartial = in_sparse.indices[:,ndims:]
  indicesflat,flat_size = flatten_indices(indicespartial, reduced_shape)
  
  segment_ids = indicesflat
  
  reduced_values = tf.unsorted_segment_sum(in_sparse.values, segment_ids,flat_size)
  reduced = tf.reshape(reduced_values, reduced_shape)
  return reduced

def simple_sparse_slice(in_sparse, start, end):
  validindices = tf.logical_and(tf.greater_equal(in_sparse.indices,start), tf.less(in_sparse.indices, end))
  validindices = tf.reduce_all(validindices,axis=-1)
  validindices = tf.where(validindices)
  validindices = tf.squeeze(validindices,-1)
  
  out_indices = tf.gather(in_sparse.indices,validindices)
  out_indices = out_indices - start
  out_values = tf.gather(in_sparse.values,validindices)
  out_shape = []
  for s,e in zip(start,end):
    out_shape.append(e-s)
  
  return SimpleSparseTensor(out_indices,out_values,out_shape)

def simple_sparse_to_dense(in_sparse):
  return tf.scatter_nd(in_sparse.indices, in_sparse.values, in_sparse.dense_shape)
