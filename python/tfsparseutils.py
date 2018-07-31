import tensorflow as tf


def flatten_indices(indices,shape):
  if indices.dtype != tf.int32:
    indices = tf.cast(indices,tf.int32)
  flat_shape = 1
  for s in shape:
    flat_shape *= int(s)
  flat_shape = (flat_shape,)
  idxmodifier = tf.cumprod(shape, exclusive=True, reverse=True)
  idxmodifier = tf.expand_dims(idxmodifier,-1)
  indicesflat = tf.matmul(indices,idxmodifier)
  indicesflat = tf.squeeze(indicesflat,-1)
  return (indicesflat, flat_shape[0])
  
def makeCache(x):
  it = tf.data.Dataset.from_tensors(x).cache().repeat().make_initializable_iterator()
  tf.add_to_collection('cache_initializers', it.initializer)
  return it.get_next()

def sparse_reduce_sum_sparse_0(in_sparse, ndims=1, doCache=False):    
  reduced_shape = in_sparse.get_shape()[ndims:]
  indicespartial = in_sparse.indices[:,ndims:]
  indicesflat,_ = flatten_indices(indicespartial, reduced_shape)
  
  reduced_indices_flat, segment_ids = tf.unique(indicesflat)
  reduced_size = tf.shape(reduced_indices_flat)[0]
  reduced_indices_flat = tf.cast(reduced_indices_flat, tf.int64)
  reduced_indices = tf.transpose(tf.unravel_index(reduced_indices_flat,reduced_shape))
  reduced_indices = tf.Print(reduced_indices,[],message="computing indices")
  
  if doCache:
    segment_ids, reduced_size, reduced_indices = makeCache((segment_ids, reduced_size, reduced_indices))
    
  reduced_values = tf.unsorted_segment_sum(in_sparse.values, segment_ids,reduced_size)
  reduced_sparse = tf.SparseTensor(reduced_indices, reduced_values, reduced_shape)
  reduced_sparse = tf.sparse_reorder(reduced_sparse)
  return reduced_sparse

def sparse_reduce_sum_sparse_m(in_sparse, ndims=1, doCache=False):    
  reduced_shape = in_sparse.get_shape()[:-ndims]
  indicespartial = in_sparse.indices[:,:-ndims]
  indicesflat,_ = flatten_indices(indicespartial, reduced_shape)
  
  reduced_indices_flat, segment_ids = tf.unique(indicesflat)
  reduced_indices_flat = tf.cast(reduced_indices_flat, tf.int64)
  reduced_indices = tf.transpose(tf.unravel_index(reduced_indices_flat,reduced_shape))
  reduced_indices = tf.Print(reduced_indices,[],message="computing indices")
  
  if doCache:
    segment_ids, reduced_indices = makeCache((segment_ids, reduced_indices))
    
  print(in_sparse.values)
  print(segment_ids)
  #print(reduced_size)
  reduced_values = tf.segment_sum(in_sparse.values, segment_ids)
  reduced_sparse = tf.SparseTensor(reduced_indices, reduced_values, reduced_shape)
  return reduced_sparse

def sparse_reduce_sum_0(in_sparse, ndims=1, reduced_shape=None, doCache=False):
  if reduced_shape is None:
    reduced_shape = in_sparse.get_shape()[:-ndims]  
  indicespartial = in_sparse.indices[:,ndims:]
  indicesflat,flat_size = flatten_indices(indicespartial, reduced_shape)
  
  segment_ids = indicesflat
  
  if doCache:
    segment_ids,flat_size = makeCache((segment_ids,flat_size))
  
  reduced_values = tf.unsorted_segment_sum(in_sparse.values, segment_ids,flat_size)
  reduced = tf.reshape(reduced_values, reduced_shape)
  return reduced

def sparse_reduce_sum_m(in_sparse, ndims = 1, reduced_shape=None, doCache=False):
  if reduced_shape is None:
    reduced_shape = in_sparse.get_shape()[:-ndims]  
  indicespartial = in_sparse.indices[:,:-ndims]
  indicesflat,flat_size = flatten_indices(indicespartial, reduced_shape)
  
  segment_ids = indicesflat
  padsize = flat_size - (tf.reduce_max(segment_ids) + 1)
  
  if doCache:
    segment_ids, padsize = makeCache((segment_ids,padsize))
  
  reduced_values = tf.segment_sum(in_sparse.values, segment_ids)
  reduced_values = tf.pad(reduced_values,[[0,padsize]])
  reduced = tf.reshape(reduced_values, reduced_shape)
  return reduced


def sparseexpmul(loga,b,doCache=True):
  fullindices = tf.concat((b.indices,loga.indices),axis=0)  
  b_size = int(b.indices.shape[0])
  in_shape = b.get_shape()
  indicesflat,_ = flatten_indices(fullindices, in_shape)
  
  reduced_indices_flat, segment_ids, counts = tf.unique_with_counts(indicesflat)
  segment_ids = segment_ids[b_size:]
  counts = counts[:b_size]
  
  #print(b_size)
  vala_size = (tf.reduce_max(segment_ids) + 1)
  padsize = b_size - vala_size
  vala_ones = tf.where(tf.equal(counts,1),tf.ones([vala_size],dtype=b.dtype), tf.zeros([vala_size],dtype=b.dtype))

  if doCache:
    segment_ids,padsize,vala_ones = makeCache((segment_ids,padsize,vala_ones))

  vala = tf.exp(loga.values)
  vala_out = tf.segment_sum(vala,segment_ids)  
  vala_out = tf.pad(vala_out,[[0,padsize]])
  val_out = (vala_out+vala_ones)*b.values
  
  reduced_sparse = tf.SparseTensor(b.indices,val_out,in_shape)
  return reduced_sparse
  
def sparse_add_alt(a,b):
  fullindices = tf.concat((a.indices,b.indices),axis=0)
  fullvalues = tf.concat((a.values,b.values),axis=0)
  in_shape = a.get_shape()

  indicesflat,_ = flatten_indices(fullindices, in_shape)
  
  reduced_indices_flat, segment_ids = tf.unique(indicesflat)
  reduced_size = tf.shape(reduced_indices_flat)[0]
  reduced_indices_flat = tf.cast(reduced_indices_flat, tf.int64)
  reduced_indices = tf.transpose(tf.unravel_index(reduced_indices_flat,reduced_shape))
  
  if doCache:
    segment_ids,reduced_size,reduced_indices = makeCache((segment_ids,reduced_size,reduced_indices))
    
  reduced_values = tf.unsorted_segment_sum(fullvalues, segment_ids,reduced_size)
  reduced_sparse = tf.SparseTensor(reduced_indices, reduced_values, in_shape)
  reduced_sparse = tf.sparse_reorder(reduced_sparse)
  return reduced_sparse
