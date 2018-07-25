import numpy as np
import tensorflow as tf
from HiggsAnalysis.CombinedLimit.h5pyutils import validateChunkSize

def maketensor(h5dset):
  
  #special handling for empty arrays
  nelems = 1
  for s in h5dset.shape:
    nelems *= s
  
  if nelems == 0:
    return tf.zeros(h5dset.shape,h5dset.dtype)
  
  #check that chunk shape is compatible with optimized reading strategy
  validateChunkSize(h5dset)
  
  #calculate grid of values to iterate over chunks
  def getGrid():
    slices = []
    for s,c in zip(h5dset.shape,h5dset.chunks):
      slices.append(slice(0,s,c))

    grid = np.mgrid[slices]
    gridr = []
    for r in grid:
      gridr.append(r.ravel())
    grid = np.transpose(np.c_[gridr])
    return grid

  grid = getGrid()

  def gen():
    for gv in grid:
      #build slice specification
      readslices = []
      for g,c in zip(gv,h5dset.chunks):
        readslices.append(slice(g,g+c))
      readslices = tuple(readslices)
      print(readslices)
      #read data from exactly one complete chunk
      aout = h5dset[readslices]
      yield aout
      
  #calculate number of chunks
  nchunks = len(grid)  
  
  #create tf Dataset which reads one chunk at a time, unbatches them along the last dimension and  batches them all together,
  #reshapes to restore the original shape, then caches the full result and returns it in an endless loop
  #(unbatching and rebatching is needed because batching only works with equally sized elements, which cannot be guaranteed in case
  #the dataset is not an exact multiple of the chunk size)
  #There are assumptions here about the chunk shape which are enforced by validateChunkSize
  
  dset = tf.data.Dataset.from_generator(gen,tf.as_dtype(h5dset.dtype))
  minibatchsize = h5dset.shape[-1]
  nbatch = int(h5dset.size/minibatchsize)
  
  dset = dset.map(lambda x: tf.reshape(x,[-1,minibatchsize]))
  dset = dset.apply(tf.contrib.data.unbatch())
  dset = dset.batch(nbatch)
  dset = dset.map(lambda x: tf.reshape(x,h5dset.shape))
  dset = dset.cache().repeat(-1)
  atensor = dset.make_one_shot_iterator().get_next()
  return atensor

