import numpy as np
import math

def makeChunkSize(shape,dtype,maxbytes = 1024**2):
  esize = np.dtype(dtype).itemsize
  maxelems = math.floor(maxbytes/esize)
  
  nelem = 1
  for s in shape:
    nelem *= s
    
  if nelem==0:
    return None
    
  chunks = []
  for s in shape:
    c = min(s,max(1,math.floor( s*maxelems/nelem )))
    nelem = math.ceil(nelem*c/s)
    chunks.append(c)        
    
  return tuple(chunks)
  
def validateChunkSize(h5dset):  
  esize = np.dtype(h5dset.dtype).itemsize

  #start from right, once first chunksize is less than shape size, all chunksizes to the left must be exactly 1
  isPartial = False
  nelem = 1
  for s,c in zip(reversed(h5dset.shape),reversed(h5dset.chunks)):
    nelem *= c
    
    if isPartial and c>1:
      raise Exception("Chunk shape not compatible with reading elements in order and one chunk at a time.")
    
    if c<s:
      isPartial = True
  
  if len(h5dset.shape)>1 and h5dset.chunks[-1]<h5dset.shape[-1]:
    raise Exception("Can't safely subdivide the last dimension, must increase cache size to get around this.")
  
  return True
