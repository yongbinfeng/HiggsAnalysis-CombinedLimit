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
  #empty datasets are by construction ok
  if h5dset.size == 0:
    return True
  
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
  
  return True

def getGrid(h5dset):
  #calculate grid of values to iterate over chunks
  slices = []
  for s,c in zip(h5dset.shape,h5dset.chunks):
    slices.append(slice(0,s,c))

  grid = np.mgrid[slices]
  gridr = []
  for r in grid:
    gridr.append(r.ravel())
  grid = np.transpose(np.c_[gridr])
  return grid

def writeInChunks(arr, h5group, outname, maxChunkBytes = 1024**2):
  nbytes = arr.size*np.dtype(arr.dtype).itemsize
  
  #special handling for empty datasets, which should not use chunked storage or compression
  if arr.size == 0:
    chunks = None
    compression = None
  else:
    chunks = makeChunkSize(arr.shape,arr.dtype,maxChunkBytes)
    compression = "gzip"
  
  h5dset = h5group.create_dataset(outname, arr.shape, chunks=chunks, dtype=arr.dtype, compression=compression)
  validateChunkSize(h5dset)
  
  #nothing to do for empty dataset
  if arr.size == 0:
    return nbytes
  
  #write in chunks, preserving sparsity if relevant
  grid = getGrid(h5dset)
  for gv in grid:
    readslices = []
    for g,c in zip(gv,h5dset.chunks):
      readslices.append(slice(g,g+c))
    readslices = tuple(readslices)
    #write data from exactly one complete chunk
    aout = arr[readslices]
    #no need to explicitly write chunk if it contains only zeros
    if np.count_nonzero(aout):
      h5dset[readslices] = aout
      
  return nbytes

  
