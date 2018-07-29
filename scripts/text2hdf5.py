#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit, modules
from optparse import OptionParser

import numpy as np
import h5py
import h5py_cache
from HiggsAnalysis.CombinedLimit.h5pyutils import makeChunkSize,validateChunkSize
import math



# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True
argv.remove( '-b-' )

from array import array

from HiggsAnalysis.CombinedLimit.DatacardParser import *
from HiggsAnalysis.CombinedLimit.ModelTools import *
from HiggsAnalysis.CombinedLimit.ShapeTools import *
from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.tfscipyhess import ScipyTROptimizerInterface

parser = OptionParser(usage="usage: %prog [options] datacard.txt -o output \nrun with --help to get list of options")
addDatacardParserOptions(parser)
parser.add_option("-P", "--physics-model", dest="physModel", default="HiggsAnalysis.CombinedLimit.PhysicsModel:defaultModel",  type="string", help="Physics model to use. It should be in the form (module name):(object name)")
parser.add_option("--PO", "--physics-option", dest="physOpt", default=[],  type="string", action="append", help="Pass a given option to the physics model (can specify multiple times)")
parser.add_option("", "--dump-datacard", dest="dumpCard", default=False, action='store_true',  help="Print to screen the DataCard as a python config and exit")
parser.add_option("","--allowNegativeExpectation", default=False, action='store_true', help="allow negative expectation")
parser.add_option("","--maskedChan", default=[], type="string",action="append", help="channels to be masked in likelihood but propagated through for later storage/analysis")
parser.add_option("-S","--doSystematics", type=int, default=1, help="enable systematics")
parser.add_option("","--chunkSize", type=int, default=4*1024**2, help="chunk size for hd5fs storage")
parser.add_option("", "--sparse", default=False, action='store_true',  help="Store normalization and systematics arrays as sparse tensors")
(options, args) = parser.parse_args()

if len(args) == 0:
    parser.print_usage()
    exit(1)

options.fileName = args[0]
if options.fileName.endswith(".gz"):
    import gzip
    file = gzip.open(options.fileName, "rb")
    options.fileName = options.fileName[:-3]
else:
    file = open(options.fileName, "r")

## Parse text file 
DC = parseCard(file, options)

if options.dumpCard:
    DC.print_structure()
    exit()

print(options)

nproc = len(DC.processes)
nsignals = len(DC.signals)

dtype = 'float64'

MB = ShapeBuilder(DC, options)

#list of processes, signals first
procs = []
for proc in DC.processes:
  if DC.isSignal[proc]:
    procs.append(proc)

for proc in DC.processes:
  if not DC.isSignal[proc]:
    procs.append(proc)    

#list of signals preserving datacard order
signals = []
for proc in DC.processes:
  if DC.isSignal[proc]:
    signals.append(proc)
      
#list of systematic uncertainties (nuisances)
systs = []
if options.doSystematics:
  for syst in DC.systs:
    systs.append(syst[0])
  
nsyst = len(systs)  
  
#list of channels, ordered such that masked channels are last
chans = []
for chan in DC.bins:
  if not chan in options.maskedChan:
    chans.append(chan)

maskedchans = []
for chan in DC.bins:
  if chan in options.maskedChan:
    chans.append(chan)
    maskedchans.append(chan)


#fill data, expected yields, and kappas into HDF5 file (with chunked storage and compression)

#n.b data and expected have shape [nbins]

#norm has shape [nprocs,nbinsfull] and keeps track of expected normalization

#logkup/down have shape [nsyst, nprocs, nbinsfull] and keep track of systematic variations
#per nuisance-parameter, per-process, per-bin

#n.b, in case of masked channels, nbinsfull includes the masked channels where nbins does not

#first loop through observed data to determine the total number of bins
nbinsfull = 0
nbinsmasked = 0
nbinschans = []
for chan in chans:
  if chan in options.maskedChan:
    nbinschan = 1
    nbinsmasked += nbinschan
  else:
    #exclude overflow/underflow bins
    nbinschan = MB.getShape(chan,"data_obs").GetSize() - 2
  
  nbinschans.append(nbinschan)
  nbinsfull += nbinschan
  
nbins = nbinsfull - nbinsmasked

chunkSize = options.chunkSize
binsSize = nbinsfull*np.dtype(dtype).itemsize
if binsSize > chunkSize:
  print("Warning: size of a single histogram is larger than the specified chunk size, increasing chunk size to %d to match" % binsSize)
  chunkSize = binsSize


#create HDF5 file (chunk cache set to the chunk size since we can guarantee fully aligned writes
outfilename = options.out.replace('.root','.hdf5')
f = h5py_cache.File(outfilename, chunk_cache_mem_size=chunkSize, mode='w')

#save some lists of strings to the file for later use
hprocs = f.create_dataset("hprocs", [len(procs)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hprocs[...] = procs

hsignals = f.create_dataset("hsignals", [len(signals)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hsignals[...] = signals

hsysts = f.create_dataset("hsysts", [len(systs)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hsysts[...] = systs

hmaskedchans = f.create_dataset("hmaskedchans", [len(maskedchans)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hmaskedchans[...] = maskedchans

#create h5py datasets with optimized chunk shapes
data_obs_shape = [nbins]
data_obs_chunks = makeChunkSize(data_obs_shape, dtype, maxbytes = chunkSize)
hdata_obs = f.create_dataset("hdata_obs", data_obs_shape, chunks=data_obs_chunks, dtype=dtype, compression="gzip")
validateChunkSize(hdata_obs)

norm_shape = [nproc,nbinsfull]
norm_chunks = makeChunkSize(norm_shape, dtype, maxbytes = chunkSize)
hnorm = f.create_dataset("hnorm", norm_shape, chunks=norm_chunks, dtype=dtype, compression="gzip")
validateChunkSize(hnorm)

logk_shape = [nsyst,nproc,nbinsfull]


if options.sparse:
  hlognorm_sparse = f.create_group("hlognorm_sparse")
  
  hlognorm_sparse_indices_shape = [0,2]
  hlognorm_sparse_indices_maxshape = [nproc*nbinsfull,2]
  hlognorm_sparse_indices_chunks = makeChunkSize(hlognorm_sparse_indices_maxshape,"int64",maxbytes=chunkSize)
  hlognorm_sparse_indices = hlognorm_sparse.create_dataset("indices",hlognorm_sparse_indices_shape,dtype="int64",maxshape=hlognorm_sparse_indices_maxshape,chunks=hlognorm_sparse_indices_chunks, compression="gzip")
  
  hlognorm_sparse_values_shape = [0]
  hlognorm_sparse_values_maxshape = [nproc*nbinsfull]
  hlognorm_sparse_values_chunks = makeChunkSize(hlognorm_sparse_values_maxshape,dtype,maxbytes=chunkSize)
  hlognorm_sparse_values = hlognorm_sparse.create_dataset("values",hlognorm_sparse_values_shape,dtype=dtype,maxshape=hlognorm_sparse_values_maxshape,chunks=hlognorm_sparse_values_chunks, compression="gzip")

  hlognorm_sparse_dense_shape = hlognorm_sparse.create_dataset("dense_shape", [2], dtype="int64", compression="gzip")
  hlognorm_sparse_dense_shape[...] = norm_shape
  
  
  hlogkavg_sparse = f.create_group("hlogkavg_sparse")
  
  hlogkavg_sparse_indices_shape = [0,3]
  hlogkavg_sparse_indices_maxshape = [nsyst*nproc*nbinsfull,3]
  hlogkavg_sparse_indices_chunks = makeChunkSize(hlogkavg_sparse_indices_maxshape,"int64",maxbytes=chunkSize)
  hlogkavg_sparse_indices = hlogkavg_sparse.create_dataset("indices",hlogkavg_sparse_indices_shape,dtype="int64",maxshape=hlogkavg_sparse_indices_maxshape,chunks=hlogkavg_sparse_indices_chunks, compression="gzip")
  
  hlogkavg_sparse_values_shape = [0]
  hlogkavg_sparse_values_maxshape = [nsyst*nproc*nbinsfull]
  hlogkavg_sparse_values_chunks = makeChunkSize(hlogkavg_sparse_values_maxshape,dtype,maxbytes=chunkSize)
  hlogkavg_sparse_values = hlogkavg_sparse.create_dataset("values",hlogkavg_sparse_values_shape,dtype=dtype,maxshape=hlogkavg_sparse_values_maxshape,chunks=hlogkavg_sparse_values_chunks, compression="gzip")

  hlogkavg_sparse_dense_shape = hlogkavg_sparse.create_dataset("dense_shape", [3], dtype="int64", compression="gzip")
  hlogkavg_sparse_dense_shape[...] = logk_shape 
  
  
  hlogkhalfdiff_sparse = f.create_group("hlogkhalfdiff_sparse")
  
  hlogkhalfdiff_sparse_indices_shape = [0,3]
  hlogkhalfdiff_sparse_indices_maxshape = [nsyst*nproc*nbinsfull,3]
  hlogkhalfdiff_sparse_indices_chunks = makeChunkSize(hlogkhalfdiff_sparse_indices_maxshape,"int64",maxbytes=chunkSize)
  hlogkhalfdiff_sparse_indices = hlogkhalfdiff_sparse.create_dataset("indices",hlogkhalfdiff_sparse_indices_shape,dtype="int64",maxshape=hlogkhalfdiff_sparse_indices_maxshape,chunks=hlogkhalfdiff_sparse_indices_chunks, compression="gzip")
  
  hlogkhalfdiff_sparse_values_shape = [0]
  hlogkhalfdiff_sparse_values_maxshape = [nsyst*nproc*nbinsfull]
  hlogkhalfdiff_sparse_values_chunks = makeChunkSize(hlogkhalfdiff_sparse_values_maxshape,dtype,maxbytes=chunkSize)
  hlogkhalfdiff_sparse_values = hlogkhalfdiff_sparse.create_dataset("values",hlogkhalfdiff_sparse_values_shape,dtype=dtype,maxshape=hlogkhalfdiff_sparse_values_maxshape,chunks=hlogkhalfdiff_sparse_values_chunks, compression="gzip")

  hlogkhalfdiff_sparse_dense_shape = hlogkhalfdiff_sparse.create_dataset("dense_shape", [3], dtype="int64", compression="gzip")
  hlogkhalfdiff_sparse_dense_shape[...] = logk_shape  

else:
  logk_chunks = makeChunkSize(logk_shape, dtype, maxbytes = chunkSize)
  logkcompression = "gzip"
  #compression not needed for empty arrays and breaks chunking logic
  if nsyst == 0:
    logkcompression = None
  hlogkavg = f.create_dataset("hlogkavg", logk_shape, chunks=logk_chunks, dtype=dtype, compression=logkcompression)
  hlogkhalfdiff = f.create_dataset("hlogkhalfdiff", logk_shape, chunks=logk_chunks, dtype=dtype, compression=logkcompression)
  if nsyst>0:
    validateChunkSize(hlogkavg)
    validateChunkSize(hlogkhalfdiff)

#fill data_obs
#counter to keep track of current bin being written
ibin = 0
for chan in chans:
  print(chan)
  if not chan in options.maskedChan:
    #get histogram, convert to np array with desired type, and exclude underflow/overflow bins
    data_obs_chan = np.array(MB.getShape(chan,"data_obs")).astype(dtype)[1:-1]
    nbinschan = data_obs_chan.shape[0]
    #write to output array and increment counter
    hdata_obs[ibin:ibin+nbinschan] = data_obs_chan
    ibin += nbinschan
    data_obs_chan = None
    
#fill norm
for iproc,proc in enumerate(procs):
  print(proc)
  #counter to keep track of current bin being written
  ibin = 0
  for chan,nbinschan in zip(chans,nbinschans):
    expchan = DC.exp[chan]
    hasproc = proc in expchan
    
    if hasproc:
      #get histogram, convert to np array with desired type, and exclude underflow/overflow bins
      norm_chan = np.array(MB.getShape(chan,proc)).astype(dtype)[1:-1]
      if norm_chan.shape[0] != nbinschan:
        raise Exception("Mismatch between number of bins in channel for data and template")
      
      if not options.allowNegativeExpectation:
        norm_chan = np.maximum(norm_chan,0.)
    else:
      #no need to explicitly fill zeros, just increment counter
      ibin += nbinschan
      continue
    
    if options.sparse:
      norm_chan_indices = np.transpose(np.nonzero(norm_chan))
      norm_chan_values = np.reshape(norm_chan[norm_chan_indices],[-1])
      
      print("iproc = %d, chan = %s, sparse length = %d" % (iproc,chan,len(norm_chan_values)))
      
      nvals_chan = len(norm_chan_values)
      oldlength = hlognorm_sparse_indices.shape[0]
      newlength = oldlength + nvals_chan
      
      out_indices = np.array([[iproc,ibin]]) + np.pad(norm_chan_indices,((0,0),(1,0)),'constant')
      norm_chan_indices = None
      
      hlognorm_sparse_indices.resize(newlength,axis=0)
      hlognorm_sparse_indices[oldlength:newlength] = out_indices
      out_indices = None
      
      hlognorm_sparse_values.resize(newlength,axis=0)
      hlognorm_sparse_values[oldlength:newlength] = np.log(norm_chan_values)
      norm_chan_values = None

    #write to (dense) output array
    hnorm[iproc,ibin:ibin+nbinschan] = norm_chan
      
    #increment counter
    ibin += nbinschan
    norm_chan = None
      
#fill logkavg and logkhalfdiff

#numerical cutoff in case of zeros in systematic variations
logkepsilon = math.log(1e-3)

for isyst,syst in enumerate(DC.systs[:nsyst]):
  name = syst[0]
  stype = syst[2]
  
  print(name)
  
  for iproc,proc in enumerate(procs):
    #counter to keep track of current bin being written
    ibin = 0
    for chan,nbinschan in zip(chans,nbinschans):
      expchan = DC.exp[chan]
      hasproc = proc in expchan
      
      if not hasproc:
        #no need to explicitly fill zeros, just increment counter
        ibin += nbinschan
        continue
      
      #retrieve nominal template to calculate ratios and protections
      norm_chan = hnorm[iproc,ibin:ibin+nbinschan]
      
      if stype=='lnN':
        ksyst = syst[4][chan][proc]
        if type(ksyst) is list:
          ksystup = ksyst[1]
          ksystdown = ksyst[0]
          if ksystup == 0. and ksystdown==0.:
            #no need to explicitly fill zeros, just increment counter
            ibin += nbinschan
            continue
          if ksystup == 0.:
            ksystup = 1.
          if ksystdown == 0.:
            ksystdown = 1.
          logkup_chan = math.log(ksystup)*np.ones([nbinschan],dtype=dtype)
          logkdown_chan = -math.log(ksystdown)*np.ones([nbinschan],dtype=dtype)
        else:
          if ksyst == 0.:
            #no need to explicitly fill zeros, just increment counter
            ibin += nbinschan
            continue
          logkup_chan = math.log(ksyst)*np.ones([nbinschan],dtype=dtype)
          logkdown_chan = logkup_chan
      elif 'shape' in stype:
        kfac = syst[4][chan][proc]
        
        if kfac>0:
          systup_chan = np.array(MB.getShape(chan,proc,name+"Up")).astype(dtype)[1:-1]
          if systup_chan.shape[0] != nbinschan:
            raise Exception("Mismatch between number of bins in channel for data and systematic variation template")
          logkup_chan = kfac*np.log(systup_chan/norm_chan)
          logkup_chan = np.where(np.equal(np.sign(norm_chan*systup_chan),1), logkup_chan, logkepsilon*np.ones_like(logkup_chan))
          systup_chan = None
          
          systdown_chan = np.array(MB.getShape(chan,proc,name+"Down")).astype(dtype)[1:-1]
          if systdown_chan.shape[0] != nbinschan:
            raise Exception("Mismatch between number of bins in channel for data and systematic variation template")
          logkdown_chan = -kfac*np.log(systdown_chan/norm_chan)
          logkdown_chan = np.where(np.equal(np.sign(norm_chan*systdown_chan),1), logkdown_chan, -logkepsilon*np.ones_like(logkdown_chan))
          systdown_chan = None
          
        else:
          #no need to explicitly fill zeros, just increment counter
          ibin += nbinschan
          continue
          
      #compute avg and halfdiff
      logkavg_chan = 0.5*(logkup_chan + logkdown_chan)
      logkhalfdiff_chan = 0.5*(logkup_chan - logkdown_chan)
      logkavg_chan = np.where(np.equal(norm_chan,0.), 0., logkavg_chan)
      logkhalfdiff_chan = np.where(np.equal(norm_chan,0.), 0., logkhalfdiff_chan)
      logkup_chan = None
      logkdown_chan = None
      
      if options.sparse:
        logkavg_chan_indices = np.transpose(np.nonzero(logkavg_chan))
        logkavg_chan_values = np.reshape(logkavg_chan[logkavg_chan_indices],[-1])
        
        print("isyst, = %d, iproc = %d, chan = %s, sparse length avg = %d" % (isyst, iproc,chan,len(logkavg_chan_values)))
        
        nvals_chan = len(logkavg_chan_values)
        oldlength = hlogkavg_sparse_indices.shape[0]
        newlength = oldlength + nvals_chan
        
        out_indices = np.array([[isyst,iproc,ibin]]) + np.pad(logkavg_chan_indices,((0,0),(2,0)),'constant')
        logkavg_chan_indices = None
        
        hlogkavg_sparse_indices.resize(newlength,axis=0)
        hlogkavg_sparse_indices[oldlength:newlength] = out_indices
        out_indices = None
        
        hlogkavg_sparse_values.resize(newlength,axis=0)
        hlogkavg_sparse_values[oldlength:newlength] = logkavg_chan_values
        logkavg_chan_values = None
        
        logkhalfdiff_chan_indices = np.transpose(np.nonzero(logkhalfdiff_chan))
        logkhalfdiff_chan_values = np.reshape(logkhalfdiff_chan[logkhalfdiff_chan_indices],[-1])
        
        print("isyst, = %d, iproc = %d, chan = %s, sparse length diff = %d" % (isyst, iproc,chan,len(logkhalfdiff_chan_values)))
        
        nvals_chan = len(logkhalfdiff_chan_values)
        oldlength = hlogkhalfdiff_sparse_indices.shape[0]
        newlength = oldlength + nvals_chan
        
        out_indices = np.array([[isyst,iproc,ibin]]) + np.pad(logkhalfdiff_chan_indices,((0,0),(2,0)),'constant')
        logkhalfdiff_chan_indices = None
        
        hlogkhalfdiff_sparse_indices.resize(newlength,axis=0)
        hlogkhalfdiff_sparse_indices[oldlength:newlength] = out_indices
        out_indices = None
        
        hlogkhalfdiff_sparse_values.resize(newlength,axis=0)
        hlogkhalfdiff_sparse_values[oldlength:newlength] = logkhalfdiff_chan_values
        logkhalfdiff_chan_values = None        
      else:
      #write to output arrays
        hlogkavg[isyst,iproc,ibin:ibin+nbinschan] = logkavg_chan
        hlogkhalfdiff[isyst,iproc,ibin:ibin+nbinschan] = logkhalfdiff_chan
      
      #increment counter
      ibin += nbinschan
      logkavg_chan = None
      logkhalfdiff_chan = None
      
if options.sparse:
  del f["hnorm"]
