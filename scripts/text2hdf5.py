#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit, modules
from optparse import OptionParser

import numpy as np
import h5py
import h5py_cache
from HiggsAnalysis.CombinedLimit.h5pyutils import makeChunkSize,validateChunkSize,writeInChunks
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

#fill data, expected yields, and kappas into numpy arrays

#n.b data and expected have shape [nbins]

#norm has shape [nbinsfull, nproc] and keeps track of expected normalization

#logk has shape [nbinsfull, nproc, nsyst, 2] and keep track of systematic variations
#per nuisance-parameter, per-process, per-bin
#the last dimension of size 2 indexes "logkavg" and "logkhalfdiff" for asymmetric uncertainties
#where logkavg = 0.5*(logkup + logkdown) and logkhalfdiff = 0.5*(logkup - logkdown)

#n.b, in case of masked channels, nbinsfull includes the masked channels where nbins does not

data_obs = np.zeros([0], dtype)

if options.sparse:
  norm_sparse_indices = np.zeros([0,2],'int64')
  norm_sparse_values = np.zeros([0],dtype)
  
  logk_sparse_indices = np.zeros([0,4],'int64')
  logk_sparse_values = np.zeros([0],dtype)
else:
  norm = np.zeros([0,nproc], dtype)
  logk = np.zeros([0,nproc,nsyst,2], dtype)

#fill data_obs, norm, and logk
#numerical cutoff in case of zeros in systematic variations
logkepsilon = math.log(1e-3)
#counter to keep track of current bin being written
ibin = 0
for chan in chans:
  print(chan)

  if not chan in options.maskedChan:
    #get histogram, convert to np array with desired type, and exclude underflow/overflow bins
    data_obs_chan = np.array(MB.getShape(chan,"data_obs")).astype(dtype)[1:-1]
    nbinschan = data_obs_chan.shape[0]
    #resize output array
    data_obs.resize([ibin+nbinschan])
    #write to output array
    data_obs[ibin:] = data_obs_chan
    data_obs_chan = None
  else:
    nbinschan = 1
    
  #resize norm and logk tenors
  if not options.sparse:
    norm.resize([ibin+nbinschan, nproc])
    logk.resize([ibin+nbinschan, nproc, nsyst, 2])
  
  expchan = DC.exp[chan]
  for iproc,proc in enumerate(procs):
    hasproc = proc in expchan
    
    if not hasproc:
      continue
    
    #get histogram, convert to np array with desired type, and exclude underflow/overflow bins
    norm_chan = np.array(MB.getShape(chan,proc)).astype(dtype)[1:-1]
    if norm_chan.shape[0] != nbinschan:
      raise Exception("Mismatch between number of bins in channel for data and template")
    
    if not options.allowNegativeExpectation:
      norm_chan = np.maximum(norm_chan,0.)
    
    if options.sparse:
      norm_chan_indices = np.transpose(np.nonzero(norm_chan))
      norm_chan_values = np.reshape(norm_chan[norm_chan_indices],[-1])
      
      print("iproc = %d, chan = %s, sparse length = %d" % (iproc,chan,len(norm_chan_values)))
      
      nvals_chan = len(norm_chan_values)
      oldlength = norm_sparse_indices.shape[0]
      newlength = oldlength + nvals_chan
      
      out_indices = np.array([[ibin,iproc]]) + np.pad(norm_chan_indices,((0,0),(0,1)),'constant')
      norm_chan_indices = None
      
      norm_sparse_indices.resize([newlength,2])
      norm_sparse_indices[oldlength:] = out_indices
      out_indices = None
      
      norm_sparse_values.resize([newlength])
      norm_sparse_values[oldlength:] = norm_chan_values
      norm_chan_values = None

    else:
      #write to (dense) output array
      norm[ibin:,iproc] = norm_chan
        
    
    for isyst,syst in enumerate(DC.systs[:nsyst]):
      name = syst[0]
      stype = syst[2]
  
      print(name)
      
      if stype=='lnN':
        ksyst = syst[4][chan][proc]
        if type(ksyst) is list:
          ksystup = ksyst[1]
          ksystdown = ksyst[0]
          if ksystup == 0. and ksystdown==0.:
            continue
          if ksystup == 0.:
            ksystup = 1.
          if ksystdown == 0.:
            ksystdown = 1.
          logkup_chan = math.log(ksystup)*np.ones([nbinschan],dtype=dtype)
          logkdown_chan = -math.log(ksystdown)*np.ones([nbinschan],dtype=dtype)
          logkavg_chan = 0.5*(logkup_chan + logkdown_chan)
          logkhalfdiff_chan = 0.5*(logkup_chan - logkdown_chan)
          logkup_chan = None
          logkdown_chan = None
        else:
          if ksyst == 0.:
            continue
          logkavg_chan = math.log(ksyst)*np.ones([nbinschan],dtype=dtype)
          logkhalfdiff_chan = np.zeros([nbinschan],dtype=dtype)
      elif 'shape' in stype:
        kfac = syst[4][chan][proc]
        
        # TODO check if it should be rather if kfac <= 0. ?
        if kfac == 0.:
          continue
        
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
        
        logkavg_chan = 0.5*(logkup_chan + logkdown_chan)
        logkhalfdiff_chan = 0.5*(logkup_chan - logkdown_chan)
        logkup_chan = None
        logkdown_chan = None
          
      #ensure that systematic tensor is sparse where normalization matrix is sparse
      logkavg_chan = np.where(np.equal(norm_chan,0.), 0., logkavg_chan)
      logkhalfdiff_chan = np.where(np.equal(norm_chan,0.), 0., logkhalfdiff_chan)
      
      if options.sparse:
        logkavg_chan_indices = np.transpose(np.nonzero(logkavg_chan))
        logkavg_chan_values = np.reshape(logkavg_chan[logkavg_chan_indices],[-1])
        
        print("isyst, = %d, iproc = %d, chan = %s, sparse length avg = %d" % (isyst, iproc,chan,len(logkavg_chan_values)))
        
        nvals_chan = len(logkavg_chan_values)
        oldlength = logk_sparse_indices.shape[0]
        newlength = oldlength + nvals_chan
        
        out_indices = np.array([[ibin,iproc,isyst,0]]) + np.pad(logkavg_chan_indices,((0,0),(0,3)),'constant')
        logkavg_chan_indices = None
        
        logk_sparse_indices.resize([newlength,4])
        logk_sparse_indices[oldlength:] = out_indices
        out_indices = None
        
        logk_sparse_values.resize([newlength])
        logk_sparse_values[oldlength:] = logkavg_chan_values
        logkavg_chan_values = None
        
        logkhalfdiff_chan_indices = np.transpose(np.nonzero(logkhalfdiff_chan))
        logkhalfdiff_chan_values = np.reshape(logkhalfdiff_chan[logkhalfdiff_chan_indices],[-1])
        
        print("isyst, = %d, iproc = %d, chan = %s, sparse length diff = %d" % (isyst, iproc,chan,len(logkhalfdiff_chan_values)))
        
        nvals_chan = len(logkhalfdiff_chan_values)
        oldlength = logk_sparse_indices.shape[0]
        newlength = oldlength + nvals_chan
        
        out_indices = np.array([[ibin,iproc,isyst,1]]) + np.pad(logkhalfdiff_chan_indices,((0,0),(0,3)),'constant')
        logkhalfdiff_chan_indices = None
        
        logk_sparse_indices.resize([newlength,4])
        logk_sparse_indices[oldlength:] = out_indices
        out_indices = None
        
        logk_sparse_values.resize([newlength])
        logk_sparse_values[oldlength:] = logkhalfdiff_chan_values
        logkhalfdiff_chan_values = None        
      else:
        #write to dense output array
        logk[ibin:,iproc,isyst,0] = logkavg_chan
        logk[ibin:,iproc,isyst,1] = logkhalfdiff_chan
      
      #free memory
      logkavg_chan = None
      logkhalfdiff_chan = None    
    
    #free memory
    norm_chan = None
    
  
  #increment counter
  ibin += nbinschan

nbinsfull = ibin


#write results to hdf5 file

procSize = nproc*np.dtype(dtype).itemsize
systSize = 2*nsyst*np.dtype(dtype).itemsize
chunkSize = np.amax([options.chunkSize,procSize,systSize])
if chunkSize > options.chunkSize:
  print("Warning: Maximum chunk size in bytes was increased from %d to %d to align with tensor sizes and allow more efficient reading/writing." % (options.chunkSize, chunkSize))


#create HDF5 file (chunk cache set to the chunk size since we can guarantee fully aligned writes
outfilename = options.out.replace('.root','.hdf5')
if options.sparse:
  outfilename = outfilename.replace('.hdf5','_sparse.hdf5')
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

writeInChunks(data_obs, f, "hdata_obs", maxChunkBytes = chunkSize)

if options.sparse:
  hnorm_sparse = f.create_group("hnorm_sparse")
  writeInChunks(norm_sparse_indices, hnorm_sparse, "indices", maxChunkBytes = chunkSize)
  writeInChunks(norm_sparse_values, hnorm_sparse, "values", maxChunkBytes = chunkSize)
  hnorm_sparse_dense_shape = hnorm_sparse.create_dataset("dense_shape", [2], dtype="int64", compression="gzip")
  hnorm_sparse_dense_shape[...] = [nbinsfull, nproc]
  
  hlogk_sparse = f.create_group("hlogk_sparse")
  writeInChunks(logk_sparse_indices, hlogk_sparse, "indices", maxChunkBytes = chunkSize)
  writeInChunks(logk_sparse_values, hlogk_sparse, "values", maxChunkBytes = chunkSize)
  hlogk_sparse_dense_shape = hlogk_sparse.create_dataset("dense_shape", [4], dtype="int64", compression="gzip")
  hlogk_sparse_dense_shape[...] = [nbinsfull, nproc, nsyst, 2]  

else:
  writeInChunks(norm, f, "hnorm", maxChunkBytes = chunkSize)
  writeInChunks(logk, f, "hlogk", maxChunkBytes = chunkSize)

