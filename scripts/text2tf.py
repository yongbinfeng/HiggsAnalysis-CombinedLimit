#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit, modules
from optparse import OptionParser

import tensorflow as tf
import numpy as np
np.random.seed(123456789)
tf.set_random_seed(123456789)

import math



# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True
argv.remove( '-b-' )

from HiggsAnalysis.CombinedLimit.DatacardParser import *
from HiggsAnalysis.CombinedLimit.ModelTools import *
from HiggsAnalysis.CombinedLimit.ShapeTools import *
from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.bfgscustom import minimize_bfgs_custom

parser = OptionParser(usage="usage: %prog [options] datacard.txt -o output \nrun with --help to get list of options")
addDatacardParserOptions(parser)
parser.add_option("-P", "--physics-model", dest="physModel", default="HiggsAnalysis.CombinedLimit.PhysicsModel:defaultModel",  type="string", help="Physics model to use. It should be in the form (module name):(object name)")
parser.add_option("--PO", "--physics-option", dest="physOpt", default=[],  type="string", action="append", help="Pass a given option to the physics model (can specify multiple times)")
parser.add_option("", "--dump-datacard", dest="dumpCard", default=False, action='store_true',  help="Print to screen the DataCard as a python config and exit")
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

nproc = len(DC.processes)
nsyst = len(DC.systs)
npoi = len(DC.signals)

dtype = 'float64'

MB = ShapeBuilder(DC, options)

#determine number of bins for each channel
nbinschan = {}
nbinstotal = 0
for chan in DC.bins:
  expchan = DC.exp[chan]
  for proc in DC.processes:
    if proc in expchan:
      datahist = MB.getShape(chan,"data_obs")
      nbins = datahist.GetNbinsX()
      nbinschan[chan] = nbins
      nbinstotal += nbins
      break

#fill data, expected yields, and kappas

#n.b data and expected have shape [nbins]

#norm has shape [nbins,nprocs] and keeps track of expected normalization

#logkup/down have shape [nbins, nprocs, nsyst] and keep track of systematic variations
#per-bin, per-process, per nuisance-parameter 

data_obs = np.empty([0],dtype=dtype)
norm = np.empty([0,nproc],dtype=dtype)
logkup = np.empty([0,nproc,nsyst],dtype=dtype)
logkdown = np.empty([0,nproc,nsyst],dtype=dtype)
for chan in DC.bins:
  expchan = DC.exp[chan]
  nbins = nbinschan[chan]
  
  datahist = MB.getShape(chan,"data_obs")
  datanp = np.array(datahist)[1:-1]
  data_obs = np.concatenate((data_obs,datanp))
  
  normchan = np.empty([nbins,0],dtype=dtype)
  logkupchan = np.empty([nbins,0,nsyst],dtype=dtype)
  logkdownchan = np.empty([nbins,0,nsyst],dtype=dtype)
  for proc in DC.processes:
    hasproc = proc in DC.exp[chan]
    
    if hasproc:
      normhist = MB.getShape(chan,proc)
      normnp = np.array(normhist).astype(dtype)[1:-1]
      normnp = np.reshape(normnp,[-1,1])
    else:
      normnp = np.zeros([nbins,1],dtype=dtype)
      
    normchan = np.concatenate((normchan,normnp),axis=1)
      
    logkupproc = np.empty([nbins,1,0],dtype=dtype)
    logkdownproc = np.empty([nbins,1,0],dtype=dtype)
    for syst in DC.systs:
      name = syst[0]
      stype = syst[2]
      
      if not hasproc:
        logkupsyst = np.zeros([nbins,1,1],dtype=dtype)
        logkdownsyst = np.zeros([nbins,1,1],dtype=dtype)
      elif stype=='lnN':
        ksyst = syst[4][chan][proc]
        if type(ksyst) is list:
          ksystup = ksyst[1]
          ksystdown = ksyst[0]
          if ksystup == 0.:
            ksystup = 1.
          if ksystdown == 0.:
            ksystdown = 1.
          logkupsyst = math.log(ksystup)*np.ones([nbins,1,1],dtype=dtype)
          logkdownsyst = -math.log(ksystdown)*np.ones([nbins,1,1],dtype=dtype)
        else:
          if ksyst == 0.:
            ksyst = 1.
          logkupsyst = math.log(ksyst)*np.ones([nbins,1,1],dtype=dtype)
          logkdownsyst = math.log(ksyst)*np.ones([nbins,1,1],dtype=dtype)
        
      elif 'shape' in stype:
        kfac = syst[4][chan][proc]
        
        if kfac>0:
          normhistup = MB.getShape(chan,proc,name+"Up")
          normnpup = np.array(normhistup).astype(dtype)[1:-1]
          normnpup = np.reshape(normnpup,[-1,1])
          logkupsyst = kfac*np.log(normnpup/normnp)
          logkupsyst = np.where(np.equal(normnp,0.),np.zeros_like(logkupsyst),logkupsyst)
          logkupsyst = np.where(np.equal(normnpup,0.),math.log(1e-6)*np.ones_like(logkupsyst),logkupsyst)
          logkupsyst = np.reshape(logkupsyst,[-1,1,1])
          
          normhistdown = MB.getShape(chan,proc,name+"Down")
          normnpdown = np.array(normhistdown).astype(dtype)[1:-1]
          normnpdown = np.reshape(normnpdown,[-1,1])
          logkdownsyst = -kfac*np.log(normnpdown/normnp)
          logkdownsyst = np.where(np.equal(normnp,0.),np.zeros_like(logkdownsyst),logkdownsyst)
          logkdownsyst = np.where(np.equal(normnpdown,0.),-math.log(1e-6)*np.ones_like(logkdownsyst),logkdownsyst)
          logkdownsyst = np.reshape(logkdownsyst,[-1,1,1])
        else:
          logkupsyst = np.zeros([normnp.shape[0],1,1],dtype=dtype)
          logkdownsyst = np.zeros([normnp.shape[0],1,1],dtype=dtype)
      else:
        raise Exception('Unsupported systematic type')

      logkupproc = np.concatenate((logkupproc,logkupsyst),axis=2)
      logkdownproc = np.concatenate((logkdownproc,logkdownsyst),axis=2)  

    logkupchan = np.concatenate((logkupchan,logkupproc),axis=1)
    logkdownchan = np.concatenate((logkdownchan,logkdownproc),axis=1)  
    
  norm = np.concatenate((norm,normchan), axis=0)

  logkup = np.concatenate((logkup,logkupchan),axis=0)
  logkdown = np.concatenate((logkdown,logkdownchan),axis=0)
    
logkavg = 0.5*(logkup+logkdown)
logkhalfdiff = 0.5*(logkup-logkdown)

#list of signals preserving datacard order
signals = []
for proc in DC.processes:
  if DC.isSignal[proc]:
    signals.append(proc)

#build matrix of signal strength effects
#hard-coded for now as one signal strength multiplier
#per signal process
kr = np.zeros([nproc,npoi],dtype=dtype)
for ipoi,signal in enumerate(signals):
  iproc = DC.processes.index(signal)
  kr[iproc][ipoi] = 1.
  

#initial value for signal strenghts
rv = np.ones([npoi]).astype(dtype)
#rv = np.zeros([npoi]).astype(dtype)

#initial value for nuisances
thetav = np.zeros([nsyst]).astype(dtype)

#combined initializer for all fit parameters
rthetav = np.concatenate((rv,thetav),axis=0)


#data
#nobs = tf.placeholder(dtype, shape=data_obs.shape)
nobs = tf.Variable(data_obs, trainable=False)
theta0 = tf.Variable(np.zeros_like(thetav), trainable=False)


#tf variable containing all fit parameters
rtheta = tf.Variable(rthetav)

#split back into signal strengths and nuisances
r = rtheta[:npoi]
theta = rtheta[npoi:]

#matrices encoding effect of signal strengths
rkr = tf.pow(r,kr)
#rkr = tf.exp(r*kr)
rnorm = tf.reduce_prod(rkr, axis=-1)

#interpolation for asymmetric log-normal
twox = 2.*theta
twox2 = twox*twox
alpha =  0.125 * twox * (twox2 * (3*twox2 - 10.) + 15.)
alpha = tf.clip_by_value(alpha,-1.,1.)
logk = logkavg + alpha*logkhalfdiff

#matrix encoding effect of nuisance parameters
snorm = tf.reduce_prod(tf.exp(logk*theta),axis=-1)

#final expected yields per-bin including effect of signal
#strengths and nuisance parmeters
pnorm = snorm*rnorm*norm
pnorm = tf.maximum(pnorm,tf.zeros_like(pnorm))
nexp = tf.reduce_sum(pnorm,axis=-1)
nexp = tf.identity(nexp,name='nexp')

nexpsafe = tf.where(tf.equal(nobs,tf.zeros_like(nobs)), tf.ones_like(nobs), nexp)
lognexp = tf.log(nexpsafe)

#final likelihood computation

#poison term
ln = tf.reduce_sum(-nobs*lognexp + nexp, axis=-1)

#constraints
lc = tf.reduce_sum(0.5*tf.square(theta - theta0))

l = ln + lc
l = tf.identity(l,name="loss")

grads = tf.gradients(l,rtheta)
grads = tf.identity(grads,"loss_grads")

#uncertainty computation
hess = tf.hessians(l,rtheta)[0]
hess = tf.identity(hess,name="loss_hessian")

invhess = tf.matrix_inverse(hess)
sigmas = tf.sqrt(tf.diag_part(invhess))

#initialize tf session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#random toy
sess.run(nobs.assign(tf.random_poisson(nexp,shape=[],dtype=dtype)))
sess.run(theta0.assign(theta + tf.random_normal(shape=thetav.shape,dtype=dtype)))

#asimov toy
#sess.run(nobs.assign(nexp))

#sess.run(rtheta.assign(1.1*rtheta))
#sess.run(rtheta.assign(1.+rtheta))


print("Running minimizer:")
#scipy-based minimizer
opts = tf.contrib.opt.ScipyOptimizerInterface(l, options={'disp': True, 'gtol' : 0., 'edmtol': 1e-5}, method=minimize_bfgs_custom).minimize(sess)

#get fit values values
rv = sess.run(r)
thetav = sess.run(theta)

#compute uncertainties
sigmasv = sess.run(sigmas)

rsigmasv = sigmasv[:npoi]
thetasigmasv = sigmasv[npoi:]

for sig,rval,sigma in zip(signals,rv,rsigmasv):
  print('%s = %f +- %f' % (sig,rval,sigma))
  
for syst,thetaval,sigma in zip(DC.systs,thetav,thetasigmasv):
  print('%s = %f +- %f' % (syst[0], thetaval, sigma))
