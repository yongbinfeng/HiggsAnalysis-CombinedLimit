#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit, modules
from optparse import OptionParser

import tensorflow as tf
import numpy as np
np.random.seed(123456789)

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

print(options)

## Parse text file 
DC = parseCard(file, options)

DC.print_structure()

if options.dumpCard:
    DC.print_structure()
    exit()

#nchan = len(DC.bins)

nproc = len(DC.processes)
nsyst = len(DC.systs)
npoi = len(DC.signals)

#dtype = 'float32'
dtype = 'float64'

MB = ShapeBuilder(DC, options)

#data_obs = None
#norm = None

print(DC.processes)
print(DC.systs[0])

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

#fill data and expected
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

    
print("data_obs:")    
print(data_obs)
print("norm:")
print(norm)
print("kup")
print(np.exp(logkup))
#print(logkup)
print("kdown")
print(np.exp(logkdown))
#print(logkdown)

signals = []
for proc in DC.processes:
  if DC.isSignal[proc]:
    signals.append(proc)

kr = np.zeros([nproc,npoi],dtype=dtype)
#for ipoi,signal in enumerate(DC.signals):
for ipoi,signal in enumerate(signals):
  iproc = DC.processes.index(signal)
  kr[iproc][ipoi] = 1.
  
print('kr')
print(kr)

x = tf.placeholder(dtype, shape=[data_obs.shape[0]])
rv = np.ones([npoi]).astype(dtype)
#rv = np.zeros([npoi]).astype(dtype)
#kr = 1.0*np.ones([nproc,npoi]).astype(dtype)
#norm = 10*np.ones([nchan,nproc]).astype(dtype)
#k = 1.05*np.ones([nchan,nproc,nsyst]).astype(dtype)
thetav = np.zeros([nsyst]).astype(dtype)
#val = 100*np.ones([nchan]).astype(dtype)

rthetav = np.concatenate((rv,thetav),axis=0)

rtheta = tf.Variable(rthetav)

r = rtheta[:npoi]
theta = rtheta[npoi:]

#r = tf.slice(rtheta,0,npoi)
#theta = tf.slice(rtheta,npoi,nsyst)

#r = tf.Variable(rv,name='r')
#theta = tf.Variable(thetav,name='theta')



#snorm = tf.reduce_prod(tf.pow(k,theta),axis=-1)
#rkr = r*kr

rkr = tf.pow(r,kr)
#rkr = tf.exp(r*kr)
rnorm = tf.reduce_prod(rkr, axis=-1)
#pnorm = rnorm*norm

twox = 2.*theta
twox2 = twox*twox
alpha =  0.125 * twox * (twox2 * (3*twox2 - 10.) + 15.)
alpha = tf.clip_by_value(alpha,-1.,1.)
logk = logkavg + alpha*logkhalfdiff

#thetabig = theta*np.ones_like(logkup)

print(logkup)
print(logkdown)
#print(logkint)
#print(thetabig)

#logk = tf.where(tf.greater(tf.abs(thetabig),0.5),logkint,tf.where(tf.greater(thetabig,0.),logkup,logkdown))
#logk = tf.where(tf.greater(thetabig,0.),logkup,logkdown)
#logk = logkup

snorm = tf.reduce_prod(tf.exp(logk*theta),axis=-1)

pnorm = snorm*rnorm*norm
#pnorm = rnorm*norm

n = tf.reduce_sum(pnorm,axis=-1)
#n = tf.where(tf.equal(x,0.),1.,n)
zdata = np.zeros([nbinstotal],dtype=dtype)
odata = np.ones([nbinstotal],dtype=dtype)
#logn = tf.where(tf.equal(x,zdata), zdata, tf.log(n))
nsafe = tf.where(tf.equal(x,zdata), odata, n)
logn = tf.log(nsafe)

xsafe = tf.where(tf.equal(x,zdata), odata, x)
logx = tf.log(xsafe)
lnx = tf.reduce_sum(-x*logx + x, axis=-1)

ln = tf.reduce_sum(-x*logn + n, axis=-1)
#ln = tf.reduce_sum(-x*tf.log(n) + n, axis=-1)
lc = tf.reduce_sum(0.5*tf.square(theta))

loffsetv = np.zeros([],dtype=dtype)
loffset = tf.Variable(loffsetv,trainable=False);



lbare = ln + lc
l = lbare + loffset
#l = ln

minvars = [r,theta]

#chisq = 2.*l

grads = tf.gradients(l,rtheta)
hess = tf.hessians(l,rtheta)[0]
invhess = tf.matrix_inverse(hess)
sigmas = tf.sqrt(tf.diag_part(invhess))

thetahess = hess[npoi:,npoi:]
invthetahess = tf.matrix_inverse(thetahess)
thetasigmas = tf.sqrt(tf.diag_part(invthetahess))

#opt = tf.train.AdamOptimizer(0.1).minimize(l)
opt = tf.contrib.opt.NadamOptimizer().minimize(l)

#opt = tf.train.AdagradOptimizer(1e-3).minimize(l)


sess = tf.Session()




sess.run(tf.global_variables_initializer())

n_exp = sess.run(n)
#data_obs = np.random.poisson(n_exp)
data_obs = n_exp 

#print('snorm:')
#print(sess.run(snorm,{x:data_obs}))
print('rkr:')
print(sess.run(rkr,{x:data_obs}))
print('rnorm:')
print(sess.run(rnorm,{x:data_obs}))
print('pnorm:')
print(sess.run(pnorm,{x:data_obs}))
print('snorm:')
print(sess.run(snorm,{x:data_obs}))
print('n:')
print(sess.run(n,{x:data_obs}))
print('sum n:')
print(sess.run(tf.reduce_sum(n),{x:data_obs}))
print('min n:')
print(sess.run(tf.reduce_min(n),{x:data_obs}))

print('min x+n:')
print(sess.run(tf.reduce_min(x+n),{x:data_obs}))

#print('logk:')
#print(sess.run(logk,{x:data_obs}))
print('ln:')
print(sess.run(ln,{x:data_obs}))
print('lc:')
print(sess.run(lc,{x:data_obs}))

print('l:')
print(sess.run(l,{x:data_obs}))

print('logn:')
print(sess.run(logn,{x:data_obs}))

#print('gradsr:')
#print(sess.run(tf.gradients(logn,r),{x:data_obs}))

#print('gradstheta:')
#print(sess.run(tf.gradients(logn,theta),{x:data_obs}))

#print('grads:')
#print(sess.run(grads,{x:data_obs}))

#print('hessian:')
#print(sess.run(hess,{x:data_obs}))


#print('invhess:')
#print(sess.run(invhess,{x:data_obs}))

#print('sigmas:')
#print(sess.run(sigmas,{x:data_obs}))

sess.run(r.assign(1.1*r))

sess.run(loffset.assign(-lnx), {x:data_obs})

def stepfunc(x):
  print(x)

class SmallEnoughException(Exception):
    pass

iter = 0
lold = 0.
def lossfunc(x):
  #print("
  lcur = x
  diff = lcur-lold
  print([iter, lcur, lold, lcur-lold])
  if iter>0 and lcur-lold < -1e-6:
    raise SmallEnoughException()
  lold = lcur
  iter += 1
  
def lossprint(x):
  print(x)

#try:
#opts = tf.contrib.opt.ScipyOptimizerInterface(l, options={'disp': True, 'gtol' : 0.,'ftol': 1e-99, 'maxls' : 200, 'maxcor' : 1000}, method='L-BFGS-B').minimize(sess, {x:data_obs}, fetches=[l],loss_callback=lossprint)
#except:
  #print("Convergence")

opts = tf.contrib.opt.ScipyOptimizerInterface(l, options={'disp': True, 'gtol' : 0.,'ftol': 0., 'maxls' : 200, 'maxcor' : 100}, method='L-BFGS-B').minimize(sess, {x:data_obs}, fetches=[l],loss_callback=lossprint)

#opts = tf.contrib.opt.ScipyOptimizerInterface(l, options={'disp': True, 'gtol': 0.}, method='BFGS').minimize(sess, {x:data_obs}, fetches=[l], loss_callback=lossprint)



print("r")
print(sess.run(r))
print("theta")
print(sess.run(theta))
#print("sigmas")
#print(sess.run(sigmas, {x:data_obs}))

sigmasv = sess.run(sigmas, {x:data_obs})

rsigmasv = sigmasv[:npoi]
thetasigmasv = sigmasv[npoi:]

for sig,sigma in zip(signals,rsigmasv):
  print([sig,sigma])
  
for syst,sigma in zip(DC.systs,thetasigmasv):
  print([syst[0],sigma])
  
print("nuissance only:")
thetasigmasv2 = sess.run(thetasigmas, {x:data_obs})
for syst,sigma in zip(DC.systs,thetasigmasv2):
  print([syst[0],sigma])


#opts = tf.contrib.opt.ScipyOptimizerInterface(l, options={'disp': True}, method='BFGS').minimize(sess, {x:data_obs}, step_callback=stepfunc)


#print(sess.run([r,theta,l],{x: data_obs}))


#for i in range(10000):
    #lo, _ = sess.run([l, opt],{x: data_obs})
    #print(sess.run([r,theta,l],{x: data_obs}))

    #print(sess.run([r,l],{x: data_obs}))
    #print(sess.run([r,theta]))
