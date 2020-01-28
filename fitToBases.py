import jax as jax
import jax.numpy as np

from absl import app
from absl import flags
from absl import logging

import os, glob, sys
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import multiprocessing as mp
from scipy.interpolate import interp1d

from modules.normal_equation import *

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "experiment", "UED",
    "Experiment to fit")

flags.DEFINE_string(
    "basis_dir", "./basisFxns",
    "Directory containing the basis functions.")

flags.DEFINE_string(
    "data_dir", "./data",
    "Directory containing the data.")

flags.DEFINE_integer(
    "Nbases", 6,
    "Number of basis functions used in the fit.")

flags.DEFINE_string(
    "start_time", "316.114000",
    "Simulation start time")

flags.DEFINE_string(
    "end_time", "320.014000",
    "Simulation start time")

flags.DEFINE_float(
    "fit_start_time", 316.114000,
    "First start time of bases when fitting")

flags.DEFINE_float(
    "fit_end_time", 316.114000,
    "Last start time of bases when fitting")

flags.DEFINE_integer(
    "Nfit_times", 1,
    "Number of sample times between [fit_start_time, end_fit_times]")

flags.DEFINE_integer(
    "q_cut_low", 10,
    "Do not use bins below q_cut for fitting")

flags.DEFINE_integer(
    "q_cut_high", 50,
    "Do not use bins above q_cut for fitting")



def get_bases(Nbases, folderName=None): 
  bases = {}
  if folderName is None:
    folderName = os.path.join(
      FLAGS.basis_dir,
      FLAGS.experiment)
  files = "{}/*.dat".format(folderName)
  for fl in glob.glob(files):
    L = int(fl[fl.find("_L-")+3:fl.find("_M-")])
    with open(fl, "rb") as file:
      bases[L] = onp.fromfile(file, onp.double)
     
      if L != 0:
        bases[L] -= np.mean(bases[L])
      #print("norm",L,np.sqrt(np.sum(bases[L]**2)), np.amax(np.abs(bases[L])))
      bases[L] /= np.sqrt(np.sum(bases[L]**2))

  basisList = []
  for l in 2*np.arange(Nbases):
    basisList.append(np.expand_dims(bases[l], axis=0))

  return np.concatenate(basisList, axis=0)


def get_data(FLAGS):

  if FLAGS.experiment == "UED":
    fName = os.path.join(FLAGS.data_dir, FLAGS.experiment,
        "legendre_coefficients[31,70,40].dat")
    with open(fName, "rb") as file:
     data = onp.fromfile(file, onp.double)
    
    data = np.reshape(data, (31, 70,40))[np.arange(6)*2,:,:]
    fName = os.path.join(FLAGS.data_dir, FLAGS.experiment,
        "legendre_coefficient_SEM[31,70,40].dat")
    with open(fName, "rb") as file:
      sem = onp.fromfile(file, onp.double)
    sem = np.reshape(sem, (31, 70,40))[np.arange(6)*2,:,:]
  else:
    raise RuntimeError
    
  return data, sem


def plot_results(res):
  X,Y = onp.meshgrid(
      np.arange(res.shape[2]),
      np.arange(res.shape[1]))
  for i in range(res.shape[0]):
    scale = onp.max([np.amax(res[i,:,:]), abs(np.amin(res[i,:,:]))])
    plt.pcolormesh(X,Y,res[i,:,:],
        vmin=-1*scale, vmax=scale,
        cmap='seismic')
    plt.savefig("plots/fitCoeffs_{}.png".format(i))
    plt.close()



temperatures  = np.arange(30, 69.6, 0.5)
intensities   = np.arange(0.25, 5.3, 0.25)

def fit_data_pool(
    q, ind, sem, data, weights, data_weights,
    start_times, sim_times,
    q_cut_low, q_cut_high,
    basis_dir, fName_suffix, Nbases):

  temp_ind = ind % len(temperatures)
  ints_ind = int((ind - temp_ind)/len(temperatures))
  q.put(fit_data(
      data, sem, weights, data_weights,
      start_times, sim_times,
      q_cut_low, q_cut_high,
      temp_ind, ints_ind,
      basis_dir, fName_suffix, Nbases))


def fit_data(
    data, sem, weights, data_weights, 
    start_times, sim_times,
    q_cut_low, q_cut_high,
    temp_ind, ints_ind,
    basis_dir, fName_suffix, Nbases):

  temp = temperatures[temp_ind]
  ints = intensities[ints_ind]

  folderName = os.path.join(
      basis_dir,
      "UED_%8.6f_%8.6f_" % (temp, ints) + fName_suffix)
  bases_orig = get_bases(Nbases, folderName)
  bases_interp = interp1d(sim_times, bases_orig)

  print("TIMES",sim_times)
  best_chiSq, best_time = onp.inf, 0
  for sTime in start_times:
    times = onp.arange(bases_orig.shape[1])*0.1 + sTime
    print("EVAL TIMES",sTime,times)
    bases = bases_interp(times)
    res = normal_equation_vmap(bases.transpose(), data, weights)

    fit = np.matmul(res, bases)
    fit_chiSq = np.mean(data_weights*\
      (((fit - data)/sem)[:,q_cut_low:q_cut_high,:])**2)\
      /np.sum(data_weights)

    print(fit_chiSq, best_chiSq)
    if fit_chiSq < best_chiSq:
      best_chiSq  = fit_chiSq
      best_time   = sTime

    return (temp_ind, ints_ind, best_time, best_chiSq)

  """
  tst = np.mean(data_weights*\
      (((fit - data)/sem)[:,FLAGS.q_cut_low:FLAGS.q_cut_high,:])**2,
      axis=-1)\
      /np.sum(data_weights)
  tst = np.mean(tst, axis=-1)
  print("ints {} temp {}:\t{}".format(ints,temp,tst))
  """

  if False and ints < 1.3 and temp > 45 and temp < 56:
    plt.plot(data[1,15,:], '-k')
    plt.plot(fit[1,15,:], '-b')
    plt.savefig("plots/afit_ints-{}_temp-{}.png".format(ints, temp))
    plt.close()
  print("done")



def main(argv):


  ###################################################################
  #####  Evaluate Chi Squared Value with the 'Best' Parameters  #####
  ###################################################################
  
  data, sem   = get_data(FLAGS)
  bases       = get_bases(FLAGS.Nbases)
  data *= 1e7
  sem *= 1e7
  weights     = 1./sem**2
  data_weights = np.sqrt(np.sum(
      np.sum((data[:,FLAGS.q_cut_low:FLAGS.q_cut_high,:])**2, axis=-1),
      axis=-1))
  data_weights = np.expand_dims(np.expand_dims(data_weights, axis=-1), axis=-1)

  res = normal_equation_vmap(bases.transpose(), data, weights)
 
  fit = np.matmul(res, bases)
  best_chiSq = np.mean((((fit - data)/sem)[:,10:,:])**2)
  print("Best chi sq: {}".format(best_chiSq))


  ###################################################################
  #####  Evaluate Chi Squared Values Over Temp and Intensities  #####
  ###################################################################

  print(bases.shape)
  sim_times = float(FLAGS.start_time) +\
      onp.arange(bases.shape[1])*\
          (float(FLAGS.end_time)-float(FLAGS.start_time))/\
          max([(bases.shape[1]-1), 1])
  fit_start_times = FLAGS.fit_start_time +\
      onp.arange(FLAGS.Nfit_times)*\
          (FLAGS.fit_end_time-FLAGS.fit_start_time)/\
          max([(FLAGS.Nfit_times-1), 1])


  processes = []
  chiSq = onp.ones((len(temperatures), len(intensities)))*onp.nan
  best_start_times = onp.ones((len(temperatures), len(intensities)))*onp.nan
  fName_suffix = FLAGS.start_time + "_" + FLAGS.end_time
  mp.set_start_method('spawn')
  que = mp.Queue()
  for i in range(5):
    p = mp.Process(target=fit_data_pool, args=(\
      que, i, data, sem, weights, data_weights,
      fit_start_times, sim_times,
      FLAGS.q_cut_low, FLAGS.q_cut_high,
      FLAGS.basis_dir, fName_suffix, FLAGS.Nbases))
    p.start()
    processes.append(p)
  for p in processes:
    p.join()

  min_chiSq = onp.inf
  min_temp, min_ints = 0, 0
  while que.qsize() != 0:
    temp_ind, ints_ind, start_time, chiSq_val = que.get()
    chiSq[temp_ind, ints_ind] = chiSq_val
    best_start_times[temp_ind, ints_ind] = start_time
    if chiSq_val < min_chiSq:
      min_chiSq = chiSq_val
      min_temp = temperatures[temp_ind]
      min_ints = intensities[ints_ind]
  print("Best Fit", min_temp, min_ints, min_chiSq)

  # Plotting Chi Squared
  logging.info("Plotting Chi Squared")
  X,Y = onp.meshgrid(intensities[:], temperatures)
  plt.pcolormesh(X, Y, chiSq[:,:],
      norm=colors.LogNorm())
  plt.colorbar()
  plt.savefig("plots/chi_square.png")
  plt.close()

  plt.pcolormesh(X, Y, best_start_times)
  plt.colorbar()
  plt.savefig("plots/start_times.png")
  plt.close()


  # Residuals from the best fit
  logging.info("Plotting Residuals")
  folderName = os.path.join(
      FLAGS.basis_dir,
      "UED_%8.6f_%8.6f_" % (min_temp, min_ints) + fName_suffix)
  bases = get_bases(FLAGS.Nbases, folderName)
  res = normal_equation_vmap(bases.transpose(), data, weights)

  fit = np.matmul(res, bases)

  X,Y = np.meshgrid(np.arange(data.shape[2]), np.arange(data.shape[1]))
  for i in range(fit.shape[0]):
    scale = 0.85*max([np.amax(fit[i,:,:] - data[i,:,:]), np.abs(np.amin(fit[i,:,:] - data[i,:,:]))])
    plt.pcolormesh(X, Y, fit[i,:,:] - data[i,:,:],
        vmin=-1*scale, vmax=scale,
        cmap='seismic')
    plt.colorbar()
    plt.savefig("plots/residuals_L-{}.png".format(i*2))
    plt.close()


  """
  for i in range(res.shape[1]):
    plt.plot(res[1,i,:])
    plt.savefig("plots/testingCoeff{}.png".format(i))
    plt.close()

    print("sssss",bases.shape,res.shape)
    fit = np.matmul(bases.transpose(), res[1,i,:])
    plt.plot(data[1,i,:])
    plt.plot(fit)
    plt.savefig("plots/testingFig{}.png".format(i))
    plt.close()
  """

  """
  testSum = np.sum(data[1,10:23,:], axis=0)
  res = normal_equation(bases.transpose(), testSum, np.ones(40))
  print(res)

  plt.plot(testSum)
  plt.plot(np.matmul(bases.transpose(), res))
  plt.savefig("plots/testSum.png")
  plt.close()
  """

if __name__ == '__main__':
  app.run(main)

