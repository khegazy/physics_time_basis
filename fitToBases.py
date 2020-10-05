from absl import app
from absl import flags
from absl import logging

import numpy as np
import csv
import os, glob, sys, shutil
import time
import copy
import h5py
import pickle as pl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import multiprocessing as mp
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool
from functools import partial
from numpy.fft import fft as fft
from scipy.optimize import curve_fit

from modules.modules import *
from modules.fitting import *
from modules.flags_UED import get_flags

FLAGS = get_flags()


def plot_results(res):
  X,Y = np.meshgrid(
      np.arange(res.shape[2]),
      np.arange(res.shape[1]))
  for i in range(res.shape[0]):
    scale = np.max([np.amax(res[i,:,:]), abs(np.amin(res[i,:,:]))])
    plt.pcolormesh(X,Y,res[i,:,:],
        vmin=-1*scale, vmax=scale,
        cmap='seismic')
    plt.savefig("plots/fitCoeffs_{}.png".format(i))
    plt.close()



def fit_data(
    data, sem, weights, data_weights, reg_scale,
    setBases, temperatures, intensities,
    start_times, sim_times, time_delta,
    t_cut_low, t_cut_high,
    temp_ind, ints_ind,
    experiment, basis_dir, fName_suffix, fit_lgInds,
    reduced_chiSq):

  temp = temperatures[temp_ind]
  ints = intensities[ints_ind]

  # Get Bases
  folderName = os.path.join(
      basis_dir, experiment, FLAGS.basis_sub_dir,
      experiment + "_%8.6f_%8.6f_" % (temp, ints) + fName_suffix)
  all_bases_orig, norms, bases_orig = get_bases(
      FLAGS, fit_lgInds, data.shape,
      normalize=False, folderName=folderName)
  bases_interp = interp1d(sim_times, bases_orig)

  bases = bases_interp(start_times).transpose((2,0,3,1))

  # Fit
  thetas = np.array(normal_equation_reg_vmap(
      bases, data, weights, reg_scale))
  loss = np.array(loss_reg_vmap(
      thetas, bases, data, weights, data_weights, reg_scale))

  
  # Find best fit over varying start times
  best_ind = np.argmin(loss)
  results = (temp_ind, ints_ind, best_ind,
      loss[best_ind], thetas[best_ind])

  return results


def fit_data_pool(
    ind, data, sem, weights, data_weights, reg_scale,
    setBases, temperatures, intensities,
    start_times, sim_times, time_delta,
    t_cut_low, t_cut_high,
    experiment, basis_dir, fName_suffix, fit_lgInds):

  temp_ind = ind % len(temperatures)
  ints_ind = int((ind - temp_ind)/len(temperatures))
  return fit_data(
      data, sem, weights, data_weights, reg_scale,
      setBases, temperatures, intensities,
      start_times, sim_times, time_delta,
      t_cut_low, t_cut_high,
      temp_ind, ints_ind,
      experiment, basis_dir, fName_suffix, fit_lgInds,
      reduced_chiSq=True)


def time_std_pool(
    time, data, sem, weights, data_weights, reg_scale,
    best_temp_ind, best_ints_ind,
    setBases, temperatures, intensities,
    sim_times, time_delta,
    t_cut_low, t_cut_high,
    experiment, basis_dir, fName_suffix, fit_lgInds):


  minChiSq  = np.inf
  endSearch = False
  eval_times = np.expand_dims(
      time + (np.arange(data.shape[-1]) + t_cut_low)*time_delta,
      axis=0)
  cnt = 0
  while not endSearch: 
    chiSqs  = []
    inds    = []
    for ti in np.arange(best_temp_ind-1, best_temp_ind+2):
      if ti < 0 or ti >= len(temperatures):
        continue
      for ii in np.arange(best_ints_ind-1, best_ints_ind+2):
        if ii < 0 or ii >= len(intensities):
          continue

        results = fit_data(
            data, sem, weights, data_weights, reg_scale,
            setBases, temperatures, intensities,
            eval_times, sim_times, time_delta,
            t_cut_low, t_cut_high,
            ti, ii,
            experiment, basis_dir, fName_suffix, fit_lgInds,
            reduced_chiSq=False)

        chiSqs.append(results[3])
        inds.append([ti,ii])

    endSearch = True
    minInd = np.argmin(np.array(chiSqs))
    if chiSqs[minInd] < minChiSq:
      endSearch = False
      minChiSq  = chiSqs[minInd]
      best_temp_ind = inds[minInd][0]
      best_ints_ini = inds[minInd][1]
  
  return (time, minChiSq, (best_temp_ind, best_ints_ind))
    





def main(argv):


  #####  Get Data, Bases, Weights, etc...  #####
  if FLAGS.setBases is not None:
    FLAGS.setBases = np.array([int(x) for x in FLAGS.setBases])
  
  temperatures  = np.arange(
      FLAGS.temp_low,
      FLAGS.temp_high + FLAGS.temp_delta/2,
      FLAGS.temp_delta)
  intensities   = np.arange(
      FLAGS.ints_low,
      FLAGS.ints_high + FLAGS.ints_delta/2,
      FLAGS.ints_delta)

  data, doms, sem, fit_lgInds, sn_map = get_data(FLAGS, plot_data=True)
  dom, plt_dom, _ = doms

  weights     = 1./sem**2
  data_weights = np.sqrt(np.sum(
    np.sum(gaussian_filter1d(data, 2.5, axis=-1)**2, axis=-1),
      axis=-1))
  # Ignore L0 projection for UED
  #if FLAGS.experiment == "UED":
  #  data_weights[0] = 0
  data_weights = np.expand_dims(np.expand_dims(data_weights, axis=-1), axis=0)

  fName_suffix = FLAGS.start_time + "_" + FLAGS.end_time
  folderName = os.path.join(
      FLAGS.basis_dir, FLAGS.experiment, FLAGS.basis_sub_dir,
      FLAGS.experiment + "_%8.6f_%8.6f_" % 
        (temperatures[0], intensities[0]) + fName_suffix)
  all_bases, norms, bases = get_bases(
      FLAGS, fit_lgInds, data.shape,
      folderName=folderName)
  
  fit_start_times = FLAGS.fit_start_time +\
      np.arange(FLAGS.Nfit_times)*\
          (FLAGS.fit_end_time-FLAGS.fit_start_time)/\
          max([(FLAGS.Nfit_times-1), 1])

  fit_times = np.ones((FLAGS.Nfit_times, data.shape[-1]))\
      *np.expand_dims(fit_start_times, -1)
  fit_times += np.arange(data.shape[-1])*FLAGS.fit_delta_time\
      + FLAGS.t_cut_low*FLAGS.fit_delta_time



  sim_times = float(FLAGS.start_time) +\
      np.arange(bases.shape[-1])*\
          (float(FLAGS.end_time)-float(FLAGS.start_time))/\
          max([(bases.shape[-1]-1), 1])


  ###################################################################
  #####  Evaluate Chi Squared Values Over Temp and Intensities  #####
  ###################################################################

  # Check if ChiSq is already saved
  if FLAGS.setBases is not None:
    suffix = "_Temp-{}-{}-{}_Ints-{}-{}-{}".format(
        temperatures[0], temperatures[-1], len(temperatures),
        intensities[0], int(100*intensities[-1])/100.0, len(intensities))
  else:
    suffix = "_Nbases-{}_Temp-{}-{}-{}_Ints-{}-{}-{}".format(
        bases.shape[0], temperatures[0], temperatures[-1], len(temperatures),
        intensities[0], int(100*intensities[-1])/100.0, len(intensities))
  if FLAGS.reg_scale > 0:
    ind = suffix.find("Temp")
    suffix = suffix[:ind] + "Reg-{}_".format(FLAGS.reg_scale) + suffix[ind:]
  
  output_prefix = os.path.join("output", FLAGS.experiment, FLAGS.basis_sub_dir)
  if not os.path.exists(output_prefix):
    os.makedirs(output_prefix)

  fit_results_fileName    = os.path.join(output_prefix,
      "{}_fit_results{}.h5".format(FLAGS.experiment, suffix))
  fit_output_fileName     = os.path.join(output_prefix,
      "{}_fit_output{}.h5".format(FLAGS.experiment, suffix))
  fit_landscape_fileName  = os.path.join(output_prefix,
      "{}_fit_landscape{}.h5".format(FLAGS.experiment, suffix))

  # If fit output/results file doesn't exist then fit
  if not os.path.exists(fit_results_fileName) and not FLAGS.debugging:
 
    pool = Pool(processes=70)
    basis_results = pool.map(partial(fit_data_pool,
        data=data,
        sem=sem,
        weights=weights,
        data_weights=data_weights,
        reg_scale=FLAGS.reg_scale,
        setBases=FLAGS.setBases,
        temperatures=temperatures,
        intensities=intensities,
        start_times=fit_times,
        sim_times=sim_times,
        time_delta=FLAGS.fit_delta_time,
        t_cut_low=FLAGS.t_cut_low,
        t_cut_high=FLAGS.t_cut_high,
        experiment=FLAGS.experiment,
        basis_dir=FLAGS.basis_dir,
        fName_suffix=fName_suffix,
        fit_lgInds=fit_lgInds),
        range(len(intensities)*len(temperatures)))

    # Parse fitting results
    temp_ind, ints_ind, start_time, chiSq_val, fit_coeffs = basis_results[0]
    chiSq             = np.ones((len(temperatures), len(intensities)))*np.nan
    best_start_times  = np.ones((len(temperatures), len(intensities)))*np.nan
    all_fit_coeffs      = np.ones((len(temperatures), len(intensities))\
        + fit_coeffs.shape)*np.nan
    best_temp, best_temp_ind = None, None
    best_ints, best_ints_ind = None, None
    best_fit_chiSq = np.inf
    best_start_time = None
    for res in basis_results:
      temp_ind, ints_ind, time_ind, chiSq_val, fit_coeffs = res
      chiSq[temp_ind, ints_ind] = chiSq_val
      best_start_times[temp_ind, ints_ind] = fit_start_times[time_ind]
      all_fit_coeffs[temp_ind, ints_ind] = fit_coeffs
      if chiSq_val < best_fit_chiSq:
        best_fit_chiSq = chiSq_val
        best_start_time = fit_start_times[time_ind]
        best_temp, best_temp_ind = temperatures[temp_ind], temp_ind
        best_ints, best_ints_ind = intensities[ints_ind], ints_ind


    ################################
    #####  Calculate Time STD  #####
    ################################

    time_std_temperatures = temperatures
    time_std_intensities  = intensities

    chiSq_time_std, time_std_range, time_std_temps, time_std_ints =\
        [], [], [], []


    std_times = np.arange(best_start_time - FLAGS.range_STD_time,
        best_start_time + FLAGS.range_STD_time + FLAGS.step_STD_time/2,
        FLAGS.step_STD_time)
    min_temp_ind = np.argmin(np.abs(temperatures[best_temp_ind]\
        - time_std_temperatures))
    min_ints_ind = np.argmin(np.abs(intensities[best_ints_ind]\
        - time_std_intensities))
    
    # Looking at only lg=2
    data_weights_std = np.zeros_like(data_weights)
    ind = np.where(fit_lgInds == 2)[0][0]
    data_weights_std[:,ind,:] = 1.
    
    pool = Pool(processes=70)
    time_results = pool.map(partial(time_std_pool,
        data=data,
        sem=sem,
        weights=weights,
        best_temp_ind=best_temp_ind,
        best_ints_ind=best_ints_ind,
        data_weights=data_weights_std,
        reg_scale=FLAGS.reg_scale,
        setBases=FLAGS.setBases,
        temperatures=time_std_temperatures,
        intensities=time_std_intensities,
        sim_times=sim_times,
        time_delta=FLAGS.fit_delta_time,
        t_cut_low=FLAGS.t_cut_low,
        t_cut_high=FLAGS.t_cut_high,
        experiment=FLAGS.experiment,
        basis_dir=FLAGS.basis_dir,
        fName_suffix=fName_suffix,
        fit_lgInds=fit_lgInds),
        std_times)

    # Parsing results
    for res in time_results:
      time, minChiSq, minInds = res
      time_std_range.append(time)
      chiSq_time_std.append(minChiSq)
      time_std_temps.append(time_std_temperatures[minInds[0]])
      time_std_ints.append(time_std_intensities[minInds[1]])
    time_std_range = np.array(time_std_range)
    chiSq_time_std = np.array(chiSq_time_std)
    time_std_temps = np.array(time_std_temps)
    time_std_ints = np.array(time_std_ints)
    

    # Fitting for t0 std
    def fitFxn(x, c, t0, off):
      return c*(x + t0)**2 + off

    mInd = np.argmin(chiSq_time_std)
    time_std_range_centered = np.array(time_std_range) - time_std_range[mInd]
    keep_inds = np.where((time_std_range_centered > -1*FLAGS.range_STD_time_cut) 
        & (time_std_range_centered < FLAGS.range_STD_time_cut))[0]
    chiSq_time_std = np.array(chiSq_time_std)[keep_inds]
    time_std_range_centered = time_std_range_centered[keep_inds]
    time_std_temps = time_std_temps[keep_inds]
    time_std_ints = time_std_ints[keep_inds]
    #time_std_range_centered *= 1000
    popt, pcov = curve_fit(fitFxn, time_std_range_centered, chiSq_time_std, 
        [1, 0, np.amin(chiSq_time_std)])
    fit_t0 = time_std_range[mInd] - popt[1]
    fit_t0_std = 1./np.sqrt(popt[0])
    fit_t0_min_chiSq = popt[2]
    print("FIT RESULTS", best_temp, best_ints,
        "{} +/- {}".format(fit_t0, fit_t0_std))

    
    #####  Saving Fitting Results  #####
    # Saving fit landscape
    with h5py.File(fit_landscape_fileName, 'w') as h5:
      # Temperature and Intensity Fit
      h5.create_dataset("fit_coeffs", data=all_fit_coeffs)
      h5.create_dataset("temp_ints_chiSq", data=chiSq)
      h5.create_dataset("temp_ints_min_chiSq", data=best_fit_chiSq)
      h5.create_dataset("temp_ints_t0", data=best_start_times)
      h5.create_dataset("temperatures", data=temperatures)
      h5.create_dataset("best_temperature", data=best_temp)
      h5.create_dataset("intensities", data=intensities)
      h5.create_dataset("best_intensity", data=best_ints)
      # T0 fit given best temperature and intensity
      h5.create_dataset("t0_chiSq", data=chiSq_time_std)
      h5.create_dataset("t0_fit_times", data=time_std_range_centered)
      h5.create_dataset("t0_min_chiSq", data=popt[2])
      h5.create_dataset("t0_time_offset", data=popt[1])
      h5.create_dataset("t0_temps_chiSq", data=time_std_temps)
      h5.create_dataset("t0_ints_chiSq", data=time_std_ints)
      h5.create_dataset("t0", data=fit_t0)
      h5.create_dataset("t0_std", data=fit_t0_std)


    ############################################################
    #####  Get Fit Coefficients with Optimal Temp/Ints/t0  #####
    ############################################################

    # Get full dataset
    FLAGS.dom_cut_low, FLAGS.dom_cut_high = None, None
    data, doms, sem, _, sn_map = get_data(FLAGS)
    dom, plt_dom, _ = doms
    weights = 1./sem**2

    # Get optimal bases
    folderName = os.path.join(
              FLAGS.basis_dir, FLAGS.experiment, FLAGS.basis_sub_dir,
              FLAGS.experiment + "_%8.6f_%8.6f_"\
                  % (best_temp, best_ints)\
                  + fName_suffix)

    all_bases_orig, norms, bases_orig = get_bases(
        FLAGS, fit_lgInds, data.shape,
        normalize=False, folderName=folderName)

    bases_interp      = interp1d(sim_times, bases_orig)
    all_bases_interp  = interp1d(sim_times, all_bases_orig)
    times = np.arange(data.shape[-1])*FLAGS.fit_delta_time\
        + fit_t0 + FLAGS.t_cut_low*FLAGS.fit_delta_time
    plt_times = np.concatenate([times, np.array([2*times[-1] - times[-2]])]) 
    
    bases = bases_interp(times)
    all_bases = all_bases_interp(times)

    bases_norm = copy.copy(bases)
    bases_norm[1:] -= np.expand_dims(np.mean(bases[1:], axis=-1), axis=-1)
    norms = np.sqrt(np.sum(bases**2, axis=-1))
    bases_norm = bases_norm/np.expand_dims(norms, axis=-1)
    
    all_bases_norm      = copy.copy(all_bases)
    all_bases_norm[1:] -= np.expand_dims(
        np.mean(all_bases[1:], axis=-1), axis=-1)
    all_norms = np.sqrt(np.sum(all_bases**2, axis=-1))
    all_bases_norm = all_bases_norm/np.expand_dims(all_norms, axis=-1)

    # Fit for optimal C coefficients
    
    fit_coeffs = normal_equation_reg(
        bases.transpose((0,2,1)),
        data,
        weights,
        FLAGS.reg_scale)

    # If bases l<0 then there are multiple l=0 bases
    singular_inds = False
    if FLAGS.setBases is not None:
      singular_inds = []
      for lg in fit_lgInds:
        inds = (lg + FLAGS.setBases) < 0 
        singular_inds.append(inds)
        if np.any(inds):
          iind = 0
          while inds[iind] and iind < len(inds):
            iind += 1
            fit_coeffs[lg,:,iind] += fit_coeffs[lg,:,inds]
          fit_coeffs[lg,:,inds] = 0

    dof = bases.shape[-1] - bases.shape[0]
    if np.any(singular_inds):
      #for lg in data.shape[0]:
      m = np.einsum('iab,ijb,icb->ijac', bases, weights, bases)*dof
      fit_coeffs_cov = np.linalg.inv(m)

    else:
      m = np.einsum('iab,ijb,icb->ijac', bases, weights, bases)*dof
      fit_coeffs_cov = np.linalg.inv(m)


    # Saving fit parameters
    with h5py.File(fit_results_fileName, 'w') as h5:
      h5.create_dataset("legendre_inds", data=fit_lgInds)
      h5.create_dataset("fit_axis", data=dom)
      h5.create_dataset("fit_coeffs", data=fit_coeffs)
      h5.create_dataset("fit_coeffs_cov", data=fit_coeffs_cov)
      h5.create_dataset("fit_bases", data=bases)
      h5.create_dataset("temperature", data=best_temp)
      h5.create_dataset("intensity", data=best_ints)
      h5.create_dataset("t0", data=fit_t0)
      h5.create_dataset("t0_std", data=fit_t0_std)


  #################################
  #####  Analyze Fit Results  #####
  #################################

  with h5py.File(fit_landscape_fileName, 'r') as h5:
    temperatures  = h5["temperatures"][:]
    intensities   = h5["intensities"][:]
    chiSq         = h5["temp_ints_chiSq"][:]
    all_fit_coeffs  = h5["fit_coeffs"][:]
    best_start_times = h5["temp_ints_t0"][:]
    t0_chiSq      = h5["t0_chiSq"][:]
    t0_fit_times  = h5["t0_fit_times"][:]
    t0_time_shift = h5["t0_time_offset"][...]
    t0_min_chiSq  = h5["t0_min_chiSq"][...]
    t0_temps_chiSq= h5["t0_temps_chiSq"][:]
    t0_ints_chiSq = h5["t0_ints_chiSq"][:]
    fit_t0_chiSq  = h5["t0_chiSq"][:]
  with h5py.File(fit_results_fileName, 'r') as h5:
    fit_bases       = h5["fit_bases"][:]
    fit_coeffs      = h5["fit_coeffs"][:]
    fit_coeffs_cov  = h5["fit_coeffs_cov"][:]
    fit_temperature = h5["temperature"][...]
    fit_intensity   = h5["intensity"][...]
    fit_t0          = h5["t0"][...]
    fit_t0_std      = h5["t0_std"][...]

  if not (len(temperatures) == chiSq.shape[0]\
      and len(intensities) == chiSq.shape[1]):
    raise RuntimeError("Shapes do not align {} {} {}".format(
        len(temperatures), len(intensities), chiSq.shape))
  
  # Get full dataset
  FLAGS.dom_cut_low = FLAGS.plot_dom_cut_low
  FLAGS.dom_cut_high = FLAGS.plot_dom_cut_high
  data, doms, sem, _, sn_map = get_data(FLAGS)
  dom, plt_dom, dom_filter = doms
  weights = 1./sem**2
  fit_coeffs = fit_coeffs[:,dom_filter,:]
  fit_coeffs_cov = fit_coeffs_cov[:,dom_filter,:,:]

  scale_dom = np.ones_like(dom)
  if FLAGS.experiment == "UED":
    with h5py.File("simulations/UED/N2O_sim_diffraction-azmAvg_align-random_Qmax-12.88.h5", "r") as h5:
      scale_dom = 1./h5["atm_diffraction"][dom_filter]


  # Get optimal bases
  folderName = os.path.join(
            FLAGS.basis_dir, FLAGS.experiment, FLAGS.basis_sub_dir,
            FLAGS.experiment + "_%8.6f_%8.6f_"\
                % (fit_temperature, fit_intensity)\
                + fName_suffix)

  all_bases_orig, _, bases_orig = get_bases(
      FLAGS, fit_lgInds, data.shape,
      normalize=False, folderName=folderName)

  bases_interp      = interp1d(sim_times, bases_orig)
  all_bases_interp  = interp1d(sim_times, all_bases_orig)
  times = np.arange(data.shape[-1])*FLAGS.fit_delta_time\
      + fit_t0 + FLAGS.t_cut_low*FLAGS.fit_delta_time
  plt_times = np.concatenate([times, np.array([2*times[-1] - times[-2]])]) 
  
  bases = bases_interp(times)
  all_bases = all_bases_interp(times)

  bases_norm = copy.copy(bases)
  bases_norm[1:] -= np.expand_dims(np.mean(bases[1:], axis=-1), axis=-1)
  norms = np.sqrt(np.sum(bases**2, axis=-1))
  bases_norm = bases_norm/np.expand_dims(norms, axis=-1)
  
  all_bases_norm      = copy.copy(all_bases)
  all_bases_norm[1:] -= np.expand_dims(
      np.mean(all_bases[1:], axis=-1), axis=-1)
  all_norms = np.sqrt(np.sum(all_bases**2, axis=-1))
  all_bases_norm = all_bases_norm/np.expand_dims(all_norms, axis=-1)


  # Get fit results
  fit = np.matmul(fit_coeffs, fit_bases)
  fit_coeffs_norm = fit_coeffs*np.expand_dims(norms, axis=-1)


  h5_output = h5py.File(fit_output_fileName, 'w')

  #####  Plotting  #####
  if FLAGS.setBases is not None:
    plot_folder = os.path.join("plots", FLAGS.experiment,
        FLAGS.basis_sub_dir, "setBases")
  else:
    plot_folder = os.path.join("plots", FLAGS.experiment,
        FLAGS.basis_sub_dir)

  if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

  # Plotting Bases Overlap
  overlap = np.matmul(all_bases_norm, all_bases_norm.transpose())
  h5_output.create_dataset("basis_overlap", data=overlap)
  fig, ax = plt.subplots()
  X,Y = np.meshgrid(
      np.arange(all_bases_norm.shape[0]+1),
      np.arange(all_bases_norm.shape[0]+1))
  pcm = ax.pcolormesh(X, Y, overlap, vmin=-1, vmax=1, cmap='seismic')
  ax.set_xticks(np.arange(all_bases_norm.shape[0]) + 0.5)
  ax.set_xticklabels(np.arange(all_bases_norm.shape[0])*2)
  ax.set_yticks(np.arange(all_bases_norm.shape[0]) + 0.5)
  ax.set_yticklabels(np.arange(all_bases_norm.shape[0])*2)
  fig.colorbar(pcm, ax=ax)  
  fig.savefig(os.path.join(plot_folder,
      "bases_overlap.png"))
  plt.close()

  # Plotting Chi Squared
  logging.info("Plotting Chi Squared")
  fig, ax = plt.subplots()
  plt_intensities = np.concatenate(
      [intensities, np.array([2*intensities[-1] - intensities[-2]])])
  plt_temperatures = np.concatenate(
      [temperatures, np.array([2*temperatures[-1] - temperatures[-2]])])
  X,Y = np.meshgrid(plt_intensities, plt_temperatures)
  pcm = ax.pcolormesh(X, Y, chiSq[:,:])#, vmax = 0.97*np.amax(chiSq))
  #pcm = ax.pcolormesh(X, Y, chiSq[:,:], norm=colors.LogNorm())#, cmap="binary")
  fig.colorbar(pcm, ax=ax)  
  #ax.set_xlim([1, 8])
  #ax.set_ylim([40, 120])
  ax.set_xlabel('Laser Intensity [$10^{12} W/cm^2$]')
  ax.set_ylabel("Temperature [K]")
  if FLAGS.experiment == "UED":
    print("skipping")
    """
    pcm.set_clim([2.5e-2, 6e-2])
    axins = ax.inset_axes([0.6, 0.5, 0.37, 0.37])
    pcm_ins = axins.pcolormesh(X, Y, chiSq[:,:], vmax=2.9e-2)
    #    norm=colors.LogNorm())#, cmap="binary")
    axins.set_xlim([2, 5])
    axins.set_ylim([70, 100])
    #axins.tick_params(axis='x', colors='w')
    #axins.tick_params(axis='y', colors='w')
    ax.indicate_inset_zoom(axins)
    cb = fig.colorbar(pcm_ins, ax=axins,
        location="top", anchor=(5,5))
    cb.minorticks_off()
    cb.set_ticks([2.8e-2, 2.9e-2])
    #cb.ax.yaxis.set_tick_params(color='w')
    #cb.ax.xaxis.set_tick_params(color='w')
    #plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='w')
    #plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color='w')

    """
  """
  else:
    ax.plot(intensities, chiSq[0,:])
    ax.set_xlim([intensities[0], intensities[-1]])
    ax.set_xlabel('Laser Intensity [$10^{12} W/cm^2$]')
    ax.set_ylabel(r'$\chi^2$')
    ax.text(0.45, 0.95, "Best Fit", fontweight='bold', transform=ax.transAxes)
    ax.text(0.45, 0.9,
        "Laser Intensity: {0:.3g}".format(fit_params["intensity"])\
          + "*$10^{12} W/cm^2$",
        transform=ax.transAxes)
  """
   
  ax.plot(fit_intensity, fit_temperature, 'wo', markersize=3)

  plt.tight_layout()
  print("saving in",os.path.join(plot_folder, "chi_square{}.png".format(suffix)))
  fig.savefig(os.path.join(plot_folder,
      "chi_square{}.png".format(suffix)))
  plt.close()
  

  print("MIN MAX", np.amin(best_start_times),np.amax(best_start_times))
  plt.pcolormesh(
      X, Y, best_start_times,
      vmin=np.amin(best_start_times), vmax=np.amax(best_start_times))
  plt.colorbar()
  plt.xlabel("Laser Intensity [$10^{12} W/cm^2$]")
  plt.ylabel("Temperature [K]")
    
  plt.savefig(os.path.join(plot_folder,
      "start_times{}.png".format(suffix)))
  plt.close()

  # Residuals from the best fit
  logging.info("Plotting Residuals")

  residuals = data - fit
  h5_output.create_dataset("residuals", data=residuals)

  plt_scales = None
  if FLAGS.experiment == "UED":
    plt_scales = [70, 3, 1.5, 1, 0.5, 0.3]
  if FLAGS.experiment == "LCLS":
    plt_scales = np.array([0.01, 0.05, 0.05, 0.5, 0.5, 0.5, 0.5])*0.12

  X,Y = np.meshgrid(plt_times, plt_dom)
  for i in range(fit.shape[0]):

    plt.pcolormesh(X, Y, fit[i,:,:],
          cmap='seismic')
    plt.colorbar()
    plt.savefig(os.path.join(plot_folder,\
        "fit{}_L-{}.png".format(i*2, suffix)))
    plt.close()
    if plt_scales is not None:
      plt.pcolormesh(X, Y, residuals[i,:,:],
          vmin=-1*plt_scales[i], vmax=plt_scales[i],
          cmap='seismic')
    else:
      plt.pcolormesh(X, Y, residuals[i,:,:],
          cmap='seismic')

    plt.colorbar()
    plt.xlabel("Time [ps]")
    if FLAGS.experiment == "UED":
      plt.ylabel('Q [$\AA^{-1}$]')
    elif FLAGS.experiment == "LCLS":
      plt.ylabel('Energy [eV]')
    plt.savefig(os.path.join(plot_folder,\
        "residuals{}_L-{}.png".format(i*2, suffix)))
    plt.close()

    """
    for j in range(data.shape[1]):
      plt.plot(times, data[i,j,:], '-k')
      plt.plot(times, fit[i,j,:], '-b')
      plt.plot(times, bases_norm[1,:]*fit_coeffs_norm[i,j,1], '-g')
      plt.savefig(os.path.join(plot_folder,
          "{}_LOfit_l{}_e{}{}.png".format(FLAGS.experiment, i, j, suffix)))
      plt.close()
    """

  #plt.plot(data[1,10,:])
  #plt.plot(fit[1,10,:])
  #plt.savefig("testingFit.png")
  #plt.close()


  # Analyze residuals as a goodness of fit
  plt_range = None
  if FLAGS.experiment == "UED":
    plt_range = [[10**1, 10**6], [1, 3*10**2], [1, 5*10**2], [1, 10], [0.5, 5], [0.5, 2]]
  elif FLAGS.experiment == "LCLS":
    plt_range = [[10**3, 10**5], [10**3, 10**5], [10**3, 10**5],
        [10**3, 10**5], [5, 7*10**2], [2, 3*10**2]]

  residuals_fft = fft(residuals, axis=-1)[:,:,:int(residuals.shape[-1]/2+1)]
  residuals_pow = np.absolute(residuals_fft)**2
  X,Y = np.meshgrid(np.arange(residuals_pow.shape[-1]+1), plt_dom)
  for i in range(fit.shape[0]):
    if plt_range is not None:
      plt.pcolormesh(X, Y, residuals_pow[i,:,:],
          #vmin=plt_range[i][0], vmax=plt_range[i][1],
          norm=colors.LogNorm(),
          cmap="Blues")
    else:
      plt.pcolormesh(X, Y, residuals_pow[i,:,:],
          norm=colors.LogNorm(),
          cmap="Blues")

    plt.colorbar()
    plt.xlim([0, residuals_pow.shape[-1]-1])
    plt.xlabel("Frequency [2$\pi$/L]")
    if FLAGS.experiment == "UED":
      plt.ylabel('Q [$\AA^{-1}$]')
    elif FLAGS.experiment == "LCLS":
      plt.ylabel('Energy [eV]')
    plt.savefig(os.path.join(plot_folder,\
        "residuals_power{}_L-{}.png".format(i*2, suffix)))
    plt.close()

    plt.plot(np.sum(residuals_pow[i], axis=0))
    plt.xlim([0, residuals_pow.shape[-1]-1])
    plt.xlabel("Frequency [2$\pi$/L]")
    #if plt_range is not None:
    #  plt.ylim(plt_range[i])
    plt.ylabel('Power')
    plt.yscale('log')
    plt.savefig(os.path.join(plot_folder,\
        "residuals_power_sum_L-{}{}.png".format(i*2, suffix)))
    plt.close()
  
  # Plot fit power spectrum
  power = fit_coeffs_norm**2
  plt_range = None
  if FLAGS.experiment == "UED":
    plt_range = [[10**4, 3*10**7], [10**2, 2*10**5], [8*10**1, 6*10**3],
        [10, 2*10**3], [5, 7*10**2], [2, 3*10**2]]
  for lg in range(power.shape[0]):
    for i in range(power.shape[-1]):
      plt.plot(dom, power[lg,:,i], label="Basis {}".format(i*2))
    plt.xlim(dom[0], 2*dom[-1]-dom[-2])
    if FLAGS.experiment == "UED":
      plt.xlabel('Q [$\AA^{-1}$]')
    elif FLAGS.experiment == "LCLS":
      plt.xlabel('Energy [eV]')
    plt.ylabel('Power')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(plot_folder,\
        "full_power_spectrum_basis-{}{}.png".format(
          2*lg, suffix)))
    plt.close()

    sum_power = np.sum(power[lg,:,:], axis=0)
    plt.bar(np.arange(len(sum_power)), sum_power, align='center')
    plt.xticks(np.arange(len(sum_power)), np.arange(len(sum_power))*2)
    plt.xlabel('Bases')
    if plt_range is not None:
      plt.ylim(plt_range[lg])
    plt.ylabel('Power')
    plt.yscale('log')
    plt.savefig(os.path.join(plot_folder,\
        "power_spectrum_scaled_basis-{}{}.png".format(
          2*lg, suffix)))
    plt.close()

    plt.bar(np.arange(len(sum_power)), sum_power, align='center')
    plt.xticks(np.arange(len(sum_power)), np.arange(len(sum_power))*2)
    plt.xlabel('Bases')
    plt.ylabel('Power')
    plt.yscale('log')
    plt.savefig(os.path.join(plot_folder,\
        "power_spectrum_basis-{}{}.png".format(
          2*lg, suffix)))
    plt.close()


  # Fit coefficients
  cov_inds = np.arange(fit_coeffs_cov.shape[-1])
  for i,(lg_C, lg_CN) in enumerate(zip(fit_coeffs, fit_coeffs_norm)):
    if FLAGS.setBases is not None:
      X,Y = np.meshgrid(
          np.arange(lg_CN.shape[-1]+1)[:-1] + i + FLAGS.setBases, plt_dom)
    else:
      X,Y = np.meshgrid(fit_lgInds, plt_dom)
    #si = np.argmax(np.abs(lg_C), axis=0)
    #scales = np.abs(lg_CN[si, np.arange(FLAGS.Nbases)])
    scale = np.amax(np.abs(lg_CN))
    fig, ax = plt.subplots()
    lbl = "Time Basis: {}"
    if lg_CN.shape[-1] == 1:
      print(scale_dom.shape, np.sqrt(fit_coeffs_cov[i,:,0]).shape)
      ax.errorbar(plt_dom[:-1], lg_CN[:,0]*scale_dom, fmt='o', color='k', 
          yerr=np.sqrt(fit_coeffs_cov[i,:,0])*np.expand_dims(scale_dom, -1),
          ecolor='gray', label=lbl.format(fit_lgInds[i]))
      if FLAGS.experiment == "UED":
        ax.set_xlabel("q $[\AA^{-1}]$")
      elif FLAGS.experiment == "LCLS":
        ax.set_xlabel("Energy [eV]")
 
    else:
      for j in range(lg_CN.shape[-1]):
        if FLAGS.setBases is not None:
          label = lbl.format(FLAGS.setBases[j] + fit_lgInds[i])
        else:
          label = lbl.format(fit_lgInds[i])
        ax.errorbar(plt_dom[:-1]*scale_dom, lg_CN[:,0],
            yerr=np.sqrt(fit_coeffs_cov[i,:,j])*scale_dom, label=label) 
        #ax.plot(plt_dom[:-1], lg_CN[:,j], label=label)
      #pcm = ax.pcolormesh(X, Y, lg_CN,#/np.expand_dims(scales, axis=0),
      #    cmap='seismic', vmax=scale, vmin=-1*scale)
      #    #norm=colors.SymLogNorm(linthresh=1e-4, linscale=1e-2,
      #    #  vmax=scale, vmin=-1*scale))
      ax.set_xlabel("Basis [L]")
      if FLAGS.experiment == "UED":
        ax.set_ylabel("q $[\AA^{-1}]$")
      elif FLAGS.experiment == "LCLS":
        ax.set_ylabel("Energy [eV]")
    
    if lg_CN.shape[-1] == 1:
      ax.set_xlim([plt_dom[0], plt_dom[-2]])
    else:
      ax.set_xlim([plt_dom[0], plt_dom[-2]])
      #fig.colorbar(pcm, ax=ax)
      #ax.set_xticks(np.arange(FLAGS.Nbases)*2)
    ax.legend()


    """
    yt = plt_dom[-1] + 0.05*(plt_dom[-1] - plt_dom[0])
    for ii, scl in enumerate(scales):
      ax.text(ii*2, yt, r"$\times$ {0:.5g}".format(scl),
          horizontalalignment='center')
    """

    plt.tight_layout()
    fig.savefig(os.path.join(plot_folder,
        "fit_coefficients_lg-{}.png".format(fit_lgInds[i])))
    plt.close()


    # Coefficient Signal to Noise
    SN = np.abs(lg_C)\
        /np.sqrt(np.abs(fit_coeffs_cov[i,:,cov_inds,cov_inds])).transpose()
    #h5_output.create_dataset("fit_coeff_signal2noise", data=SN)

    """
    fig_sn, ax_sn = plt.subplots()
    print("SN", SN)
    pcm_sn = ax_sn.pcolormesh(X, Y, SN,
        norm=colors.LogNorm())
    #cmap='seismic', vmax=scale, vmin=-1*scale)
    ax_sn.set_xlabel("Basis [L]")
    if FLAGS.experiment == "UED":
      ax_sn.set_ylabel("Q $[\AA^{-1}]$")
    elif FLAGS.experiment == "LCLS":
      ax_sn.set_ylabel("Energy [eV]")
    fig_sn.colorbar(pcm_sn, ax=ax_sn)
    ax_sn.set_xticks(np.arange(FLAGS.Nbases)*2)
    
    plt.tight_layout()
    fig_sn.savefig(os.path.join(plot_folder,
        "fit_coefficients_SN{}_lg-{}.png".format(reg_suffix[rg], 2*i)))
    plt.close()
    """


    fig_sn, ax_sn = plt.subplots()
    if FLAGS.setBases is not None:
      for isn in cov_inds:
        ax_sn.plot(Y[:-1,0], SN[:,isn],
          label="Time Basis {}".format(FLAGS.setBases[isn] + fit_lgInds[i]))
    else:
      for isn in cov_inds[1:]:
        ax_sn.plot(Y[:-1,0], SN[:,isn], label="Time Basis {}".format(fit_lgInds[isn]))
    ax_sn.legend()
    ax_sn.set_yscale('log')
    ax_sn.set_xlim([Y[0,0], Y[-1,0]])
    ax_sn.set_ylabel("Signal to Noise")
    if FLAGS.experiment == "UED":
      ax_sn.set_xlabel('Q [$\AA^{-1}$]')
    elif FLAGS.experiment == "LCLS":
      ax_sn.set_xlabel('Energy [eV]')
    fig_sn.tight_layout()
    fig_sn.savefig(os.path.join(
        plot_folder, "fit_coefficients_SN_LO_lg-{}.png".format(fit_lgInds[i])))
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
  #####  Plotting Time STD Measurement  #####
  #with open(time_std_fileName, "rb") as file:
  #  save_time_results = pl.load(file)
      
  #chiSq_time_std, time_std_range,\
  #time_std_temps, time_std_intss = save_time_results
  #  t0_chiSq      = h5["t0_chiSq"]
  #  t0_fit_times  = h5["t0_fit_std_times"]
  #  t0_temps_chiSq= h5["t0_temps_chiSq"]
  #  t0_ints_chiSq = h5["t0_ints_chiSq"]

  #time_std_range_centered = np.array(time_std_range) - fit_t0
  #keep_inds = np.where((time_std_range_centered > -1*FLAGS.range_STD_time_cut) 
  #    & (time_std_range_centered < FLAGS.range_STD_time_cut))[0]
  #t0_chiSq = np.array(t0_chiSq)[keep_inds]
  #t0_fit_times = t0_fit_times[keep_inds]

  def fitFxn(x, c, t0, off):
    return c*(x + t0)**2 + off

  fig, ax = plt.subplots()
  ax.plot(t0_fit_times - t0_time_shift, t0_chiSq, '.k')
  ax.plot(t0_fit_times - t0_time_shift, 
      fitFxn(t0_fit_times, 
        (1./fit_t0_std)**2, 
        t0_time_shift, t0_min_chiSq), '-b')
  ax.set_xlim([t0_fit_times[0], t0_fit_times[-1]])
  ax.set_xlabel("Time [ps]")
  ax.set_ylabel('$\chi^2$')
  ax.text(0.5, 0.9, "Best Fit: {0:.6g} $\pm$ {1:.3g} ps".format(
        fit_t0, fit_t0_std),
      fontsize=15, fontweight='bold',
      transform=ax.transAxes, ha='center', va='center')
  plt.tight_layout()
  fig.savefig(os.path.join(plot_folder,
      "time_std_chiSq{}.png".format(suffix)))
  plt.close()

  fig, ax1 = plt.subplots()

  ax1.set_xlabel("Time [ps]")
  ax1.set_ylabel("Temperature [K]", color="r")
  #ax1.plot(time_std_range - np.mean(np.array(time_std_range)),
  print(t0_fit_times.shape, t0_temps_chiSq.shape)
  ax1.plot(t0_fit_times, t0_temps_chiSq, '-r')

  ax2 = ax1.twinx()
  ax2.set_ylabel("Intensity $[10^{12} W/cm^2]$")
  #ax2.plot(time_std_range - np.mean(np.array(time_std_range)),
  ax2.plot(t0_fit_times, t0_ints_chiSq, '-k')
  fig.savefig(os.path.join(plot_folder, "time_std_temp_ints{}.png".format(suffix)))
  plt.close()

  h5_output.close()


  sys.exit(0)
  ################################
  #####  Calculate Time STD  #####
  ################################


  time_std_temperatures = temperatures
  time_std_intensities  = intensities
  """
  if FLAGS.experiment == "UED":
    time_std_intensities = np.concatenate([
        np.arange(0.05, 1.95, 0.05),
        np.arange(2, 
          FLAGS.ints_high + FLAGS.ints_delta/2,
          FLAGS.ints_delta)])
  """

  time_std_chiSq_fileName = os.path.join(output_prefix,
      "{}_time_std_chiSq{}.pl".format(FLAGS.experiment, suffix))
  chiSq_time_std, time_std_range, time_std_temps, time_std_intss =\
      [], [], [], []

  if os.path.exists(time_std_chiSq_fileName):
    with open(time_std_chiSq_fileName, "rb") as file:
      chiSq_time_std, time_std_range, time_std_temps, time_std_intss =\
          pl.load(file)
  else:

    std_times = np.arange(fit_params["start_time"] - FLAGS.range_STD_time,
        fit_params["start_time"] + FLAGS.range_STD_time + FLAGS.step_STD_time/2,
        FLAGS.step_STD_time)
    min_temp_ind = np.argmin(np.abs(fit_params["temp"] - time_std_temperatures))
    min_ints_ind = np.argmin(np.abs(fit_params["intensity"] - time_std_intensities))
    data_weights_std = np.zeros_like(data_weights)
    data_weights_std[:,1,:] = 1.
  
    pool = Pool(processes=1)
    results = pool.map(partial(time_std_pool,
        data=data,
        sem=sem,
        weights=weights,
        best_temp_ind=fit_params["temp_ind"],
        best_ints_ind=fit_params["ints_ind"],
        data_weights=data_weights_std,
        reg_scale=FLAGS.reg_scale,
        temperatures=time_std_temperatures,
        intensities=time_std_intensities,
        sim_times=sim_times,
        time_delta=FLAGS.fit_delta_time,
        t_cut_low=FLAGS.t_cut_low,
        t_cut_high=FLAGS.t_cut_high,
        experiment=FLAGS.experiment,
        basis_dir=FLAGS.basis_dir,
        fName_suffix=fName_suffix,
        fit_lgInds=fit_lgInds),
        std_times)

    for res in results:
      time, minChiSq, minInds = res
      time_std_range.append(time)
      chiSq_time_std.append(minChiSq)
      time_std_temps.append(time_std_temperatures[minInds[0]])
      time_std_intss.append(time_std_intensities[minInds[1]])

    saved_results = [chiSq_time_std, time_std_range, time_std_temps, time_std_intss]
    # Save ChiSq
    with open(time_std_chiSq_fileName, "wb") as file:
      pl.dump(saved_results, file)

  # Fitting for time std

  def fitFxn(x, c, t0, off):
    return c*(x + t0)**2 + off

  mInd = np.argmin(chiSq_time_std)
  time_std_range_centered = np.array(time_std_range) - time_std_range[mInd]
  keep_inds = np.where((time_std_range_centered > -1*FLAGS.range_STD_time_cut) 
      & (time_std_range_centered < FLAGS.range_STD_time_cut))[0]
  chiSq_time_std = np.array(chiSq_time_std)[keep_inds]
  time_std_range_centered = time_std_range_centered[keep_inds]
  #time_std_range_centered *= 1000
  popt, pcov = curve_fit(fitFxn, time_std_range_centered, chiSq_time_std, 
      [1, 0, np.amin(chiSq_time_std)])
  print("BEST RESULTS")
  print("\tBest Time: {} +/- {}".format(time_std_range[mInd], 1./np.sqrt(popt[0])))
  print(popt)


  #####  Plotting  #####
  fig, ax = plt.subplots()
  ax.plot(time_std_range_centered, chiSq_time_std, '.k')
  ax.plot(time_std_range_centered, fitFxn(time_std_range_centered, *popt), '-b')
  ax.set_xlim([time_std_range_centered[0], time_std_range_centered[-1]])
  ax.set_xlabel("Time [ps]")
  ax.set_ylabel('$\chi^2$')
  ax.text(0.5, 0.9, "Best Fit: {0:.6g} $\pm$ {1:.3g} ps".format(
        time_std_range[mInd], 1./np.sqrt(popt[0])), 
      fontsize=15, fontweight='bold',
      transform=ax.transAxes, ha='center', va='center')
  plt.tight_layout()
  fig.savefig(os.path.join(plot_folder, "time_std_chiSq{}.png".format(suffix)))
  plt.close()

  fig, ax1 = plt.subplots()

  ax1.set_xlabel("Time [ps]")
  ax1.set_ylabel("Temperature [K]", color="r")
  #ax1.plot(time_std_range - np.mean(np.array(time_std_range)),
  ax1.plot(time_std_range,
      time_std_temps, '-r')

  ax2 = ax1.twinx()
  ax2.set_ylabel("Intensity $[10^{12} W/cm^2]$")
  #ax2.plot(time_std_range - np.mean(np.array(time_std_range)),
  ax2.plot(time_std_range,
      time_std_intss, '-k')
  fig.savefig(os.path.join(plot_folder, "time_std_temp_ints{}.png".format(suffix)))
  plt.close()


if __name__ == '__main__':
  app.run(main)

