import numpy as np
import csv
import os, glob, sys, shutil
import time
import h5py
import pickle as pl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.interpolate import interp1d





def get_bases(FLAGS, fit_lgInds, data_shape,
    normalize=True, subtract_mean=True,
    time_cuts=None, folderName=None):

  if folderName is None:
    folderName = os.path.join(
      FLAGS.basis_dir,
      FLAGS.experiment,
      FLAGS.basis_sub_dir)
  files = "{}/*.dat".format(folderName)
  
  # Get array of which bases to gather
  bases_Linds = fit_lgInds
  if FLAGS.setBases is not None:
    bases_Linds = np.unique(np.expand_dims(fit_lgInds, 0)\
        + np.expand_dims(FLAGS.setBases, -1).flatten())
    bases_Linds = bases_Linds[bases_Linds >=0]

  basisList = [None for x in range(len(bases_Linds))]
  normsList = [None for x in range(len(bases_Linds))]

  # Gather bases
  for fl in glob.glob(files):
    L = int(fl[fl.find("_L-")+3:fl.find("_M-")])
    ind = np.where(bases_Linds == L)[0]

    # Is L in the index list
    if len(ind) != 1:
      continue
    ind = ind[0]

    # Get the basis and normalize
    with open(fl, "rb") as file:
      if time_cuts is None:
        basisList[ind] = np.fromfile(file, np.double)
      else:
        basisList[ind] = np.fromfile(file, np.double)\
            [time_cuts[0]:time_cuts[1]]

      if subtract_mean:
        if L != 0:
          basisList[ind] -= np.mean(basisList[ind])
      #print("norm",L,np.sqrt(np.sum(bases[L]**2)), np.amax(np.abs(bases[L])))
      normsList[ind] = np.sqrt(np.sum(basisList[ind]**2))
      if normalize:
        basisList[ind] /= normsList[ind]
  
  allBases = np.array(basisList)
  allNorms = np.array(normsList)
  setBases = None
  if FLAGS.setBases is not None:
    setBases = np.ones((data_shape[0], len(FLAGS.setBases), allBases.shape[1]))
    #bases[:,0] = 0
    for i,lg in enumerate(fit_lgInds):
      inds = []
      for ii in FLAGS.setBases:
        inds.append(np.where(bases_Linds == np.max(lg+ii, 0))[0])
      setBases[i,:,:] = allBases[np.concatenate(inds),:]

    return allBases, np.array(normsList), setBases
  else:
    return allBases, np.array(normsList), np.expand_dims(allBases, axis=0)




def get_data(FLAGS, subtract_mean=True, plot_data=False, crops=None):

  if FLAGS.experiment == "UED":

    fName = os.path.join(FLAGS.data_dir, FLAGS.experiment,
        "legendre_coeffs.h5")
    with h5py.File(fName, 'r') as hf:
      data  = hf["coefficients"][:]
      std   = hf["std"][:]
      dom   = hf["q"][:]
    legendre_fits = np.arange(data.shape[0])*2



    dom_filter = np.ones_like(dom).astype(bool)
    if FLAGS.dom_cut_low is not None:
      dom_filter[dom<FLAGS.dom_cut_low] = False
    if FLAGS.dom_cut_high is not None:
      dom_filter[dom>FLAGS.dom_cut_high] = False
    dom = dom[dom_filter]
    plt_dom = np.concatenate([dom, np.array([2*dom[-1] - dom[-2]])])


    """
    fName = os.path.join(FLAGS.data_dir, FLAGS.experiment, "old_analysis_2015",
        "legendre_coefficient_SEM[31,70,40].dat")
    with open(fName, "rb") as file:
      sem = np.fromfile(file, np.double)
    sem = np.reshape(sem, (31,70,40))[np.arange(FLAGS.Nfits)*2,:,:]
    """

    """
    data *= 1e7
    sem *= 1e7
    """
    legendre_fits = legendre_fits[:4]
    std = std[:4,dom_filter,FLAGS.t_cut_low:FLAGS.t_cut_high]
    data = data[:4,dom_filter,FLAGS.t_cut_low:FLAGS.t_cut_high]
    if subtract_mean:
      data -= np.expand_dims(np.mean(data, axis=-1), axis=-1)
    
    scl = [40, 13, 2.5, 1.5, 1, 0.5]
    scl = [15 for x in range(6)]
    save_folder = os.path.join("plots", FLAGS.experiment)
    if FLAGS.setBases is not None:
      save_folder = os.path.join(save_folder, "setBases")
    else:
      save_folder = os.path.join(save_folder, "NB{}".format(FLAGS.Nbases))
    X,Y = np.meshgrid(np.arange(data.shape[-1])+1, plt_dom)
    if not os.path.exists(save_folder):
      os.makedirs(save_folder)

    if plot_data:
      for i,dat in enumerate(data):
        pcm = plt.pcolormesh(X, Y, dat, vmin=-1*scl[i], vmax=scl[i], cmap="seismic")
        plt.ylabel("Q [$\AA^{-1}$]")
        plt.xlabel("Time Bins [{} fs]".format(1000*FLAGS.fit_delta_time))
        plt.colorbar(pcm)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "fit_data_lg-{}.png".format(2*i)))
        plt.close()
  else:

    data  = []
    std   = []
    for lg in np.arange(FLAGS.Nfits)*2:
      #lg = 2
      lgndr = []
      _std  = []
      data_fName = os.path.join(FLAGS.data_dir, FLAGS.experiment,
          FLAGS.data_template.format(lg))
      std_fName = os.path.join(FLAGS.data_dir, FLAGS.experiment,
          FLAGS.std_template.format(lg))
      print("name ", data_fName, std_fName)
      with open(data_fName, "r") as file:
        raw = list(csv.reader(file, delimiter='\t'))
        for i in range(1, len(raw)):
          if len(raw[i]):
            raw_row = [np.float(k) for k in raw[i] if k != '']
            lgndr.append(np.expand_dims(np.array(raw_row), axis=0))
            #lgndr.append(np.expand_dims(np.array(raw_row)[20:-10], axis=0))

      with open(std_fName, "r") as file:
        raw = list(csv.reader(file, delimiter='\t'))
        for i in range(1, len(raw)):
          if len(raw[i]):
            raw_row = [np.float(k) for k in raw[i] if k != '']
            _std.append(np.expand_dims(np.array(raw_row), axis=0))

      data.append(np.expand_dims(np.concatenate(lgndr, axis=0), axis=0))
      std.append(np.expand_dims(np.concatenate(_std, axis=0), axis=0))

    data  = np.concatenate(data, axis=0)
    std   = 1./np.sqrt(np.concatenate(std, axis=0))

    dom = np.arange(data.shape[1]) + 471
    dom_filter = np.ones_like(dom).astype(bool)
    if FLAGS.dom_cut_low is not None:
      dom_filter[dom<FLAGS.dom_cut_low] = False
    if FLAGS.dom_cut_high is not None:
      dom_filter[dom>FLAGS.dom_cut_high] = False
    dom = dom[dom_filter]
    plt_dom = np.concatenate([dom, np.array([2*dom[-1] - dom[-2]])])

    data  = data[:,dom_filter,FLAGS.t_cut_low:FLAGS.t_cut_high]
    std   = std[:,dom_filter,FLAGS.t_cut_low:FLAGS.t_cut_high]

    if plot_data:
      if FLAGS.experiment == "UED":
        scl = np.array([2, 15, 15, 1.5, 1, 0.5, 0.3, 0.2])*0.004
      else:
        scl = np.array([2, 15, 15, 15, 15, 15, 15, 15])*0.0005
      save_folder = os.path.join("plots", FLAGS.experiment,
          FLAGS.basis_sub_dir, "NB{}".format(FLAGS.Nbases))
      X,Y = np.meshgrid(np.arange(data.shape[-1])+1, plt_dom)
      if not os.path.exists(save_folder):
        os.makedirs(save_folder)

      for i,(dat,sm) in enumerate(zip(data,std)):
        pcm = plt.pcolormesh(X, Y, dat, vmin=-1*scl[i], vmax=scl[i], cmap="seismic")
        plt.ylabel("Energy [eV]")
        plt.xlabel("Time Bins [{} fs]".format(1000*FLAGS.fit_delta_time))
        plt.colorbar(pcm)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "fit_data_lg-{}.png".format(2*i)))
        plt.close()
        print("Plotting",os.path.join(save_folder, "fit_data_lg-{}.png".format(2*i)))

        pcm = plt.pcolormesh(X, Y, sm, cmap="Blues")
        plt.ylabel("Energy [eV]")
        plt.xlabel("Time Bins [{} fs]".format(1000*FLAGS.fit_delta_time))
        plt.colorbar(pcm)
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "fit_std_lg-{}.png".format(2*i)))
        plt.close()



  sn_map = np.abs(data)/std

  return data, (dom, plt_dom, dom_filter), std, legendre_fits, sn_map


