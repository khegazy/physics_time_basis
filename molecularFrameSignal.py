from absl import app
from absl import flags
from absl import logging

import sys, os, glob
import pickle as pl
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from modules.modules import *
from modules.flags_UED import get_flags
FLAGS = get_flags()


def main(argv):

  # Get Data
  FLAGS.dom_cut_low, FLAGS.dom_cut_high = None, None
  data, doms, sem, sn_map = get_data(FLAGS)
  dom, plt_dom = doms

  # Get Save Fit Parameters
  suffix = "_Nbases-{}_Temp-{}-{}-{}_Ints-{}-{}-{}".format(
      FLAGS.Nbases,
      FLAGS.temp_low, FLAGS.temp_high,
      int((FLAGS.temp_high-FLAGS.temp_low)/FLAGS.temp_delta + 1),
      FLAGS.ints_low, int(100*FLAGS.ints_high)/100.0,
      int((FLAGS.ints_high-FLAGS.ints_low)/FLAGS.ints_delta + 1))
  fitted_params_fileName = "./output/{}_best_fit_parameters{}.pl".format(
      FLAGS.experiment, suffix)
  chiSq_fileName = "./output/{}_chiSq{}.npy".format(
      FLAGS.experiment, suffix)
  start_times_fileName = "./output/{}_best_times{}.npy".format(
      FLAGS.experiment, suffix)
  C_coeffs_fileName = "./output/{}_C_coeffs{}.npy".format(
      FLAGS.experiment, suffix)
  best_C_coeffs_fileName = "./output/{}_best_C_coeffs{}.npy".format(
      FLAGS.experiment, suffix)
  with open(fitted_params_fileName, "rb") as file:
    fit_params = pl.load(file)
  with open(chiSq_fileName, "rb") as file:
    chiSq = np.load(file)
  with open(start_times_fileName, "rb") as file:
    best_start_times = np.load(file)
  with open(C_coeffs_fileName, "rb") as file:
    all_C_coeffs = np.load(file)
  with open(best_C_coeffs_fileName, "rb") as file:
    C_coeffs = np.load(file)


  # Evaluation Orientation
  theta = 0
  phi = 0
  Nlg, Nd, Nb = C_coeffs.shape
  
  X,Y = np.meshgrid(np.arange(Nb+1), np.arange(Nd+1))
  for i in range(Nlg):
    plt.pcolormesh(X, Y, C_coeffs[i,:,:])
    plt.colorbar()
    plt.savefig("c{}.png".format(i))
    plt.close()
  print("SHAPE", C_coeffs.shape, dom.shape)
  assert (Nb == FLAGS.Nbases)
  assert (Nd == len(dom))
  lgndr_l = np.arange(Nlg)*2
  bases_l = np.arange(Nb)*2
  bases_m = np.zeros(Nb)
  Bs_aligned = sp.special.sph_harm(bases_m, bases_l, phi, theta)
  Bs_aligned = np.expand_dims(
      np.expand_dims(Bs_aligned, axis=0),
      axis=0)
  #C_coeffs_interp = interp1d(np.arange(Nd), C_coeffs)
  #C_coeffs_det = C_coeffs_interp(R)
  beta = np.sum(Bs_aligned*C_coeffs, axis=-1)
  #ls = np.expand_dims(
  #    np.expand_dims(ls), axis=-1),
  #    axis=-1)
  #ms = np.expand_dims(
  #    np.expand_dims(ms), axis=-1),
  #    axis=-1)

  Nbins = int(Nd*1.05)*2 + 1
  signal = np.zeros((Nbins, Nbins))
  X,Y = np.meshgrid(
      np.arange(Nbins) - Nbins//2,
      np.arange(Nbins) - Nbins//2)
  theta_det = np.arctan2(X, -1*Y)
  R = np.sqrt(X**2 + Y**2) - (Nbins//2 - Nd)
  mask = np.ones_like(R).astype(bool)
  mask[R>Nd-1] = False
  mask[R<0] = False
  R[R>Nd-1] = Nd - 1
  R[R<0] = 0
  print("beta shape",beta.shape)
  beta_interp = interp1d(np.arange(Nd), beta, axis=-1)

  print(lgndr_l)
  legendres = np.real(sp.special.sph_harm(
      np.zeros((len(lgndr_l),1,1)),
      np.expand_dims(np.expand_dims(lgndr_l, axis=-1), axis=-1),
      0,
      np.expand_dims(theta_det, axis=0))\
      /np.sqrt((2*np.expand_dims(
        np.expand_dims(lgndr_l, axis=-1), axis=-1) + 1)/(4*np.pi)))
  print("max",np.amax(R))
  beta_det = beta_interp(R)
  beta_det[0,:,:] = 0
  print(beta_det.shape)
  #print(beta_det)
  print(beta_det.dtype, legendres.dtype)
  for i,bt in enumerate(beta_det):
    plt.pcolormesh(X,Y,np.real(bt))
    plt.colorbar()
    plt.savefig("beta{}.png".format(i))
    plt.close()
  
  for i,bt in enumerate(beta_det*legendres):
    plt.pcolormesh(X,Y,np.real(bt))
    plt.colorbar()
    plt.savefig("betaleg{}.png".format(i))
    plt.close()

  signal = np.sum(beta_det*legendres, axis=0)
  signal = np.real(signal)

  fig, ax = plt.subplots()
  pcm = ax.pcolormesh(X, Y, theta_det)
  fig.colorbar(pcm, ax=ax)
  fig.savefig("theta.png")

  fig, ax = plt.subplots()
  pcm = ax.pcolormesh(X, Y, signal)
  fig.colorbar(pcm, ax=ax)
  fig.savefig("sig.png")
  




if __name__ == '__main__':
  app.run(main)
