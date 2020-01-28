import jax as jax
import jax.numpy as np


def fit_basis_analytic(bases, data, weights):

  return normal_equation_vmap(basis, data, weights)


def normal_equation(X, Y, W):

  W = np.eye(W.shape[0])*W
  norm = np.matmul(X.transpose(), np.matmul(W, X))
  norm = np.linalg.inv(norm)

  numerator = np.matmul(X.transpose(), np.matmul(W, Y))

  return np.matmul(norm, numerator)

normal_equation_vmap = jax.vmap(
    jax.vmap(normal_equation, (None, 0, 0)),
    (None, 0, 0))
