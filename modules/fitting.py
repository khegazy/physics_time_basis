import sys
import numpy as np
import jax as jax
import jax.numpy as jnp


def fit_basis_analytic(bases, data, weights):

  return normal_equation(basis, data, weights)


def normal_equation(X, Y, W):

  W = np.expand_dims(W, axis=-1)\
      *np.expand_dims(np.expand_dims(np.eye(W.shape[-1]), 0), 0)
  W = np.expand_dims(W, axis=0)
  X = np.expand_dims(np.expand_dims(X, 1), 1)
  #print(X.shape, W.shape, Y.shape)
  m = np.matmul(W, X)
  norm = np.matmul(X.transpose((0,1,2,4,3)), np.matmul(W, X))
  norm = np.linalg.inv(norm)

  Y = np.expand_dims(np.expand_dims(Y, axis=-1), 0)
  #print(np.matmul(W, Y).shape)
  numerator = np.matmul(X.transpose((0,1,2,4,3)), np.matmul(W, Y))
  fit = np.matmul(norm, numerator)

  return np.reshape(fit, list(fit.shape)[:-1])



def normal_equation_reg(X, Y, W, reg_scale):

  W = jnp.expand_dims(W, axis=-1)\
      *jnp.expand_dims(jnp.expand_dims(jnp.eye(W.shape[-1]), 0), 0)
  #W = jnp.expand_dims(W, axis=0)
  X = jnp.expand_dims(X, 1)
  #print("1",X.shape, W.shape, Y.shape)
  #m = jnp.matmul(W, X)
  #norm = np.matmul(X.transpose((0,1,2,4,3)), np.matmul(W, X))
  norm = jnp.matmul(X.transpose((0,1,3,2)), jnp.matmul(W, X))
  #x_overlap = jnp.abs(jnp.matmul(X.transpose((0,1,3,2)), X))
  #x_overlap *= 1 - jnp.eye(X.shape[-1])
  
  reg_overlap = jnp.expand_dims(jnp.expand_dims(
      jnp.eye(X.shape[-1]), 0), 0)
  #reg_overlap = jnp.matmul(X.transpose((0,1,3,2)), X)
  #reg_overlap *= 2*(jnp.eye(X.shape[-1])-0.5)

  #    jnp.expand_dims(theta_sign, -1)*x_overlap*jnp.expand_dims(theta_sign, 0)
  reg_scale = jnp.expand_dims(jnp.expand_dims(
      reg_scale/jnp.var(Y, axis=-1), -1), -1)

  denom = jnp.linalg.inv(norm + reg_scale*reg_overlap)
  #denom = jnp.linalg.inv(norm)

  Y = jnp.expand_dims(Y, axis=-1)
  #print("2", jnp.matmul(W, Y).shape)
  numerator = jnp.matmul(X.transpose((0,1,3,2)), jnp.matmul(W, Y))
  #print("asdf", denom.shape, numerator.shape)
  fit = jnp.matmul(denom, numerator)
  
  return jnp.reshape(fit, list(fit.shape)[:-1])

normal_equation_reg_vmap = jax.vmap(jax.jit(normal_equation_reg),
    in_axes=(0, None, None, None))

def loss_reg(theta, X, Y, W, data_weights, reg_scale):

  #W = jnp.expand_dims(W, axis=-1)*jnp.eye(W.shape[-1])
  #    *jnp.expand_dims(jnp.expand_dims(jnp.eye(W.shape[-1]), 0), 0)
  #W = jnp.expand_dims(jnp.expand_dims(W, axis=0), axis=0)
  #X = jnp.expand_dims(jnp.expand_dims(X, 0), 0)
  #print("shapes",theta.shape,X.shape, W.shape, Y.shape)
  #m = jnp.matmul(W, X)
  #norm = np.matmul(X.transpose((0,1,2,4,3)), np.matmul(W, X))
  l_chiSq = jnp.einsum('jia,jka->jki', X, theta) - Y
  #print("lch", np.sum(np.isinf(np.array(l_chiSq))))
  #print("l shape", l_chiSq.shape)
  chiSq = jnp.mean(l_chiSq*W*l_chiSq, axis=-1)
  #x_overlap = jnp.matmul(X.transpose(), X)
  ##x_overlap *= 1 - jnp.eye(X.shape[-1])
  reg_overlap = jnp.sum(theta**2, axis=-1)
  #x_overlap *= 2*(jnp.eye(X.shape[-1]) - 0.5)
  #reg_overlap = jnp.einsum('jka,ab,jkb->jk',
  #    theta, x_overlap, theta)

  reg_scale = jnp.expand_dims(jnp.expand_dims(
      reg_scale/jnp.var(Y, axis=-1), -1), -1)
  #print("shapessss", chiSq.shape,reg_overlap.shape, x_overlap.shape)
  data_weights = jnp.var(Y, axis=-1)
  loss = np.sum((data_weights*(chiSq + reg_scale*reg_overlap))[0:])\
      /(chiSq.shape[1]*np.sum(data_weights[0:]))
  #loss = chiSq + reg_scale*reg_overlap
  #loss = reg_scale*reg_overlap
  return loss

loss_reg_vmap = jax.vmap(loss_reg,
    in_axes=(0, 0, None, None, None, None))


"""
def normal_equation(X, Y, W):

  print("inp", X.shape, Y.shape, W.shape)
  W = np.expand_dims(W, axis=-1)\
      *np.expand_dims(np.expand_dims(np.eye(W.shape[-1]), 0), 0)
  print("Ws", W.shape)
  X_overlap = np.matmul(X.transpose(), X)
  X_overlap_diag = np.zeros_like(X_overlap)
  inds = np.arange(X_overlap.shape[0])
  X_overlap_diag[inds,inds] = X_overlap[inds,inds]

  norm = np.matmul(X.transpose(), np.matmul(W, X))
  norm = np.linalg.inv(norm)

  Y = np.expand_dims(Y, axis=-1)
  numerator = np.matmul(X.transpose(), np.matmul(W, Y))
  fit = np.matmul(norm, numerator)
  sys.exit(0)
  
  return np.reshape(fit, list(fit.shape)[:-1])
"""

def get_loss_fxn(X, Y, W):
  W_shape = W.shape
  X_shape = X.shape
  print("Ws", W.shape)

  def loss_fxn(theta, X, Y, W):
   
    #a = jnp.ones([9, 5, 7, 4])
    #c = jnp.ones([9, 5, 4, 3])
    #b = jnp.matmul(a, c)
    print(theta.shape, X.shape, Y.shape, W.shape)
    W = jnp.expand_dims(W, axis=-1)\
        *jnp.expand_dims(jnp.expand_dims(jnp.eye(W_shape[-1]), 0), 0)
    W = jnp.expand_dims(W, axis=0)
    #X = jnp.expand_dims(jnp.expand_dims(X, 1), 1)
    #m = jnp.transpose(X, (0,1,2,4,3))
    X_overlap = np.einsum("iab,icb->iac", X, X)*(jnp.eye(X_shape[-1]) + 1 % 2)
    print(X_overlap.shape)
    L_overlap = np.einsum("ijka,iab,ijkb->ijk", theta, X_overlap, theta)
    l_chiSq = np.matmul(X, theta) - Y
    L = np.matmul(l_chiSq, np.matmul(W, l_chiSq)) + L_overlap #jnp.sum(theta) + jnp.sum(W) + jnp.sum(X)
    """
    X_overlap = jnp.matmul(X.transpose((0,1,2,4,3)), X)*(jnp.eye(X_shape[0]) + 1 % 2)

    L_overlap = jnp.matmul(jnp.abs(theta), 
        jnp.matmul(jnp.abs(X_overlap), jnp.theta))
    l_chiSq = jnp.matmul(X, theta) - Y
    L = jnp.matmul(l_chiSq, jnp.matmul(W, l_chiSq)) + L_overlap
    """
    #L = np.sum(theta)

    return L

  return loss_fxn

