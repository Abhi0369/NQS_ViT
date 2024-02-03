import jax.numpy as jnp
import jax
from jax import vmap, jit, grad
from functools import partial
from Heisenberg_2d import matrix_elements, gen_configs
from VMC import run

@partial(jit, static_argnums=(2, 3, 4, 5))
def compute_Elocal(params, initial_state, nqs_apply, Jz, n_sweeps, warm_up, key):
    """
    Compute the local energy for a given state using the Variational Monte Carlo method.

    Args:
        params: Parameters of the variational quantum state (NQS) model.
        initial_state: Initial configuration/state of the system.
        nqs_apply: Function to apply the NQS model.
        Jz: Coupling constant for the z-component in the Heisenberg model.
        n_sweeps: Number of Monte Carlo sweeps for sampling.
        warm_up: Number of warm-up steps before measurements start.
        key: Random key for stochastic operations.

    Returns:
        A tuple containing the loss, mean local energy, and variance of local energy.
    """
    samples, Elocals = run(initial_state, n_sweeps, warm_up, nqs_apply, params, key, Jz)
    mean_Elocal = jnp.mean(Elocals)
    var_Elocal = jnp.var(Elocals)
    log_psi = nqs_apply({'params': params}, jnp.expand_dims(samples, -1))
    loss = 2 * jnp.real(jnp.mean((Elocals - mean_Elocal) * jnp.conj(log_psi)))
    return loss, mean_Elocal, var_Elocal

@partial(jit, static_argnums=(2, 3, 4, 5))
def cost(params, initial_state, nqs_apply, Jz, n_sweeps, warm_up, key):
    """
    Wrapper function to compute the gradient of the mean local energy.

    Args:
        params: Parameters of the variational quantum state (NQS) model.
        initial_state: Initial configuration/state of the system.
        nqs_apply: Function to apply the NQS model.
        Jz: Coupling constant for the z-component in the Heisenberg model.
        n_sweeps: Number of Monte Carlo sweeps for sampling.
        warm_up: Number of warm-up steps before measurements start.
        key: Random key for stochastic operations.

    Returns:
        A tuple containing the loss, mean local energy, variance of local energy, and gradients.
    """
    def mean_Elocal_fn(p):
        loss, Eloc_m, var = compute_Elocal(p, initial_state, nqs_apply, Jz, n_sweeps, warm_up, key)
        return loss

    loss, mean_Elocal, var_Elocal = compute_Elocal(params, initial_state, nqs_apply, Jz, n_sweeps, warm_up, key)
    grads = grad(mean_Elocal_fn, holomorphic=False)(params)
    return loss, mean_Elocal, var_Elocal, grads



#######################################################################################

@partial(jit, static_argnums=(1, 3))
def single_sample_eloc(state, nqs_apply, params, J):
    """
    Compute the local energy for a single sample state.

    Args:
        state: A single configuration/state of the system.
        nqs_apply: Function to apply the NQS model.
        params: Parameters of the variational quantum state (NQS) model.
        J: Coupling constant for the Heisenberg model.

    Returns:
        The local energy of the given state.
    """
    E_diag, spin_flip_loc = matrix_elements(state, J)
    new_configs = gen_configs(state, spin_flip_loc)
    
    log_psi_s = nqs_apply({'params': params}, jnp.expand_dims(state, (0, -1)))
    log_psi_sprime = nqs_apply({'params': params}, jnp.expand_dims(new_configs, (-1)))
    log_psi_sprime = jnp.where(log_psi_sprime == log_psi_s, -jnp.inf, log_psi_sprime)
    
    ratio = jnp.exp(log_psi_sprime - log_psi_s)
    E_off = -2 * ratio.sum()
    E_local = E_diag + E_off
    return E_local

@partial(jit, static_argnums=(1, 3))
def cost2(samples, nqs_apply, params, J):
    """
    Compute the loss for a batch of samples based on their local energies.

    Args:
        samples: A batch of configuration/states of the system.
        nqs_apply: Function to apply the NQS model.
        params: Parameters of the variational quantum state (NQS) model.
        J: Coupling constant for the Heisenberg model.

    Returns:
        The computed loss for the batch of samples.
    """
    Elocs = vmap(single_sample_eloc, in_axes=(0, None, None, None))(samples, nqs_apply, params, J)
    log_psi = nqs_apply({'params': params}, jnp.expand_dims(samples, -1))
    loss = 2 * jnp.real(jnp.mean((Elocs - Elocs.mean()) * jnp.conj(log_psi)))
    return loss
