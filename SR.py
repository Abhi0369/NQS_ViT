import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from jax.scipy.linalg import inv,solve
from jax.scipy.sparse.linalg import cg
from functools import partial

def reshape_update(flat_update, params):
    """
    Reshape a flat array of parameter updates into the structured format of `params`.
    
    Args:
        flat_update: A 1D JAX array containing concatenated updates for all parameters.
        params: A nested dictionary of model parameters.
    
    Returns:
        A nested dictionary of updates structured like `params`.
    """
    update_dict = {}
    i = 0  # Pointer to track position in flat_update

    def reshape_recursive(sub_params, sub_update_dict, pointer):
        """
        Recursively traverse the nested parameter dictionary to reshape updates.
        """
        for key, value in sub_params.items():
            if isinstance(value, dict):  # If the value is still a dict, recurse further
                sub_update_dict[key] = {}
                pointer = reshape_recursive(value, sub_update_dict[key], pointer)
            else:  # We've hit an actual parameter array, reshape
                size = value.size  # Determine size of parameter
                shape = value.shape  # Determine shape of parameter
                sub_update_dict[key] = flat_update[pointer:pointer+size].reshape(shape)
                pointer += size  # Move the pointer
        return pointer  # Return updated pointer

    reshape_recursive(params, update_dict, i)
    return update_dict

@partial(jit, static_argnums=(0,))
def compute_log_derivatives(model, params, samples):
    """
    Compute the log psi derivatives for given samples.
    
    Args:
        model: NQS model.
        params: The parameters of the model.
        samples: An array of input samples.
    
    Returns:
        An array of log psi derivatives for the samples.
    """
    def single_sample_log_derivative(sample):
        """Compute the log psi derivative for a single sample."""
        sample = jnp.expand_dims(sample, 0)
        grads = grad(lambda p: model.apply({'params': p}, jnp.expand_dims(sample,-1)), holomorphic=True)(params)
        grads_processed = jax.tree_map(lambda g: jnp.concatenate([jnp.ravel(g.real), jnp.ravel(g.imag)]), grads)
        return jnp.concatenate([g for g in jax.tree_util.tree_leaves(grads_processed)])
    
    O_k =vmap(single_sample_log_derivative)(samples)
    
    return O_k

@partial(jit, static_argnums=(0,))
def sr(model, params, samples, Eloc, learning_rate, reg_coef=1e-4):
    """
    Perform a stochastic reconfiguration (SR) update on the model parameters.
    
    Args:
        model: The model to update.
        params: Current parameters of the model.
        samples: Input samples for computing log derivatives.
        Eloc: Local energies for the samples.
        learning_rate: Learning rate for the update.
        reg_coef: Regularization coefficient for stability.
    
    Returns:
        new_params: Updated model parameters.
        delta: The parameter update vector.
    """
    O_k = compute_log_derivatives(model, params, samples)
    O_k_mean = jnp.mean(O_k, axis=0, keepdims=True)
    O_k_p = O_k - O_k_mean
    
    S_kk = jnp.einsum('ik,kj->ij',jnp.conjugate(O_k_p).T, O_k_p) / O_k.shape[0]
    F_p = jnp.mean(Eloc[:, None] * O_k - jnp.mean(Eloc) * O_k, axis=0)
    S_kk_reg = S_kk + reg_coef * jnp.eye(S_kk.shape[0])

    # Solve the linear system using conjugate gradient
    delta, _ = cg(S_kk_reg, F_p)

    # Apply the update
    update = -learning_rate * delta
    update_dict = reshape_update(update, params)
    new_params = jax.tree_map(lambda p, u: p + u, params, update_dict)

    return new_params, delta
