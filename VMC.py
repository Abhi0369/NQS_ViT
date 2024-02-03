import jax
from jax import jit, random, grad, lax, vmap
import jax.numpy as jnp
from Heisenberg_2d import matrix_elements, gen_configs
from functools import partial

@partial(jit, static_argnums=(3,))
def single_move(current_state, key, params, nqs_apply, J):
    """
    Perform a single Monte Carlo move and compute the local energy.

    Args:
        current_state: The current state of the system.
        key: PRNG key for random number generation.
        params: Parameters of the neural quantum state (NQS) model.
        nqs_apply: Function to apply the NQS model.
        J: Coupling constant of the Heisenberg model.

    Returns:
        new_state: The new state after the move.
        key: Updated PRNG key.
        E_local: Local energy of the new state.
    """
    key, subkey = random.split(key)  # Split the key for randomness

    # i,j = random.randint(subkey,shape = (2,),minval=0, maxval=current_state.shape[1])
    # flipped_state = current_state.at[i,j].multiply(-1)

    # Randomly select indices for swapping spins
    i1, j1, i2, j2 = random.randint(subkey, shape=(4,), minval=0, maxval=current_state.shape[1])

    # Perform the swap
    flipped_state = current_state.at[i1, j1].set(current_state[i2, j2])
    flipped_state = flipped_state.at[i2, j2].set(current_state[i1, j1])
    
    # Compute log-psi values for the current and flipped states
    log_psi_current = nqs_apply({'params': params}, jnp.expand_dims(current_state, (0,-1)))
    log_psi_flipped = nqs_apply({'params': params}, jnp.expand_dims(flipped_state, (0,-1)))
    
    # Compute the acceptance probability
    ratio = jnp.square(jnp.abs(jnp.exp(log_psi_flipped - log_psi_current)))
    acceptance_probability = jnp.minimum(ratio, 1.0)
    
    key, subkey = random.split(key)  # Split the key again for randomness
    random_num = random.uniform(subkey)
    
    # Accept or reject the new state based on acceptance probability
    new_state = lax.cond(random_num < acceptance_probability,
                         lambda _: flipped_state,
                         lambda _: current_state,
                         operand=None)
    
    # Compute energy components
    E_diag, spin_flip_loc = matrix_elements(new_state, J)
    new_configs = gen_configs(new_state, spin_flip_loc)

    log_psi_sprime = nqs_apply({'params': params}, jnp.expand_dims(new_configs, (-1)))
    log_psi_s = nqs_apply({'params': params}, jnp.expand_dims(new_state, (0,-1)))
    log_psi_sprime = jnp.where(log_psi_sprime == log_psi_s, -jnp.inf, log_psi_sprime)

    ratio_psi = jnp.exp(log_psi_sprime - log_psi_s)
    E_off = -2 * ratio_psi.sum()
    
    E_local = E_diag + E_off
    
    return new_state, key, E_local

@partial(jit, static_argnums=(1, 2, 3, 6))
def run(initial_state, num_steps, warm_up, nqs_apply, params, key, J):
    """
    Run the Monte Carlo simulation for a given number of steps.

    Args:
        initial_state: The initial state of the system.
        num_steps: Number of Monte Carlo steps/samples to perform.
        warm_up: Number of warm-up steps before sampling.
        nqs_apply: Function to apply the NQS model.
        params: Parameters of the NQS model.
        key: PRNG key for random number generation.
        J: Coupling constant of the Heisenberg model.

    Returns:
        samples: Sampled states after warm-up.
        E_locals: Local energies of the sampled states.
    """
    def warmup_step(carry, _):
        """Performs a single warm-up step."""
        state, key = carry
        state, key, _ = single_move(state, key, params, nqs_apply, J)
        return (state, key), None

    def sampling_step(carry, _):
        """Performs a single sampling step."""
        state, key = carry
        state, key, E_local = single_move(state, key, params, nqs_apply, J)
        return (state, key), (state, E_local)

    # Perform warm-up phase
    (current_state, key), _ = lax.scan(warmup_step, (jnp.squeeze(initial_state), key), None, length=warm_up)

    # Perform sampling phase
    (final_state, final_key), (samples, E_locals) = lax.scan(sampling_step, (current_state, key), None, length=num_steps)

    return samples, E_locals
