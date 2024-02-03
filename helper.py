import jax
from jax import jit, vmap, random
from functools import partial
import jax.numpy as jnp


def generate_state(shape,subkey):
    n = shape[1] * shape[2]  # total number of spins
    if n % 2 != 0:
        raise ValueError("The total number of spins must be even to have zero magnetization")

    # Create an array with equal number of -1 and +1
    half_n = n // 2
    state = jnp.concatenate([jnp.ones(half_n), -jnp.ones(half_n)])

    # Shuffle the array to randomize the spin configuration
    key, subkey = random.split(subkey)
    state = random.permutation(subkey, state)

    # Reshape the array to the desired shape
    state = state.reshape(shape)

    return state


