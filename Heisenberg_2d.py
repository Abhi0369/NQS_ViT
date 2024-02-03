from jax import vmap, jit, lax
import jax.numpy as jnp
from functools import partial

@partial(jit, static_argnums=(0,))
def compute_bonds(L):
    """
    Compute the neighboring bonds for a square lattice of size LxL.

    Args:
        L: The length of the square lattice.

    Returns:
        A JAX array of shape (num_bonds, 2, 2), where each entry represents
        a bond between two sites [(i1, j1), (i2, j2)].
    """
    bonds = []
    for i in range(L):
        for j in range(L):
            # Identify the right and down neighbors with periodic boundary conditions
            right_neighbor = ((i + 1) % L, j)
            down_neighbor = (i, (j + 1) % L)
            bonds.append([(i, j), right_neighbor])
            bonds.append([(i, j), down_neighbor])

    return jnp.array(bonds, dtype=jnp.int32)

@jit
def matrix_elements(spins, J):
    """
    Calculate the diagonal part of the Hamiltonian matrix and the locations of spin flips.

    Args:
        spins: A JAX array representing the spin configuration.
        J: The coupling constant.

    Returns:
        E_diag: The diagonal energy of the given spin configuration.
        spin_flips: An array indicating the locations of possible spin flips.
    """
    L = spins.shape[1]
    bonds = compute_bonds(L)
    
    # Initialize array for storing spin flip locations
    spin_flips = jnp.zeros((2 * L * L, 4), dtype=jnp.int32)
    
    def body_fun(i, spin_flips):
        """Update spin flips based on the bond configuration."""
        si, sj = bonds[i]
        Si = spins[si[0], si[1]]
        Sj = spins[sj[0], sj[1]]
        diff = jnp.not_equal(Si, Sj)

        # Update spin flips if spins are different
        updated_spin_flips = lax.cond(
            diff,
            lambda _: spin_flips.at[i].set([si[0], si[1], sj[0], sj[1]]),
            lambda _: spin_flips,
            None
        )
        return updated_spin_flips

    # Loop over all bonds to update spin flips
    spin_flips = lax.fori_loop(0, len(bonds), body_fun, spin_flips)

    def single_bond_energy(bond):
        """Calculate energy for a single bond."""
        si, sj = bond
        Si = spins[si[0], si[1]]
        Sj = spins[sj[0], sj[1]]
        return J * Si * Sj

    # Calculate total diagonal energy
    E_diag = vmap(single_bond_energy)(bonds)
    E_diag = jnp.sum(E_diag)

    return E_diag, spin_flips

@jit
def gen_configs(state, spin_flip_loc):
    """
    Generate new configurations by flipping spins at specified locations.

    Args:
        state: The current spin configuration.
        spin_flip_loc: Locations for spin flips.

    Returns:
        new_states: New configurations after applying spin flips.
    """
    def flip_spins(state, flip):
        """Flip spins at specified locations."""
        i1, j1, i2, j2 = flip
        new_state = state.at[i1, j1].set(state[i2, j2])
        new_state = new_state.at[i2, j2].set(state[i1, j1])
        return new_state

    # Vectorize spin flipping across all specified locations
    new_states = vmap(flip_spins, in_axes=(None, 0))(state, spin_flip_loc)
    
    return new_states
