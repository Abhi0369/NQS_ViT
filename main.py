import argparse
from jax import random
import jax.numpy as jnp
from jax.lib import xla_bridge
from ViT import VisionTransformer 

from SR import sr
import jax
import optax
import numpy as np
import cProfile
from VMC import run
from helper import generate_state
from cost import cost,cost2


reg_coef = 1e-4 
configurations = []
Energies =[]

device=xla_bridge.get_backend().platform
if device!='gpu':
    print("Using CPU, consider switching to GPU if available")



def main(epochs, batch_size, learning_rate, n, n_sweeps, warm_up, SR, Jz):
    key = random.PRNGKey(369)  # Initialize the random key for JAX

    # Initialize the VisionTransformer model with given parameters
    nqs = VisionTransformer(patch_size=2, hidden_size=16, lattice_size=n, num_heads=8, num_layers=1, num_classes=1, 
                            num_channels=1,use_cls_token=False,use_relative_pos_embedding=False,use_scale_norm=False
                            )
    
    # Initialize model parameters
    params = jax.device_put(nqs.init(key, jnp.ones([1, n, n, 1]))['params'])
    # Calculate total number of parameters
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Number of parameters: {total_params}")

    # Initialize the optimizer
    optimizer = optax.sgd(learning_rate)
    optimizer_state = optimizer.init(params)

    Energies, configurations = [], []

    # Generate the initial state
    initial_state = jax.device_put(generate_state([1, n, n, 1], key))

    if SR:
        print("Optimizing parameters using Stochastic Reconfiguration")
    else:
        print("Optimizing parameters using SGD")

    for epoch in range(epochs):
        key, subkey = random.split(key)  # Split the key for this epoch
        samples, elocs = run(initial_state, n_sweeps, warm_up, nqs.apply, params, subkey, Jz)
        if SR:
            # samples, elocs = run(initial_state, n_sweeps, warm_up, nqs.apply, params, key, Jz)
            # Update parameters using SR
            params, delta = sr(nqs, params, samples, elocs, learning_rate, reg_coef)
            # Log energy and configurations
            Energies.append(elocs)
            configurations.append(samples)
            print(f"Iteration {epoch}, Energy: {jnp.real(elocs.mean())}, variance: {jnp.var(elocs)}")
        else:
      
            loss, eloc, var, grads = cost(params, initial_state, nqs.apply, Jz, n_sweeps, warm_up, subkey)
            # grads = jax.grad(lambda p:cost2(samples,nqs.apply,p,Jz),holomorphic=False)(params)

            # Update optimizer state and parameters
            updates, optimizer_state = optimizer.update(grads, optimizer_state)
        

            params = optax.apply_updates(params, updates)
            # Log energy
            Energies.append(jnp.real(eloc))
            print(f"Iteration {epoch}, Loss: {loss}, Energy: {jnp.real(eloc)}, Variance: {var}")

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a NQS simulation with a Vision Transformer")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to run")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Learning rate for the optimizer")
    parser.add_argument("--n", type=int, default=2, help="Size of the lattice (n x n)")
    parser.add_argument("--n_sweeps", type=int, default=400, help="Number of sweeps/samples per epoch")
    parser.add_argument("--warm_up", type=int,default=100,help="Number of Warm up steps ")
    parser.add_argument("--patch_size",type=int,default=1,help="Patch size fot ViT")
    parser.add_argument("--SR",type=bool,default=False, help="Whether to use SR")
    parser.add_argument("--Jz", type=float,default=10,help="Coupling strength")
    args = parser.parse_args()


    main(args.epochs, args.batch_size, args.learning_rate, args.n, args.n_sweeps,args.warm_up,args.SR,args.Jz)


np.savez_compressed("Samples",configurations)
np.savez_compressed("Energies",Energies)
