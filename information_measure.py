import numpy as np
from scipy.stats import unitary_group

import random
from random import choices

from functools import partial
from itertools import product, combinations
from opt_einsum import contract

import tensorcircuit as tc
import qutip as qt

import jax
import jax.numpy as jnp


def psiR_purity(psi_RA, Na):
    '''
    given bipartite pure state, evaluate the purity of the reduced state of subsystem
    '''
    rho_RA = jnp.einsum('i,j->ij', psi_RA, psi_RA.conj())
    rho_R = jnp.einsum('ijkj->ik', rho_RA.reshape(2**Na, 2**Na, 2**Na, 2**Na))
    
    return jnp.real(jnp.trace(rho_R @ rho_R))
    
psiR_purity_vec = jax.jit(jax.vmap(psiR_purity, in_axes=(0, None)), static_argnums=(1, ))


def psiR_vNEntropy(psi_RA, Na):
    '''
    given bipartite pure state, evaluate the von-Neumann entropy of the reduced state of subsystem
    '''
    rho_RA = jnp.einsum('i,j->ij', psi_RA, psi_RA.conj())
    rho_R = jnp.einsum('ijkj->ik', rho_RA.reshape(2**Na, 2**Na, 2**Na, 2**Na))
    
    vals = jnp.linalg.eigvalsh(rho_R)
    vals = jnp.where(vals > 0, vals, 1e-14)
    return -jnp.sum(vals * jnp.log2(vals))

psiR_vNEntropy_vec = jax.jit(jax.vmap(psiR_vNEntropy, in_axes=(0, None)), static_argnums=1)

@partial(jax.jit)
def rho_vNEntropy(rho):
    '''
    given a mixed state, evaluate the von-Neumann entropy
    '''
    vals = jnp.linalg.eigvalsh(rho)
    vals = jnp.where(vals > 0, vals, 1e-14)
    return -jnp.sum(vals * jnp.log2(vals))

rho_vNEntropy_vec = jax.jit(jax.vmap(rho_vNEntropy))


def MI_rhoRA(rho_RA, Na):
    '''
    mutual information of a bipartite mixed state
    '''
    Sra = rho_vNEntropy(rho_RA)

    rho_RA = rho_RA.reshape(2**Na, 2**Na, 2**Na, 2**Na)
    rho_R = jnp.einsum('ijkj->ik', rho_RA)
    rho_A = jnp.einsum('ijil->jl', rho_RA)
    
    Sr, Sa = rho_vNEntropy(rho_R), rho_vNEntropy(rho_A)
    return Sr + Sa - Sra

MI_rhoRA_vec = jax.jit(jax.vmap(MI_rhoRA, in_axes=(0, None)), static_argnums=1)


def MIr2_rhoRA(rho_RA, Na):
    '''
    Renyi-2 extension mutual information of a bipartite mixed state
    '''
    S2ra = -jnp.log2(jnp.real(jnp.trace(rho_RA @ rho_RA)))

    rho_RA = rho_RA.reshape(2**Na, 2**Na, 2**Na, 2**Na)
    rho_R = jnp.einsum('ijkj->ik', rho_RA)
    rho_A = jnp.einsum('ijil->jl', rho_RA)
    
    S2r = -jnp.log2(jnp.real(jnp.trace(rho_R @ rho_R)))
    S2a = -jnp.log2(jnp.real(jnp.trace(rho_A @ rho_A)))
    return S2r + S2a - S2ra

MIr2_rhoRA_vec = jax.jit(jax.vmap(MIr2_rhoRA, in_axes=(0, None)), static_argnums=1)

def holevoInfo(psis, weights):
    rhos = jnp.einsum('bi,bj->bij', psis, psis.conj())
    rho = jnp.sum(weights[:, jnp.newaxis, jnp.newaxis] * rhos, axis=0)
    return rho_vNEntropy(rho)

holevoInfo_vec = jax.jit(jax.vmap(holevoInfo, in_axes=(0, 0)))