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

import datetime

K = tc.set_backend('jax')
tc.set_dtype('complex128')


def maxEntangle_state(N):
    '''
    create maximally entangled state
    N: number of qubit in one party
    '''
    d = 2**N
    Psi = 0
    for i in range(d):
        Psi += qt.tensor(qt.basis(d, i), qt.basis(d, i))
    return Psi.unit()

def CQstate(psis, Nr):
    rho = 0
    for i in range(len(psis)):
        bin_i = list(np.binary_repr(i, width=Nr))
        bin_i = [int(x) for x in bin_i]
        state_r = qt.basis([2]*Nr, bin_i)
        rho += qt.tensor(qt.ket2dm(state_r), qt.ket2dm(psis[i]))
    return rho.unit()

@partial(jax.jit, static_argnums=(1, 2))
def randomMeasure(inputs, Na, Nb, key):
    '''
    given samples of input pure states, perform projective measurement on
    the ancillary system, and collect post measurement pure states
    '''
    num = len(inputs)
    psis = inputs.reshape(num, 4**Na, 2**Nb)
    probs = jnp.linalg.norm(psis, axis=1)**2.
    res = jax.random.categorical(key, jnp.log(probs))
    post_states = psis[jnp.arange(num), :, res]
    post_states /= jnp.sqrt(probs[jnp.arange(num), res])[:, jnp.newaxis]
    
    return post_states


def seqModel_RA(z, Us, psi_RA, Na, Nb, T):
    '''
    sequential model with random unitary, output the corresponding probability 
    of projective measurement trajectories, and corresponding state
    '''
    zero_B = qt.basis(2**Nb, 0).full().squeeze()
    
    prob_z = []
    for t in range(T):
        inputs = jnp.kron(psi_RA, zero_B)
        c = tc.Circuit(2*Na + Nb, inputs=inputs)
        c.any(*list(range(Na, 2*Na+Nb)), unitary=Us[t])
        # post-selection on measure
        psi = c.state().reshape(4**Na, 2**Nb)
        post_psi = psi[:, z[t]]
        p_zt = jnp.linalg.norm(post_psi)**2
        psi_RA = post_psi / jnp.sqrt(p_zt)
        prob_z.append(p_zt)
    return jnp.array(prob_z), psi_RA

seqModel_RA_vec = K.jit(K.vmap(seqModel_RA, vectorized_argnums=0), static_argnums=(3, 4, 5))


def globalCircuit(inputs, U, Na, Nb):
    '''
    apply a global unitary to the system
    '''
    c = tc.Circuit(2*Na + Nb, inputs=inputs)
    c.any(*list(range(Na, 2*Na+Nb)), unitary=U)
    return c.state()

globalCircuit_vec = K.jit(K.vmap(globalCircuit, vectorized_argnums=0), static_argnums=(2, 3))


def seqModel_RAmc(Us, psi_RA, Na, Nb, T, num):
    '''
    sequential model with random unitary, output post-measurement state 
    through monte-carlo sampling
    '''
    psis_RA = jnp.stack([psi_RA]*num)
    zero_B = qt.basis(2**Nb, 0).full().squeeze()

    seed = int(1e6 * datetime.datetime.now().timestamp())
    key = jax.random.PRNGKey(seed)
    for t in range(T):
        inputs = jnp.einsum('bi, j->bij', psis_RA, zero_B).reshape(num, 4**Na*2**Nb)
        psis = globalCircuit_vec(inputs, Us[t], Na, Nb)

        key, subkey = jax.random.split(key)
        psis_RA = randomMeasure(psis, Na, Nb, subkey)
    
    return psis_RA

def seqModel_CQ_RA(z, Us, psis_A, Na, Nb, T):
    '''
    sequential model with random unitary, output the corresponding probability 
    of projective measurement trajectories, and corresponding state
    '''
    num = len(psis_A)
    zero_B = qt.basis(2**Nb, 0).full().squeeze()

    prob_z = []
    psis_A_t = jnp.copy(psis_A)
    weights = 1./num*np.ones(num)
    for t in range(T):
        norms_t = jnp.zeros(num)
        for j in range(num):
            inputs = jnp.kron(psis_A_t[j], zero_B)
            c = tc.Circuit(Na + Nb, inputs=inputs)
            c.any(*list(range(Na+Nb)), unitary=Us[t])
            # post-selection on measure
            psi = c.state().reshape(2**Na, 2**Nb)
            post_psi = psi[:, z[t]] # unnormalized
            norms_t = norms_t.at[j].set(jnp.linalg.norm(post_psi)**2)
            psi_A = post_psi / jnp.sqrt(norms_t[j]) # normalize
            psis_A_t = psis_A_t.at[j].set(psi_A) # update state in CQ
        # measurement outcome probability at step t
        p_zt = np.sum(weights * norms_t)
        prob_z.append(p_zt)
        # update weight for each state
        weights *= norms_t/p_zt
    
    return jnp.array(prob_z), psis_A_t, weights

seqModel_CQ_RA_vec = K.jit(K.vmap(seqModel_CQ_RA, vectorized_argnums=0), static_argnums=(3, 4, 5))

def globalCircuit_CQ(inputs, U, Na, Nb):
    '''
    apply a global unitary to the system
    '''
    c = tc.Circuit(Na + Nb, inputs=inputs)
    c.any(*list(range(Na+Nb)), unitary=U)
    return c.state()

globalCircuit_CQ_vec = K.jit(K.vmap(globalCircuit_CQ, vectorized_argnums=0), static_argnums=(2, 3))

@partial(jax.jit, static_argnums=(2, 3))
def randomMeasure_CQ(inputs, weights, Na, Nb, key):
    '''
    given samples of input pure states, perform projective measurement on
    the ancillary system, and collect post measurement pure states
    '''
    num, ensemble_size = inputs.shape[:2]
    psis = inputs.reshape(num, ensemble_size, 2**Na, 2**Nb)
    norms = jnp.linalg.norm(psis, axis=2)**2.
    probs = jnp.sum(weights[:, :, jnp.newaxis] * norms, axis=1)

    res = jax.random.categorical(key, jnp.log(probs))
    post_states = psis[jnp.arange(num), :, :, res]
    post_states /= jnp.sqrt(norms[jnp.arange(num), :, res])[:, :, jnp.newaxis]
    
    weights_new = weights * norms[jnp.arange(num), :, res]/probs[jnp.arange(num), res][:, jnp.newaxis]
    return post_states, weights_new

def seqModel_CQ_RAmc(Us, psis_A, Na, Nb, T, num):
    '''
    sequential model with random unitary, output post-measurement state 
    through monte-carlo sampling
    '''
    ensemble_size = len(psis_A)
    psis_A_t = np.stack([psis_A]*num)
    zero_B = qt.basis(2**Nb, 0).full().squeeze()

    seed = int(1e6 * datetime.datetime.now().timestamp())
    key = jax.random.PRNGKey(seed)

    weights = 1./ensemble_size * np.ones((num, ensemble_size))
    for t in range(T):
        norms_t = np.zeros((num, ensemble_size))
        psis_AB_t = np.zeros((num, ensemble_size, 2**(Na+Nb)), dtype=complex)
        for j in range(ensemble_size):
            inputs = jnp.einsum('bi, j->bij', psis_A_t[:, j], zero_B).reshape(num, 2**(Na+Nb))
            psis_AB_t[:, j] = globalCircuit_CQ_vec(inputs, Us[t], Na, Nb)

        # post-selection
        key, subkey = jax.random.split(key)
        psis_A_t, weights = randomMeasure_CQ(psis_AB_t, weights, Na, Nb, subkey)

    return psis_A_t, weights


def seqModelTraceOut_RA(Us, rho_RA, Na, Nb, T):
    '''
    sequential model with random unitary and traceout ancillary system, 
    output the reduced state
    '''
    zero_B = qt.ket2dm(qt.basis(2**Nb, 0)).full()

    for t in range(T):
        inputs = jnp.kron(rho_RA, zero_B)
        c = tc.DMCircuit(2*Na + Nb, dminputs=inputs)
        c.any(*list(range(Na, 2*Na+Nb)), unitary=Us[t])
        
        # traceout the bath system
        rho = c.state()
        rho_RA = jnp.einsum('ijkj->ik', rho.reshape(4**Na, 2**Nb, 4**Na, 2**Nb))
    
    return rho_RA

seqModelTraceOut_RA_vec = K.jit(K.vmap(seqModelTraceOut_RA, vectorized_argnums=0), 
                                static_argnums=(2, 3, 4))

@partial(jax.jit, static_argnums=(1, 2))
def randomMeasure_result(inputs, Na, Nb, key):
    '''
    given samples of input pure states, perform projective measurement on
    the ancillary system, output measurement result and corresponding post measurement pure states
    '''
    num = len(inputs)
    psis = inputs.reshape(num, 4**Na, 2**Nb)
    probs = jnp.linalg.norm(psis, axis=1)**2.
    res = jax.random.categorical(key, jnp.log(probs))
    post_states = psis[jnp.arange(num), :, res]
    post_states /= jnp.sqrt(probs[jnp.arange(num), res])[:, jnp.newaxis]
    
    return res, post_states

def seqModelnoReset_RAmc(Us, psi_RA, Na, Nb, T, num):
    '''
    sequential model with random unitary without reset, output post-measurement state 
    through monte-carlo sampling
    '''
    psis_RA = jnp.stack([psi_RA]*num)
    basis_B = jnp.eye(2**Nb, dtype=complex)
    
    d_all = 4**Na * 2**Nb

    m_res = np.zeros(num, dtype=int)
    
    seed = int(1e6 * datetime.datetime.now().timestamp())
    key = jax.random.PRNGKey(seed)
    for t in range(T):
        inputs = jnp.einsum('bi, bj->bij', psis_RA, basis_B[m_res]).reshape(num, d_all)
        psis = globalCircuit_vec(inputs, Us[t], Na, Nb)

        key, subkey = jax.random.split(key)
        m_res, psis_RA = randomMeasure_result(psis, Na, Nb, subkey)

    return psis_RA

def seqModelTraceOutCReset_RA(Us, Na, Nb, T):
    '''
    sequential model with random unitary and traceout ancillary system without reset, 
    output the reduced state
    '''
    rho_RA = qt.ket2dm(maxEntangle_state(Na)).full()
    rho_B = qt.ket2dm(qt.basis(2**Nb, 0)).full()

    for t in range(T):
        inputs = jnp.kron(rho_RA, rho_B)
        c = tc.DMCircuit(2*Na + Nb, dminputs=inputs)
        c.any(*list(range(Na, 2*Na+Nb)), unitary=Us[t])
        
        # traceout
        rho = c.state()
        post_rho = rho.reshape(4**Na, 2**Nb, 4**Na, 2**Nb)[:, jnp.arange(2**Nb), :, jnp.arange(2**Nb)]
        prob = jnp.real(jnp.einsum('mii->m', post_rho))

        rho_RA = jnp.einsum('ijkj->ik', rho.reshape(4**Na, 2**Nb, 4**Na, 2**Nb))
        rho_B =  jnp.diag(prob) # input a mixed state in corresponding measurement basis and probability

    return rho_RA

seqModelTraceOutCReset_RA_vec = K.jit(K.vmap(seqModelTraceOutCReset_RA, vectorized_argnums=0), 
                                static_argnums=(1, 2, 3))


def seqModelTraceOutMaxMix_RA(Us, Na, Nb, T):
    '''
    sequential model with random unitary and traceout ancillary system,
    from second step, input the anciallry system a maximally mixed state 
    output the reduced state
    '''
    rho_RA = qt.ket2dm(maxEntangle_state(Na)).full()
    rho_B = qt.ket2dm(qt.basis(2**Nb, 0)).full()

    for t in range(T):
        inputs = jnp.kron(rho_RA, rho_B)
        c = tc.DMCircuit(2*Na + Nb, dminputs=inputs)
        c.any(*list(range(Na, 2*Na+Nb)), unitary=Us[t])
        
        # traceout
        rho_RA = jnp.einsum('ijkj->ik', c.state().reshape(4**Na, 2**Nb, 4**Na, 2**Nb))
        rho_B =  jnp.eye(2**Nb)/2**Nb

    return rho_RA


seqModelTraceOutMaxMix_RA_vec = K.jit(K.vmap(seqModelTraceOutMaxMix_RA, vectorized_argnums=0), 
                                static_argnums=(1, 2, 3))



def seqModelTraceOutNoReset_RA(z, Us, psi_RA, Na, Nb, T):
    basis_B = jnp.eye(2**Nb, dtype=complex)
    
    m_res = 0
    d_all = 2**(2*Na + Nb)
    prob_z = []
    for t in range(T):
        inputs = jnp.einsum('i, j->ij', psi_RA, basis_B[m_res]).reshape(d_all)
        c = tc.Circuit(2*Na + Nb, inputs=inputs)
        c.any(*list(range(Na, 2*Na+Nb)), unitary=Us[t])
        # post-selection on measure
        psi = c.state().reshape(4**Na, 2**Nb)
        post_psi = psi[:, z[t]]
        p_zt = jnp.linalg.norm(post_psi)**2
        psi_RA = post_psi / jnp.sqrt(p_zt)
        prob_z.append(p_zt)
        m_res = z[t]
    return jnp.array(prob_z), psi_RA

seqModelTraceOutNoReset_RA_vec = K.jit(K.vmap(seqModelTraceOutNoReset_RA, vectorized_argnums=0), 
                                       static_argnums=(3, 4, 5))
