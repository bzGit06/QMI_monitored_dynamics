import numpy as np

def purityA_global(dA, dB):
    '''
    purity of subsystem A given a global unitary applied on AB
    followed by traceout on B, RA is initially maximally entangled
    '''
    return (dA**2 * (dB**2 + dB - 1) - dB)/(dA * (dA**2 * dB**2 - 1))

def purityB_global(dA, dB):
    return (dB**2 + dB*(dA**2 - 1) - 1)/(dA**2 * dB**2 - 1)


def MIR2_traceout(Na, Nb, t):
    '''
    Renyi-2 extension of mutual information between R and A, 
    where a sequence of unitaries applied on AB followed by traceout on B,
    RA is initially maximally entangled
    '''
    dA, dB = 2**Na, 2**Nb
    r = dB*(dA**2 - 1)/(dA**2 * dB**2 - 1)
    
    c0 = (1 + dB)/(dA**2 * dB + 1)
    c1 = (1-dA**2)/(dA**3 * dB + dA)
    Pu_A = dA*c0 + c1 * r**t

    c3 = (dA**2 - 1)*dB/(dA**2 * dB + 1)
    Pu_B = c0 + c3 * r**t

    SA = -np.log2(Pu_A)
    SB = -np.log2(Pu_B)
    return SA + Na - SB

def avgPu(Na, Nb, t):
    '''
    measurement-averaged purity of reduced state A_t,
    where a sequence of unitaries applied on AB followed by projective measurement on B,
    RA is initially maximally entangled
    '''
    dA, dB = 2.**Na, 2.**Nb

    c = 0.5 * dA**(-(2*t+1)) * dB**(-t)
    r1 = (dA*dB-1.)**t * (dA+1.)**(t+1.)
    r2 = (dA*dB+1.)**t * (dA-1.)**(t+1.)

    return c * (r1 - r2)

def avgS2(Na, Nb, t):
    '''
    measurement-averaged Renyi-2 entropy of reduced state A_t,
    where a sequence of unitaries applied on AB followed by projective measurement on B,
    RA is initially maximally entangled
    '''

    dA, dB = 2.**Na, 2.**Nb

    c = 1. + (2*t+1.)*Na + t*Nb - t*np.log2(dA*dB+1.) - (t+1.)*np.log2(dA+1.)
    r1 = (1. - 2./(dA*dB+1.))**t
    r2 = (1. - 2./(dA+1.))**(t+1.)
    return c - np.log2(r1 - r2)

def MIR2_traceout_MM(Na, Nb, t):
    '''
    Renyi-2 extension of mutual information between R and A,
    where a sequence of unitaries applied on AB followed by traceout B and
    maximally mixed state initialization, RA is initially maximally entangled
    '''
    dA, dB = 2.**Na, 2.**Nb

    q = (dA**2-1.)/(dA**2 * dB**2 - 1.)
    c1 = dA**2 * (dB**2+dB-1) - dB
    purity_A = (c1 * q**t + dA**2-1 - (dA**2 * dB**2-1)*q**t)/(dA*(dA**2 - 1))
    Sa = -np.log2(purity_A)
    
    c2 = dB*(dA**2 + dB-1.) - 1
    purity_RA = (dA**2*c2 * q**t + dA**2-1 - (dA**2 * dB**2-1)*q**t)/(dA**2*(dA**2 - 1))
    Sra = -np.log2(purity_RA)
    return Na + Sa - Sra

def tau_traceout(Na, Nb, eps):
    '''
    exact expression for lifetime of memory in the system with traceout and reset
    '''
    dA, dB = 2.**Na, 2.**Nb

    c1 = dA**2 * (dA**(2*eps)-1) * (dB + 1)
    c2 = (dA**2 - 1) * (dA**(2*eps) + dA**2 * dB)

    c3 = (dA**2 - 1)*dB/(dA**2 * dB**2 - 1)
    return np.log(c1/c2) / np.log(c3)