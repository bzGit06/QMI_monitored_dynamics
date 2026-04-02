'''
Autthor: Runzhe (Ricardo) Mo, rzmo0924@berkeley.edu
Date: 1969-12-31 16:00:00
LastEditors: Ricardo Mo rzmo0924@berkeley.edu
LastEditTime: 2025-03-05 17:40:45
FilePath: /QRC_memory_NS/MCM_utils.py
Description: 

'''
import qiskit
import numpy as np
import qutip as qt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import random_unitary, partial_trace, entropy
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.primitives import BitArray

from qiskit_ibm_runtime import QiskitRuntimeService, IBMBackend
from qiskit.circuit.library import XGate, YGate
from qiskit.transpiler import InstructionProperties, PassManager
from qiskit.transpiler.passes.scheduling import ASAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.transpiler.passes import BasisTranslator
from copy import deepcopy



from itertools import combinations

# Helper function for calculating the mutual information of a bipartite quantum state
def n_qubit_qmi(density_matrix, subsys1, subsys2, ancillasys, base=2) -> float:
    """the helper function to calculate the quantum mutual information between two subsystems.
    This method relies on the the density matrix of the system.

    Args:
        density_matrix (qiskit.quantum_info.states.DensityMatrix): the density matrix of the system
        subsys1 (List[int]): the list of qubits in the first subsystem
        subsys2 (List[int]): the list of qubits in the second subsystem
        ancillasys (List[int]): the list of qubits in the ancilla subsystem to be traced out
        base (int): the base of the logarithm
    Returns:
        qmi (float): the quantum mutual information between the two subsystems
    """

    # calculate the reduced density matrix of whole system except ancilla
    rho_AB = partial_trace(density_matrix, ancillasys)
    # calculate the reduced density matrix of the first subsystem
    rho_A = partial_trace(density_matrix, subsys2 + ancillasys)
    # calculate the reduced density matrix of the second subsystem
    rho_B = partial_trace(density_matrix, subsys1 + ancillasys)
    
    # calculate the mutual information
    return entropy(rho_A, base=base) + entropy(rho_B, base=base) - entropy(rho_AB, base=base)

def n_qubit_cmi(density_matrix, subsys1, subsys2, ancillasys) -> float:
    """the helper function to calculate the classical mutual information between two subsystems.
    This method relies on the the density matrix of the system.

    Args:
        density_matrix (qiskit.quantum_info.states.DensityMatrix): the density matrix of the system
        subsys1 (List[int]): the list of qubits in the first subsystem
        subsys2 (List[int]): the list of qubits in the second subsystem
        ancillasys (List[int]): the list of qubits in the ancilla subsystem to be traced out
    Returns:
        cmi (float): the classical mutual information between the two subsystems   
    """

    rho_AB = partial_trace(density_matrix, ancillasys)
    rho_A = partial_trace(rho_AB, subsys2)
    rho_B = partial_trace(rho_AB, subsys1)
    pr_A = np.diag(rho_A.data)
    pr_B = np.diag(rho_B.data)
    pr_AB = np.diag(rho_AB.data)

    cmi = 0
    for i in range(pr_A.shape[0]):
        for j in range(pr_B.shape[0]):
            whole_index = j * pr_B.shape[0] + i
            if pr_AB[whole_index] != 0 and pr_A[i] != 0 and pr_B[j] != 0:
                cmi += pr_AB[whole_index] * np.log2(pr_AB[whole_index] / (pr_A[i] * pr_B[j]))

    return cmi

def bitarray_to_CMI(bitarray: BitArray, data_size) -> float:
    """
    Calculate the classical mutual information (CMI) from a bitarray of measurement results.
    
    Args:
        bitarray (BitArray): The bitarray of measurement results.
        data_size (int): The number of data/idle qubits.
    
    Returns:
        float: The classical mutual information (CMI) between the data and idle qubits.
    """
    # assert len(data_index_list) == len(idle_index_list), "Data and idle index lists must have the same length."

    # Get counts for data and idle qubits
    data_counts = bitarray.slice_bits(list(range(data_size, 2*data_size))).get_counts()
    idle_counts = bitarray.slice_bits(list(range(0, data_size))).get_counts()
    joint_counts = bitarray.get_counts()

    total_cmi = 0
    shots = bitarray.num_shots
    
    # for data_bitstring, data_count in data_counts.items():
    #     for idle_bitstring, idle_count in idle_counts.items():
    #         joint_bitstring = data_bitstring + idle_bitstring
    #         if joint_bitstring not in joint_counts:
    #             continue
    #         joint_prob = joint_counts[joint_bitstring] / shots
    #         total_cmi += joint_prob * np.log2(joint_prob / (data_count/shots * idle_count/shots))
    for joint_bitstring, joint_count in joint_counts.items():
        data_bitstring = joint_bitstring[::-1][data_size:2*data_size][::-1]
        idle_bitstring = joint_bitstring[::-1][:data_size][::-1]
        # idle_bitstring = "".join([joint_bitstring[index] for index in idle_index_list])
        pr_A = data_counts[data_bitstring] / shots
        pr_B = idle_counts[idle_bitstring] / shots
        joint_prob = joint_count / shots
        total_cmi += joint_prob * np.log2(joint_prob / (pr_A * pr_B))
    
    return total_cmi.real
# ========================================================================
# The helper function to compose the original (QUDDPM) fast-scrambling circuit to approximate the Haar random unitary ========================================

def scrambling_circuit_d(n, d, phis, gs=None):
    '''
    Generate a n-qubit scrambling circuit with step t.
    The circuit uses the fast-scrambling circuit structure with randomly generated angles.

    Args:
        n (int): the number of qubits
        d (int): the number of depth in one scrambling unitary
        phis (np.ndarray): the angles of single-qubit rotation gates in diffusion circuit, shape: (t, 3*n)
        gs (np.ndarray): the angle of RZZ gates in diffusion circuit when n>=2
    '''

    # Initialize the circuit
    qc = QuantumCircuit(n)
    for tt in range(d):
        # Apply single-qubit rotations on each qubit
        phis_tt = phis[tt]
        for i in range(n):
            qc.rz(phi=phis_tt[3*i], qubit=i)
            qc.ry(theta=phis_tt[3*i+1], qubit=i)
            qc.rz(phi=phis_tt[3*i+2], qubit=i)
        if n > 1:
            # Apply RZZ gate
            for j, k in combinations(range(n), 2):
                qc.rzz(gs[tt]/(2*np.sqrt(n)), j, k)
        qc.barrier()
    return qc

def set_scramblie_circ(n, d, seed, scattering_coef):
    np.random.seed(seed)
    diff_hs_t = np.linspace(scattering_coef["hs_lower"], scattering_coef["hs_upper"], d).reshape((d, 1))
    phis = np.random.rand(d, 3*n) * scattering_coef["phis_scale"] + scattering_coef["phis_shift"]
    phis = phis * diff_hs_t
    if n > 1:
        # set homogeneous RZZ gate angles
        gs = np.random.rand(d) * scattering_coef["gs_scale"] + scattering_coef["gs_shift"]
        gs = gs * diff_hs_t.reshape((d,))
        
        circ = scrambling_circuit_d(n, d, phis, gs)
    else:
        circ = scrambling_circuit_d(n, d, phis)
    return circ

# =======================================================================================================================================

# The helper function to compose the  fast-scrambling circuit with ECR gate to approximate the Haar random unitary
def scrambling_circuit_ECR_d(n, d, seed, all_to_all=False):
    """
    Generate a n-qubit scrambling circuit with depth d.
    The circuit consists of single-qubit Haar random unitaries and global interaction with ECR gates.

    Args:
        d (int): the number of depth in one scrambling unitary
        n (int): the number of qubits
        seed (int): the random seed to generate the Haar random unitaries
    """

    # generate the random seed list for each single-qubit Haar random unitary
    np.random.seed(seed)
    seeds = np.random.randint(0, 2**31, (d, n))

    # Initialize the circuit
    qc = QuantumCircuit(n)
    for tt in range(d):
        # Apply single-qubit Haar random unitaries
        for i in range(n):
            single_haar = random_unitary(2, seed=seeds[tt, i])
            qc.append(single_haar, [i])
        # Apply ECR gates
        if n > 1:
            if all_to_all:
                for j, k in combinations(range(n), 2):
                    qc.ecr(j, k)
            else:
                for i in range(n//2):
                    qc.ecr(2*i, 2*i+1)
                for i in range((n-1)//2):
                    qc.ecr(2*i+1, 2*i+2)
        qc.barrier()
    return qc


def scrambling_circuit_CZ_d(n, d, seed, all_to_all=False):
    np.random.seed(seed)
    seeds = np.random.randint(0, 2**31, (d, n))

    # Initialize the circuit
    qc = QuantumCircuit(n)
    for tt in range(d):
        # Apply single-qubit Haar random unitaries
        for i in range(n):
            single_haar = random_unitary(2, seed=seeds[tt, i])
            qc.append(single_haar, [i])
        # Apply CZ gates
        # if n > 1:
        #     for j, k in combinations(range(n), 2):
        #         qc.cz(j, k)
        if n > 1:
            if all_to_all:
                for j, k in combinations(range(n), 2):
                    qc.cz(j, k)
            else:
                for i in range(n//2):
                    qc.cz(2*i, 2*i+1)
                for i in range((n-1)//2):
                    qc.cz(2*i+1, 2*i+2)
        qc.barrier()
    return qc


# =======================================================================================================================================

# Ising Model Hamiltonian
def IsingHamiltonian(N, seed):
    # Generate a random Ising Hamiltonian with N spins/qubits

    np.random.seed(seed)
    hx0_vec_all = 1 + 0.3 * np.random.randn(N) # Eta of x
    hz0_vec_all = 0 + 0.3 * np.random.randn(N) # Eta of z

    J_vec_all = 0 + 1*np.random.randn(N*(N-1)//2)

    Xs = []
    Zs = []
    for i in range(N):
        X = qt.tensor([qt.sigmax() if k==i else qt.qeye(2) for k in range(N)]) # Pauli X operator on qubit i
        Z = qt.tensor([qt.sigmaz() if k==i else qt.qeye(2) for k in range(N)]) # Pauli Z operator on qubit i
        Xs.append(X)
        Zs.append(Z)

    H = 0
    for i in range(N):
        H += hx0_vec_all[i] * Xs[i]
        H += hz0_vec_all[i] * Zs[i]

    for cc, (i, j) in enumerate(combinations(range(N), 2)):
        H += J_vec_all[cc] * Zs[i] * Zs[j]

    return H

# =======================================================================================================================================
# QUDDPM backward circuit module

def back_circuit_module_prev(n, na, L, params):
    """
    The backward denoise parameteric quantum circuit module.

    The first na qubits are the ancilla qubits
    params: [num_layer, 2*(n+na)]
    """
    n_tot = n + na
    q_reg = QuantumRegister(n_tot, name="data")
    c_reg = ClassicalRegister(na, name="ancilla_meas")
    qc = QuantumCircuit(q_reg, c_reg)

    # repeat the hardware-efficient ansatz for multiple layers
    for l in range(L):
        params_l = params[l]
        for i in range(n_tot):
            qc.rx(params_l[2*i], qubit=q_reg[i])
            qc.ry(params_l[2*i+1], qubit=q_reg[i])
        for i in range(n_tot//2):
            qc.cz(q_reg[2*i], q_reg[2*i+1])
        for i in range((n_tot-1)//2):
            qc.cz(q_reg[2*i+1], q_reg[2*i+2])
    # qc.measure(q_reg[:na], c_reg)
    # qc.reset(q_reg[:na])
    return qc

# def set_denoising_circ_prev(n, na, L, t, T, params):
#     """
#     When trying to train the denoising circuit at step t, we set up the t-1 denoising circuits with freezed
#     parameters.

#     The first na qubits are the ancilla qubits.
#     Args:
#         params: [T-t, num_layer, 2*(n+na)], in a reverse order(e.g., the params[0] is for the last denoising layer(T-1))
#     """

#     n_tot = n + na
#     q_reg = QuantumRegister(n_tot, name="data")
#     c_reg = ClassicalRegister(na, name="ancilla_meas")
#     qc = QuantumCircuit(q_reg, c_reg)

#     for i in range(t):
#         qc.compose(back_circuit_module_prev(n, na, L, params[T-1-i]), qubits=q_reg, clbits=c_reg, inplace=True)
#         qc.barrier()

#     return qc

def single_denoising_circ(n, na, L, t, T, params):
    """
    Only set up the denoising circuit at step t, ignoring the previous steps.
    The first na qubits are the ancilla qubits.
    Args:
        params: [num_layer, 2*(n+na)]
    """

    n_tot = n + na
    q_reg = QuantumRegister(n_tot, name="data")
    c_reg = ClassicalRegister(na, name="ancilla_meas")
    qc = QuantumCircuit(q_reg, c_reg)
    
    qc.compose(back_circuit_module_prev(n, na, L, params[T-1-t]), qubits=q_reg, clbits=c_reg, inplace=True)
    
    return qc

# =======================================================================================================================================
# Dynamical decoupling
def adding_dynamical_decoupling(tcirc, backend:IBMBackend):
    """
    Adds dynamical decoupling sequences to the given quantum circuit.

    Args:
        tcirc (QuantumCircuit): The transpiled quantum circuit.
        service (QiskitRuntimeService): The Qiskit Runtime service instance.
        backend (IBMBackend): The IBM backend instance.

    Returns:
        QuantumCircuit: The quantum circuit with dynamical decoupling sequences added.
    """

    basis_gates = list(backend.target.operation_names)
    target = deepcopy(backend.target)
    X, Y = XGate(), YGate()
    dd_sequence = [X, Y, X, Y]  # Example DD sequence (XY4) 
    
    # Since Y is not a native gate, modify the Y gate properties
    # Y gate has some duration and error properties as X gate
    y_gate_properties = {}
    for qubit in range(target.num_qubits):
        y_gate_properties.update(
            {
                (qubit,): InstructionProperties(
                    duration=target["x"][(qubit,)].duration,
                    error=target["x"][(qubit,)].error,
                )
            }
        )
    target.add_instruction(YGate(), y_gate_properties)

    dd_pm = PassManager(
        [
            ASAPScheduleAnalysis(target=target),
            PadDynamicalDecoupling(target=target, dd_sequence=dd_sequence),
        ]
    )
    tcirc_dd = dd_pm.run(tcirc)
    
    tcirc_dd = BasisTranslator(sel, basis_gates)(tcirc_dd)
    return tcirc_dd

# Post-processing functions for measurement data
def postprocessing_CMI_experiment(result, t:int, data_size: int, ancilla_size: int, shots:int, repeat: int, measurement_conditioned: bool) -> np.ndarray:
    """
    Postprocess the CMI experiment results from the job.
    
    Args:
        result: The qiskit PrimitiveResult to analyze
        backend (IBMBackend): The backend to run the experiment.
        data_size (int): The number of the data qubits.
        ancilla_size (int): The number of the ancilla/bath qubits.
        repeat (int): The number of repetitions for the experiment.
        measurement_conditioned (bool): Whether the measurement is conditioned.
    
    Returns:
        np.ndarray: The processed CMI results.
    """
    cmi_list = np.zeros(repeat)
    if measurement_conditioned and t!=0:
        for r in range(repeat):
            result_data_r = result[r]
            ancilla_bitarray = result_data_r.data.ancilla_meas

            result_data = result_data_r.join_data()
            
            total_cmi = 0
            for ancilla_bitstring, ancilla_count in ancilla_bitarray.get_counts().items():
                # reverse the order of bitstring to match the qiskit's bit order
                bitstring_list = [int(digit) for digit in ancilla_bitstring][::-1]

                # postselect the bitarray which ancilla measurement result matches the ancilla_bitstring
                postselect_bitarray = result_data.postselect(indices=list(range(t*ancilla_size)), 
                                                            selection=bitstring_list)
                # Slice the ancilla out
                sliced_postselect_bitarray = postselect_bitarray.slice_bits(list(range(t*ancilla_size, 2*data_size + t*ancilla_size)))

                # Calculate the CMI for the postselected bitarray
                conditional_cmi_meas = bitarray_to_CMI(sliced_postselect_bitarray, data_size)
                # print(f'bitstring: {ancilla_bitstring}, prob: {ancilla_count/shots:04f}, cmi: {conditional_cmi_meas:04f}')
                total_cmi += conditional_cmi_meas * ancilla_count / shots
            cmi_list[r] = total_cmi
        return cmi_list
    else:
        for r in range(repeat):
            result_data_r = result[r]
            unconditional_bitarray = result_data_r.data.system_meas
            cmi_list[r] = bitarray_to_CMI(unconditional_bitarray, data_size)
        return cmi_list