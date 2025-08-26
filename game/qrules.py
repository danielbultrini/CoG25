#%%
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import Aer, statevector_simulator
from qiskit.quantum_info import partial_trace

def liveliness(nhood):
    v=nhood
    a = v[0][0][0]+v[0][1][0]+v[0][2][0]+v[1][0][0]+v[1][2][0]+v[2][0][0]+v[2][1][0]+v[2][2][0]
    return np.abs(a)

def SQGOL(nhood):
    a = liveliness(nhood)
    value =  nhood[1][1]
    alive = np.array([1.0,0.0])
    dead = np.array([0.0,1.0])
    B = np.array([[0,0],[1,1]])
    D = np.array([[1,1],[0,0]])
    S = np.array([[1,0],[0,1]])
    if a <= 1:
        value =  dead
    elif (a > 1 and a <= 2):
        value = ((np.sqrt(2)+1)*2-(np.sqrt(2)+1)*a)*dead+(a-1)*value#(((np.sqrt(2)+1)*(2-a))**2+(a-1)**2)
    elif (a > 2 and a <= 3):
        value = (((np.sqrt(2)+1)*3)-(np.sqrt(2)+1)*a)*value+(a-2)*alive#(((np.sqrt(2)+1)*(3-a))**2+(a-2)**2)
    elif (a > 3 and a < 4):
        value = ((np.sqrt(2)+1)*4-(np.sqrt(2)+1)*a)*alive+(a-3)*dead#(((np.sqrt(2)+1)*(4-a))**2+(a-3)**2)
    elif a >= 4:
        value = dead
    value = value/np.linalg.norm(value)
    return value

def find_closest_pure_state(density_matrix):
    """
    Finds the closest pure state to a given density matrix for a single qubit.

    The closest pure state is the eigenvector corresponding to the largest
    eigenvalue of the density matrix.

    Args:
        density_matrix: A 2x2 NumPy array representing the density matrix.
                        It must be Hermitian with a trace of 1.

    Returns:
        A tuple containing:
        - closest_pure_state_vector (np.ndarray): The state vector |ψ⟩.
        - closest_pure_state_density_matrix (np.ndarray): The density matrix |ψ⟩⟨ψ|.

    Raises:
        ValueError: If the input is not a valid 2x2 density matrix.
    """
    # --- Input Validation ---
    if not isinstance(density_matrix, np.ndarray) or density_matrix.shape != (2, 2):
        raise ValueError("Input must be a 2x2 NumPy array.")
    if not np.isclose(np.trace(density_matrix), 1):
        raise ValueError("The trace of the density matrix must be 1.")
    if not np.allclose(density_matrix, density_matrix.conj().T):
        raise ValueError("The density matrix must be Hermitian.")

    # --- Core Algorithm ---
    # For a Hermitian matrix, eigh is preferred as it's more efficient
    # and guarantees real eigenvalues and orthonormal eigenvectors.
    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)

    # Find the index of the largest eigenvalue
    largest_eigenvalue_index = np.argmax(eigenvalues)

    # The corresponding eigenvector is the state vector of the closest pure state
    closest_pure_state_vector = eigenvectors[:, largest_eigenvalue_index]

    # Construct the density matrix for this pure state: |ψ⟩⟨ψ|
    # Reshape the vector to be a column vector for the outer product
    column_vector = closest_pure_state_vector.reshape(2, 1)

    return closest_pure_state_vector


def init_quantum(nhood):
    v=nhood
    qr = QuantumRegister(9,'qr')
    qc = QuantumCircuit(qr,name='conway')
    v = np.array(v)[0]
    for qubit,state in enumerate(v):
        qc.initialize(state,qubit)
    qc.h(qr[4])
    
    qc.cy(qr[0],qr[4])
    qc.crz(np.pi/8,qr[1],qr[4])
    qc.cx(qr[3],qr[4])
    qc.crz(-np.pi/8,qr[5],qr[4])
    qc.cry(np.pi/8,qr[7],qr[4])
    qc.h(qr[4])

    job = Aer.get_backend('statevector_simulator').run(qc)
    results = job.result().get_statevector()
    value = partial_trace(results,[0,1,2,3,5,6,7,8])
    value = find_closest_pure_state(value.data)
    return value

def DSQGOL(nhood):

    a = liveliness(nhood)

    value =  nhood[1][1][0]
    value =  nhood[1][1]
    alive = [1,0]
    dead = [0,1]

    if value[0] > 0.98:
        if (a <= 1.5 ):
            value = init_quantum(nhood)
        elif (a > 1.5 and a <= 2.5):
            value = init_quantum(nhood)
        elif (a > 2.5 and a <= 3.5):
            value = init_quantum(nhood)
        elif (a > 3.5):
            value = init_quantum(nhood)
    elif a < 0.02:
        if (a < 1 ):
            value = init_quantum(nhood)
        elif (a > 1 and a <= 1.5):
            value = init_quantum(nhood)
        elif (a > 1.5 and a <= 2.5):
            value = init_quantum(nhood)
        elif (a > 2.5 and a <= 3.5):
            value = init_quantum(nhood)
        elif (a > 3.5):
            value = init_quantum(nhood)
    else:
        if (a < 1 ):
            value = init_quantum(nhood)
        elif (a > 1 and a <= 1.5):
            value = init_quantum(nhood)
        elif (a > 1.5 and a <= 2.5):
            value = init_quantum(nhood)
        elif (a > 2.5 and a <= 3.5):
            value = init_quantum(nhood)
        elif (a > 3.5):
            value=init_quantum(nhood)
    return value
