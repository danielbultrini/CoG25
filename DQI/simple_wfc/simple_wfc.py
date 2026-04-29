import os
import sys
import numpy as np
import warnings
import pprint

root = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if root not in sys.path:
    sys.path.insert(0, root)

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

from dqi.qiskit.initialization.state_preparation.gates import UnaryAmplitudeEncoding
from dqi.qiskit.initialization.calculate_w import get_optimal_w
from dqi.qiskit.dicke_state_preparation.gates import UnkGate
from dqi.qiskit.decoding.gates import GJEGate
from dqi.utils.counts import post_selection_counts, combine_counts
from dqi.qiskit.decoding.BPQM.qiskit_impl.linearcode import LinearCode
from dqi.qiskit.decoding.BPQM.qiskit_impl.decoders import create_init_qc, decode_single_syndrome
from dqi.qiskit.decoding.BPQM.qiskit_impl.cloner import VarNodeCloner

warnings.filterwarnings("ignore")
pp = pprint.PrettyPrinter(depth=4)

def generate_wfc_constraints(grid_width, grid_height, rule="different"):
    """Generates the linear system (B matrix and v vector) for the WFC grid."""
    num_variables = grid_width * grid_height
    equations = []
    
    def get_index(x, y):
        return y * grid_width + x

    for y in range(grid_height):
        for x in range(grid_width):
            current_idx = get_index(x, y)
            
            # Right neighbor constraint
            if x < grid_width - 1:
                right_idx = get_index(x + 1, y)
                eq = [0] * num_variables
                eq[current_idx] = 1
                eq[right_idx] = 1
                equations.append(eq)
                
            # Bottom neighbor constraint
            if y < grid_height - 1:
                bottom_idx = get_index(x, y + 1)
                eq = [0] * num_variables
                eq[current_idx] = 1
                eq[bottom_idx] = 1
                equations.append(eq)
                
    B_matrix = np.array(equations)
    num_equations = len(equations)
    
    if rule == "different":
        v_vector = np.ones(num_equations, dtype=int)
    elif rule == "same":
        v_vector = np.zeros(num_equations, dtype=int)
    else:
        raise ValueError("Rule must be 'different' or 'same'")
        
    return B_matrix, v_vector

def decode_wfc_output(counts_dict, grid_width, grid_height, num_variables):
    """Parses simulator counts into a 2D spatial grid."""
    if not counts_dict:
        raise ValueError("No valid counts returned from the simulator.")
        
    best_measurement = list(counts_dict.keys())[0]
    split_measurement = best_measurement.split()
    
    solution_string = None
    for segment in split_measurement:
        if len(segment) == num_variables:
            solution_string = segment
            break
            
    if not solution_string:
        solution_string = best_measurement[-num_variables:]
        
    solution_string = solution_string[::-1] #qiskit ordering
    solution_array = np.array(list(solution_string), dtype=int)
    
    return solution_array.reshape((grid_height, grid_width)), best_measurement

def run_quantum_wfc_GJ(grid_width, grid_height, rule="different", shots=100000):
    """
    Runs the Decoded Quantum Interferometry algorithm to solve a WFC generation task.
    
    Args:
        grid_width (int): Width of the procedural grid.
        grid_height (int): Height of the procedural grid.
        rule (str): "different" (checkerboard) or "same" (solid biomes).
        shots (int): Number of shots for the AerSimulator.
        
    Returns:
        wfc_grid (np.ndarray): The final 2D procedural generation map.
        circuit (QuantumCircuit): The generated DQI circuit.
        sorted_counts (dict): The raw sorted measurement counts.
    """
    B, v = generate_wfc_constraints(grid_width, grid_height, rule)
    
    n = len(B[0])  # Number of variables (grid cells)
    m = len(B)     # Number of constraints (equations)
    p, r, ell = 2, 1, 2  # DQI State prep params
    
    # Initialization Circuit
    w = get_optimal_w(m, ell, p, r)
    init_qregs = QuantumRegister(m, name='k')
    initialize_circuit = QuantumCircuit(init_qregs)
    WUE_Gate = UnaryAmplitudeEncoding(num_bit=m, weights=w)
    initialize_circuit.append(WUE_Gate, range(m))

    # Dicke State Circuit
    dicke_qregs = QuantumRegister(m, name='y')
    dicke_circuit = QuantumCircuit(dicke_qregs)
    max_errors = int(np.nonzero(w)[0][-1]) if np.any(w) else 0
    dicke_circuit.append(UnkGate(m, max_errors), range(m))
    # Master Circuit Setup
    dicke_cregs = ClassicalRegister(m, name='cy')
    syndrome_qregs = QuantumRegister(n, name='syndrome')
    syndrome_cregs = ClassicalRegister(syndrome_qregs.size, name='csolution')
    
    dqi_circuit = QuantumCircuit(dicke_qregs, syndrome_qregs, syndrome_cregs, dicke_cregs)
    dqi_circuit.compose(initialize_circuit, inplace=True)
    dqi_circuit.barrier()
    dqi_circuit.compose(dicke_circuit, inplace=True)

    # V Phase Flips
    v_phase_qregs = QuantumRegister(m, name='v')
    v_phase_flip_circuit = QuantumCircuit(v_phase_qregs)
    for i in range(len(v)):
        if v[i] == 1:
            v_phase_flip_circuit.z(i)
    dqi_circuit.barrier()
    dqi_circuit.compose(v_phase_flip_circuit, qubits=range(m), inplace=True)

    # B Matrix Circuit
    B_circuit = QuantumCircuit(dicke_qregs, syndrome_qregs)
    for i in range(n):
        for j in range(m):
            if B[j][i] == 1:  # Using B directly instead of B.T based on matrix construction
                B_circuit.cx(j, m+i)
    dqi_circuit.barrier()
    dqi_circuit.compose(B_circuit, qubits=list(range(m+n)), inplace=True)
    dqi_circuit.barrier()
    
# Decoding Circuit
    if n > m:
        decoding_circuit = QuantumCircuit(syndrome_qregs, name="GJE")
        GJE_gate = GJEGate(B.T) 
        decoding_circuit.append(GJE_gate, range(n))
        dqi_circuit.append(decoding_circuit, range(m, m+n))
        for i in range(m):
            dqi_circuit.cx(syndrome_qregs[i], dicke_qregs[i])
        dqi_circuit.append(decoding_circuit.inverse(), range(m, m+n))
    else:
        decoding_circuit = QuantumCircuit(dicke_qregs[:n], name="GJE")
        GJE_gate = GJEGate(B.T) 
        decoding_circuit.append(GJE_gate, range(n))
        dqi_circuit.append(decoding_circuit.inverse(), range(n))
        for i in range(n):
            dqi_circuit.cx(syndrome_qregs[i], dicke_qregs[i])
                        
    for i in range(n):
        dqi_circuit.h(m+i)
        
    # Measurement
    dqi_circuit.barrier()
    dqi_circuit.measure(dicke_qregs, dicke_cregs)
    dqi_circuit.measure(syndrome_qregs[::-1], syndrome_cregs)

    # Simulation
    simulator = AerSimulator()
    transpiled_circuit = transpile(dqi_circuit, backend=simulator)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    counts = result.get_counts(dqi_circuit)
    
    # Post-process
    sorted_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    
    # Decode final shape
    wfc_grid, _ = decode_wfc_output(sorted_counts, grid_width, grid_height, n)
    
    return wfc_grid, dqi_circuit, sorted_counts



def run_quantum_wfc_BP(grid_width, grid_height, rule="different", shots=100000):
    """
    Runs the Decoded Quantum Interferometry algorithm to solve a WFC generation task.
    
    Args:
        grid_width (int): Width of the procedural grid.
        grid_height (int): Height of the procedural grid.
        rule (str): "different" (checkerboard) or "same" (solid biomes).
        shots (int): Number of shots for the AerSimulator.
        
    Returns:
        wfc_grid (np.ndarray): The final 2D procedural generation map.
        circuit (QuantumCircuit): The generated DQI circuit.
        sorted_counts (dict): The raw sorted measurement counts.
    """
    B, v = generate_wfc_constraints(grid_width, grid_height, rule)
    
    n = len(B[0])  # Number of variables (grid cells)
    m = len(B)     # Number of constraints (equations)
    p, r, ell = 2, 1, 2  # DQI State prep params
    
    # Initialization Circuit
    w = get_optimal_w(m, ell, p, r)
    init_qregs = QuantumRegister(m, name='k')
    initialize_circuit = QuantumCircuit(init_qregs)
    WUE_Gate = UnaryAmplitudeEncoding(num_bit=m, weights=w)
    initialize_circuit.append(WUE_Gate, range(m))

    # Dicke State Circuit
    dicke_qregs = QuantumRegister(m, name='y')
    dicke_circuit = QuantumCircuit(dicke_qregs)
    max_errors = int(np.nonzero(w)[0][-1]) if np.any(w) else 0
    dicke_circuit.append(UnkGate(m, max_errors), range(m))
    
    theta = 0.2 * np.pi
    cloner = VarNodeCloner(theta)
    code = LinearCode(None, B.T)
    syndrome_qc = QuantumCircuit(code.hk)

    decoded_bits, decoded_qubits, qc_decode = decode_single_syndrome(
    syndrome_qc=syndrome_qc,
    code=code,
    prior=0.5,
    theta=theta,
    height=2,
    shots=1024,
    debug=True,
    run_simulation=False
    )

    # Master Circuit Setup
    dicke_cregs = ClassicalRegister(m, name='cy')
    syndrome_qregs = QuantumRegister(n, name='syndrome')
    syndrome_cregs = ClassicalRegister(syndrome_qregs.size, name='csolution')
    ancilla_qregs =  QuantumRegister(qc_decode.num_qubits - m -n , name='ancilla')
    # Build the DQI circuit by composing initialization and Dicke circuits
    dqi_circuit = QuantumCircuit(dicke_qregs, syndrome_qregs, syndrome_cregs, dicke_cregs, ancilla_qregs)
    dqi_circuit.compose(initialize_circuit, inplace=True)
    dqi_circuit.barrier()
    dqi_circuit.compose(dicke_circuit, inplace=True)

    # V Phase Flips
    v_phase_qregs = QuantumRegister(m, name='v')
    v_phase_flip_circuit = QuantumCircuit(v_phase_qregs)
    for i in range(len(v)):
        if v[i] == 1:
            v_phase_flip_circuit.z(i)
    dqi_circuit.barrier()
    dqi_circuit.compose(v_phase_flip_circuit, qubits=range(m), inplace=True)

    # B Matrix Circuit
    B_circuit = QuantumCircuit(dicke_qregs, syndrome_qregs)
    for i in range(n):
        for j in range(m):
            if B[j][i] == 1:  # Using B directly instead of B.T based on matrix construction
                B_circuit.cx(j, m+i)
    dqi_circuit.barrier()
    dqi_circuit.compose(B_circuit, qubits=list(range(m+n)), inplace=True)
    dqi_circuit.barrier()
    
    dqi_circuit.compose(qc_decode, 
                        qubits=list(range(dicke_qregs.size, 
                                          qc_decode.num_qubits)) + 
                                list(range(dicke_qregs.size)),
                        inplace=True)                        
    for i in range(n):
        dqi_circuit.h(m+i)
        
    # Measurement
    dqi_circuit.barrier()
    dqi_circuit.measure(dicke_qregs, dicke_cregs)
    dqi_circuit.measure(syndrome_qregs[::-1], syndrome_cregs)

    # Simulation
    simulator = AerSimulator()
    transpiled_circuit = transpile(dqi_circuit, backend=simulator)
    result = simulator.run(transpiled_circuit, shots=shots).result()
    counts = result.get_counts(dqi_circuit)
    
    # Post-process
    sorted_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    
    # Decode final shape
    wfc_grid, _ = decode_wfc_output(sorted_counts, grid_width, grid_height, n)
    
    return wfc_grid, dqi_circuit, sorted_counts


if __name__ == "__main__":
    # Define grid dimensions
    W, H = 3, 3 
    
    print(f"Running Quantum WFC on {W}x{H} grid...")
    final_grid, qc, counts = run_quantum_wfc_GJ(grid_width=W, grid_height=H, rule="different")
    
    print("\n--- Final Procedurally Generated Grid ---")
    print(final_grid)
    
    print("\n--- Visualized Biomes ---")
    biome_map = {0: "🟦", 1: "🟩"}
    for row in final_grid:
        print("".join([biome_map[cell] for cell in row]))
        
    # qc.draw('mpl')