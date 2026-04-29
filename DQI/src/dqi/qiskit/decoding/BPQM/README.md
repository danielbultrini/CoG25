# Belief Propagation with Quantum Messages (BPQM)

This repository contains a Qiskit implementation of the BPQM algorithm from the paper **"Quantum message-passing algorithm for optimal and efficient decoding"** by Christophe Piveteau and Joe Renes ([arXiv:2109.08170](https://arxiv.org/abs/2109.08170)).

Originally forked from [https://github.com/ChriPiv/quantum-message-passing-paper](https://github.com/ChriPiv/quantum-message-passing-paper) and updated with required things to make it work with moden python

## Directory Structure

```
BPQM/
├── qiskit_impl/          # Qiskit implementation
│   ├── __init__.py
│   ├── linearcode.py     # Binary linear block codes
│   ├── cloner.py         # Quantum cloning utilities
│   ├── decoders.py       # BPQM decoding algorithms
│   └── bpqm.py          # Core BPQM circuit construction
└── main.ipynb           # Example notebook demonstrating usage
```

## Setup

1. **Python version**: 3.10 - 3.12 (recommended: 3.11)
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Key Components

### LinearCode
Represents binary linear block codes with:
- Generator matrix G
- Parity check matrix H
- Factor graph construction for message passing
- Computation graph unrolling for BPQM

### Cloner
Implements quantum cloning operations:
- `VarNodeCloner`: Standard 1→2 symmetric cloning
- `ExtendedVarNodeCloner`: Extended cloning for 1→n copies
- Uses the ENU (equatorial) cloner described in the paper

### Decoders
Main decoding functions:
- `decode_bpqm`: Full BPQM decoding with success probability calculation
- `decode_single_codeword`: Decode a corrupted codeword
- `decode_single_syndrome`: Syndrome-based decoding for error correction
- `create_init_qc`: Initialize quantum states for decoding

### BPQM Circuit Construction
- `tree_bpqm`: Recursively builds BPQM circuits from computation graphs
- `combine_variable`: Variable node operations
- `combine_check`: Check node operations with syndrome information

## Usage Examples

### Basic BPQM Decoding (Qiskit)

```python
import numpy as np
from qiskit_impl.linearcode import LinearCode
from qiskit_impl.cloner import VarNodeCloner
from qiskit_impl.decoders import decode_bpqm

# Define an [8,4] code (Section 6 of the paper)
G = np.array([
    [1, 0, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 1],
])
H = np.array([
    [1, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 1],
])
code = LinearCode(G, H)

# Set channel parameter and cloner
theta = 0.2 * np.pi
cloner = VarNodeCloner(theta)

# Decode a single bit (bit 4) with unrolling depth 2
p_bit = decode_bpqm(
    code, theta, cloner=cloner, height=2,
    mode='bit', bit=4, only_zero_codeword=True
)
print(f"Success probability for bit 4: {p_bit:.4f}")
# Expected: ~0.8602

# Decode the full codeword
p_codeword = decode_bpqm(
    code, theta, cloner=cloner, height=2,
    mode='codeword', order=[0,1,2,3], only_zero_codeword=True
)
print(f"Success probability for codeword: {p_codeword:.4f}")
# Expected: ~0.6893
```

### Syndrome Decoding (Qiskit)

```python
from qiskit import QuantumCircuit
from qiskit_impl.decoders import decode_single_syndrome

# Create syndrome circuit
syndrome = np.array([0, 0, 1, 1])
syndrome_qc = QuantumCircuit(len(syndrome))
for i, s in enumerate(syndrome):
    if s == 1:
        syndrome_qc.x(i)

# Decode
decoded_bits, _, _ = decode_single_syndrome(
    syndrome_qc=syndrome_qc,
    code=code,
    prior=0.5,  # Uniform prior
    theta=0.2 * np.pi,
    height=2,
    shots=1024
)
print(f"Decoded error pattern: {decoded_bits}")
```

## Important Notes on Syndrome Decoding

1. **Multiple Valid Solutions**: For syndrome decoding, there can be multiple error patterns that produce the same syndrome. The decoder may find any valid error pattern, not necessarily the original one.

2. **Probabilistic Nature**: BPQM is a probabilistic decoder. Results may vary between runs, especially for ambiguous cases.

3. **Prior Probability**: The `prior` parameter significantly affects decoding performance:
   - Use high prior (e.g., 0.8-0.9) when errors are unlikely
   - Use 0.5 for uniform prior (no preference)
   - Adjust based on expected error rate

4. **Performance**: Cirq implementation is typically 3-5x faster than Qiskit for the same operations.

## Running Tests

```bash
# Test Qiskit implementation
python test_bpqm_qiskit.py

# Test Cirq implementation  
python test_bpqm_cirq.py
```

### Test Suite Contents

Both test suites include:

1. **[8,4] Code BPQM Decoding**: Validates theoretical success probabilities
   - Single-bit decoding: ~0.8602
   - Full codeword decoding: ~0.6893

2. **Corrupted Codeword Recovery**: Tests error correction capability

3. **Syndrome Decoding**: Tests syndrome-based decoding with various error patterns

4. **Factor Graph Construction**: Verifies graph structure consistency

5. **Different Tree Heights**: Tests performance with different unrolling depths

Based on:
- **Paper**: "Quantum message-passing algorithm for optimal and efficient decoding" by Christophe Piveteau and Joe Renes ([arXiv:2109.08170](https://arxiv.org/abs/2109.08170))
- **Original implementation**: [ChriPiv/quantum-message-passing-paper](https://github.com/ChriPiv/quantum-message-passing-paper)
