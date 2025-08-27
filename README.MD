Workshop tutorial for CoG25 - Daniel Bultrini
minimal environment is python with qiskit 2.x , qiskit-aer 0.17 and pygame 2.x . For the purposes of the workshop you can just do 
```
pip install qiskit
pip install qiskit-aer
pip install pygame
```
You should change and play around with the code in `qrules.py`, specifically the `init_quantum` function. You can use more or less qubits here, but you must trace out all qubits but one or do something else that returns a valid single qubit statevector as an output (a normalizecd complex vector/list of length two).

To learn how to use qiskit you might want to look at the tutorials/lessons on this page: https://quantum.cloud.ibm.com/learning/en


thanks goes to Enrique de la Torre and Xiang (@ttosouth) for having worked on this with me many years ago. This version is updated for IEE CoG

Please ask any questions in the issues.
