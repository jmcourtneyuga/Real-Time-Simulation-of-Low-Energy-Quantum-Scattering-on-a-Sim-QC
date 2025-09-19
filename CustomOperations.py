#!/usr/bin/env python
# coding: utf-8

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer  
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.circuit.library import XGate, CCXGate, CRXGate
from qiskit.quantum_info import Statevector
import re

def cqft(qc, register, n_qubits):

    #I'm applying the swap first because bit order is backwards!
    for i in range(n_qubits // 2):
        qc.swap(register[i], register[n_qubits - 1 - i])
    
    # Then apply standard QFT
    for i in range(n_qubits):
        qc.h(register[i])
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            qc.cp(angle, register[j], register[i])

    qc.x(register[n_qubits-1])  # Flip the highest bit to swap halves

def ciqft(qc, register, n_qubits):

    # Flip the highest bit back
    qc.x(register[n_qubits-1])  # Flip highest bit again

    # First reverse qubit order (part of QFT^(-1))
    for i in reversed(range(n_qubits)):
        for j in reversed(range(i + 1, n_qubits)):
            angle = -np.pi / (2 ** (j - i))
            qc.cp(angle, register[j], register[i])
        qc.h(register[i])
    
    # Then reverse bits AFTER inverse QFT to flip bit order back
    for i in range(n_qubits // 2):
        qc.swap(register[i], register[n_qubits - 1 - i])
