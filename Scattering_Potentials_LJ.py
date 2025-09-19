#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import os
import sys
import getpass
import subprocess
import datetime
import random

# Get username
USERNAME = getpass.getuser()
HOME_PATH = f"/home/{USERNAME}"
SCRATCH_PATH = f"/scratch/{USERNAME}"

# Define paths
LIB_DIR = os.path.join(HOME_PATH, "lib")
CACHE_DIR = os.path.join(SCRATCH_PATH, "coefficient_cache")
OUTPUT_DIR = os.path.join(SCRATCH_PATH, "simulation_output")
DATA_DIR = os.path.join(SCRATCH_PATH, "simulation_data")

# Create directories
for directory in [LIB_DIR, CACHE_DIR, OUTPUT_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)
    print(f"Created/verified directory: {directory}")


# In[3]:


# Compile C Library
def compile_c_library():

    #Look for the source file
    c_source_path = os.path.join(HOME_PATH, "precomp_coefficients.c")
    lib_path = os.path.join(LIB_DIR, "precomp_coefficients.so")
    
    # Check if source file exists
    if not os.path.exists(c_source_path):
        print(f"Error: C code not found at {c_source_path}")
        return False
    
    # Build compile command
    compile_cmd = [
        "gcc", "-O3", "-fopenmp", "-fPIC", "-shared",
        "-o", lib_path,
        c_source_path,
        "-lm"
    ]
    
    print(f"Compiling C library...")
    print(f"Command: {' '.join(compile_cmd)}")

    #I made a safe compilation but it's not robust i.e. if the compilation fails everything stops, and it doesn't safely fix the error or
    #make a strong second attempt to compile
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"C code Successfully compiled to {lib_path}")
            # Check file size to ensure it compiled properly
            file_size = os.path.getsize(lib_path)
            print(f"Library size: {file_size:,} bytes")
            return True
        else:
            print(f"Compilation failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error during compilation: {e}")
        return False

# Compile the library
if compile_c_library():
    print("\nLibrary ready")
else:
    print("\nLibrary not read, compilation failed")


# # Basic Imports

# In[4]:


#Magic numbers: 
#QUBITS_NUM
#ANCILLA_QUBITS
#START_TIME
#TIMESTEP
#STOP_TIME
#VERTICAL_OFFSET
#MASS M
#BOX SIZE l
#HBAR
#X_0 INITIAL POSITION
#P_0 INITIAL MOMENTUM
#INITIAL WAVEPACKET WIDTH \delta
#POLYNOMIAL_TYPE
#POLYNOMIAL PARAMS


# In[5]:


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, RYGate, CXGate, MCPhaseGate
from qiskit.circuit.library import XGate, RXGate, CCXGate, CRXGate, RZGate, CRZGate, SwapGate, QFT
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector, partial_trace, DensityMatrix
from qiskit_aer import Aer

from scipy.optimize import differential_evolution, minimize, OptimizeResult, brentq
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numdifftools as nd
import numpy as np
from decimal import Decimal, getcontext

import itertools
from itertools import product, combinations
from functools import partial, lru_cache
from collections import defaultdict
from queue import Queue
from tqdm import tqdm
import multiprocessing as mp
import threading
import pickle
import time
from datetime import datetime
import traceback
import tempfile
import os
import json
import ctypes
import struct


# ### Custom Imports|

# In[6]:


from CustomOperations import * #imports cqft, ciqft


# # How many qubits?

# In[7]:


QUBITS_NUM = 11 #number of Position Register qubits (longer runtime scales exponentially with number of qubits)
start_time = 0.0 #start time
timestep = 30.0 #Trotter step size (smaller step size, longer runtime, scales linearly)
stop_time = 3000010.0 #stop time (longer stop_time, longer runtime total, scales linearly)
trial = 1.0
vertical_offset = 0.0


# In[8]:


current_time = int(time.time())
simulator_backend = Aer.get_backend('qasm_simulator')


# # **Magic** Parameters

# In[9]:


#By Magic Parameters, I mean this in the way Computer scientists use the word "magic"
#These are numbers that dictate how the whole system behaves

#If using VQE for state preparation, you won't need x_0, p_0 or δ. The variational ansatz you import will produce a good approximation
#of the wavepacket shape. You may need to adjust scaling or where the potential is centered based on the potential's ground state
#The code is not robust to errors, so if you input an invalid term in the magic parameters, it'll throw up a red ERROR message instead
#of actually handling the error.
#If you're familiar with error messages though, it should be pretty clear what's going wrong.


#This isn't a mysterious code. Some parts are rather convoluted, but the algorithm is super simple, being a Suzuki-Trotter evolution of a Hamiltonian
#H = T+V. The partial tracing of the density matrix is simple, the readouts should be legible. The only part which I recommend you leave alone is the 
#specific parameterization for the potential energy terms. I made those myself, those numbers are correct and directly correspond to the polynomial degree.


# In[10]:


m_1 = 72835 #argon
m_2 = 72835 #argon
L = 60 #box size
ħ = 1.0 #atomic units for hbar

d = L/2 #Box is defined from (-d to d)
N = 2**QUBITS_NUM #number of states
Δ_x = L / N #position grid spacing
Δ_p  = 2*np.pi / (N * Δ_x) #momentum grid spacing

x_0 = 20 - Δ_x/2 #need Δ_x shift to initialize in the true middle, due to an even number of states.
p_0 = -0.8 #momentum of initialized Gaussian wavepacket
δ = 0.55 #width of initialized Gaussian wavepacket

epsilon = 0.000379067
sigma = 6.430738

Nyquist = np.pi / Δ_x #max / min momentum

PolynomialType = "Lennard-Jones"
current_params = [epsilon, sigma]

#reduced mass
m = (m_1 * m_2) / (m_1 + m_2)

statevector = []
System = f"Ar-Ar_fscattering_{L}L_{timestep}timestep"
current_date = datetime.now().strftime("%Y-%m-%d")


# # DEFINE KINETIC COEFFICIENTS

# In[11]:


β_kinetic = (-Nyquist - p_0) / (Δ_p)
γ_kinetic = (Δ_p)**2 / (2*m*ħ)

θ_0 = -timestep * (γ_kinetic * β_kinetic**2)  # Global Phase
θ_1 = -2 * timestep * γ_kinetic * β_kinetic  # Linear Term
θ_2 = -timestep * γ_kinetic  # Quadratic term
kinetic_coeffs = [θ_0, θ_1, θ_2]


# # Helper Functions

# In[12]:


def record_highbit_state_z_basis_single_tau(main_circuit, position_register, QUBITS_NUM, tau, filename="ancilla_states.txt"):
    simulator = Aer.get_backend('statevector_simulator')
    transpiled_circuit = transpile(main_circuit, simulator)
    result = simulator.run(transpiled_circuit).result()
    statevector = Statevector(result.get_statevector())
    pauli_z = np.array([[1, 0], [0, -1]])
    ancilla_index = main_circuit.find_bit(position_register[QUBITS_NUM-1]).index
    z_expectation_value = np.real(statevector.expectation_value(Operator(pauli_z), [ancilla_index]))
    population_fraction = (z_expectation_value + 1) / 2
    with open(filename, "a") as file:
        file.write(f"{tau}\t{population_fraction}\n")


# # Record Momentum Distribution after CQFT

# In[13]:


class BatchedDataRecorder:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.position_buffer = []
        self.momentum_buffer = []
        self.ancilla_buffer = []
        self.lock = threading.Lock()
        self.write_threads = []
        
    def add_position_record(self, time, probabilities, filename):
        with self.lock:
            # Store exact copy to avoid any modification issues
            self.position_buffer.append((time, np.copy(probabilities), filename))
            
            if len(self.position_buffer) >= self.batch_size:
                self._flush_position_buffer_async()
                
    def add_momentum_record(self, time, probabilities, filename):
        with self.lock:
            # Store exact copy
            self.momentum_buffer.append((time, np.copy(probabilities), filename))
            
            if len(self.momentum_buffer) >= self.batch_size:
                self._flush_momentum_buffer_async()
                
    def add_ancilla_record(self, time, population_fraction, filename):
        with self.lock:
            self.ancilla_buffer.append((time, population_fraction, filename))
            
            if len(self.ancilla_buffer) >= self.batch_size:
                self._flush_ancilla_buffer_async()
                
    def _flush_position_buffer_async(self):
        if not self.position_buffer:
            return
            
        # Group by filename
        files_data = {}
        for time, probs, filename in self.position_buffer:
            if filename not in files_data:
                files_data[filename] = []
            files_data[filename].append((time, probs))
        
        self.position_buffer.clear()
        
        # Start async write for each file
        for filename, data in files_data.items():
            thread = threading.Thread(
                target=self._write_position_data,
                args=(filename, data)
            )
            thread.start()
            self.write_threads.append(thread)
            
    def _flush_momentum_buffer_async(self):
        if not self.momentum_buffer:
            return
            
        # Group by filename
        files_data = {}
        for time, probs, filename in self.momentum_buffer:
            if filename not in files_data:
                files_data[filename] = []
            files_data[filename].append((time, probs))
        
        self.momentum_buffer.clear()
        
        # Start async write for each file
        for filename, data in files_data.items():
            thread = threading.Thread(
                target=self._write_momentum_data,
                args=(filename, data)
            )
            thread.start()
            self.write_threads.append(thread)
            
    def _flush_ancilla_buffer_async(self):
        if not self.ancilla_buffer:
            return
            
        # Group by filename
        files_data = {}
        for time, pop, filename in self.ancilla_buffer:
            if filename not in files_data:
                files_data[filename] = []
            files_data[filename].append((time, pop))
        
        self.ancilla_buffer.clear()
        
        # Start async write for each file
        for filename, data in files_data.items():
            thread = threading.Thread(
                target=self._write_ancilla_data,
                args=(filename, data)
            )
            thread.start()
            self.write_threads.append(thread)
            
    def _write_position_data(self, filename, data):
        with open(filename, 'a') as f:
            for time, probs in data:
                line = f"{time:.1f}\t" + "\t".join(f"{x:.14f}" for x in probs) + "\n"
                f.write(line)
                
    def _write_momentum_data(self, filename, data):
        with open(filename, 'a') as f:
            for time, probs in data:
                line = f"{time:.1f}\t" + "\t".join(f"{x:.14f}" for x in probs) + "\n"
                f.write(line)
                
    def _write_ancilla_data(self, filename, data):
        with open(filename, 'a') as f:
            for time, pop in data:
                f.write(f"{time}\t{pop}\n")
                
    def flush_all(self):
        with self.lock:
            self._flush_position_buffer_async()
            self._flush_momentum_buffer_async()
            self._flush_ancilla_buffer_async()
            
        # Wait for all writes to complete
        self.wait_for_writes()
        
    def wait_for_writes(self):
        for thread in self.write_threads:
            if thread.is_alive():
                thread.join()
        self.write_threads.clear()
        
    def cleanup_finished_threads(self):
        self.write_threads = [t for t in self.write_threads if t.is_alive()]


# In[14]:


def record_highbit_state_z_basis_single_tau(main_circuit, position_register, QUBITS_NUM, tau, recorder, filename):
    simulator = Aer.get_backend('statevector_simulator')
    transpiled_circuit = transpile(main_circuit, simulator)
    result = simulator.run(transpiled_circuit).result()
    statevector = Statevector(result.get_statevector())
    pauli_z = np.array([[1, 0], [0, -1]])
    ancilla_index = main_circuit.find_bit(position_register[QUBITS_NUM-1]).index
    z_expectation_value = np.real(statevector.expectation_value(Operator(pauli_z), [ancilla_index]))
    population_fraction = (z_expectation_value + 1) / 2
    
    recorder.add_ancilla_record(tau, population_fraction, filename)


def record_momentum_distribution(circuit, position_register, tau, recorder, filename):
    #Uses batched input/output
    simulator = Aer.get_backend('statevector_simulator')
    transpiled_circuit = transpile(circuit, simulator)
    result = simulator.run(transpiled_circuit).result()
    statevector = Statevector(result.get_statevector())
    rho_position = DensityMatrix(statevector)
    probabilities = np.real(np.diag(rho_position.data))
    probabilities = np.flip(probabilities)
    
    recorder.add_momentum_record(tau, probabilities, filename)


# # Calculate Potentials for Visualization

# In[15]:


class QuantumPotential:
    def __init__(self, params, vertical_offset=0.0):
        self.params = params
        self.vertical_offset = vertical_offset
        self.d = d
        self.Δ_x = Δ_x
        
    def calculate_classical(self, x_values, scale_factor=1.0):
        raise NotImplementedError
        
    def apply_to_circuit(self, circuit, position_register, τ):
        raise NotImplementedError


# # Initialize Gaussian Wavepacket

# In[16]:


def initialize_gaussian_wavepacket(x_0, p_0, δ):
    psi = np.zeros(N, dtype=complex)
    norm_factor = (1/(2*np.pi*δ**2))**(1/4)
    
    for position in range(N):
        x = -d + position * Δ_x
        gaussian = np.exp(-(x - x_0)**2 / (4*δ**2))
        psi[position] = gaussian
    
    psi = norm_factor * psi
    final_norm = np.sqrt(np.sum(np.abs(psi)**2))
    psi = psi / final_norm
    
    return psi


# # Get Expectation Values Over Time

# In[17]:


def calculate_expectation_value(probabilities, positions, box_size=4):
    x_min = 0
    x_max = L
    box_size = x_max - x_min
    scaling = box_size / len(positions)
    real_positions = np.linspace(x_min, x_max, len(positions))
    expectation = np.sum(probabilities * real_positions)
    return expectation

def calculate_p_expectation_value(probabilities, positions, box_size=4):
    x_min = -d
    x_max = d
    box_size = x_max - x_min
    scaling = box_size / len(positions)
    real_positions = np.linspace(x_min, x_max, len(positions))
    expectation = np.sum(probabilities * real_positions)
    return expectation


# In[18]:


def global_phase(circuit, angle, register):
    circuit.p(angle, register[0])
    circuit.x(register[0])
    circuit.p(angle, register[0])
    circuit.x(register[0])
    return circuit


# In[19]:


def kinetic_first_order(circuit, angle, register, QUBITS_NUM):
    for qubit in range(QUBITS_NUM):
        position_scaling = 2**qubit
        circuit.p(angle * position_scaling, register[qubit])
    return circuit


# In[20]:


def kinetic_second_order(circuit, angle, register, QUBITS_NUM):
    for control in range(QUBITS_NUM):
        bit_order = 2 * control
        position_scaling = 2**bit_order
        circuit.p(angle * position_scaling, register[control])
        
        for target in range(QUBITS_NUM):
            if target != control:
                bit_order = control + target
                position_scaling = 2**bit_order
                circuit.cp(angle * position_scaling, register[control], register[target])
    return circuit


# In[21]:


def kinetic_term(circuit, position_register, QUBITS_NUM, τ, kinetic_coeffs):
    θ_0, θ_1, θ_2 = kinetic_coeffs 
    global_phase(circuit, θ_0, position_register)
    kinetic_first_order(circuit, θ_1, position_register, QUBITS_NUM)
    kinetic_second_order(circuit, θ_2, position_register, QUBITS_NUM)
    return circuit


# In[22]:


class EffectivePotential(QuantumPotential):
    def __init__(self, lj_params, centrifugal_params, vertical_offset=0.0, 
                 coefficients_by_power=None):
        all_params = lj_params + centrifugal_params
        super().__init__(all_params, vertical_offset)
    
        self.epsilon = lj_params[0]
        self.sigma = lj_params[1]
        self.partial_wave = centrifugal_params[0]
        self.m = centrifugal_params[1]

        self.coefficients_by_power = coefficients_by_power or {}
    
    def calculate_classical(self, x_values, scale_factor=1.0):
        x_safe = np.where(x_values > 0, x_values, 1e-10)
        
        # LJ potential
        sigma_over_r = self.sigma / x_safe
        V_lj = 4 * self.epsilon * ((sigma_over_r)**12 - (sigma_over_r)**6)
        
        # Centrifugal potential (only if ℓ > 0)
        if self.partial_wave > 0:
            V_cent = (self.partial_wave * (self.partial_wave + 1)) / (2 * self.m * x_safe**2)
        else:
            V_cent = 0
        
        return (V_lj + V_cent) * scale_factor + self.vertical_offset
    
    def apply_to_circuit(self, circuit, position_register, τ):
        n_qubits = len(position_register)
        
        for power, coefficients in self.coefficients_by_power.items():
            discretization_factor = (1.0 / (L/N)) ** power 
            
            for indices_tuple, physical_coefficient in coefficients.items():
                angle = -τ * physical_coefficient * discretization_factor
                
                num_controls = len(indices_tuple)
                
                if num_controls == 1:
                    circuit.p(angle, position_register[indices_tuple[0]])
                else:
                    controls = [position_register[i] for i in indices_tuple[:-1]]
                    target = position_register[indices_tuple[-1]]
                    circuit.append(MCPhaseGate(angle, num_controls-1), controls + [target])
        
        return circuit


# In[23]:


def sum_cached_coefficients(lj_cached_coeffs, centrifugal_cached_coeffs, lj_params, cent_params):
    combined_by_power = {}
    
    # Extract physical parameters
    epsilon = lj_params[0]  
    sigma = lj_params[1]
    l = cent_params[0]  # Angular momentum quantum number
    mu = cent_params[1]  # Reduced mass
    
    # LJ repulsive term:
    if lj_cached_coeffs and 12 in lj_cached_coeffs:
        if epsilon != 0: 
            combined_by_power[12] = {}
            physical_coeff_12 = 4 * epsilon * (sigma**12)
            for indices_tuple, precomp_coeff in lj_cached_coeffs[12].items():
                combined_by_power[12][indices_tuple] = physical_coeff_12 * precomp_coeff
    
    # LJ attractive term:
    if lj_cached_coeffs and 6 in lj_cached_coeffs:
        if epsilon != 0: 
            combined_by_power[6] = {}
            physical_coeff_6 = -4 * epsilon * (sigma**6)
            for indices_tuple, precomp_coeff in lj_cached_coeffs[6].items():
                combined_by_power[6][indices_tuple] = physical_coeff_6 * precomp_coeff
    
    # Centrifugal term:
    if centrifugal_cached_coeffs and 2 in centrifugal_cached_coeffs:
        if l > 0: 
            combined_by_power[2] = {}
            physical_coeff_2 = l * (l + 1) / (2 * mu)
            for indices_tuple, precomp_coeff in centrifugal_cached_coeffs[2].items():
                combined_by_power[2][indices_tuple] = physical_coeff_2 * precomp_coeff
    
    return combined_by_power


# In[24]:


# Set decimal precision for Python side
getcontext().prec = 50  # 50 decimal places

class FastLJCoefficientsHP:
    def __init__(self, lib_path=None):
        if lib_path is None:
            lib_path = os.path.join(LIB_DIR, "precomp_coefficients.so")
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"High-precision library not found at {lib_path}")
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures using double interface
        # The C code internally uses long double but exports as double
        self.lib.compute_lj_coefficients.argtypes = [
            ctypes.c_int,     # n_qubits
            ctypes.c_double,  # epsilon (not used)
            ctypes.c_double,  # sigma (not used)
            ctypes.POINTER(ctypes.c_double),  # coeffs_12
            ctypes.POINTER(ctypes.c_double),  # coeffs_6
        ]
        
        self.lib.compute_arbitrary_power_coefficients.argtypes = [
            ctypes.c_int,     # n_qubits
            ctypes.c_int,     # power
            ctypes.POINTER(ctypes.c_double),  # output
        ]
        
        self.lib.print_precision_info.argtypes = []
        self.lib.test_precision.argtypes = []
        
        # For string-based high precision export
        self.lib.compute_lj_coefficients_string.argtypes = [
            ctypes.c_int,                    # n_qubits
            ctypes.POINTER(ctypes.c_char),   # coeffs_12_strings
            ctypes.POINTER(ctypes.c_char),   # coeffs_6_strings
            ctypes.c_int                     # string_length
        ]
        
        # Print precision info on initialization
        print("Initializing high-precision library...")
        self.lib.print_precision_info()
    
    def compute_lj_coefficients(self, n_qubits, epsilon, sigma, L, N):
        n_states = 1 << n_qubits
        string_length = 64 
    
        use_string_export = True
        
        if use_string_export and hasattr(self.lib, 'compute_lj_coefficients_string'):
            coeffs_12_strings = (ctypes.c_char * (n_states * string_length))()
            coeffs_6_strings = (ctypes.c_char * (n_states * string_length))()
            
            print(f"Computing high-precision coefficients for {n_qubits} qubits (string export)...")
            self.lib.compute_lj_coefficients_string(
                n_qubits, 
                coeffs_12_strings, 
                coeffs_6_strings, 
                string_length
            )
            
            coeffs_12 = []
            coeffs_6 = []
            for i in range(n_states):
                str_12 = coeffs_12_strings[i*string_length:(i+1)*string_length].decode('utf-8').strip('\x00')
                str_6 = coeffs_6_strings[i*string_length:(i+1)*string_length].decode('utf-8').strip('\x00')
                
                try:
                    coeffs_12.append(Decimal(str_12) if str_12 else Decimal('0'))
                    coeffs_6.append(Decimal(str_6) if str_6 else Decimal('0'))
                except:
                    coeffs_12.append(Decimal('0'))
                    coeffs_6.append(Decimal('0'))
        else:
            # Fallback: Use double interface (less precision)
            coeffs_12_double = (ctypes.c_double * n_states)()
            coeffs_6_double = (ctypes.c_double * n_states)()
            
            print(f"Computing coefficients for {n_qubits} qubits (double precision fallback)...")
            self.lib.compute_lj_coefficients(
                n_qubits, 0.0, 0.0, 
                coeffs_12_double, coeffs_6_double
            )
            
            # Convert to Decimal
            coeffs_12 = [Decimal(repr(coeffs_12_double[i])) for i in range(n_states)]
            coeffs_6 = [Decimal(repr(coeffs_6_double[i])) for i in range(n_states)]
        
        # Apply physical scaling with high precision
        epsilon = Decimal(repr(epsilon))
        sigma = Decimal(repr(sigma))
        L = Decimal(repr(L))
        N = Decimal(repr(N))
        
        # Convert to Python dictionaries
        result = {6: {}, 12: {}}
        
        for i in range(1, n_states):
            indices = self._bitset_to_indices(i, n_qubits)
            
            try:
                coeff_12 = coeffs_12[i]
                coeff_6 = coeffs_6[i]
                
                if coeff_12 != 0:
                    result[12][tuple(indices)] = float(coeff_12)
                    
                if coeff_6 != 0:
                    result[6][tuple(indices)] = float(coeff_6)
                    
            except Exception as e:
                print(f"Warning: Error processing coefficient {i}: {e}")
                continue
        
        print(f"Computed LJ coefficients:")
        print(f"  12th power: {len(result[12])} non-zero coefficients")
        print(f"  6th power: {len(result[6])} non-zero coefficients")
        
        # Show sample coefficients
        if result[12]:
            # Check some critical single-bit states
            for bit in [11, 12, 13]:
                state_tuple = (bit,)
                if state_tuple in result[12]:
                    print(f"  State 2^{bit} = {2**bit}: coeff_12 = {result[12][state_tuple]:.10e}")
        
        return result
    
    def compute_centrifugal_coefficients(self, n_qubits, l_value, m, L, N):
        n_states = 1 << n_qubits
        coeffs_2 = (ctypes.c_double * n_states)()
        
        # Get coefficients for power 2
        self.lib.compute_arbitrary_power_coefficients(n_qubits, 2, coeffs_2)
        
        # High-precision scaling
        l = Decimal(repr(l_value))
        mu = Decimal(repr(m))
        L = Decimal(repr(L))
        N = Decimal(repr(N))
        
        physical_coeff = l * (l + Decimal('1')) / (Decimal('2') * mu)
        
        print(f"Centrifugal scaling for l = {l_value}:")
        print(f"  Physical coefficient: {physical_coeff:.25e}")
        
        result = {}
        
        for i in range(1, n_states):
            indices = self._bitset_to_indices(i, n_qubits)
            try:
                precomp_coeff = Decimal(repr(coeffs_2[i]))
                if precomp_coeff != 0:
                    result[tuple(indices)] = float(precomp_coeff)
                    
            except Exception as e:
                print(f"Error processing coefficient {i}: {e}")
                continue
        
        print(f"  {len(result)} non-zero coefficients")
        return {2: result}
    
    def _bitset_to_indices(self, bitset, n_qubits):
        indices = []
        for i in range(n_qubits):
            if bitset & (1 << i):
                indices.append(i)
        return sorted(indices)
    
    def verify_precision(self):
        print("\nRunning precision verification...")
        if hasattr(self.lib, 'test_precision'):
            self.lib.test_precision()


# In[25]:


def compile_library():
    c_source_path = os.path.join(HOME_PATH, "precomp_coefficients.c")
    lib_path = os.path.join(LIB_DIR, "precomp_coefficients.so")
    
    if not os.path.exists(c_source_path):
        print(f"ERROR: C source file not found at {c_source_path}")
        return False
    
    # Compile with extended precision flags
    compile_cmd = [
        "gcc", 
        "-O3",                    # Optimization
        "-fopenmp",               # OpenMP parallelization
        "-fPIC",                  # Position independent code
        "-shared",                # Shared library
        "-mlong-double-128",      # Force 128-bit long double if available
        "-std=c11",               # C11 standard
        "-Wall",                  # All warnings
        "-Wextra",              
        "-o", lib_path,
        c_source_path,
        "-lm"                  
    ]
    
    print(f"Compiling high-precision C library...")
    print(f"Command: {' '.join(compile_cmd)}")
    
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully compiled to {lib_path}")
            file_size = os.path.getsize(lib_path)
            print(f"Library size: {file_size:,} bytes")
            return True
        else:
            print(f"Compilation failed:")
            print(result.stderr)
            print("\nRetrying without 128-bit long double flag...")
            compile_cmd.remove("-mlong-double-128")
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Successfully compiled with default long double precision")
                return True
            else:
                print(f"Compilation failed again:")
                print(result.stderr)
                return False
    except Exception as e:
        print(f"Error during compilation: {e}")
        return False

if compilecoeff_library():
    print("\nHigh-precision library ready!")


# In[26]:


def compute_and_cache_coefficients(potential_type, params, n_qubits,
                                   max_power=12, n_cores=None,
                                   cache_dir=None, use_high_precision=True):
    if cache_dir is None:
        cache_dir = CACHE_DIR
    
    os.makedirs(cache_dir, exist_ok=True)
    
    precision_tag = "hp" if use_high_precision else "dp"
    param_str = "_".join([f"{p}".replace('.', 'p') for p in params])
    cache_key = f"{potential_type}_{n_qubits}q_L{L}_N{N}_{precision_tag}_{param_str}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    lock_file = os.path.join(cache_dir, f"{cache_key}.lock")

    got_lock = False
    try:
        while not os.path.exists(cache_file):
            try:
                os.close(os.open(lock_file, os.O_CREAT | os.O_EXCL))
                got_lock = True
                print(f"Process {os.getpid()} acquired lock for {cache_key}")
                break 
            except FileExistsError:
                time.sleep(random.uniform(0.5, 1.5))

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache loading failed for {os.getpid()}: {e}, recomputing...")

        if got_lock:
            print(f"Process {os.getpid()} computing {potential_type} coefficients...")
            if use_high_precision:
                fast_computer = FastLJCoefficientsHP()
            else:
                raise NotImplementedError("Standard precision not implemented, and the high precision code failed :(")

            if potential_type == "Lennard-Jones":
                epsilon, sigma = params
                coefficients = fast_computer.compute_lj_coefficients(n_qubits, epsilon, sigma, L, N)
            elif potential_type == "Centrifugal":
                l_value, m = params
                coefficients = fast_computer.compute_centrifugal_coefficients(n_qubits, l_value, m, L, N)
            else:
                raise ValueError(f"Unknown potential type: {potential_type}")

            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(coefficients, f)
            print(f"Process {os.getpid()} saved coefficients to {cache_file}")
            
            return coefficients

    finally:
        if got_lock and os.path.exists(lock_file):
            os.remove(lock_file)


# # Create Position Animation

# In[27]:


def create_quantum_evolution_animation(
    l_value,
    timestep_sampling=100,
    polynomial_type="Lennard-Jones",
    polynomial_params=None,
    centrifugal_params=None,
    show_potential=True,
):
    # Construct file paths based on partial wave
    System = f"Ar-Ar_scattering_{L}L_{timestep}ts"
    current_date = datetime.now().strftime("%Y-%m-%d")
    subfolder = os.path.join(OUTPUT_DIR, f"{QUBITS_NUM}q_PartialWaveSim_{System}_{current_date}", f"l_{l_value:04d}")statevector_filename = os.path.join(subfolder, f"statevectors_l{l_value}.txt")
    momentum_filename = os.path.join(subfolder, f"momentum_l{l_value}.txt")
    
    if not os.path.exists(statevector_filename) or not os.path.exists(momentum_filename):
        print(f"Data files not found for l={l_value}")
        return None
    
    # Load data with sampling
    position_data = np.loadtxt(statevector_filename, delimiter='\t')
    momentum_data = np.loadtxt(momentum_filename, delimiter='\t')
    
    # Sample every Nth timestep
    position_data = position_data[::timestep_sampling]
    momentum_data = momentum_data[::timestep_sampling]
    
    time = position_data[:, 0]
    positions = position_data[:, 1:]
    momentum_distributions = momentum_data[:, 1:]
    
    num_frames = len(time)
    
    # Position values
    position_values = np.linspace(-d, d, N)
    momentum_values = np.linspace(-Nyquist, Nyquist, N)
    
    # Calculate expectation values
    expectation_values = [calculate_expectation_value(positions[i], position_values) for i in range(num_frames)]
    momentum_expectation_values = [calculate_p_expectation_value(momentum_distributions[i], momentum_values) 
                                  for i in range(num_frames)]
    
    # Setup plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Position distribution subplot
    ax1.set_xlabel('Position (a.u.)')
    ax1.set_ylabel('Probability Amplitude')
    bar_width = (position_values[1] - position_values[0]) * 0.8
    bars = ax1.bar(position_values, positions[0], width=bar_width, color='blue', alpha=0.6)
    ax1.set_xlim(-d, d)
    ax1.set_title(f'l={l_value} Partial Wave, {polynomial_type} Potential')
    
    # Add potential overlay
    if show_potential and polynomial_params is not None:
        x_potential = position_values.copy()
        x_potential[np.abs(x_potential) < 0.001] = 0.001
        
        # Calculate effective potential (LJ + centrifugal)
        epsilon_eff = polynomial_params[0]
        sigma_eff = polynomial_params[1]
        
        V_lj = 4 * epsilon_eff * ((sigma_eff/np.abs(x_potential))**12 - (sigma_eff/np.abs(x_potential))**6)
        V_cent = 0 if l_value == 0 else (l_value * (l_value + 1)) / (2 * m * x_potential**2)
        V_total = V_lj + V_cent
        
        ax1_pot = ax1.twinx()
        ax1_pot.plot(x_potential, V_total, 'r-', linewidth=2, label='Total Potential')
        ax1_pot.set_ylabel('Potential Energy (a.u.)', color='r')
        ax1_pot.tick_params(axis='y', labelcolor='r')
        ax1_pot.set_ylim(np.min(V_total[np.isfinite(V_total)]), 
                         min(np.max(V_total[np.isfinite(V_total)]), 0.01))
    
    time_box = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=14,
                       verticalalignment='top', bbox=dict(facecolor='black', color='white'))
    
    # Position density evolution
    ax2.set_xlabel('Time (a.u.)')
    ax2.set_ylabel('Position (a.u.)')
    ax2.set_title('Position Space Density Evolution')
    
    position_density = np.zeros((N, num_frames))
    position_density_plot = ax2.imshow(position_density, aspect='auto',
                                      extent=[time[0], time[-1], d, -d],
                                      cmap='hot', interpolation='gaussian')
    plt.colorbar(position_density_plot, ax=ax2, label='Probability Amplitude')
    position_expectation_line, = ax2.plot([], [], 'c-', linewidth=2, label='<x>')
    ax2.legend(loc='upper right')
    
    # Momentum distribution
    ax3.set_xlabel('Momentum (a.u.)')
    ax3.set_ylabel('Probability Amplitude')
    ax3.set_title('Momentum Distribution')
    momentum_line, = ax3.plot(momentum_values, momentum_distributions[0], 'g-', linewidth=2)
    ax3.set_xlim(-Nyquist, Nyquist)
    ax3.set_ylim(0, np.max(momentum_distributions) * 1.1)
    
    plt.tight_layout()
    
    def update(frame):
        for i, bar in enumerate(bars):
            bar.set_height(positions[frame, i])
        
        time_box.set_text(f't = {time[frame]:.1f}')
        
        position_expectation_line.set_data(time[:frame + 1], expectation_values[:frame + 1])
        
        for t in range(frame + 1):
            position_density[:, t] = positions[t, :]
        position_density_plot.set_array(position_density)
        
        momentum_line.set_ydata(momentum_distributions[frame])
        
        return list(bars) + [time_box, position_expectation_line, position_density_plot, momentum_line]
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True, interval=50, repeat=False)
    
    # Save animation in same directory as data
    animation_filename = os.path.join(subfolder, f"evolution_l{l_value}_sampled{timestep_sampling}.gif")
    
    try:
        ani.save(animation_filename, writer='pillow', fps=20)
        print(f"Animation saved: {animation_filename}")
    except Exception as e:
        print(f"Error saving animation for l={l_value}: {e}")
    
    plt.close(fig)
    return animation_filename


# # Create Momentum Animation

# In[28]:


def create_momentum_evolution_animation(
    l_value,
    timestep_sampling=100,
    polynomial_type="Lennard-Jones",
):
    System = f"Ar-Ar_scattering_{L}L_{timestep}ts"
    current_date = datetime.now().strftime("%Y-%m-%d")
    subfolder = os.path.join(OUTPUT_DIR, f"{QUBITS_NUM}q_PartialWaveSim_{System}_{current_date}", f"l_{l_value:04d}")
    momentum_filename = os.path.join(subfolder, f"momentum_l{l_value}.txt")
    
    if not os.path.exists(momentum_filename):
        print(f"Momentum data file not found for l={l_value}")
        return None
    
    # Load data with sampling
    momentum_data = np.loadtxt(momentum_filename, delimiter='\t')
    momentum_data = momentum_data[::timestep_sampling]
    
    time = momentum_data[:, 0]
    momentum_distributions = momentum_data[:, 1:]
    num_frames = len(time)
    
    # Momentum values
    momentum_values = np.linspace(-Nyquist, Nyquist, N)
    
    # Calculate momentum expectation values
    momentum_expectation_values = [calculate_p_expectation_value(momentum_distributions[i], momentum_values) 
                                  for i in range(num_frames)]
    
    # Calculate FFT for frequency analysis
    fft_results = []
    for i in range(num_frames):
        fft_magnitude = np.abs(np.fft.fft(momentum_distributions[i]))
        fft_results.append(fft_magnitude)
    fft_results = np.array(fft_results)
    
    freq_values = np.fft.fftfreq(N, d=timestep * timestep_sampling)
    positive_freq_mask = freq_values >= 0
    freq_values_positive = freq_values[positive_freq_mask]
    
    # Setup plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Momentum density heatmap
    ax1.set_xlabel('Time (a.u.)')
    ax1.set_ylabel('Momentum (a.u.)')
    ax1.set_title(f'l={l_value} Momentum Space Evolution')
    
    small = 1e-10
    momentum_density_log = np.log10(momentum_distributions.T + small)
    
    momentum_density_plot = ax1.imshow(momentum_density_log, aspect='auto',
                                      extent=[time[0], time[-1], Nyquist, -Nyquist],
                                      cmap='viridis', interpolation='gaussian')
    plt.colorbar(momentum_density_plot, ax=ax1, label='log₁₀(Probability)')
    
    momentum_expectation_line, = ax1.plot([], [], 'w-', linewidth=2, label='<p>')
    ax1.legend(loc='upper right')
    
    # Current momentum distribution
    ax2.set_xlabel('Momentum (a.u.)')
    ax2.set_ylabel('Probability Amplitude')
    ax2.set_title('Current Momentum Distribution')
    momentum_line, = ax2.plot(momentum_values, momentum_distributions[0], 'b-', linewidth=2)
    ax2.set_xlim(-Nyquist, Nyquist)
    ax2.set_ylim(0, np.max(momentum_distributions) * 1.1)
    ax2.grid(True, alpha=0.3)
    
    time_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black'))
    
    # Frequency spectrum
    ax3.set_xlabel('Frequency (a.u.⁻¹)')
    ax3.set_ylabel('Spectral Amplitude')
    ax3.set_title('Frequency Spectrum')
    
    initial_fft_positive = fft_results[0][positive_freq_mask]
    frequency_line, = ax3.plot(freq_values_positive, initial_fft_positive, 'g-', linewidth=2)
    
    max_freq_to_show = freq_values_positive[len(freq_values_positive)//4]
    ax3.set_xlim(0, max_freq_to_show)
    ax3.set_ylim(0, np.max(fft_results[:, positive_freq_mask]) * 1.1)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    def update(frame):
        momentum_expectation_line.set_data(time[:frame + 1], momentum_expectation_values[:frame + 1])
        momentum_line.set_ydata(momentum_distributions[frame])
        time_text.set_text(f't = {time[frame]:.1f} a.u.')
        
        current_fft_positive = fft_results[frame][positive_freq_mask]
        frequency_line.set_ydata(current_fft_positive)
        
        return [momentum_expectation_line, momentum_line, time_text, frequency_line]
    
    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True, interval=50, repeat=False)
    
    # Save animation in same directory as data
    animation_filename = os.path.join(subfolder, f"momentum_l{l_value}_sampled{timestep_sampling}.gif")
    
    try:
        ani.save(animation_filename, writer='pillow', fps=20)
        print(f"Momentum animation saved: {animation_filename}")
    except Exception as e:
        print(f"Error saving momentum animation for l={l_value}: {e}")
    
    plt.close(fig)
    return animation_filename


# # Main Function

# In[29]:


def manage_resume_logic(sv_filename, mom_filename, start_time, stop_time, timestep, qubits_num, initial_state_params):
    x_0, p_0, delta = initial_state_params
    n_states = 2**qubits_num
    
    # Always start fresh - clear any existing files
    print(f"Process {os.getpid()}: Starting new simulation from t={start_time:.1f}")
    
    # Clear existing files to ensure clean start
    for filename in [sv_filename, mom_filename]:
        with open(filename, 'w') as f:
            f.write('')  # Create empty file
    
    # Initialize fresh wavepacket
    initial_psi = initialize_gaussian_wavepacket(x_0, p_0, delta)
    current_statevector = np.zeros(n_states, dtype=complex)
    current_statevector[:len(initial_psi)] = initial_psi
    
    return start_time, current_statevector


# In[30]:


def get_last_timestep(filename):
        if not os.path.exists(filename):
            return None
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        return float(last_line.split('\t')[0])
        except:
            return None
        return None


# In[31]:


def run_single_partial_wave_standalone(l_value, lj_base_coeffs, centrifugal_base_coeffs, params):
    FOLDER_NAME = params['folder_name']
    QUBITS_NUM = params['qubits_num']
    timestep = params['timestep']
    start_time = params['start_time']
    stop_time = params['stop_time']
    current_params = params['current_params']
    m = params['m']
    x_0, p_0, delta = params['x_0'], params['p_0'], params['delta']
    kinetic_coeffs = params['kinetic_coeffs']
    vertical_offset = params['vertical_offset']

    subfolder = os.path.join(FOLDER_NAME, f"l_{l_value:04d}")
    os.makedirs(subfolder, exist_ok=True)
    trotter_sv_filename = os.path.join(subfolder, f"statevectors_l{l_value}.txt")
    trotter_mom_filename = os.path.join(subfolder, f"momentum_l{l_value}.txt")

    try:
        # Initialize batched recorder
        recorder = BatchedDataRecorder(batch_size=100)
        
        # Always start fresh - no resume logic
        resume_time, current_statevector = manage_resume_logic(
            trotter_sv_filename, trotter_mom_filename, start_time, stop_time,
            timestep, QUBITS_NUM, (x_0, p_0, delta)
        )
        
        if resume_time >= stop_time:
            print(f"Process {os.getpid()}: l={l_value} already complete.")
            return l_value, True, "Already completed"

        # Coefficient summing
        centrifugal_params = [l_value, m]
        coefficients_by_power = sum_cached_coefficients(
            lj_base_coeffs, centrifugal_base_coeffs, current_params, centrifugal_params
        )
        
        pot = EffectivePotential(
            current_params, centrifugal_params, vertical_offset,
            coefficients_by_power=coefficients_by_power
        )
        
        # Construct circuit
        position_register = QuantumRegister(QUBITS_NUM, name="position")
        simulator = Aer.get_backend('statevector_simulator')
        current_time = resume_time

        print(f"Process {os.getpid()}: Starting l={l_value} from t={current_time:.1f}")

        # Main evolution loop
        step_count = 0
        
        while current_time < stop_time:
            circuit_mom = QuantumCircuit(position_register)
            circuit_mom.initialize(current_statevector, list(position_register))
            cqft(circuit_mom, position_register, QUBITS_NUM)
            
            record_momentum_distribution_batched(
                circuit_mom, position_register, current_time, 
                recorder, trotter_mom_filename
            )

            circuit_evol = QuantumCircuit(position_register)
            circuit_evol.initialize(current_statevector, list(position_register))
            
            # Kinetic evolution
            cqft(circuit_evol, position_register, QUBITS_NUM)
            kinetic_term(circuit_evol, position_register, QUBITS_NUM, timestep, kinetic_coeffs)
            ciqft(circuit_evol, position_register, QUBITS_NUM)
            
            # Potential evolution
            pot.apply_to_circuit(circuit_evol, position_register, timestep)

            # Get evolved state
            result = simulator.run(circuit_evol).result()
            current_state = Statevector(result.get_statevector())
            current_statevector = current_state.data

            # Record Positions
            rho = DensityMatrix(current_state).data
            state_vector_probs = np.real(np.diag(rho))
            recorder.add_position_record(
                current_time + timestep, state_vector_probs, 
                trotter_sv_filename
            )

            current_time += timestep
            step_count += 1
            
            # Periodically clean up finished threads
            if step_count % 1000 == 0:
                recorder.cleanup_finished_threads()
                print(f"Process {os.getpid()}: l={l_value}, t={current_time:.1f}/{stop_time}")

        # Final flush, wait for writes
        print(f"Process {os.getpid()}: Flushing data for l={l_value}...")
        recorder.flush_all()
        
        print(f"Process {os.getpid()}: Completed l={l_value}")
        return l_value, True, None

    except Exception as e:
        error_msg = f"Error for l={l_value}: {e}"
        print(f"Process {os.getpid()}: {error_msg}")
        with open(os.path.join(subfolder, "error.txt"), 'w') as f:
            f.write(f"{datetime.now()}\n{error_msg}\n{traceback.format_exc()}")
        return l_value, False, str(e)


# In[32]:


def run_parallel_partial_waves_main(n_cores, lj_base_coeffs, centrifugal_base_coeffs):

    l_values = list(range(0, 31))

    System = f"Ar-Ar_scattering_{QUBITS_NUM}q_{timestep}ts"
    current_date = datetime.now().strftime("%Y-%m-%d")
    FOLDER_NAME = os.path.join(OUTPUT_DIR,
                               f"{QUBITS_NUM}q_PartialWaveSim_{System}_{current_date}")
    os.makedirs(FOLDER_NAME, exist_ok=True)

    print(f"Output folder: {FOLDER_NAME}")

    shared_params = {
        'folder_name': FOLDER_NAME,
        'cache_dir': CACHE_DIR,
        'qubits_num': QUBITS_NUM,
        'N': N, 'L': L, 'd': d,
        'timestep': timestep,
        'start_time': start_time, 'stop_time': stop_time,
        'current_params': current_params,
        'polynomial_type': PolynomialType,
        'm': m,
        'mass': m, 'x_0': x_0, 'p_0': p_0, 'delta': δ,
        'kinetic_coeffs': kinetic_coeffs,
        'vertical_offset': vertical_offset
    }

    params_file = os.path.join(FOLDER_NAME, "simulation_parameters.json")
    with open(params_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        params_to_save = shared_params.copy()
        for key, value in params_to_save.items():
            if isinstance(value, np.ndarray):
                params_to_save[key] = value.tolist()
        json.dump(params_to_save, f, indent=4)
    print(f"Simulation parameters saved to {params_file}")

    print(f"\nStarting parallel simulations for {len(l_values)} partial waves...")
    print(f"Using {n_cores} parallel cores for l = {min(l_values)} to {max(l_values)}.")

    worker_args = [
        (l, lj_base_coeffs, centrifugal_base_coeffs, shared_params) for l in l_values
    ]
    
    start_time_total = datetime.now()
    results = []
    with mp.Pool(processes=n_cores) as pool:
        with tqdm(total=len(l_values), desc="Partial waves completed") as pbar:
            for result in pool.starmap(run_single_partial_wave_standalone, worker_args):
                results.append(result)
                pbar.update(1)

    # Process results
    successful_runs = [r[0] for r in results if r[1]]
    failed_runs = [(r[0], r[2]) for r in results if not r[1]]
    end_time_total = datetime.now()

    print_summary(FOLDER_NAME, len(l_values), n_cores, successful_runs, failed_runs, start_time_total, end_time_total)

def print_summary(folder, total, n_cores, successful, failed, start_time, end_time):
    duration = end_time - start_time
    summary = (
        f"\n{'='*60}\n"
        f"All simulations completed\n"
        f"Total time: {duration}\n"
        f"Successful runs: {len(successful)}/{total}\n"
        f"Failed runs: {len(failed)}/{total}\n"
        f"Results saved in: {folder}\n"
        f"{'='*60}\n"
    )
    print(summary)

    summary_file = os.path.join(folder, "simulation_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Completion time: {datetime.now()}\n" + summary)
        if failed:
            f.write("\nFailed l values and errors:\n")
            for l_val, err in failed:
                f.write(f"  l={l_val}: {err}\n")


# In[ ]:


def main(n_cores=None):
    epsilon = 0.00011
    sigma = 5.2
    epsilon_scaled = 4*epsilon

    if n_cores is None:
        n_cores = min(32, mp.cpu_count())
    print(f"System has {mp.cpu_count()} cores available. Using {n_cores} for this simulation.")

    print("\n" + "="*60)
    print("Pre-calculating or loading base coefficients...")
    start_coeff_time = time.time()

    lj_base_coeffs = compute_and_cache_coefficients(
        "Lennard-Jones", [epsilon_scaled, sigma], QUBITS_NUM,
        max_power=12, cache_dir=CACHE_DIR
    )

    centrifugal_base_coeffs = compute_and_cache_coefficients(
        "Centrifugal", [1.0, 1.0], QUBITS_NUM,
        max_power=2, cache_dir=CACHE_DIR
    )

    end_coeff_time = time.time()
    print(f"Base coefficients ready in {end_coeff_time - start_coeff_time:.2f} seconds.")
    print("="*60 + "\n")

    run_parallel_partial_waves_main(n_cores, lj_base_coeffs, centrifugal_base_coeffs)

    print("\n" + "="*60)
    print("Simulations complete. Generating animations for successful runs...")
    print("="*60 + "\n")

    l_values_to_animate = list(range(0,1))

    # Timestep sampling rate. A higher number means fewer frames,
    #    a smaller file size, and faster generation.
    animation_sampling_rate = 200

    # The animation functions need to find the output folder.
    System = f"Ar-Ar_scattering_{L}L_{timestep}ts"
    current_date = datetime.now().strftime("%Y-%m-%d")
    FOLDER_NAME = os.path.join(OUTPUT_DIR,
                               f"{QUBITS_NUM}q_PartialWaveSim_{System}_{current_date}")

    # Check if the main output folder exists to avoid errors
    if not os.path.exists(FOLDER_NAME):
        print(f"Error: Output folder not found at {FOLDER_NAME}")
        print("Cannot generate animations. Please ensure simulations ran correctly.")
        return

    # Loop through each l-value and attempt to create its animations
    for l_val in l_values_to_animate:
        # Check if the data for this specific l-value exists, as some may have failed
        subfolder = os.path.join(FOLDER_NAME, f"l_{l_val:04d}")
        if os.path.exists(subfolder) and os.path.exists(os.path.join(subfolder, f"statevectors_l{l_val}.txt")):
            print(f"\n Generating animations for l={l_val}")
            try:
                # Call the function to create the position-space evolution animation
                create_quantum_evolution_animation(
                    l_value=l_val,
                    timestep_sampling=animation_sampling_rate,
                    polynomial_type=PolynomialType,
                    polynomial_params=current_params,
                    centrifugal_params=[l_val, m],
                    show_potential=True
                )

                # Call the function to create the momentum-space evolution animation
                create_momentum_evolution_animation(
                    l_value=l_val,
                    timestep_sampling=animation_sampling_rate,
                    polynomial_type=PolynomialType
                )
            except Exception as e:
                print(f"Could not generate animation for l={l_val}. Error: {e}")
        else:
            print(f"Skipping animations for l={l_val} (data not found).")

    print("\n" + "="*60)
    print("All animations generated.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()


# In[ ]:




