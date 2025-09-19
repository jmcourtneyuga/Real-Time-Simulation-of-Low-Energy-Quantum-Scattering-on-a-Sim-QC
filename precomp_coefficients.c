// precomp_coefficients.c - High precision version with correct exports
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <float.h>

// Use long double for high precision
typedef long double hp_float;

// Print precision info
void print_precision_info() {
    printf("Long double precision: %d decimal digits\n", LDBL_DIG);
    printf("Long double epsilon: %.40Le\n", LDBL_EPSILON);
    printf("Sizeof long double: %zu bytes\n", sizeof(long double));
}

// Compute base
hp_float compute_coefficient_internal(unsigned int index_bitset) {
    int n = __builtin_popcount(index_bitset);
    if (n == 0) return 0.0L;
    
    hp_float coefficient = 0.0L;
    
    unsigned int subset = index_bitset;
    while (subset > 0) {
        int subset_size = __builtin_popcount(subset);
        hp_float denominator = 0.0L;
        
        for (int j = 0; j < 32; j++) {
            if (subset & (1U << j)) {
                denominator += (hp_float)(1U << j);
            }
        }
        
        int sign = ((n - subset_size) % 2 == 0) ? 1 : -1;
        coefficient += (hp_float)sign / denominator;
        
        subset = (subset - 1) & index_bitset;
    }
    
    return coefficient;
}

// Compute all  coefficients
void compute_all_coefficients_internal(int n_qubits, hp_float* output) {
    unsigned int max_set = (1U << n_qubits);
    
    for (unsigned int i = 0; i < max_set; i++) {
        output[i] = 0.0L;
    }
    
    #pragma omp parallel for
    for (unsigned int bitset = 1; bitset < max_set; bitset++) {
        output[bitset] = compute_coefficient_internal(bitset);
    }
}

// Compute power coefficients with Kahan summation for precision
void compute_power_coefficients_internal(
    int n_qubits,
    int target_power,
    const hp_float* coeff_base,
    hp_float* output
) {
    unsigned int max_set = (1U << n_qubits);
    
    for (unsigned int i = 0; i < max_set; i++) {
        output[i] = 0.0L;
    }
    
    if (target_power == 1) {
        memcpy(output, coeff_base, max_set * sizeof(hp_float));
        return;
    }
    
    if (target_power == 2) {
        #pragma omp parallel for
        for (unsigned int target = 1; target < max_set; target++) {
            hp_float sum = 0.0L;
            hp_float c = 0.0L;
            
            for (unsigned int s1 = 1; s1 <= target; s1++) {
                if ((s1 & target) == s1) {
                    for (unsigned int s2 = 1; s2 <= target; s2++) {
                        if ((s2 & target) == s2 && (s1 | s2) == target) {
                            hp_float prod = coeff_base[s1] * coeff_base[s2];
                            hp_float y = prod - c;
                            hp_float t = sum + y;
                            c = (t - sum) - y;
                            sum = t;
                        }
                    }
                }
            }
            output[target] = sum;
        }
        return;
    }
    
    // For any other power, use the general multiplication approach
    hp_float* temp = (hp_float*)calloc(max_set, sizeof(hp_float));
    hp_float* current = (hp_float*)calloc(max_set, sizeof(hp_float));
    
    memcpy(current, coeff_base, max_set * sizeof(hp_float));
    
    for (int p = 2; p <= target_power; p++) {
        #pragma omp parallel for
        for (unsigned int target = 1; target < max_set; target++) {
            hp_float sum = 0.0L;
            hp_float c = 0.0L;
            
            for (unsigned int s1 = 1; s1 <= target; s1++) {
                if ((s1 & target) == s1) {
                    for (unsigned int s2 = 1; s2 <= target; s2++) {
                        if ((s2 & target) == s2 && (s1 | s2) == target) {
                            hp_float prod = current[s1] * coeff_base[s2];
                            hp_float y = prod - c;
                            hp_float t = sum + y;
                            c = (t - sum) - y;
                            sum = t;
                        }
                    }
                }
            }
            temp[target] = sum;
        }
        memcpy(current, temp, max_set * sizeof(hp_float));
    }
    
    memcpy(output, current, max_set * sizeof(hp_float));
    free(temp);
    free(current);
}

// Wrapper with double compatibility
// These convert between double and long double

void compute_lj_coefficients(
    int n_qubits,
    double epsilon,    // Not used, but kept for interface
    double sigma,      // Not used, but kept for interface
    double* coeffs_12_out,
    double* coeffs_6_out
) {
    unsigned int max_set = (1U << n_qubits);
    
    // Allocate high precision arrays
    hp_float* coeff_base = (hp_float*)calloc(max_set, sizeof(hp_float));
    hp_float* coeffs_12 = (hp_float*)calloc(max_set, sizeof(hp_float));
    hp_float* coeffs_6 = (hp_float*)calloc(max_set, sizeof(hp_float));
    
    // Compute with high precision
    compute_all_coefficients_internal(n_qubits, coeff_base);
    compute_power_coefficients_internal(n_qubits, 12, coeff_base, coeffs_12);
    compute_power_coefficients_internal(n_qubits, 6, coeff_base, coeffs_6);
    
    // Convert to double for output (with potential precision loss)
    for (unsigned int i = 0; i < max_set; i++) {
        coeffs_12_out[i] = (double)coeffs_12[i];
        coeffs_6_out[i] = (double)coeffs_6[i];
    }
    
    free(coeff_base);
    free(coeffs_12);
    free(coeffs_6);
}

void compute_arbitrary_power_coefficients(
    int n_qubits,
    int power,
    double* output
) {
    unsigned int max_set = (1U << n_qubits);
    
    hp_float* coeff_base = (hp_float*)calloc(max_set, sizeof(hp_float));
    hp_float* coeffs = (hp_float*)calloc(max_set, sizeof(hp_float));
    
    compute_all_coefficients_internal(n_qubits, coeff_base);
    compute_power_coefficients_internal(n_qubits, power, coeff_base, coeffs);
    
    // Convert to double
    for (unsigned int i = 0; i < max_set; i++) {
        output[i] = (double)coeffs[i];
    }
    
    free(coeff_base);
    free(coeffs);
}

// Exporters
// These maintain full long double precision by exporting as strings

void compute_lj_coefficients_string(
    int n_qubits,
    char* coeffs_12_strings,  // Array of strings
    char* coeffs_6_strings,   // Array of strings
    int string_length          // Length of each string
) {
    unsigned int max_set = (1U << n_qubits);
    
    hp_float* coeff_base = (hp_float*)calloc(max_set, sizeof(hp_float));
    hp_float* coeffs_12 = (hp_float*)calloc(max_set, sizeof(hp_float));
    hp_float* coeffs_6 = (hp_float*)calloc(max_set, sizeof(hp_float));
    
    compute_all_coefficients_internal(n_qubits, coeff_base);
    compute_power_coefficients_internal(n_qubits, 12, coeff_base, coeffs_12);
    compute_power_coefficients_internal(n_qubits, 6, coeff_base, coeffs_6);
    
    // Export as strings with full precision
    for (unsigned int i = 0; i < max_set; i++) {
        snprintf(&coeffs_12_strings[i * string_length], string_length, "%.30Le", coeffs_12[i]);
        snprintf(&coeffs_6_strings[i * string_length], string_length, "%.30Le", coeffs_6[i]);
    }
    
    free(coeff_base);
    free(coeffs_12);
    free(coeffs_6);
}

// Test function to verify precision
void test_precision() {
    print_precision_info();
    
    // Test a simple coefficient
    unsigned int test_set = 0b111;  // Indices {0, 1, 2}
    hp_float coeff = compute_coefficient_internal(test_set);
    
    printf("Test coefficient for {0,1,2}: %.30Le\n", coeff);
    
    // Verify it matches expected value
    hp_float expected = 1.0L/1.0L - 1.0L/2.0L - 1.0L/4.0L - 1.0L/3.0L + 1.0L/5.0L + 1.0L/6.0L - 1.0L/7.0L;
    printf("Expected value: %.30Le\n", expected);
    printf("Difference: %.30Le\n", fabsl(coeff - expected));
}
