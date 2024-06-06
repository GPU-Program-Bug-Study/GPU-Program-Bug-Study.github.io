from cmath import inf
import math
import re
import numpy as np
import sys

max_fp = 3.40E38
#min_fp = -3.40E38
min_fp = 1.17549e-38

CPU_err = 1
GPU_err = 1

epsilon = 1E-7 #0.0000001
data_epsilon = 0.0001

def verify_cpu_results(result, flag):
    if flag == 1.0:
        if result == 1.0:
             e = "Error: Division by zero occurred"
             CPU_err = 1
             return CPU_err, e
        elif result == 2.0:
            e = "Error: Overflow occurred"
            CPU_err = 1
            return CPU_err, e
        elif result == 3.0:
            e = "Error: Invalid operation occurred"
            CPU_err = 1
            return CPU_err, e
        elif result == 4.0:
            e = "Error: Underflow occurred"
            CPU_err = 1
            return CPU_err, e
        else:
            e = "Unknown error"
            CPU_err = 1
            return CPU_err, e
    else:
        #e = result 
        CPU_err = 0
        return CPU_err, result

def verify_gpu_result(result):
    if math.isnan(result):
        GPU_err = 1
        g = "Error: Invalid operation occurred"
        return GPU_err, g
    elif result > max_fp:
        GPU_err = 1
        g ="Error Positive overflow detected: Value exceeds maximum representable float"
        return GPU_err, g
    elif result < min_fp and result > 0:
        GPU_err = 1
        g = "Error: Positive underflow detected: Value is smaller than minimum representable positive float"
        return GPU_err, g
    elif result < -max_fp:
        GPU_err = 1
        g ="Error: Negative overflow detected: Value exceeds minimum representable float"
        return GPU_err, g
    elif result > -min_fp and result < 0:
        GPU_err = 1
        g ="Error: Negative underflow detected: Value is larger than maximum representable negative float"
        return GPU_err, g
    else: 
        GPU_err = 0
        return GPU_err, result


def check_for_rounding_errors(g_result, c_result):
    diff = abs(g_result - c_result)
    if diff > epsilon:
        g = "Rounding Error has been detected"
        round_err = 1
        return round_err, g

    else:
        g = "No rounding error"
        round_err = 0
        return round_err, g

def check_for_environmental_errors(gpu_elapsed, cpu_elapsed):
    if (gpu_elapsed) > 2*cpu_elapsed:
        g = "Environmental Error has been detected. CPU is much faster than GPU"
        env_err = 1
        return env_err, g
    else:
        g = "No environmental error"
        env_err = 0
        return env_err, g

def check_for_datatype_errors(gpu_result, double_gpu_result, gpu_elapsed, double_gpu_elapsed):
    diff = abs(gpu_result-double_gpu_result)
    if diff > data_epsilon or (gpu_elapsed > 1.5*double_gpu_elapsed):
        g = "Data Type Error has been detected."
        data_err = 1
        return data_err, g

    else:
        g = "No data type error"
        data_err = 0
        return data_err, g


def check_for_casting_errors(x0,g_result):
    diff = abs(g_result - x0)
    if diff > 0.99:
        g = "Casting Error has been detected"
        round_err = 1
        return round_err, g

    else:
        g = "No casting error"
        round_err = 0
        return round_err, g




    """elif math.isinf(result):
        if result > 0:
            print("Error: Overflow: Positive infinity")
        else:
            print("Error: Overflow: Negative infinity")
    elif abs(result) < np.finfo(float).tiny:
        print("Error: Underflow occured")
    else:
        print("Normal result", result)"""



