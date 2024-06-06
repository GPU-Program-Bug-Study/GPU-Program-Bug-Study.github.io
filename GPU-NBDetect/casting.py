#!/usr/bin/env python3

import argparse
import math
import subprocess
import socket
import os
#import bo_analysis'
import numpy as np
import ctypes
import sys
import shutil
compute_cap = 'sm_35'
from detection import *
from input_gen import *
import time

# Generates CUDA code for a given math function
def generate_CUDA_code(fun_name: str, params: list, directory: str) -> str:
  file_name = 'cuda_code_'+fun_name+'.cu'
  with open(directory+'/'+file_name, 'w') as fd:
    fd.write('// Atomatically generated - do not modify\n\n')
    fd.write('#include <stdio.h>\n\n')
    fd.write('__global__ void kernel_1(\n')
    signature = ""
    param_names = ""
    for i in range(len(params)):
      if params[i] == 'float':
        signature += 'float x'+str(i)+','
        param_names += 'x'+str(i)+','
    fd.write('  '+signature)
    fd.write('float *ret) {\n')
    fd.write('   *ret = '+fun_name+'('+param_names[:-1]+');\n')
    fd.write('}\n\n')

    fd.write('extern "C" {\n')
    fd.write('float kernel_wrapper_1('+signature[:-1]+') {\n')
    fd.write('  float *dev_p;\n')
    fd.write('  cudaMalloc(&dev_p, sizeof(float));\n')
    fd.write('  kernel_1<<<1,1>>>('+param_names+'dev_p);\n')
    fd.write('  float res;\n')
    fd.write('  cudaMemcpy (&res, dev_p, sizeof(float), cudaMemcpyDeviceToHost);\n')
    fd.write('  return res;\n')
    fd.write('  }\n')
    fd.write(' }\n\n\n')
  return file_name


def double_generate_CUDA_code(fun_name: str, params: list, directory: str) -> str:
  file_name = 'double_cuda_code_'+fun_name+'.cu'
  with open(directory+'/'+file_name, 'w') as fd:
    fd.write('// Atomatically generated - do not modify\n\n')
    fd.write('#include <stdio.h>\n\n')
    fd.write('__global__ void kernel_1(\n')
    signature = ""
    param_names = ""
    for i in range(len(params)):
      if params[i] == 'float':
        signature += 'double x'+str(i)+','
        param_names += 'x'+str(i)+','
    fd.write('  '+signature)
    fd.write('double *ret) {\n')
    fd.write('   *ret = '+fun_name+'('+param_names[:-1]+');\n')
    fd.write('}\n\n')

    fd.write('extern "C" {\n')
    fd.write('double double_kernel_wrapper_1('+signature[:-1]+') {\n')
    fd.write('  double *dev_p;\n')
    fd.write('  cudaMalloc(&dev_p, sizeof(double));\n')
    fd.write('  kernel_1<<<1,1>>>('+param_names+'dev_p);\n')
    fd.write('  double res;\n')
    fd.write('  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);\n')
    fd.write('  return res;\n')
    fd.write('  }\n')
    fd.write(' }\n\n\n')
  return file_name


# Generates C++ code for a given math function
"""def generate_CPP_code(fun_name: str, params: list, directory: str) -> str:
  file_name = 'c_code_'+fun_name+'.c'
  with open(directory+'/'+file_name, 'w') as fd:
    fd.write('// Atomatically generated - do not modify\n\n')
    fd.write('#include <cmath>\n\n')
    fd.write('#include <fenv.h>\n\n')
    fd.write('float cpp_kernel_1( ')
    signature = ""
    param_names = ""
    for i in range(len(params)):
      if params[i] == 'float':
        signature += 'float x'+str(i)+','
        param_names += 'x'+str(i)+','
    fd.write(signature[:-1]+') {\n')
    fd.write('float *result; \n\n')
    fd.write('    // Enable floating-point exceptions\n')
    fd.write('    feclearexcept(FE_ALL_EXCEPT);\n')
    fd.write('    *result = cosh(a);\n')
    fd.write('    int exceptions = fetestexcept(FE_ALL_EXCEPT);\n')
    fd.write('    if (exceptions & FE_DIVBYZERO) {\n')
    fd.write('        return "Floating-point exception occurred: Division by zero\\n;" \n')
    fd.write('    }\n')
    fd.write('    else if (exceptions & FE_OVERFLOW) {\n')
    fd.write('        return "Floating-point exception occurred: Overflow\\n";\n')
    fd.write('    }\n')
    fd.write('    else if (exceptions & FE_INVALID) {\n')
    fd.write('        return "Floating-point exception occurred: Invalid operation\\n";\n')
    fd.write('    }\n')
    fd.write('    else if (exceptions & FE_UNDERFLOW) {\n')
    fd.write('        return "Floating-point exception occurred: Underflow\\n");\n')
    fd.write('    }\n') 
    fd.write('  else {\n')
    fd.write('    return result;\n')
    fd.write('}\n\n')
  return file_name
    #fd.write('   return '+fun_name+'('+param_names[:-1]+');\n')
    #fd.write('}\n\n')
  #return file_name"""


def generate_CPP_code(fun_name: str, params: list, directory: str) -> str:
    file_name = 'c_code_' + fun_name + '.c'
    with open(directory + '/' + file_name, 'w') as fd:
        fd.write('// Automatically generated - do not modify\n\n')
        fd.write('#include <cmath>\n\n')
        fd.write('#include <fenv.h>\n\n')
        fd.write('extern "C" float cpp_kernel_1(')
        signature = ""
        param_names = ""
        for i in range(len(params)):
            if params[i] == 'float':
                signature += 'float x' + str(i) + ',' 
                param_names += 'x' + str(i) + ','
        fd.write(signature[:-1] + ', int *flag) {\n')
        fd.write('    float result;\n\n')
        fd.write('    // Enable floating-point exceptions\n')
        fd.write('    feclearexcept(FE_ALL_EXCEPT);\n')
        fd.write('    result =' +fun_name+'(' + param_names[:-1] + ');\n')
        fd.write('    int exceptions = fetestexcept(FE_ALL_EXCEPT);\n')
        fd.write('    if (exceptions & FE_DIVBYZERO) {\n')
        fd.write('        *flag = 1;\n')
        fd.write('      return 1.0;\n')
        fd.write('    }\n')
        fd.write('    else if (exceptions & FE_OVERFLOW) {\n')
        fd.write('        *flag = 1;\n')
        fd.write('        return 2.0;\n')
        fd.write('    }\n')
        fd.write('    else if (exceptions & FE_INVALID) {\n')
        fd.write('        *flag = 1;\n')
        fd.write('        return 3.0;\n')
        fd.write('    }\n')
        fd.write('    else if (exceptions & FE_UNDERFLOW) {\n')
        fd.write('        *flag = 1;\n')
        fd.write('        return 4.0;\n')
        fd.write('    }\n')
        fd.write('    else {\n')
        fd.write('        *flag = 0;\n')
        fd.write('        return result;\n')
        fd.write('    }\n')
        fd.write('}\n\n')
    return file_name

#------------------------------------------------------------------------------
# Compilation & running external programs
#------------------------------------------------------------------------------

def run_command(cmd: str):
  try:
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as e:
    print(e.output)
    exit()



def compile_CUDA_code(file_name: str, d: str):
  global compute_cap 
  shared_lib = d+'/'+file_name+'.so'
  #cmd = 'nvcc '+' -arch='+compute_cap+' '+d+'/'+file_name+' -o '+shared_lib+' -shared -Xcompiler -fPIC'
  cmd = 'nvcc -shared '+d+'/'+file_name+' -o '+shared_lib+' -Xcompiler -fPIC '   #--use_fast_math 
  print('Running:', cmd)
  run_command(cmd)
  return shared_lib




def compile_CPP_code(file_name: str, d: str):
  shared_lib = d+'/'+file_name+'.so'
  #cmd = 'g++ -shared'+d+'/'+file_name+' -o '+d+'/'+file_name+'.so -shared -fPIC'
  cmd = 'g++ -shared '+d+'/'+file_name+' -o '+shared_lib+' -fPIC'
  print('Running:', cmd)
  run_command(cmd)

  """
  # Run the compilation command and capture its output
  process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdout, stderr = process.communicate()
    
    # Decode the byte strings to UTF-8
  stdout_str = stdout.decode('utf-8')
  stderr_str = stderr.decode('utf-8')
    
    # Print compilation output
  print('Compilation Output (stdout):')
  print(stdout_str)
    
  print('Compilation Error Output (stderr):')
  print(stderr_str)
    
    # Check if compilation was successful
  if process.returncode == 0:
        print('Compilation successful!')
  else:
        print('Compilation failed!') """

    

  return shared_lib

#------------------------------------------------------------------------------
# File and directory creation 
#------------------------------------------------------------------------------

def dir_name():
  return '_tmp_'+socket.gethostname()+"_"+str(os.getpid())

def create_experiments_dir() -> str:
    p = dir_name()
    print("Creating dir:", p)
    try:
        os.mkdir(p)
    except OSError:
        print ("Creation of the directory %s failed" % p)
        exit()
    return p

#------------------------------------------------------------------------------
# Function Classes
#------------------------------------------------------------------------------
class SharedLib:
  def __init__(self, path, inputs):
    self.path = path
    self.inputs = int(inputs)

class FunctionSignature:
  def __init__(self, fun_name, input_types):
    self.fun_name = fun_name
    self. input_types = input_types

#------------------------------------------------------------------------------
# Main driver
#------------------------------------------------------------------------------

#FUNCTION:acos (double)
##FUNCTION:acosh (double)
#SHARED_LIB:./app_kernels/CFD_Rodinia/cuda_code_cfd.cu.so, N
#SHARED_LIB:./app_kernels/backprop_Rodinia/cuda_code_backprop.cu.so, N
def parse_functions_to_test(fileName):
  #function_signatures = []
  #shared_libs = []
  ret = []
  with open(fileName, 'r') as fd:
    for line in fd:
      # Comments
      if line.lstrip().startswith('#'):
        continue
      # Empty line
      if ''.join(line.split()) == '':
        continue

      if line.lstrip().startswith('FUNCTION:'):
        no_spaces = ''.join(line.split())
        signature = no_spaces.split('FUNCTION:')[1]
        fun = signature.split('(')[0]
        params = signature.split('(')[1].split(')')[0].split(',')
        ret.append(FunctionSignature(fun, params))
        #function_signatures.append((fun, params))

      if line.lstrip().startswith('SHARED_LIB:'):
        lib_path = line.split('SHARED_LIB:')[1].split(',')[0].strip()
        inputs = line.split('SHARED_LIB:')[1].split(',')[1].strip()
        #shared_libs.append((lib_path, inputs))
        ret.append(SharedLib(lib_path, inputs))

  #return (function_signatures, shared_libs)
  return ret




# Namespace(af='ei', function=['./function_signatures.txt'], number_sampling='fp', range_splitting='many', samples=30)
def areguments_are_valid(args):
  if args.af != 'ei' and args.af != 'ucb' and args.af != 'pi':
    return False
  if args.samples < 1:
    return False
  if args.range_splitting != 'whole' and args.range_splitting != 'two' and args.range_splitting != 'many':
    return False
  if args.number_sampling != 'fp' and args.number_sampling != 'exp':
    return False
  return True

#file_name+'.so'
def call_GPU_kernel_1(x0, shared_lib):
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, shared_lib)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_float
  res = E.kernel_wrapper_1(ctypes.c_float(x0))
  return res


def call_double_GPU_kernel_1(x0, shared_lib):
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, shared_lib)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.double_kernel_wrapper_1.restype = ctypes.c_double
  res = E.double_kernel_wrapper_1(ctypes.c_double(x0))
  return res

def call_CPU_kernel_1(x0, shared_lib, flag):
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, shared_lib)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.cpp_kernel_1.argtypes = [ctypes.c_float, ctypes.POINTER(ctypes.c_int)]
  E.cpp_kernel_1.restype = ctypes.c_float
  #E.cpp_kernel_1.restype = ctypes.c_float
  res = E.cpp_kernel_1(ctypes.c_float(x0), flag)
  return res


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='GPU Numerical Bug Detection tool')
  parser.add_argument('function', metavar='FUNCTION_TO_TEST', nargs=1, help='Function to test (file or shared library .so)')
  parser.add_argument('-a', '--af', default='ei', help='Acquisition function: ei, ucb, pi')
  parser.add_argument('-n', '--number-sampling', default='fp', help='Number sampling method: fp, exp')
  parser.add_argument('-r', '--range-splitting', default='many', help='Range splitting method: whole, two, many')
  parser.add_argument('-s', '--samples', type=int, default=30, help='Number of BO samples (default: 30)')
  parser.add_argument('--random_sampling', action='store_true', help='Use random sampling')
  parser.add_argument('--random_sampling_unb', action='store_true', help='Use random sampling unbounded')
  parser.add_argument('-c', '--clean', action='store_true', help='Remove temporal directories (begin with _tmp_)')
  args = parser.parse_args()

  # --------- Cleaning -------------
  if (args.clean):
    print('Removing temporal dirs...')
    this_dir = './'
    for fname in os.listdir(this_dir):
      if fname.startswith("_tmp_"):
        #os.remove(os.path.join(my_dir, fname))
        shutil.rmtree(os.path.join(this_dir, fname))
    exit()

    """
  # --------- Checking arguments for BO approach ---------
  if (not areguments_are_valid(args)):
    print('Invalid input!')
    parser.print_help()"""

  input_file = args.function[0]
  functions_to_test = []
  if input_file.endswith('.txt'):
    functions_to_test = parse_functions_to_test(input_file)
  else:
    exit()

  # Create directory to save experiments
  d = create_experiments_dir()

  """
  # --------------- BO approach -----------------
  # Set BO  max iterations
  bo_analysis.set_max_iterations(args.samples) """

  # Generate CUDA and compile them
  for i in functions_to_test:
    if type(i) is FunctionSignature:
      log_file_name = os.path.join("./results", f"{i.fun_name}_log.txt")
      with open(log_file_name, "w") as log_file:

        print(i.fun_name,i.input_types,d, file=log_file)
      
        #-----generate code ----------
        f = generate_CUDA_code(i.fun_name, i.input_types, d)
        double_f = double_generate_CUDA_code(i.fun_name,i.input_types,d)
        #g = generate_CPP_code(i.fun_name,i.input_types,d )


        g_shared_lib = compile_CUDA_code(f, d)
        #flag = ctypes.c_int()
        #c_shared_lib = compile_CPP_code(g,d)

        double_g_shared_lib = compile_CUDA_code(double_f,d)
        x0 = 0.5#-2.932960271835327##3.40282347E+38  #cosh minor difference 2.0 #3.40282347E+38
        iterations = 100

        for i in range(iterations):
          if i > 0:
            fuzzer = GrayBoxFuzzer(x0)  # Replace with the path to the target program
            mutated_input = fuzzer.fuzz(num_iterations=1)
            x0 = mutated_input

          print("==================Analysis begins==================",file=log_file)
          print("Input is ", x0, file=log_file)
       
       
          gpu_start_time = time.time()
          gpu_result = call_GPU_kernel_1(x0, g_shared_lib)
          gpu_end_time = time.time()
          gpu_elapsed = gpu_end_time - gpu_start_time

      #print(gpu_result)
          double_gpu_start_time = time.time()
          double_gpu_result = call_double_GPU_kernel_1(x0,double_g_shared_lib)
          double_gpu_end_time = time.time()
          double_gpu_elapsed = double_gpu_end_time-double_gpu_start_time
        
          #cpu_start_time = time.time()
          #cpu_result = call_CPU_kernel_1(x0, c_shared_lib, ctypes.byref(flag))
          #cpu_end_time= time.time()
          #cpu_elapsed = cpu_end_time - cpu_start_time
      
          gpu_err, g_result = verify_gpu_result(gpu_result)
          #cpu_err, c_result = verify_cpu_results(cpu_result, flag.value)
          print("GPU result", g_result,file=log_file)
          print("GPU time", gpu_elapsed, file =log_file)
          #print("CPU result",c_result,file=log_file)
          #print("CPU time", cpu_elapsed, file =log_file)

        #double_gpu_err, double_g_result = verify_gpu_result(double_gpu_result)
          print("Double GPU result", double_gpu_result,file=log_file)
          print("Double GPU time", double_gpu_elapsed, file =log_file)




          if gpu_err==1 :
            print("An floating point exception is detected. No further analysis",file=log_file)
            continue

        #print("Checking for Rounding Error", file=log_file)
        
          c_err, c = check_for_casting_errors(x0,g_result)

          if c_err == 1:
              print("An casting error has been detected. No further Analysis", file = log_file)



        








      #print(cpu_result, flag.value)

      


        #num_inputs = len(i.input_types)
    


    # Run CUDA and C code 
  




    """
    # Random Sampling
    if args.random_sampling or args.random_sampling_unb:
      print('******* RANDOM SAMPLING on:', shared_lib)
      # Total samples per each input depends on:
      # 18 ranges, 30 max samples (per range), n inputs
      inputs = num_inputs
      max_iters = 30 * int(math.pow(18, inputs))
      unbounded = False
      if args.random_sampling_unb:
        unbounded = True
      bo_analysis.optimize_randomly(shared_lib, inputs, max_iters, unbounded)
      bo_analysis.print_results_random(shared_lib)

    # Run BO optimization
    print('*** Running BO on:', shared_lib)
    bo_analysis.optimize(shared_lib,
                        args.number_sampling, 
                        num_inputs, 
                        args.range_splitting)
    bo_analysis.print_results(shared_lib, args.number_sampling, args.range_splitting)"""



"""
void print_result(float result, int flag) {
    if (flag == 1) {
        switch (static_cast<int>(result)) {
            case 1:
                std::cout << "Overflow occurred" << std::endl;
                break;
            case 2:
                std::cout << "Division by zero occurred" << std::endl;
                break;
            case 3:
                std::cout << "Invalid operation occurred" << std::endl;
                break;
            case 4:
                std::cout << "Underflow occurred" << std::endl;
                break;
            default:
                std::cout << "Unknown error" << std::endl;
                break;
        }
    } else {
        std::cout << "Correct result: " << result << std::endl;
    }
}

int main() {
    float x0 = 10.0f;
    int flag;
    float result = cpp_kernel_1(x0, flag);
    print_result(result, flag);
    return 0;
}"""