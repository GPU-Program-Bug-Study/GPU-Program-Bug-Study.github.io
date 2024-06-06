import subprocess
import random
import struct

class GrayBoxFuzzer:
    def __init__(self, target_program):
        #self.target_program = ""#target_program
        self.custom_input = target_program

    def generate_float_input(self):
        # Generate a random floating-point input
        return random.uniform(-3.40E38, 3.40E38)  # Example: Generates a float between -1e6 and 1e6

    def is_within_range(self, value):
        # Check if the value falls within the supported floating-point range
        return value >= -3.40E38 and value <= 3.40E38


    def fpmutate(self, input):
        # Mutation strategy for floating-point numbers
        mutation_type = random.choice(['bitflip_s', 'bitflip_e', 'bitflip_m', 'addition_e', 'addition_m', 'decrease_e', 'decrease_m', 'random_bit'])
        float_bits = struct.unpack('!I', struct.pack('!f', input))[0]  #32bit representation


        if mutation_type == 'bitflip_s':

            sign_bit = (float_bits >> 31) & 0x1
            flip_sign = bool(random.getrandbits(1))

    # Flip the sign bit if the random boolean is True
            if flip_sign:
                flipped_sign_bit = 1 - sign_bit
            else:
                flipped_sign_bit = sign_bit

    # Reconstruct the modified floating-point number
            modified_float_bits = (float_bits & 0x7FFFFFFF) | (flipped_sign_bit << 31)


        elif mutation_type == 'bitflip_e':

            exponent = (float_bits >> 23) & 0xFF
            bit_position = random.randint(0, 7)
            flipped_exponent = exponent ^ (1 << bit_position)
            modified_float_bits = (float_bits & 0x807FFFFF) | (flipped_exponent << 23)


        elif mutation_type == 'bitflip_m':
            mantissa = float_bits & 0x7FFFFF
            bit_position = random.randint(0, 22)
            #print(bit_position)
            flipped_mantissa = mantissa ^ (1 << bit_position)
            modified_float_bits = (float_bits & 0xFF800000) | flipped_mantissa


        elif mutation_type == 'addition_e':

            exponent = (float_bits >> 23) & 0xFF

            # Generate a random integer in the range 1 to 32
            increment = random.randint(1, 32)

            # Add the random integer to the exponent
            modified_exponent = min(exponent + increment, 255)  # Ensure exponent remains within valid range

            # Reconstruct the modified floating-point number
            modified_float_bits = (float_bits & 0x807FFFFF) | (modified_exponent << 23)


        elif mutation_type == 'addition_m':
            mantissa = float_bits & 0x007FFFFF

            # Generate a random integer p in the range 1 to 33
            p = random.randint(1, 33)

            # Calculate the integer value 2^(p-1)
            increment = 2 ** (p - 1)

            # Add the calculated integer to the mantissa
            modified_mantissa = min(mantissa + increment, 0x007FFFFF)  # Ensure mantissa remains within valid range

            # Reconstruct the modified floating-point number
            modified_float_bits = (float_bits & 0xFF800000) | modified_mantissa

           

        elif mutation_type == 'decrease_e':
            exponent = (float_bits >> 23) & 0xFF

            # Generate a random integer in the range 1 to 32
            increment = random.randint(1, 32)

            # Add the random integer to the exponent
            modified_exponent = max(exponent - increment, 0)  # Ensure exponent remains within valid range

            # Reconstruct the modified floating-point number
            modified_float_bits = (float_bits & 0x807FFFFF) | (modified_exponent << 23)

            

        elif mutation_type == 'decrease_m':
            mantissa = float_bits & 0x007FFFFF

            # Generate a random integer p in the range 1 to 33
            p = random.randint(1, 33)

            # Calculate the integer value 2^(p-1)
            increment = 2 ** (p - 1)

            # Add the calculated integer to the mantissa
            modified_mantissa = max(mantissa - increment, 0)  # Ensure mantissa remains within valid range

            # Reconstruct the modified floating-point number
            modified_float_bits = (float_bits & 0xFF800000) | modified_mantissa

        elif mutation_type == 'random_bit':
            exponent = (float_bits >> 23) & 0xFF

            # Extracting the mantissa bits (23 bits)
            mantissa = float_bits & 0x007FFFFF

            # Generate random bit positions for both exponent and mantissa
            exponent_bit_position = random.randint(0, 7)
            mantissa_bit_position = random.randint(0, 22)

            # Flip the randomly chosen bit in the exponent
            mutated_exponent = exponent ^ (1 << exponent_bit_position)

            # Flip the randomly chosen bit in the mantissa
            mutated_mantissa = mantissa ^ (1 << mantissa_bit_position)

            # Reconstruct the modified floating-point number
            modified_float_bits = (float_bits & 0x80000000) | (mutated_exponent << 23) | mutated_mantissa


            # Random bit mutation
            ##bit_to_flip = random.randint(0, 31)
            ##mutant = input ^ (1 << bit_to_flip)

        #return struct.unpack('!f', struct.pack('!I', int(mutant)))[0]
        modified_float_value = struct.unpack('!f', struct.pack('!I', modified_float_bits))[0]
        if self.is_within_range(modified_float_value):
            return modified_float_value
        else: 
            return input+ 0.5

        #print(modified_float_value)

        #return modified_float_value
    

    def execute_program(self, input_value):
        # Execute the target program with the input and capture output
        command = [self.target_program, str(input_value)]
        try:
            result = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
            return result.strip()  # Assuming the program prints the result
        except subprocess.CalledProcessError as e:
            return "Error: " + e.output.strip()

    def fuzz(self, num_iterations):
        for i in range(num_iterations):
            #input_value = self.generate_float_input()
            ##input_value = 123.456
            ##print("generated", input_value)
            ##mutated_input_value = self.mutate_float_input(input_value)
            mutated_input_value = self.fpmutate(self.custom_input)
            self.custom_input = mutated_input_value
            #print(mutated_input_value)
            return mutated_input_value
            ###result = self.execute_program(mutated_input_value)
            

if __name__ == "__main__":
    # Example usage
    fuzzer = GrayBoxFuzzer(123.456)  # Replace with the path to the target program
    fuzzer.fuzz(num_iterations=1)



"""import random

# Function to flip one bit at a random position
def bitflip(value):
    # Convert value to integer
    int_value = int(value)
    # Convert integer value to binary string
    binary_str = bin(int_value)[2:]
    # Randomly select a bit position to flip
    bit_pos = random.randint(0, len(binary_str) - 1)
    # Flip the selected bit
    flipped_bit = '1' if binary_str[bit_pos] == '0' else '0'
    # Construct the new binary string with the flipped bit
    new_binary_str = binary_str[:bit_pos] + flipped_bit + binary_str[bit_pos + 1:]
    # Convert the binary string back to integer
    return int(new_binary_str, 2)

# Function to add an integer to the value
def addition(value, max_addition):
    # Add a random integer in the range 1 to max_addition
    return value + random.randint(1, max_addition)

# Function to subtract an integer from the value
def decrease(value, max_decrease):
    # Subtract a random integer in the range 1 to max_decrease
    return value - random.randint(1, max_decrease)

# Function to apply havoc mutation
def havoc(value, max_exponent_addition, max_mantissa_addition, max_exponent_decrease, max_mantissa_decrease):
    # Randomly select a mutation type
    mutation_type = random.choice(['bitflip_e', 'bitflip_m', 'addition_e', 'addition_m', 'decrease_e', 'decrease_m', 'random_bit'])
    
    if mutation_type == 'bitflip_e':
        return bitflip(value)
    elif mutation_type == 'bitflip_m':
        return bitflip(value)
    elif mutation_type == 'addition_e':
        return addition(value, max_exponent_addition)
    elif mutation_type == 'addition_m':
        return addition(value, max_mantissa_addition)
    elif mutation_type == 'decrease_e':
        return decrease(value, max_exponent_decrease)
    elif mutation_type == 'decrease_m':
        return decrease(value, max_mantissa_decrease)
    elif mutation_type == 'random_bit':
        # For random bit mutation, select a random number of bits to flip
        num_bits = random.choice([4, 8, 16, 32])
        return bitflip(value, num_bits)

# Define the seed input value
seed_input = 0.5

# Define the mutation parameters
max_exponent_addition = 32
max_mantissa_addition = 32
max_exponent_decrease = 32
max_mantissa_decrease = 32

# Mutation process
mutated_inputs = []
for _ in range(10):  # Generate 10 mutated inputs
    mutated_input = havoc(seed_input, max_exponent_addition, max_mantissa_addition, max_exponent_decrease, max_mantissa_decrease)
    mutated_inputs.append(mutated_input)

# Print mutated inputs
print("Mutated inputs:", mutated_inputs)"""





"""import random
import struct

def generate_single_float():
    # Generate a single float within the valid range
    fraction = random.uniform(0.0, 1.0)
    exponent = random.uniform(-126, 127)
    sign_bit = random.choice([0, 1])

    # Combine sign, exponent, and fraction bits to create the float representation
    float_bits = (sign_bit << 31) | ((int(exponent) + 127) << 23) | int(fraction * (1 << 23))

    # Pack the float bits into a bytes object and unpack it as a single-precision float
    return struct.unpack('f', struct.pack('I', float_bits))[0]

# Example usage
generated_number = generate_single_float()
print("Generated number:", generated_number) """


"""import numpy as np
import math

# Define the functions to test
def test_functions(x):
    functions = [math.cosh, math.sinh, math.tanh]  
    for func in functions:
        try:
            _ = func(x)
        except FloatingPointError:
            return True  # Exception triggered 
    return False  # No exception triggered 

# Simulated Annealing (SA) optimization algorithm
def simulated_annealing():
    current_input = np.random.uniform(-3.40282347E+38, 3.40282347E+38)
    print()
    current_score = test_functions(current_input)
    best_input = current_input
    best_score = current_score
    temperature = 100.0
    cooling_rate = 0.01

    while temperature > 1e-3:
        new_input = current_input + np.random.normal(0, 1)
        new_score = test_functions(new_input)
        
        # Metropolis criterion
        if new_score < current_score or np.random.uniform(0, 1) < np.exp((current_score - new_score) / temperature):
            current_input = new_input
            current_score = new_score
        
        if new_score < best_score:
            best_input = new_input
            best_score = new_score
        
        temperature *= 1 - cooling_rate
    
    return best_input

# trial:
best_input = simulated_annealing()
print("Input that triggers floating-point exceptions:", best_input) """








"""from z3 import *
import math
import random

def generate_float_constraint():
    # Define symbolic input variable
    #x = Real('x')

    #x = FP('x', Float32()) 
    x = FPSort(8, 24).cast_to_fp(FP('x', FPSort(8, 24)))


    # Define constraints
    lower_bound = random.uniform(-10, 0)  # Random lower bound for x
    upper_bound = random.uniform(0, 100)   # Random upper bound for x
    constraints = [
        x >= lower_bound,
        x <= upper_bound
    ]

    return x, constraints

def test_cosh(x_value):
    # Test the cosh function with the generated input value
    result = math.cosh(x_value)
    return result

def main():
    # Perform 10 iterations
    for i in range(10):
        print(f"Iteration {i+1}:")
        
        # Generate constraints for floating-point input
        x, constraints = generate_float_constraint()

        # Create a solver
        solver = Solver()

        # Add constraints to the solver
        for constraint in constraints:
            solver.add(constraint)

        # Check if there exists a solution
        if solver.check() == sat:
            # Get a model (solution)
            model = solver.model()

            # Extract the value of the input variable
            x_value = model[x].as_decimal(5)  # Get x value with 5 decimal places
            #x_value_str = model[x].sexpr()
        # Convert the string representation to a float
            x_value = float(x_value)
            print("Generated input value (x):", x_value)

            # Test the cosh function with the generated input value
            result = test_cosh(float(x_value))
            print("Result of cosh function:", result)
        else:
            print("No solution .")
        
        print() 

if __name__ == "__main__":
    main()  """


"""
        # Convert the floating-point value to its 32-bit representation (unsigned integer)
        float_bits = struct.unpack('!I', struct.pack('!f', input))[0]
        print(float_bits)

        # Extracting the sign bit
        sign_bit = (float_bits >> 31) & 0x1

    # Extracting the exponent bits (8 bits)
        exponent = (float_bits >> 23) & 0xFF

    # Extracting the mantissa bits (23 bits)
        mantissa = float_bits & 0x7FFFFF

        #print("Sign:", sign_bit)
        print("Exponent:", exponent)
        #print("Mantissa:", mantissa)
        # Choose a random bit position within the exponent (0 to 7)
        bit_position = random.randint(0, 7)
        print(bit_position)

        # Flip the bit at the chosen position
        flipped_exponent = exponent ^ (1 << bit_position)

    # Reconstruct the modified floating-point number
        modified_float_bits = (float_bits & 0x807FFFFF) | (flipped_exponent << 23)

    # Convert the modified bits back to a floating-point number
        modified_float_value = struct.unpack('!f', struct.pack('!I', modified_float_bits))[0]
        print(modified_float_value) 
"""


"""
    def mutate_float_input(self, input_value):
        mutation_type = random.choice(['bitflip', 'addition', 'subtraction', 'random_perturbation'])
        
        if mutation_type == 'bitflip':
            print("bitflip")
            # Bitflip mutation
            # Convert the float to its IEEE 754 binary representation
            binary_repr = bin(struct.unpack('!I', struct.pack('!f', input_value))[0])[2:]  # Remove '0b' prefix
            # Choose a random bit to flip
            bit_to_flip = random.randint(0, len(binary_repr) - 1)
            # Flip the chosen bit
            mutated_binary_repr = binary_repr[:bit_to_flip] + str(1 - int(binary_repr[bit_to_flip])) + binary_repr[bit_to_flip + 1:]
            # Convert the mutated binary representation back to float
            mutated_input_value = struct.unpack('!f', int(mutated_binary_repr, 2).to_bytes(4, byteorder='big', signed=False))[0]

        elif mutation_type == 'addition':
            print("addition")
            # Addition mutation
            # Add a small random value to the input
            mutation_value = random.uniform(-10000, 10000)      #0.000001, 0.001)
            mutated_input_value = input_value + mutation_value
        
        elif mutation_type == 'subtraction':
            print("substraction")
            # Subtraction mutation
            # Subtract a small random value from the input
            mutation_value = random.uniform(-10000, 10000)    #0.000001, 0.001)
            mutated_input_value = input_value - mutation_value
        
        elif mutation_type == 'random_perturbation':
            print("random")
            # Random perturbation mutation
            # Perturb the input by a small random value
            mutation_value = random.uniform(-1000, 1000)
            mutated_input_value = input_value + mutation_value


        # Check if the mutated input value is within range
        if self.is_within_range(mutated_input_value):
            return mutated_input_value
        else:
            print("none inputs,", mutated_input_value)
            return None  # Return None if mutated input value is not within range
        
        #return mutated_input_value """