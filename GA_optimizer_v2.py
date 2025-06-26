import itertools
import pandas as pd
import multiprocessing as mp
from brian2 import *  # Brian2 must be imported for the simulation
import pygad
import time
import sys

start_time = time.time()
first_generation_passed = False
second_generation_passed = False
first_generation_end_time = None

# Import the simulation function from your model file
from optimization_model import opt_ring_attractor  # your simulation function
# Also import any utility functions if needed (e.g., for computing firing rates, etc.)
from utils import *




if len(sys.argv) > 1:
    num_processes = int(sys.argv[1])
    print(f"Using {num_processes} processes for parallel processing.")
    population_size = int(sys.argv[2])
    print(f"Using population size: {population_size}")
else:
    print("COMMAND LINE THINGY NOT WORKING")
    exit()


# Set the connectivity profile to optimize
# Options: 'mexican_hat', 'cosine'
connectivity_profile = 'mexican_hat'  # Change this to 'cosine' to optimize the cosine profile

stim_center = 1.571
stim_width = 0.5

# Fixed parameters for the simulation (not searched over)
fixed_params = {
    'tau': 10,           # in our simulation, run_ring_attractor converts this to ms.
    'sigma_noise': 1,    # similarly converted to mV inside run_ring_attractor.
    'syn_profile': connectivity_profile,  # Set the connectivity profile
}

# Set the ranges to sample the initial values from
# Note: during optimization, the values can go outside these ranges,
# but it is possible to set limits for that as well.
if connectivity_profile == 'mexican_hat':
    sigma_exc_range = [0.05, 0.2]   # excitatory spread
    sigma_inh_range = [0.1, 0.3]    # inhibitory spread
    g_exc_range   = [0.5, 1.0]      # excitatory gain
    g_inh_range   = [-1.0, -0.3]    # inhibitory gain
elif connectivity_profile == 'cosine':
    g_cosine_range = [0.01, 0.1]    # cosine gain
    glob_inh_range = [True, False]  # global inhibition flag
    w_inh_range = [-2.0, -0.5]      # global inhibition weight
else:
    raise ValueError("Unsupported connectivity profile. Choose 'mexican_hat' or 'cosine'.")


# GA hyperparameters
num_generations = 10
num_parents_mating = 2
mutation_type = "adaptive" # Options: "random", "swap", "inversion", "scramble", "adaptive", or a custom function
mutation_num_genes = [2,1] # Number of genes to mutate in each solution. If using mutation_type="adaptive", this should be a list of 2 numbers: the first for lower-than-average fitness solutions, the second for higher-than-average fitness solutions.

parent_selection_type = "rank" # Options: "sss" (steady state selection), "rws" (roulette wheel selection),
                              # "sus" (stochastic universal selection), "rank", "tournament", "random", or a custom function
crossover_type = "single_point" # Options: "single_point", "two_points", "uniform", "scattered", or a custom function
keep_elitism = 5



# Number of genes based on the connectivity profile
if connectivity_profile == 'mexican_hat':
    num_genes = 4  # sigma_exc, sigma_inh, g_exc, g_inh
elif connectivity_profile == 'cosine':
    num_genes = 3 # g_cosine, glob_inh, w_inh
else:  
    raise ValueError("Unsupported connectivity profile. Choose 'mexican_hat' or 'cosine'.")


def on_generation(ga_instance):
    global first_generation_passed
    global first_generation_end_time
    global second_generation_passed

    if not first_generation_passed:
        first_generation_passed = True
        first_generation_end_time = time.time()
        print(f"First generation completed in {first_generation_end_time - start_time:.2f} seconds")
    elif not second_generation_passed:
        second_generation_passed = True
        # Write times to a file
        with open(f'times{population_size}.txt', 'a') as f:
            f.write(f"Processes: {num_processes}, Time: {time.time() - first_generation_end_time:.2f} seconds\n")
    
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    gens_completed = ga_instance.generations_completed
    print(f"Generation {gens_completed} - Best composite error: {1/(solution_fitness)} (sigma_exc: {solution[0]}, sigma_inh: {solution[1]}, g_exc: {solution[2]} mV, g_inh: {solution[3]} mV)")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")


def fitness_func(ga_instance, solution, solution_idx):

    params = fixed_params.copy()

    if connectivity_profile == 'mexican_hat':
        params.update({
            'sigma_exc': solution[0],
            'sigma_inh': solution[1],
            'g_exc': solution[2]*mV,
            'g_inh': solution[3]*mV
        })
    elif connectivity_profile == 'cosine':
        raise NotImplementedError("Cosine profile optimization not yet implemented in GA fitness function.")
    else:
        raise ValueError("Unsupported connectivity profile. Choose 'mexican_hat' or 'cosine'.")
    
    try:
        # Run the simulation.
        # opt_ring_attractor returns: (GT_center, GT_input, out_rates, out_pva_angle, out_pva_magnitude)
        GT_center, GT_input, out_rates, out_pva_angle, out_pva_magnitude = opt_ring_attractor(params, stim_center=stim_center, stim_width=stim_width)
        
        # Compute the circular standard deviation (spread) from the PVA magnitude.
        circular_std = np.sqrt(-2 * np.log(out_pva_magnitude + 1e-8))
        
        # Compute the center error and the confidence weighted center error (CWCE).
        center_err, cwce = conf_weighted_CE(out_pva_angle, GT_center, out_pva_magnitude)
        
        # Compute the angular Z-score:
        # This expresses the misalignment (center_err) in units of the circular standard deviation,
        # analogous to a z-score in linear statistics.
        angular_Zscore = center_err / circular_std
        
        # Compute the NMSE between the observed firing rates and the ideal Gaussian profile.
        nmse = compute_nmse_normalized(out_rates, GT_input, norm_type='max')
        
        # Combine the errors into one composite score.
        # Adjust weights to prioritize center accuracy if desired
        w_center = 0.3
        w_Zscore = 0.2
        w_nmse = 0.5
        composite_error = w_center * cwce + w_Zscore * angular_Zscore + w_nmse * nmse

        # Check if the composite error is NaN or infinite. In that case, set it to a very high value,
        # so that the fitness value will be very low.
        if np.isnan(composite_error) or np.isinf(composite_error):
            composite_error = 1e10

    # For now, if there is any exception raised, just give very low fitness value to this solution.
    except Exception as e:
        composite_error = 1e10


    fitness_value = 1.0 / (composite_error + 1e-8)  # Avoid division by zero

    return fitness_value



ga_instance = pygad.GA(num_generations=num_generations,
                       sol_per_pop=population_size,
                       fitness_func=fitness_func,
                       num_genes=num_genes,
                       num_parents_mating=num_parents_mating,
                       mutation_type=mutation_type,
                       mutation_num_genes=mutation_num_genes,
                       on_generation=on_generation,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       parallel_processing=["process", num_processes],
                       keep_elitism=keep_elitism)


# Initialize the population with random values within the specified ranges
if connectivity_profile == 'mexican_hat':
    sigma_exc_range = [0.05, 0.2]   # excitatory spread
    sigma_inh_range = [0.1, 0.3]    # inhibitory spread
    g_exc_range   = [0.5, 1.0]      # excitatory gain
    g_inh_range   = [-1.0, -0.3]    # inhibitory gain

    ga_instance.initialize_population(low = [sigma_exc_range[0], sigma_inh_range[0], g_exc_range[0], g_inh_range[0]],
                                      high = [sigma_exc_range[1], sigma_inh_range[1], g_exc_range[1], g_inh_range[1]],
                                      allow_duplicate_genes=True,
                                      mutation_by_replacement=False,
                                      gene_type=[float, float, float, float])
elif connectivity_profile == 'cosine':
    raise NotImplementedError("Cosine profile optimization not yet implemented in GA initialization.")
else:
    raise ValueError("Unsupported connectivity profile. Choose 'mexican_hat' or 'cosine'.")





if __name__ == '__main__':
    print(f"Starting optimization for {connectivity_profile} connectivity profile")
    
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(f"""Best solution found:
    composite error: {1/solution_fitness}
    sigma_exc: {solution[0]}
    sigma_inh: {solution[1]}
    g_exc: {solution[2]} mV
    g_inh: {solution[3]} mV
""")

    print(f"Total time taken for optimization: {time.time() - start_time:.2f} seconds")
    exit()


