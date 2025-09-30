import itertools
import pandas as pd
import multiprocessing as mp
from brian2 import *  # Brian2 must be imported for the simulation
import pygad
import time
import sys
import os
import argparse
import matplotlib.pyplot as plt

# For keeping track of how much time each generation takes
start_time = time.time()
previous_gen_start_time = start_time

# For collecting stats, i.e. the errors of the best specimens in each generation,
# the means/standard deviations of the errors of the specimens that don't crash,
# and the variation within the population over time (mean of standard deviations of the genes).
best_errors = []
mean_errors_working_specimens = []
mean_center_error_working_specimens = []
mean_nmse_working_specimens = []
mean_spread_diff_error_working_specimens = []
mean_aim_spread_error_working_specimens = []
mean_frequency_error_working_specimens = []
mean_rate_change_error_working_specimens = []
std_errors_working_specimens = []
std_center_error_working_specimens = []
std_nmse_working_specimens = []
std_spread_diff_error_working_specimens = []
std_aim_spread_error_working_specimens = []
std_frequency_error_working_specimens = []
std_rate_change_error_working_specimens = []
mean_gene_stddevs = []

# Import the simulation function from your model file
from optimization_model_faithful import opt_ring_attractor  # your simulation function
# Also import any utility functions if needed (e.g., for computing firing rates, etc.)
from utils import *


# Check if directory 'GA_results' exists, if not create it
if not os.path.exists('GA_results'):
    os.makedirs('GA_results')

# Create a unique directory for the current run based on the timestamp
current_results_dirname = f"GA_results/GA_run_{time.strftime('%Y%m%d_%H%M%S')}"
if not os.path.exists(current_results_dirname):
    os.makedirs(current_results_dirname)


# Set up argparse. Read in settings from command line, or use defaults.
parser = argparse.ArgumentParser(description='Run GA optimization for ring attractor model.')

parser.add_argument('--num_processes', type=int, default=1, help='Number of processes for parallel processing.')
parser.add_argument('--population_size', type=int, default=100, help='Population size for the genetic algorithm.')
parser.add_argument('--random_seed', type=int, default=24, help='Random seed for reproducibility.')
parser.add_argument('--num_generations', type=int, default=500, help='Number of generations for the genetic algorithm.')
parser.add_argument('--num_parents_mating', type=int, default=50, help='Number of parents mating in each generation.')
parser.add_argument('--mutation_type', type=str, default='adaptive', choices=['random', 'swap', 'inversion', 'scramble', 'adaptive', ], help='Type of mutation to use in the genetic algorithm.') # Options: "random", "swap", "inversion", "scramble", "adaptive", or a custom function
parser.add_argument('--mutation_probability', type=float, default=[0.5, 0.25], nargs=2, help='Probability of mutation for each gene. If using adaptive mutation, this should be a list of two values: the first for lower-than-average fitness solutions, the second for higher-than-average fitness solutions. If using random mutation, this should be a single value for all solutions.')
parser.add_argument('--parent_selection_type', type=str, default='tournament_nsga2', choices=['nsga2', 'tournament_nsga2'], help='Parent selection method for the genetic algorithm.')
parser.add_argument('--connectivity_profile', type=str, default='cosine', choices=['cosine'], help='Connectivity profile to optimize.')
parser.add_argument('--crossover_type', type=str, default='single_point', choices=['single_point', 'two_points', 'uniform', 'scattered'], help='Crossover type for the genetic algorithm.')
parser.add_argument('--keep_elitism', type=int, default=1, help='Number of best solutions to keep in the next generation.')
parser.add_argument('--rand_mut_min_val', type=float, default=-0.1, help='Minimum value for random mutation.')
parser.add_argument('--rand_mut_max_val', type=float, default=0.1, help='Maximum value for random mutation.')
parser.add_argument('--initial_ranges_cos', type=float, nargs=8, default=[0.0001, 100, -100, 0.0, 23, 100, 0.5, 100], help='Initial ranges for Cosine connectivity profile: g_cosine_min, g_cosine_max, w_inh_min, w_inh_max, Iff_min, Iff_max, tau_s_min, tau_s_max.')
parser.add_argument('--gene_spaces_cos', type=float, nargs=8, default=[0.0001, 100, -100, 0.0, 23, 100, 0.5, 100], help='Cosine gene spaces/limits for the optimization: g_cosine_min, g_cosine_max, w_inh_min, w_inh_max, Iff_min, Iff_max, tau_s_min, tau_s_max.')
parser.add_argument('--stim_center', type=float, default=1.571, help='Center of the stimulus for the ring attractor model.')
parser.add_argument('--stim_width', type=float, default=0.5, help='Width of the stimulus for the ring attractor model.')
parser.add_argument('--tau', type=float, default=10.0, help='Time constant for the ring attractor model in ms.')
parser.add_argument('--sigma_noise', type=float, default=1.0)
parser.add_argument('--num_neurons', type=int, default=120, help='Number of neurons in the ring attractor model.')
parser.add_argument('--input_on', type=float, default=0.5, help='Duration of the input stimulus in seconds.')
parser.add_argument('--input_off', type=float, default=0.2, help='Duration of the input off period in seconds.')


args = parser.parse_args()
num_neurons = args.num_neurons
aim_spread = int(num_neurons / 10)
spread_variation = int(num_neurons / 20)
spread_variation_squared = spread_variation ** 2
rand_seed = args.random_seed
num_processes = args.num_processes
population_size = args.population_size
num_generations = args.num_generations
num_parents_mating = args.num_parents_mating
mutation_type = args.mutation_type
mutation_probability = args.mutation_probability
connectivity_profile = args.connectivity_profile
parent_selection_type = args.parent_selection_type
crossover_type = args.crossover_type
keep_elitism = args.keep_elitism
rand_mut_min_val = args.rand_mut_min_val
rand_mut_max_val = args.rand_mut_max_val


initial_ranges_cos = args.initial_ranges_cos

gene_spaces_cos = args.gene_spaces_cos

stim_center = args.stim_center
stim_width = args.stim_width

tau = args.tau 
sigma_noise = args.sigma_noise

input_on = args.input_on
input_off = args.input_off

# Write the parameters to a file
with open(f"{current_results_dirname}/exp_params.txt", "w") as f:
    f.write(f"Random seed: {rand_seed}\n")
    f.write(f"Number of processes: {num_processes}\n")
    f.write(f"Population size: {population_size}\n")
    f.write(f"Number of generations: {num_generations}\n")
    f.write(f"Number of parents mating: {num_parents_mating}\n")
    f.write(f"Mutation type: {mutation_type}\n")
    f.write(f"Mutation probability: {mutation_probability}\n")
    f.write(f"Parent selection type: {parent_selection_type}\n")
    f.write(f"Connectivity profile: {connectivity_profile}\n")
    f.write(f"Crossover type: {crossover_type}\n")
    f.write(f"Keep elitism: {keep_elitism}\n")
    f.write(f"Random mutation min value: {rand_mut_min_val}\n")
    f.write(f"Random mutation max value: {rand_mut_max_val}\n")
    f.write(f"Initial ranges for Cosine connectivity profile: {initial_ranges_cos}\n")
    f.write(f"Input on duration: {input_on} seconds\n")
    f.write(f"Input off duration: {input_off} seconds\n")


# If adaptive mutation is used, there is a probability for lower-than-average fitness solutions 
# and a probability for higher-than-average fitness solutions.
if not (mutation_type in ['adaptive']):
    mutation_probability = mutation_probability[0]  # Use the first value for all solutions

# Set seed for reproducibility
np.random.seed(rand_seed)
seed(rand_seed)



# Fixed parameters for the simulation (not searched over)
fixed_params = {
    'tau': tau,           # in our simulation, run_ring_attractor converts this to ms.
    'sigma_noise': sigma_noise,   # similarly converted to mV inside run_ring_attractor.
    'syn_profile': connectivity_profile,  # Set the connectivity profile
    'num_neurons': num_neurons,  # Number of neurons in the ring attractor
    'input_on': input_on,
    'input_off': input_off
}


# Set the ranges to sample the initial values from
# Note: during optimization, the values can go outside these ranges,
# but it is possible to set limits for that as well.
if connectivity_profile == 'cosine':
    g_cosine_range = initial_ranges_cos[:2]    # cosine gain
    w_inh_range = initial_ranges_cos[2:4]      # global inhibition weight
    Iff_range = initial_ranges_cos[4:6]        # feedforward input strength
    tau_s_range = initial_ranges_cos[6:8]      # synaptic time constant
    gene_0_stddevs = []
    gene_1_stddevs = []
    gene_2_stddevs = []
    gene_3_stddevs = []

    g_cosine_space = {'low': gene_spaces_cos[0], 'high': gene_spaces_cos[1]}
    w_inh_space = {'low': gene_spaces_cos[2], 'high': gene_spaces_cos[3]}
    Iff_space = {'low': gene_spaces_cos[4], 'high': gene_spaces_cos[5]}
    tau_s_space = {'low': gene_spaces_cos[6], 'high': gene_spaces_cos[7]}
else:
    raise ValueError("Unsupported connectivity profile. Choose 'cosine'.")




# Number of genes based on the connectivity profile
if connectivity_profile == 'cosine':
    num_genes = 4  # g_cosine, w_inh, Iff, tau_s
else:  
    raise ValueError("Unsupported connectivity profile. Choose 'cosine'.")


# Function that is called after each generation to collect statistics and print results
def on_generation(ga_instance):
    global previous_gen_start_time
    global best_errors
    global mean_errors_working_specimens
    global std_errors_working_specimens
    global mean_center_error_working_specimens
    global mean_nmse_working_specimens
    global mean_spread_diff_error_working_specimens
    global mean_aim_spread_error_working_specimens
    global mean_frequency_error_working_specimens
    global mean_rate_change_error_working_specimens
    global std_center_error_working_specimens
    global std_nmse_working_specimens
    global std_spread_diff_error_working_specimens
    global std_aim_spread_error_working_specimens
    global std_frequency_error_working_specimens
    global std_rate_change_error_working_specimens
    global current_results_dirname
    global gene_0_stddevs, gene_1_stddevs, gene_2_stddevs, gene_3_stddevs
    global mean_gene_stddevs
    

    # Calculate the population standard deviation for each gene
    # This gives a measure of the variation within the population
    all_genomes = ga_instance.population
    
    gene_std_devs = np.std(all_genomes, axis=0)
    mean_gene_stddevs.append(np.mean(gene_std_devs))
    if connectivity_profile == 'cosine':
        gene_0_stddevs.append(gene_std_devs[0])
        gene_1_stddevs.append(gene_std_devs[1])
        gene_2_stddevs.append(gene_std_devs[2])
        gene_3_stddevs.append(gene_std_devs[3])
        np.savetxt(f"{current_results_dirname}/gene_0_stddevs.txt", np.array(gene_0_stddevs))
        np.savetxt(f"{current_results_dirname}/gene_1_stddevs.txt", np.array(gene_1_stddevs))
        np.savetxt(f"{current_results_dirname}/gene_2_stddevs.txt", np.array(gene_2_stddevs))
        np.savetxt(f"{current_results_dirname}/gene_3_stddevs.txt", np.array(gene_3_stddevs))
    else:
        raise ValueError("Unsupported connectivity profile. Choose 'cosine'.")
    
    np.savetxt(f"{current_results_dirname}/mean_gene_stddevs.txt", np.array(mean_gene_stddevs))



    # Retrieve the fitnesses of working and failed specimens
    # The fitness function is set up such that a negative fitness is given
    # to specimens that fail the simulation for some reason (e.g., NaN values). 
    pop_fitnesses = np.array(ga_instance.last_generation_fitness)
    positive_fitnesses = pop_fitnesses[pop_fitnesses[:, 0] >= 0]
    negative_fitnesses = pop_fitnesses[pop_fitnesses[:, 0] < 0]

    # Calculate the errors for working specimens
    # Simply the inverse of the fitness value calculation as can be seen in the fitness_func.
    center_fitness_vals = positive_fitnesses[:, 0]
    center_errors = (1.0 / center_fitness_vals) - 1e-8
    mean_center_errors = np.mean(center_errors)
    std_center_errors = np.std(center_errors)
    mean_center_error_working_specimens.append(mean_center_errors)
    std_center_error_working_specimens.append(std_center_errors)
    nmse_fitness_vals = positive_fitnesses[:, 1]
    nmses = (1.0 / nmse_fitness_vals) - 1e-8
    mean_nmses = np.mean(nmses)
    std_nmses = np.std(nmses)
    mean_nmse_working_specimens.append(mean_nmses)
    std_nmse_working_specimens.append(std_nmses)
    spread_diff_fitness_vals = positive_fitnesses[:, 2]
    spread_diff_errors = (1.0 / spread_diff_fitness_vals) - 1e-8
    mean_spread_diff_errors = np.mean(spread_diff_errors)
    std_spread_diff_errors = np.std(spread_diff_errors)
    mean_spread_diff_error_working_specimens.append(mean_spread_diff_errors)
    std_spread_diff_error_working_specimens.append(std_spread_diff_errors)
    aim_spread_fitness_vals = positive_fitnesses[:, 3]
    aim_spread_errors = (1.0 / aim_spread_fitness_vals) - 1e-8
    mean_aim_spread_errors = np.mean(aim_spread_errors)
    std_aim_spread_errors = np.std(aim_spread_errors)
    mean_aim_spread_error_working_specimens.append(mean_aim_spread_errors)
    std_aim_spread_error_working_specimens.append(std_aim_spread_errors)
    frequency_fitness_vals = positive_fitnesses[:, 4]
    frequency_errors = (1.0 / frequency_fitness_vals) - 1e-8
    mean_frequency_errors = np.mean(frequency_errors)
    std_frequency_errors = np.std(frequency_errors)
    mean_frequency_error_working_specimens.append(mean_frequency_errors)
    std_frequency_error_working_specimens.append(std_frequency_errors)
    rate_change_fitness_vals = positive_fitnesses[:, 5]
    rate_change_errors = (1.0 / rate_change_fitness_vals) - 1e-8
    mean_rate_change_errors = np.mean(rate_change_errors)
    std_rate_change_errors = np.std(rate_change_errors)
    mean_rate_change_error_working_specimens.append(mean_rate_change_errors)
    std_rate_change_error_working_specimens.append(std_rate_change_errors)

    mean_specimen_fitnesses = np.mean(np.stack([center_fitness_vals, nmse_fitness_vals, spread_diff_fitness_vals, aim_spread_fitness_vals, frequency_fitness_vals, rate_change_fitness_vals]), axis=0)


    mean_specimen_errors = np.mean(np.stack([center_errors, nmses, spread_diff_errors, aim_spread_errors, frequency_errors, rate_change_errors]), axis=0)
    mean_errors = np.mean(mean_specimen_errors)
    std_errors = np.std(mean_specimen_errors)
    mean_errors_working_specimens.append(mean_errors)
    std_errors_working_specimens.append(std_errors)

    
    

    np.savetxt(f"{current_results_dirname}/mean_errors_working_specimens.txt", np.array(mean_errors_working_specimens))
    np.savetxt(f"{current_results_dirname}/std_errors_working_specimens.txt", np.array(std_errors_working_specimens))
    np.savetxt(f"{current_results_dirname}/mean_center_error_working_specimens.txt", np.array(mean_center_error_working_specimens))
    np.savetxt(f"{current_results_dirname}/std_center_error_working_specimens.txt", np.array(std_center_error_working_specimens))
    np.savetxt(f"{current_results_dirname}/mean_nmse_working_specimens.txt", np.array(mean_nmse_working_specimens))
    np.savetxt(f"{current_results_dirname}/std_nmse_working_specimens.txt", np.array(std_nmse_working_specimens))
    np.savetxt(f"{current_results_dirname}/mean_spread_diff_error_working_specimens.txt", np.array(mean_spread_diff_error_working_specimens))
    np.savetxt(f"{current_results_dirname}/std_spread_diff_error_working_specimens.txt", np.array(std_spread_diff_error_working_specimens))
    np.savetxt(f"{current_results_dirname}/mean_aim_spread_error_working_specimens.txt", np.array(mean_aim_spread_error_working_specimens))
    np.savetxt(f"{current_results_dirname}/std_aim_spread_error_working_specimens.txt", np.array(std_aim_spread_error_working_specimens))
    np.savetxt(f"{current_results_dirname}/mean_frequency_error_working_specimens.txt", np.array(mean_frequency_error_working_specimens))
    np.savetxt(f"{current_results_dirname}/std_frequency_error_working_specimens.txt", np.array(std_frequency_error_working_specimens))
    np.savetxt(f"{current_results_dirname}/mean_rate_change_error_working_specimens.txt", np.array(mean_rate_change_error_working_specimens))
    np.savetxt(f"{current_results_dirname}/std_rate_change_error_working_specimens.txt", np.array(std_rate_change_error_working_specimens))

    print(f"Generation mean FITNESS (working solutions): {np.mean(mean_specimen_fitnesses):.4f} +/- {np.std(mean_specimen_fitnesses):.4f} | {len(positive_fitnesses)} working, {len(negative_fitnesses)} failed solutions")
    print(f"Generation mean ERROR (working solutions): {mean_errors:.4f} +/- {std_errors:.4f}")
    print(f"      mean center error: {mean_center_errors:.4f} +/- {std_center_errors:.4f}")
    print(f"      mean NMSE: {mean_nmses:.4f} +/- {std_nmses:.4f}")
    print(f"      mean spread difference error: {mean_spread_diff_errors:.4f} +/- {std_spread_diff_errors:.4f}")
    print(f"      mean aim spread error: {mean_aim_spread_errors:.4f} +/- {std_aim_spread_errors:.4f}")
    print(f"      mean frequency error: {mean_frequency_errors:.4f} +/- {std_frequency_errors:.4f}")
    print(f"      mean rate change error: {mean_rate_change_errors:.4f} +/- {std_rate_change_errors:.4f}")
    print(f"Generation mean gene standard deviations: {np.mean(gene_std_devs):.4f}")

    solution, solution_fitness, solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    best_errors.append(1/(solution_fitness) - 1e-8)
    np.savetxt(f"{current_results_dirname}/best_errors.txt", np.array(best_errors))
    
    gens_completed = ga_instance.generations_completed
    with open(f"{current_results_dirname}/exp_results.txt", "a") as f:
        if connectivity_profile == 'cosine':
            f.write(f"Generation {gens_completed} - Best specimen's errors: {1/(solution_fitness) - 1e-8} (g_cosine: {solution[0]}, w_inh: {solution[1]}, Iff: {solution[2]}, tau_s: {solution[3]})\n")
        f.write(f"Generation mean FITNESS: {np.mean(positive_fitnesses):.4f} +/- {np.std(positive_fitnesses):.4f} | {len(positive_fitnesses)} working, {len(negative_fitnesses)} failed solutions\n")
        f.write(f"Generation mean ERROR (working solutions): {mean_errors:.4f} +/- {std_errors:.4f}\n")
        f.write(f"      mean center error: {mean_center_errors:.4f} +/- {std_center_errors:.4f}\n")
        f.write(f"      mean NMSE: {mean_nmses:.4f} +/- {std_nmses:.4f}\n")
        f.write(f"      mean spread difference error: {mean_spread_diff_errors:.4f} +/- {std_spread_diff_errors:.4f}\n")
        f.write(f"      mean aim spread error: {mean_aim_spread_errors:.4f} +/- {std_aim_spread_errors:.4f}\n")
        f.write(f"      mean frequency error: {mean_frequency_errors:.4f} +/- {std_frequency_errors:.4f}\n")
        f.write(f"      mean rate change error: {mean_rate_change_errors:.4f} +/- {std_rate_change_errors:.4f}\n")
        f.write(f"Generation mean gene standard deviations: {np.mean(gene_std_devs):.4f}\n")
        f.write(f"Generation time: {time.time() - previous_gen_start_time:.2f} seconds\n")
    if connectivity_profile == 'cosine':
        print(f"Generation {gens_completed} - Best specimen's errors: {1/(solution_fitness) - 1e-8} (g_cosine: {solution[0]}, w_inh: {solution[1]}, Iff: {solution[2]}, tau_s: {solution[3]})")
    print(f"Generation time: {time.time() - previous_gen_start_time:.2f} seconds\n")

    previous_gen_start_time = time.time()


# Function that takes the parameter values given by a specimen, runs a simulation
# with those values, calculates the composite error, and returns the fitness value.
# Basically the same as in the grid search code.
# Specimens for which the simulation fails are given a composite error of -1,
# so that they can easily be identified and excluded from the statistics.
def fitness_func(ga_instance, solution, solution_idx):
    global aim_spread, spread_variation, spread_variation_squared
    np.random.seed(rand_seed)
    seed(rand_seed)

    params = fixed_params.copy()

    if connectivity_profile == 'cosine':
        params.update({
            'g_cosine': solution[0],
            'w_inh_val': solution[1],
            'Iff_val': solution[2],
            'tau_s': solution[3]
        })
    else:
        raise ValueError("Unsupported connectivity profile. Choose 'cosine'.")
    
    try:
        # Run the simulation.
        GT_center, GT_input, out_rates, out_pva_angle, out_pva_magnitude, spread_difference, mid_sim_spread, max_firing_rate, Iff_firing_rate, highest_rate_1, highest_rate_2 = opt_ring_attractor(params, stim_center=stim_center, stim_width=stim_width)
        
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

        # Normalize the spread difference by the number of neurons, and take the
        # absolute value to punish increases and decreases equally. Add 1 to make sure
        # everything is above zero.
        spread_diff_error = np.abs(spread_difference / num_neurons) + 1

        # Force bump spread at halfway the simulation to be near num_neurons/10.
        # Spreads within +/- num_neurons/20 of this are punished lightly, and outside
        # this range are punished quadratically with a maximum of 5.
        lower_lim = (spread_variation - 1)
        upper_lim = (3*spread_variation + 1)
        if (mid_sim_spread > lower_lim) and (mid_sim_spread < upper_lim):
            aim_spread_error = -2/((mid_sim_spread - lower_lim)*(mid_sim_spread - upper_lim) + 1e-8)
        elif mid_sim_spread <= 0 or mid_sim_spread >= num_neurons:
            aim_spread_error = 5
        else:
            aim_spread_error = np.clip((1/(spread_variation_squared)) * (mid_sim_spread - aim_spread)**2, a_max=5)
        



        # If no neurons are active, give high punishment
        if np.any(out_rates > 0):
            lowest_active_neuron_rate = np.min(out_rates[out_rates > 0])*Hz
            highest_active_neuron_rate = np.max(out_rates)*Hz

            # Highest active neuron should be as low as possible (to avoid saturation)
            high_error = highest_active_neuron_rate / max_firing_rate
            # Lowest active neuron should be above 40% of the Iff firing rate, but not
            # necessarily as high as possible.
            low_error = np.clip((0.4 * Iff_firing_rate - lowest_active_neuron_rate) / (0.4 * Iff_firing_rate), a_min=0)
            frequency_error = high_error + low_error
        else:
            frequency_error = 1000


        # If firing rates change over time, punish this.
        rate_change_error = np.abs(highest_rate_2 - highest_rate_1) / highest_rate_1
        
        
        # Combine the errors into one composite score, just to be able to
        # quickly check NaN or infinite values.
        composite_error = center_err + nmse + spread_diff_error + aim_spread_error + frequency_error + rate_change_error

        # Check if the composite error is NaN or infinite. In that case, set all errors to -1
        if np.isnan(composite_error) or np.isinf(composite_error):
            isnan_indices = np.isnan([center_err, nmse, spread_diff_error, aim_spread_error, frequency_error, rate_change_error])
            isinf_indices = np.isinf([center_err, nmse, spread_diff_error, aim_spread_error, frequency_error, rate_change_error])

            print(f"Composite error is NaN for solution {solution_idx}. Setting errors to -1. The values that caused NaN are: cwce: {isnan_indices[0]}, angular_Zscore: {isnan_indices[1]}, nmse: {isnan_indices[2]}, spread_err: {isnan_indices[3]}, normalized_mid_sim_spread: {isnan_indices[4]}, frequency_error: {isnan_indices[5]}, rate_increase_error: {isnan_indices[6]}, out_rates: {np.any(np.isnan(out_rates))}, GT_input: {np.any(np.isnan(GT_input))}. Values that are inf are: cwce: {isinf_indices[0]}, angular_Zscore: {isinf_indices[1]}, nmse: {isinf_indices[2]}, spread_err: {isinf_indices[3]}, normalized_mid_sim_spread: {isinf_indices[4]}, frequency_error: {isinf_indices[5]}, rate_increase_error: {isinf_indices[6]}.")
            

            center_err = -1
            nmse = -1
            spread_diff_error = -1
            aim_spread_error = -1
            frequency_error = -1
            rate_change_error = -1

    # For now, if there is any exception raised, just give very low fitness value to this solution.
    except Exception as e:
        center_err = -1
        nmse = -1
        spread_diff_error = -1
        aim_spread_error = -1
        frequency_error = -1
        rate_change_error = -1

        
    fitness_center = 1 / (center_err + 1e-8)
    fitness_nmse = 1 / (nmse + 1e-8)
    fitness_spread_diff = 1 / (spread_diff_error + 1e-8)
    fitness_aim_spread = 1 / (aim_spread_error + 1e-8)
    fitness_frequency = 1 / (frequency_error + 1e-8)
    fitness_rate_change = 1 / (rate_change_error + 1e-8)

    return [fitness_center, fitness_nmse, fitness_spread_diff, fitness_aim_spread, fitness_frequency, fitness_rate_change]


# Create the GA instance with the specified parameters
ga_instance = pygad.GA(num_generations=num_generations,
                       sol_per_pop=population_size,
                       fitness_func=fitness_func,
                       num_genes=num_genes,
                       num_parents_mating=num_parents_mating,
                       mutation_type=mutation_type,
                       mutation_probability=mutation_probability,
                       on_generation=on_generation,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       parallel_processing=["process", num_processes],
                       keep_elitism=keep_elitism,
                       random_seed=rand_seed,
                       random_mutation_min_val= rand_mut_min_val,
                       random_mutation_max_val= rand_mut_max_val,
                       gene_space=[g_cosine_space, w_inh_space, Iff_space, tau_s_space]
)







if __name__ == '__main__':

    # Run the GA optimization. At the end, save all statistics once again, and create plots.

    print(f"Starting optimization for {connectivity_profile} connectivity profile")
    
    ga_instance.run()

     
    solution, solution_fitness, solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    
    if connectivity_profile == 'cosine':
        print(f"""Best solution found:
        errors: {1/solution_fitness}
        g_cosine: {solution[0]} mV
        w_inh: {solution[1]}
        Iff: {solution[2]}
        tau_s: {solution[3]} ms
    """)
        
        with open(f"{current_results_dirname}/exp_results.txt", "a") as f:
            f.write(f"Best solution found:\n")
            f.write(f"errors: {1/solution_fitness}\n")
            f.write(f"g_cosine: {solution[0]} mV\n")
            f.write(f"w_inh: {solution[1]}\n")
            f.write(f"Iff: {solution[2]}\n")
            f.write(f"tau_s: {solution[3]} ms\n")
    else:
        raise ValueError("Unsupported connectivity profile. Choose 'cosine'.")
    
    
    
    ga_instance.save(f"{current_results_dirname}/ga_instance_final")

    print(f"Total time taken for optimization: {time.time() - start_time:.2f} seconds")

    np.savetxt(f"{current_results_dirname}/mean_errors_working_specimens.txt", np.array(mean_errors_working_specimens))
    np.savetxt(f"{current_results_dirname}/std_errors_working_specimens.txt", np.array(std_errors_working_specimens))
    np.savetxt(f"{current_results_dirname}/best_errors.txt", np.array(best_errors))
    np.savetxt(f"{current_results_dirname}/mean_gene_stddevs.txt", np.array(mean_gene_stddevs))
    np.savetxt(f"{current_results_dirname}/gene_0_stddevs.txt", np.array(gene_0_stddevs))
    np.savetxt(f"{current_results_dirname}/gene_1_stddevs.txt", np.array(gene_1_stddevs))
    np.savetxt(f"{current_results_dirname}/gene_2_stddevs.txt", np.array(gene_2_stddevs))
    np.savetxt(f"{current_results_dirname}/gene_3_stddevs.txt", np.array(gene_3_stddevs))

    best_errors = np.array(best_errors)

    plt.figure()
    plt.plot(best_errors[:, 0])
    plt.xlabel('Generation')
    plt.ylabel('Center Error')
    plt.title('Best Center Error Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/best_center_errors.png")

    plt.figure()
    plt.plot(best_errors[:, 1], label='Best NMSE')
    plt.xlabel('Generation')
    plt.ylabel('NMSE')
    plt.title('Best NMSE Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/best_nmse_errors.png")

    plt.figure()
    plt.plot(best_errors[:, 2], label='Best Spread Difference Error')
    plt.xlabel('Generation')
    plt.ylabel('Spread Difference Error')
    plt.title('Best Spread Difference Error Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/best_spread_diff_errors.png")

    plt.figure()
    plt.plot(best_errors[:, 3], label='Best Aim Spread Difference')
    plt.xlabel('Generation')
    plt.ylabel('Aim Spread Difference Error')
    plt.title('Best Aim Spread Difference Error Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/best_aim_spread_diff_errors.png")

    plt.figure()
    plt.plot(best_errors[:, 4], label='Best Frequency Error')
    plt.xlabel('Generation')
    plt.ylabel('Frequency Error')
    plt.title('Best Frequency Error Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/best_frequency_errors.png")

    plt.figure()
    plt.plot(best_errors[:, 5], label='Best Rate Change Error')
    plt.xlabel('Generation')
    plt.ylabel('Rate Change Error')
    plt.title('Best Rate Change Error Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/best_rate_change_errors.png")



    plt.figure()
    plt.plot(mean_errors_working_specimens, label='Mean Errors (Working Specimens)')
    plt.fill_between(range(len(mean_errors_working_specimens)), 
                     np.array(mean_errors_working_specimens) - np.array(std_errors_working_specimens), 
                     np.array(mean_errors_working_specimens) + np.array(std_errors_working_specimens), 
                     alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel('Mean Error')
    plt.title('Mean Errors Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/mean_errors.png")

    plt.figure()
    plt.plot(mean_center_error_working_specimens, label='Mean Center Error (Working Specimens)')
    plt.fill_between(range(len(mean_center_error_working_specimens)), 
                     np.array(mean_center_error_working_specimens) - np.array(std_center_error_working_specimens), 
                     np.array(mean_center_error_working_specimens) + np.array(std_center_error_working_specimens), 
                     alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel('Mean Center Error')
    plt.title('Mean Center Error Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/mean_center_errors.png")

    plt.figure()
    plt.plot(mean_nmse_working_specimens, label='Mean NMSE (Working Specimens)')
    plt.fill_between(range(len(mean_nmse_working_specimens)), 
                     np.array(mean_nmse_working_specimens) - np.array(std_nmse_working_specimens), 
                     np.array(mean_nmse_working_specimens) + np.array(std_nmse_working_specimens), 
                     alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel('Mean NMSE')
    plt.title('Mean NMSE Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/mean_nmses.png")

    plt.figure()
    plt.plot(mean_spread_diff_error_working_specimens, label='Mean Spread Difference Error (Working Specimens)')
    plt.fill_between(range(len(mean_spread_diff_error_working_specimens)), 
                     np.array(mean_spread_diff_error_working_specimens) - np.array(std_spread_diff_error_working_specimens), 
                     np.array(mean_spread_diff_error_working_specimens) + np.array(std_spread_diff_error_working_specimens), 
                     alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel('Mean Spread Difference Error')
    plt.title('Mean Spread Difference Error Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/mean_spread_diff_errors.png")

    plt.figure()
    plt.plot(mean_aim_spread_error_working_specimens, label='Mean Aim Spread Error (Working Specimens)')
    plt.fill_between(range(len(mean_aim_spread_error_working_specimens)), 
                     np.array(mean_aim_spread_error_working_specimens) - np.array(std_aim_spread_error_working_specimens), 
                     np.array(mean_aim_spread_error_working_specimens) + np.array(std_aim_spread_error_working_specimens), 
                     alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel('Mean Aim Spread Error')
    plt.title('Mean Aim Spread Error Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/mean_fourth_errors.png")

    plt.figure()
    plt.plot(mean_frequency_error_working_specimens, label='Mean Frequency Error (Working Specimens)')
    plt.fill_between(range(len(mean_frequency_error_working_specimens)),
                     np.array(mean_frequency_error_working_specimens) - np.array(std_frequency_error_working_specimens), 
                     np.array(mean_frequency_error_working_specimens) + np.array(std_frequency_error_working_specimens), 
                     alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel('Mean Frequency Error')
    plt.title('Mean Frequency Error Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/mean_fifth_errors.png")

    plt.figure()
    plt.plot(mean_rate_change_error_working_specimens, label='Mean Rate Change Error (Working Specimens)')
    plt.fill_between(range(len(mean_rate_change_error_working_specimens)),
                     np.array(mean_rate_change_error_working_specimens) - np.array(std_rate_change_error_working_specimens), 
                     np.array(mean_rate_change_error_working_specimens) + np.array(std_rate_change_error_working_specimens), 
                     alpha=0.2)
    plt.xlabel('Generation')
    plt.ylabel('Mean Frequency Error')
    plt.title('Mean Frequency Error Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/mean_sixth_errors.png")

    plt.figure()
    plt.plot(mean_gene_stddevs, label='Mean Gene Standard Deviations')
    plt.xlabel('Generation')
    plt.ylabel('Mean Gene Std Dev')
    plt.title('Mean Gene Standard Deviations Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/mean_gene_stddevs.png")

    plt.figure()
    plt.plot(gene_0_stddevs, label='g_cosine Std Dev')
    plt.xlabel('Generation')
    plt.ylabel('g_cosine Std Dev')
    plt.title('g_cosine Standard Deviations Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/g_cosine_stddevs.png")

    plt.figure()
    plt.plot(gene_1_stddevs, label='w_inh Std Dev')
    plt.xlabel('Generation')
    plt.ylabel('w_inh Std Dev')
    plt.title('w_inh Standard Deviations Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/w_inh_stddevs.png")

    plt.figure()
    plt.plot(gene_2_stddevs, label='I_ff Std Dev')
    plt.xlabel('Generation')
    plt.ylabel('I_ff Std Dev')
    plt.title('I_ff Standard Deviations Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/I_ff_stddevs.png")

    plt.figure()
    plt.plot(gene_3_stddevs, label='tau_s Std Dev')
    plt.xlabel('Generation')
    plt.ylabel('tau_s Std Dev')
    plt.title('tau_s Standard Deviations Over Generations')
    plt.legend()
    plt.savefig(f"{current_results_dirname}/tau_s_stddevs.png")
