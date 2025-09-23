import os
import itertools
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from brian2 import *  # Brian2 must be imported for the simulation
import time

# Import the simulation function from your model file
from optimization_model_faithful_test import opt_ring_attractor  # your simulation function
# Also import any utility functions if needed (e.g., for computing firing rates, etc.)
from utils import *


# Set the connectivity profile to optimize
# Options: 'cosine'
connectivity_profile = 'cosine'  


date = time.strftime('%Y%m%d_%H%M%S')

output_folder = "Optimization Results"

result_dir = os.path.join(output_folder, f"optimization_results_{connectivity_profile}_{date}")

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

num_neurons = 120



if connectivity_profile == 'cosine':
    # Cosine profile parameters - optimize g_cosine, glob_inh, and w_inh
    g_cosine_range = np.linspace(0.01, 0.5, 10)   # cosine gain with smaller scale
    # g_cosine_range = np.asarray([0.01, 0.33496, 0.5])
    glob_inh_range = [True]                # global inhibition flag
    w_inh_range = np.linspace(-1.0, 0.0, 10)    # global inhibition weight
    # w_inh_range = np.asarray([-2.0, -0.33478, 0.0])
    # Create parameter grid with conditional logic
    param_grid = []
    for g in g_cosine_range:
        # With global inhibition (g_cosine, True, and each w_inh value)
        for w in w_inh_range:
            param_grid.append((g, True, w))

    # Store the ranges in a txt file in the results dir
    with open(os.path.join(result_dir, "parameter_ranges.txt"), "w") as f:
        f.write("Cosine Profile Parameter Ranges:\n")
        f.write(f"Gain Cosine: {g_cosine_range}\n")
        f.write(f"Global Inhibition: {glob_inh_range}\n")
        f.write(f"Weight Inhibition: {w_inh_range}\n")
else:
    raise ValueError("Unsupported connectivity profile. Choose 'cosine'.")


total_runs = len(param_grid)

# Fixed parameters for the simulation (not searched over)
fixed_params = {
    'tau': 10,           # in our simulation, run_ring_attractor converts this to ms.
    'sigma_noise': 0.1,    # similarly converted to mV inside run_ring_attractor.
    'syn_profile': connectivity_profile,  # Set the connectivity profile
    'num_neurons': num_neurons
}



def worker_run(params_tuple):
    """
    Worker function for a single simulation run on one set of parameters.
    Returns a dictionary with the parameters and performance metrics.
    """
    # Build the full parameter dictionary based on connectivity profile
    params = fixed_params.copy()

    
    if connectivity_profile == 'cosine':
        # Three parameters being optimized: g_cosine, glob_inh, and w_inh (if glob_inh is True)
        g_cosine, glob_inh, w_inh = params_tuple
        params.update({
            'g_cosine': g_cosine,
            'placeholder': glob_inh
        })
        # Only add w_inh if global inhibition is enabled
        if glob_inh:
            params.update({'w_inh_val': w_inh})
    # print("test4")
    try:
        # Run the simulation.
        # opt_ring_attractor returns: (GT_center, GT_input, out_rates, out_pva_angle, out_pva_magnitude)
        GT_center, GT_input, out_rates, out_pva_angle, out_pva_magnitude, spread_difference = opt_ring_attractor(params)
        # print("test5")
        # Compute the circular standard deviation (spread) from the PVA magnitude.
        circular_std = np.sqrt(-2 * np.log(out_pva_magnitude + 1e-8))
        # print("test6")
        # Compute the center error and the confidence weighted center error (CWCE).
        center_err, cwce = conf_weighted_CE(out_pva_angle, GT_center, out_pva_magnitude)
        # print("test7")
        # Compute the angular Z-score:
        # This expresses the misalignment (center_err) in units of the circular standard deviation,
        # analogous to a z-score in linear statistics.
        angular_Zscore = center_err / circular_std
        # print("test8")
        # Compute the NMSE between the observed firing rates and the ideal Gaussian profile.
        nmse = compute_nmse_normalized(out_rates, GT_input, norm_type='max')
        # print("test9")

        # Normalize the spread difference by the number of neurons, and take the
        # absolute value to punish increases and decreases equally. Add 1 to make sure
        # everything is above zero.
        spread_err = np.abs(spread_difference / num_neurons) + 1
        # print("test10")
        # Combine the errors into one composite score.
        # Adjust weights to prioritize center accuracy if desired
        w_center = 0.25
        w_Zscore = 0.25
        w_nmse = 0.25
        w_spread = 0.25
        composite_error = w_center * cwce + w_Zscore * angular_Zscore + w_nmse * nmse + w_spread * spread_err
        # print(f"composite_error: {composite_error}")
        # Create result dictionary with profile-specific parameters
        result = {
            'composite_error': float(composite_error),
            'center_error': float(center_err),
            'pva_magnitude': float(out_pva_magnitude),
            'circular_std': float(circular_std),
            'cwce': float(cwce),
            'angular_Zscore': float(angular_Zscore),
            'nmse': float(nmse),
            'spread_error': float(spread_err),
            'error_message': np.nan
        }
        # print("test12")
        # Add profile-specific parameters to results
        if connectivity_profile == 'cosine':
            result.update({
                'g_cosine': g_cosine,
                'glob_inh': glob_inh,
                'w_inh': w_inh if glob_inh else np.nan
            })
        # print("test13")    
        return result
    
    except Exception as e:
        print(e)
        # Create error result with profile-specific parameters
        result = {
            'center_error': np.nan,
            'pva_magnitude': np.nan,
            'circular_std': np.nan,
            'cwce': np.nan,
            'angular_Zscore': np.nan,
            'nmse': np.nan,
            'composite_error': np.nan,
            'spread_error': np.nan,
            'error_message': str(e)
        }
        # print("test14")
        # Add profile-specific parameters to error results
        if connectivity_profile == 'cosine':
            result.update({
                'g_cosine': g_cosine,
                'glob_inh': glob_inh,
                'w_inh': w_inh if glob_inh else np.nan
            })
        # print("test15")
        return result

if __name__ == '__main__':
    print(f"Starting grid search optimization for {connectivity_profile} connectivity profile")
    print(f"Total parameter combinations to evaluate: {total_runs}")
    
    # num_proc = mp.cpu_count()   # Adjust the number of worker processes based on your system
    num_proc = 16
    results = []
    print(f"Using {num_proc} processes for grid search.")
    # Use Pool.imap_unordered with tqdm for progress tracking.
    with mp.Pool(processes=num_proc) as pool:
        # print("test1")
        for res in tqdm(pool.imap_unordered(worker_run, param_grid), total=total_runs, desc="Grid Search"):
            # print("test2")
            results.append(res)
    
    # Convert results to a pandas DataFrame for easier sorting and saving.
    results_df = pd.DataFrame(results)
    
    # # Create folder for results if it doesn't exist.
    # output_folder = "Optimization Results"
    # os.makedirs(output_folder, exist_ok=True)
    
    # Save the complete results to a CSV file with profile name in the filename inside the folder
    results_filename = os.path.join(result_dir, f"optimization_results_{connectivity_profile}.csv")
    results_df.to_csv(results_filename, index=False)
    print(f"Results saved to {results_filename}")
    
    # Results Analysis
    # If there are any errors, you might want to filter them out for selecting the best parameters.
    valid_results = results_df[~results_df['composite_error'].isna()]
    
    if not valid_results.empty:
        sortBy = 'composite_error'
        # Sort the DataFrame based on composite_error (lower is better).
        best_result = valid_results.sort_values(by=[sortBy], ascending=[True]).iloc[0]
        print(f"Best parameter set found for {connectivity_profile} profile (sorted by {sortBy}):")
        print(best_result)

        # Save best result to txt file in results_dir
        best_result_filename = os.path.join(result_dir, f"best_result_{connectivity_profile}.txt")
        with open(best_result_filename, 'w') as f:
            f.write(best_result.to_string())
        print(f"Best result saved to {best_result_filename}")

        # Assuming results_df is already loaded with columns: 'w_inh', 'g_cosine', 'composite_error'
        # Mask composite_error > 1 as NaN for plotting
        plot_df = results_df.copy()
        plot_df.loc[plot_df['composite_error'] > 1, 'composite_error'] = np.nan
        pivot = plot_df.pivot_table(index='g_cosine', columns='w_inh', values='composite_error')

        plt.figure(figsize=(8, 6))
        plt.imshow(pivot, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Composite Error')
        plt.xlabel('w_inh')
        plt.ylabel('g_cosine')
        plt.title('Composite Error Grid')
        plt.xticks(ticks=np.arange(len(pivot.columns)), labels=[f"{x:.2f}" for x in pivot.columns])
        plt.yticks(ticks=np.arange(len(pivot.index)), labels=[f"{y:.2f}" for y in pivot.index])
        plt.savefig(os.path.join(result_dir, f'composite_error_heatmap_{connectivity_profile}.png'))
    else:
        print("No valid simulation results found.")
