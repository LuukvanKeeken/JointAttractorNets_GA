import os
import itertools
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from brian2 import *  # Brian2 must be imported for the simulation
import time

# Import the simulation function from your model file
from optimization_model_faithful import opt_ring_attractor  # your simulation function
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
    # Cosine profile parameters - optimize g_cosine and w_inh
    g_cosine_range = np.linspace(0.001, 1.0, 22)   # cosine gain with smaller scale
    w_inh_range = np.linspace(-1.0, 0.0, 22)    # global inhibition weight
    Iff_range = np.linspace(80, 80, 1)
    tau_s_range = np.linspace(0.5, 100, 22)
    # Create parameter grid with conditional logic
    param_grid = []
    for g in g_cosine_range:
        for w in w_inh_range:
            for Iff in Iff_range:
                for tau_s in tau_s_range:
                    param_grid.append((g, w, Iff, tau_s))

    # Store the ranges in a txt file in the results dir
    with open(os.path.join(result_dir, "parameter_ranges.txt"), "w") as f:
        f.write("Cosine Profile Parameter Ranges:\n")
        f.write(f"Gain Cosine: {g_cosine_range}\n")
        f.write(f"Weight Inhibition: {w_inh_range}\n")
        f.write(f"Iff: {Iff_range}\n")
        f.write(f"Tau_s: {tau_s_range}\n")
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
        # Two parameters being optimized: g_cosine and w_inh
        g_cosine, w_inh, Iff, tau_s = params_tuple
        params.update({
            'g_cosine': g_cosine,
            'w_inh_val': w_inh,
            'Iff_val': Iff,
            'tau_s': tau_s
        })
        
    
    try:
        # Run the simulation.
        # opt_ring_attractor returns: (GT_center, GT_input, out_rates, out_pva_angle, out_pva_magnitude)
        GT_center, GT_input, out_rates, out_pva_angle, out_pva_magnitude, spread_difference, mid_sim_spread, max_firing_rate, Iff_firing_rate, highest_rate_1, highest_rate_2 = opt_ring_attractor(params)
        
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
        aim_spread = num_neurons/10
        lower_lim = (int(num_neurons/20) - 1)
        upper_lim = (3*int(num_neurons/20) + 1)
        if (mid_sim_spread > lower_lim) and (mid_sim_spread < upper_lim):
            aim_spread_error = -2/((mid_sim_spread - lower_lim)*(mid_sim_spread - upper_lim) + 1e-8)
        elif mid_sim_spread <= 0 or mid_sim_spread >= num_neurons:
            aim_spread_error = 5
        else:
            aim_spread_error = min(5, 1/((num_neurons/20)**2) * (mid_sim_spread - aim_spread)**2)
        



        # If no neurons are active, give high punishment
        if np.any(out_rates > 0):
            lowest_active_neuron_rate = np.min(out_rates[out_rates > 0])*Hz
            highest_active_neuron_rate = np.max(out_rates)*Hz

            # Highest active neuron should be as low as possible (to avoid saturation)
            high_error = highest_active_neuron_rate / max_firing_rate
            # Lowest active neuron should be above 40% of the Iff firing rate, but not
            # necessarily as high as possible.
            low_error = max(0, (0.4 * Iff_firing_rate - lowest_active_neuron_rate) / (0.4 * Iff_firing_rate))
            frequency_error = high_error + low_error
        else:
            frequency_error = 1000

        
        # If firing rates change over time, punish this.
        rate_change_error = np.abs(highest_rate_2 - highest_rate_1) / highest_rate_1
    

        
        # Combine the errors into one composite score.
        # Adjust weights to prioritize center accuracy if desired
        w_center = 0.1667
        w_nmse = 0.1667
        w_spread_diff = 0.1667
        w_aim_spread = 0.1667
        w_frequency = 0.1667
        w_rate_change = 0.1667
        composite_error = w_center * center_err + w_nmse * nmse + w_spread_diff * spread_diff_error + w_aim_spread * aim_spread_error + w_frequency * frequency_error + w_rate_change * rate_change_error

        # Create result dictionary with profile-specific parameters
        result = {
            'composite_error': float(composite_error),
            'center_error': float(center_err),
            'pva_magnitude': float(out_pva_magnitude),
            'circular_std': float(circular_std),
            'cwce': float(cwce),
            'angular_Zscore': float(angular_Zscore),
            'nmse': float(nmse),
            'spread_diff_error': float(spread_diff_error),
            'aim_spread_error': float(aim_spread_error),
            'frequency_error': float(frequency_error),
            'rate_change_error': float(rate_change_error),
            'error_message': np.nan
        }
        
        # Add profile-specific parameters to results
        if connectivity_profile == 'cosine':
            result.update({
                'g_cosine': g_cosine,
                'w_inh': w_inh,
                'Iff': Iff,
                'tau_s': tau_s
            })
            
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
            'spread_diff_error': np.nan,
            'aim_spread_error': np.nan,
            'frequency_error': np.nan,
            'rate_change_error': np.nan,
            'error_message': str(e)
        }
        
        # Add profile-specific parameters to error results
        if connectivity_profile == 'cosine':
            result.update({
                'g_cosine': g_cosine,
                'w_inh': w_inh,
                'Iff': Iff,
                'tau_s': tau_s
            })
        
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
        
        for res in tqdm(pool.imap_unordered(worker_run, param_grid), total=total_runs, desc="Grid Search"):
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

        # For each unique Iff value, create a separate heatmap
        for Iff_val in sorted(results_df['Iff'].unique()):
            sub_df = results_df[results_df['Iff'] == Iff_val].copy()
            sub_df.loc[sub_df['composite_error'] > 1, 'composite_error'] = np.nan
            pivot = sub_df.pivot_table(index='g_cosine', columns='w_inh', values='composite_error')

            plt.figure(figsize=(8, 6))
            plt.imshow(pivot, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Composite Error')
            plt.xlabel('w_inh')
            plt.ylabel('g_cosine')
            plt.title(f'Composite Error Grid (Iff={Iff_val})')

            # Limit ticks to 10 evenly spaced values for coarse view
            num_xticks = min(10, len(pivot.columns))
            num_yticks = min(10, len(pivot.index))
            xtick_indices = np.linspace(0, len(pivot.columns)-1, num_xticks, dtype=int)
            ytick_indices = np.linspace(0, len(pivot.index)-1, num_yticks, dtype=int)
            plt.xticks(ticks=xtick_indices, labels=[f"{pivot.columns[i]:.2f}" for i in xtick_indices])
            plt.yticks(ticks=ytick_indices, labels=[f"{pivot.index[i]:.2f}" for i in ytick_indices])
            plt.savefig(os.path.join(result_dir, f'composite_error_heatmap_{connectivity_profile}_Iff_{Iff_val:.2f}.png'))
            plt.close()
    else:
        print("No valid simulation results found.")
