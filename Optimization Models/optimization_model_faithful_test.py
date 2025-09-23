from brian2 import *
import numpy as np

import sys
sys.path.append('Neuron and Synapse Models')
from neuronModels import *
from faithfulRingAttractorClass import *
from faithfulRingAttractorClass import FaithfulRingAttractor
sys.path.append('Tools')
from utils import *

def opt_ring_attractor(params, stim_center=0, stim_width=0.5):
    """
    Run the ring attractor simulation with the specified parameters.
    
    Parameters:
        params (dict): A dictionary containing parameter values, e.g.,
                       {
                         'tau': value,
                         'sigma_noise': value,
                         'sigma_exc': value,
                         'sigma_inh': value,
                         'g_exc': value,
                         'g_inh': value,
                         ... (others as needed)
                       }
    Returns:
        result: Any outcome from the simulation you wish to optimize (e.g., a cost metric)
    """
    # --- Simulation parameters ---


    # External input parameters (could also be passed in or kept fixed)
    stimulus_center = stim_center  
    stimulus_width = stim_width

    autapse = True

    defaultclock.dt = 0.1*ms
    num_neurons = 120

    # Use parameters from the dict, with appropriate units:
    tau = params.get('tau', 10)*ms
    tau_s = 13 * ms
    sigma_noise = params.get('sigma_noise', 0.1)*mV
    V_rest = -70*mV
    I0 = 10 * mV
    # sim_duration = duration_val*second
    g_cosine = params.get('g_cosine')*mV
    w_inh_v = params.get('w_inh_val')*mV
    
    Iff_val = params.get('Iff_val', 80)*mA
    I0_CONST = Iff_val * ohm
    
    # velocity_duration = velocity_duration_val 

    # Define neuron positions
    positions = linspace(0, 2*pi, num_neurons, endpoint=False)

    # Calculate external input
    d = np.angle(np.exp(1j * (positions - stimulus_center)))
    I_ext_array = I0 * np.exp(-(d**2) / (2 * stimulus_width**2)) + I0_CONST

    # Set up neuron model
    neuron_eq = Equations(LIF_xi_vel_eq, tau=tau, V_rest=V_rest, sigma_noise=sigma_noise)
    

    # Set up ring attractor
    Vth = -48 * mV
    V_reset = -80 * mV
    refractory_period = 5 * ms

    # Create the ring attractor network
    ringAttractor = FaithfulRingAttractor(neuron_eq, 
                        num_neurons, 
                        Vth, V_reset, refractory_period,
                        autapse=autapse, profile='cosine',
                        w_sub=w_inh_v, normalized=False,
                        g_cosine=g_cosine, g_sine=1*mV)
    # Set external input
    ringAttractor.ring_pool.I_ext = I_ext_array
    ringAttractor.ring_pool.I_vel = 0.0*volt

    

    # Setup monitors
    spikemon = SpikeMonitor(ringAttractor.ring_pool)
    statemon = StateMonitor(ringAttractor.ring_pool, 'V', record=True)
    inputmon = StateMonitor(ringAttractor.ring_pool, 'I_ext', record=True)
    isynmon = StateMonitor(ringAttractor.ring_pool, 'I_syn', record=True)

    # Clipping - Reverse Potential Behaviour: Define a network operation to enforce the lower bound on the membrane potential
    @network_operation(dt=defaultclock.dt)
    def enforce_lower_bound():
        # Using the built-in clip function (from numpy)
        ringAttractor.ring_pool.V[:] = clip(ringAttractor.ring_pool.V[:], V_reset, inf*volt)

    # Set of Brian objects to be added to the network
    localObjects = [enforce_lower_bound,
                    spikemon, statemon, inputmon, isynmon]
    
    net = Network(ringAttractor.BrianObjects+localObjects)

    input_on = 0.05 * second
    input_off = 0.95 * second
    velocity_on = 0 * second
    end_duration = 0.95 * second

    total_duration = input_on + input_off + velocity_on + end_duration

    # Run simulation
    net.run(input_on)

    # Turn off input for the second half
    ringAttractor.ring_pool.I_ext = I0_CONST
    net.run(input_off)

    # Turn on velocity input
    ringAttractor.ring_synapses_asym.vel_in = 0.0
    ringAttractor.ring_synapses_asym.vel_on = True

    net.run(velocity_on)

    # Turn off velocity input
    ringAttractor.ring_synapses_asym.vel_in = 0.0
    ringAttractor.ring_synapses_asym.vel_on = False

    # Turn off velocity input and run for the rest of the duration
    net.run(end_duration)

    
    firing_rates = compute_firing_rate(spikemon, num_neurons,
                                       start_time=input_on, end_time=total_duration)
                                    #    start_time=0.95*sim_duration, end_time=sim_duration)
                                    
    pva_angle, pva_magnitude = calculate_PVA(firing_rates, positions)
    
    # Positive spread difference means an increase in bump spread
    # between t1 and t2, negative means a decrease.
    spread_difference, spread_t1, spread_t2 = calculate_spreads(spikemon, t1=input_on, t2=total_duration)

    # Return simulation results
    return stimulus_center, I_ext_array, firing_rates, pva_angle, pva_magnitude, spread_difference

if __name__ == '__main__':
    # Example: run with default parameters when this file is executed directly
    default_params = {'tau': 10, 'sigma_noise': 1.0, 'sigma_exc': 0.125, 'sigma_inh': 0.1, 'g_exc': 1.0*mV, 'g_inh': -1.0*mV, 'syn_profile': 'mexican_hat', 'autapse': True, 'glob_inh': False}
    GT_center, GT_input, out_rates, out_pva_angle, out_pva_magnitude = opt_ring_attractor(default_params,  stim_center=0, stim_width=0.5)   
    print("Ground Truth Center:", GT_center)
    print("Ground Truth Input:", GT_input)
    print("Observed Firing Rates:", out_rates)
    print("PVA Angle:", out_pva_angle)
    print("PVA Magnitude:", out_pva_magnitude)