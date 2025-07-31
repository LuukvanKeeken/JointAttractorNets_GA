#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include<random>

#include "code_objects/ring_neurons_spike_resetter_codeobject.h"
#include "code_objects/ring_neurons_spike_resetter_codeobject_1.h"
#include "code_objects/ring_neurons_spike_thresholder_codeobject.h"
#include "code_objects/ring_neurons_spike_thresholder_codeobject_1.h"
#include "code_objects/ring_neurons_stateupdater_codeobject.h"
#include "code_objects/ring_neurons_stateupdater_codeobject_1.h"
#include "code_objects/ring_synapses_asym_group_variable_set_conditional_codeobject.h"
#include "code_objects/ring_synapses_asym_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/ring_synapses_asym_pre_codeobject.h"
#include "code_objects/ring_synapses_asym_pre_codeobject_1.h"
#include "code_objects/ring_synapses_asym_pre_push_spikes.h"
#include "code_objects/ring_synapses_asym_synapses_create_generator_codeobject.h"
#include "code_objects/ring_synapses_group_variable_set_conditional_codeobject.h"
#include "code_objects/ring_synapses_group_variable_set_conditional_codeobject_1.h"
#include "code_objects/ring_synapses_pre_codeobject.h"
#include "code_objects/ring_synapses_pre_codeobject_1.h"
#include "code_objects/ring_synapses_pre_push_spikes.h"
#include "code_objects/ring_synapses_synapses_create_generator_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"
#include "code_objects/spikemonitor_codeobject_1.h"
#include "code_objects/statemonitor_codeobject.h"
#include "code_objects/statemonitor_codeobject_1.h"


void brian_start()
{
	_init_arrays();
	_load_arrays();
	// Initialize clocks (link timestep and dt to the respective arrays)
    brian::defaultclock.timestep = brian::_array_defaultclock_timestep;
    brian::defaultclock.dt = brian::_array_defaultclock_dt;
    brian::defaultclock.t = brian::_array_defaultclock_t;
    brian::networkoperation_clock.timestep = brian::_array_networkoperation_clock_timestep;
    brian::networkoperation_clock.dt = brian::_array_networkoperation_clock_dt;
    brian::networkoperation_clock.t = brian::_array_networkoperation_clock_t;
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}


