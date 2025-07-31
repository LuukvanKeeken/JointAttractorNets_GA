import pygad


ga_instance = pygad.load("GA_results/GA_run_20250630_143429/ga_instance_final")

solution, solution_fitness, solution_idx = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)
    
print(f"""Best solution found:
composite error: {1/solution_fitness}
sigma_exc: {solution[0]}
sigma_inh: {solution[1]}
g_exc: {solution[2]} mV
g_inh: {solution[3]} mV
""")