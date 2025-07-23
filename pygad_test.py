import numpy as np
import pygad

coef = np.random.randint(20, size=(1,10))
print("random coef:", coef)
desired_output = np.random.randint(100)
print("desired output:", desired_output)

def fitness_func(ga_instance, solution, solution_idx):
    print(type(solution))
    print(type(tuple(solution)))
    exit()
    print("solution_idx:", solution_idx)
    output = np.sum(solution * coef)
    fitness = 1.0 / (np.abs(output - desired_output) + 1e-6)  # Avoid division by zero
    return fitness

sol_per_pop = 100
num_generations = 1000
num_genes = len(coef[0])
num_parent_mating = 5

ga_instance = pygad.GA(num_generations=num_generations,
                       sol_per_pop=sol_per_pop,
                       fitness_func=fitness_func,
                        num_genes=num_genes,
                       num_parents_mating=num_parent_mating,
                       mutation_type="random")


ga_instance.run()


print("Parameters of the best solution : {solution}".format(solution=ga_instance.best_solution()))
print("Prediction = ", np.sum(ga_instance.best_solution()[0] * coef))