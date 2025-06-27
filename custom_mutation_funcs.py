import numpy
import random

def adaptive_perturbation_mutation(offspring, ga_instance):

    """
    Applies the adaptive mutation which changes the values of a number of genes randomly. In adaptive mutation, the number of genes to mutate differs based on the fitness value of the solution.
    The random value is selected either using the 'gene_space' parameter or the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
    It accepts a single parameter:
        -offspring: The offspring to mutate.
    It returns an array of the mutated offspring.
    """

    # If the attribute 'gene_space' exists (i.e. not None), then the mutation values are selected from the 'gene_space' parameter according to the space of values of each gene. Otherwise, it is selected randomly based on the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
    # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation. Otherwise, the 'mutation_num_genes' parameter is used.

    if ga_instance.mutation_probability is None:
        # When the 'mutation_probability' parameter does not exist (i.e. None), then the parameter 'mutation_num_genes' is used in the mutation.
        if not (ga_instance.gene_space is None):
            # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
            # offspring = ga_instance.adaptive_mutation_by_space(offspring)
            raise NotImplementedError("Adaptive perturbation mutation with 'gene_space' is not implemented in this version.")
        else:
            # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
            average_fitness, offspring_fitness = ga_instance.adaptive_mutation_population_fitness(offspring)

            # Adaptive random mutation changes one or more genes in each offspring randomly.
            # The number of genes to mutate depends on the solution's fitness value.
            for offspring_idx in range(offspring.shape[0]):
                ## TODO Make edits to work with multi-objective optimization.
                # Compare the fitness of each offspring to the average fitness of each objective function.
                fitness_comparison = offspring_fitness[offspring_idx] < average_fitness

                # Check if the problem is single or multi-objective optimization.
                if type(fitness_comparison) in [bool, numpy.bool_]:
                    # Single-objective optimization problem.
                    if fitness_comparison:
                        adaptive_mutation_num_genes = ga_instance.mutation_num_genes[0]
                    else:
                        adaptive_mutation_num_genes = ga_instance.mutation_num_genes[1]
                else:
                    # Multi-objective optimization problem.

                    # Get the sum of the pool array (result of comparison).
                    # True is considered 1 and False is 0.
                    fitness_comparison_sum = sum(fitness_comparison)
                    # Check if more than or equal to 50% of the objectives have fitness greater than the average.
                    # If True, then use the first percentage. 
                    # If False, use the second percentage.
                    if fitness_comparison_sum >= len(fitness_comparison)/2:
                        adaptive_mutation_num_genes = ga_instance.mutation_num_genes[0]
                    else:
                        adaptive_mutation_num_genes = ga_instance.mutation_num_genes[1]

                mutation_indices = numpy.array(random.sample(range(0, ga_instance.num_genes), adaptive_mutation_num_genes))
                for gene_idx in mutation_indices:

                    range_min, range_max = ga_instance.get_mutation_range(gene_idx)

                    # Generating a random value.
                    random_value = numpy.random.uniform(low=range_min, 
                                                        high=range_max, 
                                                        size=1)[0]
                    # Change the random mutation value data type.
                    random_value = ga_instance.change_random_mutation_value_dtype(random_value, gene_idx)

                    # Round the gene.
                    random_value = ga_instance.round_random_mutation_value(random_value, gene_idx)

                    offspring[offspring_idx, gene_idx] += random_value

                    if ga_instance.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = ga_instance.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                                min_val=range_min,
                                                                                                max_val=range_max,
                                                                                                mutation_by_replacement=ga_instance.mutation_by_replacement,
                                                                                                gene_type=ga_instance.gene_type,
                                                                                                num_trials=10)
            
    else:
        # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation.
        if not (ga_instance.gene_space is None):
            # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
            # offspring = ga_instance.adaptive_mutation_probs_by_space(offspring)
            raise NotImplementedError("Adaptive perturbation mutation with 'gene_space' is not implemented in this version.")
        else:
            # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
            average_fitness, offspring_fitness = ga_instance.adaptive_mutation_population_fitness(offspring)

            # Adaptive random mutation changes one or more genes in each offspring randomly.
            # The probability of mutating a gene depends on the solution's fitness value.
            for offspring_idx in range(offspring.shape[0]):
                ## TODO Make edits to work with multi-objective optimization.
                # Compare the fitness of each offspring to the average fitness of each objective function.
                fitness_comparison = offspring_fitness[offspring_idx] < average_fitness

                # Check if the problem is single or multi-objective optimization.
                if type(fitness_comparison) in [bool, numpy.bool_]:
                    # Single-objective optimization problem.
                    if offspring_fitness[offspring_idx] < average_fitness:
                        adaptive_mutation_probability = ga_instance.mutation_probability[0]
                    else:
                        adaptive_mutation_probability = ga_instance.mutation_probability[1]
                else:
                    # Multi-objective optimization problem.

                    # Get the sum of the pool array (result of comparison).
                    # True is considered 1 and False is 0.
                    fitness_comparison_sum = sum(fitness_comparison)
                    # Check if more than or equal to 50% of the objectives have fitness greater than the average.
                    # If True, then use the first percentage. 
                    # If False, use the second percentage.
                    if fitness_comparison_sum >= len(fitness_comparison)/2:
                        adaptive_mutation_probability = ga_instance.mutation_probability[0]
                    else:
                        adaptive_mutation_probability = ga_instance.mutation_probability[1]

                probs = numpy.random.random(size=offspring.shape[1])
                for gene_idx in range(offspring.shape[1]):

                    range_min, range_max = ga_instance.get_mutation_range(gene_idx)

                    if probs[gene_idx] <= adaptive_mutation_probability:
                        # Generating a random value.
                        random_value = numpy.random.uniform(low=range_min, 
                                                            high=range_max, 
                                                            size=1)[0]
                        # Change the random mutation value data type.
                        random_value = ga_instance.change_random_mutation_value_dtype(random_value, gene_idx)

                        # Round the gene.
                        random_value = ga_instance.round_random_mutation_value(random_value, gene_idx)

                        offspring[offspring_idx, gene_idx] += random_value

                        if ga_instance.allow_duplicate_genes == False:
                            offspring[offspring_idx], _, _ = ga_instance.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                                min_val=range_min,
                                                                                                max_val=range_max,
                                                                                                mutation_by_replacement=ga_instance.mutation_by_replacement,
                                                                                                gene_type=ga_instance.gene_type,
                                                                                                num_trials=10)
            

    return offspring




def random_perturbation_mutation(offspring, ga_instance):
     # If the mutation values are selected from the mutation space, the attribute 'gene_space' is not None. Otherwise, it is None.
    # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation. Otherwise, the 'mutation_num_genes' parameter is used.

    if ga_instance.mutation_probability is None:
        # When the 'mutation_probability' parameter does not exist (i.e. None), then the parameter 'mutation_num_genes' is used in the mutation.
        if not (ga_instance.gene_space is None):
            # When the attribute 'gene_space' exists (i.e. not None), the mutation values are selected randomly from the space of values of each gene.
            # offspring = ga_instance.mutation_by_space(offspring)
            raise NotImplementedError("Perturbation mutation with 'gene_space' is not implemented in this version.")
        else:
            # Random mutation changes one or more genes in each offspring randomly.
            for offspring_idx in range(offspring.shape[0]):
                mutation_indices = numpy.array(random.sample(range(0, ga_instance.num_genes), ga_instance.mutation_num_genes))
                for gene_idx in mutation_indices:

                    range_min, range_max = ga_instance.get_mutation_range(gene_idx)

                    # Generating a random value.
                    random_value = numpy.random.uniform(low=range_min, 
                                                        high=range_max, 
                                                        size=1)[0]
                    # Change the random mutation value data type.
                    random_value = ga_instance.change_random_mutation_value_dtype(random_value, gene_idx)

                    # Round the gene.
                    random_value = ga_instance.round_random_mutation_value(random_value, gene_idx)

                    offspring[offspring_idx, gene_idx] += random_value

                    if ga_instance.allow_duplicate_genes == False:
                        offspring[offspring_idx], _, _ = ga_instance.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                            min_val=range_min,
                                                                                            max_val=range_max,
                                                                                            mutation_by_replacement=ga_instance.mutation_by_replacement,
                                                                                            gene_type=ga_instance.gene_type,
                                                                                            num_trials=10)

            
    else:
        # When the 'mutation_probability' parameter exists (i.e. not None), then it is used in the mutation.
        if not (ga_instance.gene_space is None):
            # When the attribute 'gene_space' does not exist (i.e. None), the mutation values are selected randomly based on the continuous range specified by the 2 attributes 'random_mutation_min_val' and 'random_mutation_max_val'.
            # offspring = ga_instance.mutation_probs_by_space(offspring)
            raise NotImplementedError("Perturbation mutation with 'gene_space' is not implemented in this version.")
        else:
            # Random mutation changes one or more genes in each offspring randomly.
            for offspring_idx in range(offspring.shape[0]):
                probs = numpy.random.random(size=offspring.shape[1])
                for gene_idx in range(offspring.shape[1]):

                    range_min, range_max = ga_instance.get_mutation_range(gene_idx)

                    if probs[gene_idx] <= ga_instance.mutation_probability:
                        # Generating a random value.
                        random_value = numpy.random.uniform(low=range_min, 
                                                            high=range_max, 
                                                            size=1)[0]
                        # Change the random mutation value data type.
                        random_value = ga_instance.change_random_mutation_value_dtype(random_value, gene_idx)

                        # Round the gene.
                        random_value = ga_instance.round_random_mutation_value(random_value, gene_idx)

                        offspring[offspring_idx, gene_idx] += random_value

                        if ga_instance.allow_duplicate_genes == False:
                            offspring[offspring_idx], _, _ = ga_instance.solve_duplicate_genes_randomly(solution=offspring[offspring_idx],
                                                                                                min_val=range_min,
                                                                                                max_val=range_max,
                                                                                                mutation_by_replacement=ga_instance.mutation_by_replacement,
                                                                                                gene_type=ga_instance.gene_type,
                                                                                                num_trials=10)
            return offspring

    return offspring