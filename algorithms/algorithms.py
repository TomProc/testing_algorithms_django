import numpy as np
import random
import json
import pickle


class AlgorithmState:
    def __init__(self, iteration, population, fitness, best_solution, best_fitness):
        self.iteration = iteration
        self.population = population
        self.fitness = fitness
        self.best_solution = best_solution
        self.best_fitness = best_fitness


def save_state(filename, state):
    with open(filename, 'wb') as f:
        pickle.dump(state, f)


def load_state(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def benchmark_algorithm(algorithm_function, algorithm_name, functions, max_iterations, population_sizes, trials):
    results = []

    for function_name, (objective_function, properties) in functions.items():
        for dimensions in properties()['dimensions']:
            for pop_size in population_sizes:
                for max_iter in max_iterations:
                    best_fitness_list = []
                    for trial in range(trials):
                        lower_boundary = properties()['lower_boundary']
                        upper_boundary = properties()['upper_boundary']
                        best_solution, best_fitness = algorithm_function(
                            objective_function, dimensions, lower_boundary, upper_boundary, pop_size, max_iter
                        )
                        best_fitness_list.append(best_fitness)

                    avg_fitness = np.mean(best_fitness_list)
                    std_fitness = np.std(best_fitness_list)
                    results.append({
                        'algorithm': algorithm_name,
                        'function': function_name,
                        'dimensions': dimensions,
                        'population_size': pop_size,
                        'max_iterations': max_iter,
                        'average_fitness': avg_fitness,
                        'std_dev_fitness': std_fitness,
                        'trials': trials
                    })

    benchmark_file = f'benchmark_{algorithm_name.lower().replace(" ", "_")}.json'
    with open(benchmark_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Benchmark results saved to {benchmark_file}")


def jellyfish_search(objective_function, dimensions, lower_boundary, upper_boundary, population_size, max_iterations):
    jellyfish = np.random.uniform(lower_boundary, upper_boundary, (population_size, dimensions))
    fitness = np.array([objective_function(position) for position in jellyfish])
    best_solution = jellyfish[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    beta = 3.0
    gamma = 0.1

    for iteration in range(max_iterations):
        for current in range(population_size):
            time_control = abs((1 - iteration / max_iterations) * (2 * np.random.rand() - 1))

            if time_control >= 0.5:
                micro = np.mean(jellyfish, axis=0)
                ocean_current = best_solution - beta * np.random.rand() * micro
                jellyfish[current] = jellyfish[current] + np.random.rand() * ocean_current
            else:
                if np.random.rand() > 1 - time_control:
                    jellyfish[current] = jellyfish[current] + gamma * np.random.rand() * (upper_boundary - lower_boundary)
                else:
                    random_number = np.random.randint(population_size)
                    direction = (
                        jellyfish[random_number] - jellyfish[current]
                        if fitness[current] >= fitness[random_number]
                        else jellyfish[current] - jellyfish[random_number]
                    )
                    step = np.random.rand() * direction
                    jellyfish[current] += step

            jellyfish[current] = np.clip(jellyfish[current], lower_boundary, upper_boundary)
            new_fitness = objective_function(jellyfish[current])

            if new_fitness <= fitness[current]:
                fitness[current] = new_fitness
                if new_fitness <= best_fitness:
                    best_solution = jellyfish[current].copy()
                    best_fitness = new_fitness

    return best_solution, best_fitness


def artificial_bee_colony(objective_function, dimensions, lower_boundary, upper_boundary, population_size, max_iterations):
    solutions = np.random.uniform(lower_boundary, upper_boundary, (population_size, dimensions))
    values = np.full(population_size, np.inf)
    best_solution = None
    best_value = np.inf
    limit = 0.5 * max_iterations

    def initialize():
        nonlocal solutions, values, best_value
        solutions = np.random.uniform(lower_boundary, upper_boundary, (population_size, dimensions))
        values = np.array([np.inf] * population_size)
        best_value = np.inf

    def remember_best_solution():
        nonlocal best_solution, best_value
        best_index = np.argmin(values)
        if values[best_index] < best_value:
            best_value = values[best_index]
            best_solution = solutions[best_index]

    def employed_bees_phase():
        for i in range(population_size):
            k = random.choice([j for j in range(population_size) if j != i])
            phi = np.random.uniform(-1, 1, dimensions)
            candidate = solutions[i] + phi * (solutions[i] - solutions[k])
            candidate = np.clip(candidate, lower_boundary, upper_boundary)

            candidate_value = objective_function(candidate)
            if candidate_value < values[i]:
                solutions[i] = candidate
                values[i] = candidate_value
                iterations_without_improvement[i] = 0
            else:
                iterations_without_improvement[i] += 1

    def onlooker_bees_phase():
        fitness = 1 / (1 + values)
        probabilities = fitness / np.sum(fitness)

        for _ in range(population_size):
            i = np.random.choice(range(population_size), p=probabilities)
            k = random.choice([j for j in range(population_size) if j != i])
            phi = np.random.uniform(-1, 1, dimensions)
            candidate = solutions[i] + phi * (solutions[i] - solutions[k])
            candidate = np.clip(candidate, lower_boundary, upper_boundary)

            candidate_value = objective_function(candidate)
            if candidate_value < values[i]:
                solutions[i] = candidate
                values[i] = candidate_value
                iterations_without_improvement[i] = 0
            else:
                iterations_without_improvement[i] += 1

    def scout_bees_phase():
        for i in range(population_size):
            if iterations_without_improvement[i] > limit:
                solutions[i] = np.random.uniform(lower_boundary, upper_boundary, dimensions)
                values[i] = objective_function(solutions[i])
                iterations_without_improvement[i] = 0

    initialize()
    values = np.array([objective_function(solutions[i]) for i in range(population_size)])
    remember_best_solution()

    iterations_without_improvement = np.zeros(population_size)

    for _ in range(max_iterations):
        employed_bees_phase()
        remember_best_solution()

        onlooker_bees_phase()
        remember_best_solution()

        scout_bees_phase()
        remember_best_solution()

    return best_solution, best_value




