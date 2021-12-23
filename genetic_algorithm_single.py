from ioh import get_problem
from ioh import logger
import sys
import numpy as np
from numpy.random import rand
from typing import List
from model import Population, Individual
from mutation import Mutation
from crossover import NPointCrossover, UniformCrossover, Crossover
from selection import ProportionalSelection, TournamentSelection, Selection
from common import create_population, get_budget, get_optimum_for

ONE_MAX = {
  'id': 1,
  'name': 'one_max',
  'bits_length': 50,
  'budget_limit': None,
  'seed': 41
}
LEADING_ONES = {
  'id': 2,
  'name': 'leading_ones',
  'bits_length': 50,
  'budget_limit': None,
  'seed': 421
}
LABS = {
  'id': 18,
  'name': 'labs',
  'bits_length': 32,
  'budget_limit': None,
  'seed': 51
}

ONE_MAX_CONFIGURATION = [
  (UniformCrossover(pc=0.5), Mutation(pm=0.02, is_dynamic=False), TournamentSelection(k=3, factor=1.0), 4),
  (UniformCrossover(pc=0.5), Mutation(pm=0.02, is_dynamic=True), TournamentSelection(k=3, factor=1.0), 4),
  (UniformCrossover(pc=0.875), Mutation(pm=0.02, is_dynamic=True), TournamentSelection(k=2, factor=1.0), 6),
  (UniformCrossover(pc=1.0), Mutation(pm=0.02, is_dynamic=True), ProportionalSelection(factor=1.0), 4),
  (NPointCrossover(pc=1, nc=6), Mutation(pm=0.02, is_dynamic=True), ProportionalSelection(factor=1.0), 4),
  (UniformCrossover(pc=1.0), Mutation(pm=0.02, is_dynamic=True), TournamentSelection(k=3, factor=2.0), 6),
]

LEADING_ONES_CONFIGURATION = [
  (NPointCrossover(pc=0.625, nc=4), Mutation(is_dynamic=False, pm=0.02), TournamentSelection(k=2, factor=1.0), 4),
  (NPointCrossover(pc=0.625, nc=2), Mutation(is_dynamic=False, pm=0.02), ProportionalSelection(factor=1.0), 10),
  (NPointCrossover(pc=0.75, nc=2), Mutation(is_dynamic=False, pm=0.02), ProportionalSelection(factor=2.0), 10),
  (NPointCrossover(pc=0.75, nc=2), Mutation(is_dynamic=False, pm=0.02), TournamentSelection(k=3, factor=0.5), 4),
  (NPointCrossover(pc=0.625, nc=4), Mutation(is_dynamic=False, pm=0.02), TournamentSelection(k=3, factor=1.5), 4),
  (NPointCrossover(pc=0.625, nc=4), Mutation(is_dynamic=False, pm=0.02), TournamentSelection(k=2, factor=1.0), 4),
  (NPointCrossover(pc=0.4, nc=4), Mutation(is_dynamic=False, pm=0.02), ProportionalSelection(factor=1.0), 10),
  (NPointCrossover(pc=0.625, nc=6), Mutation(is_dynamic=False, pm=0.02), ProportionalSelection(factor=2.0), 6),
]

LABS_CONFIGURATION = [
  (NPointCrossover(pc=0.5, nc=2, is_dynamic=False), Mutation(pm=0.02, is_dynamic=True),
   ProportionalSelection(factor=1.0), 50),
  (NPointCrossover(pc=0.5, nc=2, is_dynamic=True), Mutation(pm=0.02, is_dynamic=True),
   ProportionalSelection(factor=2.0), 50),
  (NPointCrossover(pc=0.7, nc=2, is_dynamic=False), Mutation(pm=0.02, is_dynamic=False),
   ProportionalSelection(factor=1.0), 100),
  (NPointCrossover(pc=0.7, nc=2, is_dynamic=True), Mutation(pm=0.02, is_dynamic=False),
   TournamentSelection(k=3, factor=1.0), 50),
  (NPointCrossover(pc=0.7, nc=2, is_dynamic=True), Mutation(pm=0.02, is_dynamic=False),
   TournamentSelection(k=3, factor=1.0), 100),
  (NPointCrossover(pc=0.9, nc=4), Mutation(pm=0.02, is_dynamic=True), TournamentSelection(k=3, factor=1.0), 300),
]

def offsprings_from(pairs, crossover, mutation: Mutation, fitnesses) -> List[Individual]:
  offsprings = list()
  for pair in pairs:
    o1, o2 = crossover.apply(pair, fitnesses)
    o1 = mutation.apply(o1)
    o2 = mutation.apply(o2)
    offsprings.append(o1)
    offsprings.append(o2)

  return offsprings

def next_population_from(offsprings: Population, population: Population) -> Population:
  parents_and_offsprings = Population(
    individuals=population.individuals.copy() + offsprings.individuals.copy()
  )
  sorted_individuals = sorted(
    parents_and_offsprings.individuals,
    key=lambda x: x.fitness,
    reverse=True
  )

  return Population(individuals=sorted_individuals[:population.size()])

def genetic_search(
  func,
  selection: Selection,
  crossover: Crossover,
  mutation: Mutation,
  budget=None,
  n=50,
  pop_size=10,
):
  score = list()
  max_budget = get_budget(budget, func)
  optimum = get_optimum_for(func)

  # 10 independent runs for each algorithm on each problem.
  for r in range(10):
    remaining_budget = max_budget
    f_opt = sys.float_info.min
    x_opt = None

    population = create_population(pop_size, n)
    remaining_budget = population.evaluate_fitnesses(func, remaining_budget)

    while func.state.evaluations < max_budget and not func.state.optimum_found:
      pairs = selection.select_pairs(population)
      average_fitness = population.ave_fitness()
      if mutation.is_dynamic:
        mutation.pm = 1 / (2 * (average_fitness + 1) - n) if average_fitness >= n / 2 else 1 / n
      offsprings = offsprings_from(pairs, crossover, mutation, population.get_fitnesses())
      offsprings_population = Population(individuals=offsprings)
      remaining_budget = offsprings_population.evaluate_fitnesses(func, remaining_budget)
      population = next_population_from(offsprings_population, population)

      best_individual = population.best_individual()

      if f_opt < best_individual.fitness:
        f_opt = best_individual.fitness
        x_opt = best_individual.genotype
      if f_opt >= optimum:
        score.append(max_budget - remaining_budget)
        break

    if func.state.evaluations >= max_budget:
      score.append(max_budget + 1)

    func.reset()
  return f_opt, x_opt, score

def get_logger_for(problem, crossover, mutation, selection, pop_size):
  return logger.Analyzer(
    root="data",
    folder_name=f'{problem}/ga_pop-{pop_size}_{selection.info()}_{crossover.info()}_{mutation.info()}',
    algorithm_name=f'ga_pop-{pop_size}_{selection.info()}_{crossover.info()}_{mutation.info()}',
    algorithm_info=f"GA execution on problem {problem} with "
                   f"population_size: {pop_size}, "
                   f"selection: {selection.info()}, "
                   f"mutation: {mutation.info()}, "
                   f"crossover: {crossover.info()}"
  )

def execute_experiment(configuration, problem):
  (crossover, mutation, selection, population_size) = configuration
  algorithm = problem['name']
  algorithm_id = problem['id']
  N = problem['bits_length']
  budget_limit = problem['budget_limit']

  l_problem = get_logger_for(algorithm, crossover, mutation, selection, population_size)
  problem = get_problem(algorithm_id, dim=N, iid=1, problem_type='PBO')
  problem.attach_logger(l_problem)

  best_fitness, _, problem_score = genetic_search(
    problem,
    selection=selection,
    crossover=crossover,
    mutation=mutation,
    pop_size=population_size,
    n=N,
    budget=budget_limit
  )

  del l_problem
  log = f'{population_size},{selection.info()},{crossover.info()},{mutation.info()} - Budget used per iteration: {np.array(problem_score, dtype=int)} - max fitness: {best_fitness}'
  print(log)
  return log

if __name__ == '__main__':

  np.random.seed(ONE_MAX['seed'])
  for config in ONE_MAX_CONFIGURATION:
    execute_experiment(config, ONE_MAX)

  np.random.seed(LEADING_ONES['seed'])
  for config in LEADING_ONES_CONFIGURATION:
    execute_experiment(config, LEADING_ONES)

  np.random.seed(LABS['seed'])
  for config in LABS_CONFIGURATION:
    execute_experiment(config, LABS)
