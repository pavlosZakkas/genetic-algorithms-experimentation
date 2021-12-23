import os
import random
from ioh import get_problem
from ioh import logger
import sys
import numpy as np
from typing import List
from model import Population, Individual
from mutation import Mutation
from crossover import NPointCrossover, UniformCrossover, Crossover
from selection import ProportionalSelection, TournamentSelection, Selection
import threading
import concurrent.futures
import argparse
from common import create_population, get_budget, get_optimum_for

global problem_scores
global problem_runs
global problem_fitnesses
lock = threading.Lock()

# POPULATIONS = [10, 20]
POPULATIONS = [1, 2, 4, 10, 20, 50, 100, 300]

pc_range = np.linspace(0.5, 1.0, num=5)
ONE_CROSSOVERS = list(map(lambda pc: NPointCrossover(pc=pc, nc=1), pc_range))
TWO_CROSSOVERS = list(map(lambda pc: NPointCrossover(pc=pc, nc=2), pc_range))
FOUR_CROSSOVERS = list(map(lambda pc: NPointCrossover(pc=pc, nc=4), pc_range))
SIX_CROSSOVERS = list(map(lambda pc: NPointCrossover(pc=pc, nc=6), pc_range))
UNIFORM_CROSSOVERS = list(map(lambda pc: UniformCrossover(pc=pc), pc_range))

pm_range = np.linspace(0.02, 0.3, num=4)
STATIC_MUTATIONS = list(map(lambda pm: Mutation(pm=pm, is_dynamic=False), pm_range))
DYNAMIC_MUTATIONS = [Mutation(pm=0.03, is_dynamic=True)]

ALL_CROSSOVERS = ONE_CROSSOVERS + TWO_CROSSOVERS + FOUR_CROSSOVERS + SIX_CROSSOVERS + UNIFORM_CROSSOVERS
ALL_SELECTIONS = [ProportionalSelection(), TournamentSelection(k=2), TournamentSelection(k=3)]
ALL_MUTATIONS = STATIC_MUTATIONS + DYNAMIC_MUTATIONS

ONE_MAX = {
  'id': 1,
  'name': 'one_max',
  'bits_length': 50,
  'budget_limit': 500
}
LEADING_ONES = {
  'id': 2,
  'name': 'leading_ones',
  'bits_length': 50,
  'budget_limit': 10000
}
LABS = {
  'id': 18,
  'name': 'labs',
  'bits_length': 32,
  'budget_limit': None
}

PROBLEMS = {
  'one_max': ONE_MAX,
  'leading_ones': LEADING_ONES,
  'labs': LABS,
}

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
  for r in range(25):
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

  lock.acquire()
  problem_fitnesses.append(best_fitness)
  problem_scores.append(np.array(problem_score, dtype=int))
  problem_runs.append(f'ga_pop-{population_size}_{selection.info()}_{crossover.info()}_{mutation.info()}')
  lock.release()

  del l_problem
  log = f'{population_size},{selection.info()},{crossover.info()},{mutation.info()} - {np.mean(np.array(problem_score, dtype=int))} - {best_fitness}'
  print(log)
  return log

def parsed_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--threads', help='Number of threads to parallelize executions')
  parser.add_argument('--problem', help='Problem to execute ("one_max", "leading_ones", "labs")')
  return parser.parse_args()

if __name__ == '__main__':
  args = parsed_arguments()
  threads = int(args.threads) if args.threads != None else 16
  problem = PROBLEMS[args.problem]

  problem_scores = list()
  problem_runs = list()
  problem_fitnesses = list()
  configurations = list()

  for crossover in ALL_CROSSOVERS:
    for mutation in ALL_MUTATIONS:
      for population_size in POPULATIONS:
        for selection in ALL_SELECTIONS:
          configurations.append((crossover, mutation, selection, population_size))

  random.shuffle(configurations)
  try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
      futures = {executor.submit(execute_experiment, p, problem) for p in configurations}

    for future in concurrent.futures.as_completed(futures):
      pass
  except KeyboardInterrupt:
    print('Experiment was interrupted')
  except Exception as e:
    print(e)
  finally:
    if not os.path.exists('./results'):
      os.mkdir('./results')
    np.save(f'./results/{problem["name"]}_runs.npy', np.array(problem_runs, dtype=object))
    np.save(f'./results/{problem["name"]}_fitnesses.npy', np.array(problem_fitnesses, dtype=object))
    np.save(f'./results/{problem["name"]}.npy', np.array(problem_scores, dtype=object))
