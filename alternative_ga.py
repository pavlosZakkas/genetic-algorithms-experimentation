from ioh import get_problem
from ioh import logger
from common import create_population, get_budget, get_optimum_for
import sys
import numpy as np
from numpy.random import rand
from model import Population
from mutation import Mutation
from crossover import BiasedUniformCrossover

global problem_scores
global problem_runs
global problem_fitnesses

ONE_MAX = {
  'id': 1,
  'name': 'one_max',
  'bits_length': 50,
  'budget_limit': 10000,
  'seed': 42
}
LEADING_ONES = {
  'id': 2,
  'name': 'leading_ones',
  'bits_length': 50,
  'budget_limit': None,
  'seed': 90
}
LABS = {
  'id': 18,
  'name': 'labs',
  'bits_length': 32,
  'budget_limit': None,
  'seed': 42
}

def alt_genetic_search(
  func,
  pc,
  pm,
  lamda=1,
  budget=None,
  n=50,
):
  score = list()
  max_budget = get_budget(budget, func)
  optimum = get_optimum_for(func)

  # 10 independent runs for each algorithm on each problem.
  for r in range(10):
    remaining_budget = max_budget
    f_opt = sys.float_info.min
    x_opt = None

    population = create_population(1, n)
    individual = population.individuals[0]
    remaining_budget = population.evaluate_fitnesses(func, remaining_budget)

    while func.state.evaluations < max_budget and not func.state.optimum_found:
      mutated_offsprings = [Mutation(pm=pm).apply(individual) for i in range(lamda)]
      mutated_population = Population(individuals=mutated_offsprings)
      remaining_budget = mutated_population.evaluate_fitnesses(func, remaining_budget)
      best_mutated_offspring = mutated_population.best_individual()

      crossovered_offsprings = [
        BiasedUniformCrossover(pc=pc).apply(individual, best_mutated_offspring)
        for i in range(lamda)
      ]
      crossovered_population = Population(individuals=crossovered_offsprings)
      remaining_budget = crossovered_population.evaluate_fitnesses(func, remaining_budget)
      best_crossovered_offspring = crossovered_population.best_individual()

      if best_crossovered_offspring.fitness > individual.fitness:
        individual = best_crossovered_offspring
        population = Population(individuals=[individual])

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

def get_logger_for(problem, pc, pm, lamda):
  return logger.Analyzer(
    root="data",
    folder_name=f'{problem}/alt_ga_lamda-{lamda}_pc-{pc}_pm-{pm}',
    algorithm_name=f'alt_ga_lamda-{lamda}_pc-{pc}_pm-{pm}',
    algorithm_info=f"alt GA execution on problem {problem} with pc: {pc} and pm: {pm}"
  )

def execute_experiment(pc, pm, lamda, problem):
  algorithm = problem['name']
  algorithm_id = problem['id']
  N = problem['bits_length']
  budget_limit = problem['budget_limit']

  l_problem = get_logger_for(algorithm, pc, pm, lamda)
  problem = get_problem(algorithm_id, dim=N, iid=1, problem_type='PBO')
  problem.attach_logger(l_problem)

  best_fitness, _, problem_score = alt_genetic_search(
    problem,
    pc=pc,
    pm=pm,
    lamda=lamda,
    n=N,
    budget=budget_limit
  )

  del l_problem
  log = f'alt_ga_lambda-{lamda}_pc-{pc}_pm-{pm} - Budget used per iteration: {np.array(problem_score, dtype=int)} - max fitness: {best_fitness}'
  print(log)
  return log

if __name__ == '__main__':

  for problem in [ONE_MAX, LEADING_ONES, LABS]:
    np.random.seed(problem['seed'])
    for lamda in [4, 8, 12]:
      pm = lamda / problem['bits_length']
      pc = 1 / lamda
      execute_experiment(pc, pm, lamda, problem)
