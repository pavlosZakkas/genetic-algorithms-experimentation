import numpy as np
from model import Population, Individual

def create_population(pop_size, n=50) -> Population:
  individuals = []
  for i in range(pop_size):
    new = np.random.randint(2, size=n, dtype=int)
    individuals.append(Individual(genotype=list(new)))

  return Population(individuals=individuals)

def get_budget(budget, func):
  if budget is None:
    budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)

  return budget

def get_optimum_for(func):
  if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
    optimum = 8
  else:
    optimum = func.objective.y
  return optimum
