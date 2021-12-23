from numpy.random import rand, binomial, sample
from model import Individual

class Mutation():
  def __init__(self, pm, is_dynamic=False):
    self.pm = pm
    self.is_dynamic = is_dynamic

  def info(self):
    return f'mut-dynamic' if self.is_dynamic else f'mut-{self.pm}'

  def apply(self, individual):
    mutated = individual.copy_genotype()
    for i in range(mutated.size()):
      # check if a mutation will take place
      if rand() < self.pm:
        # flipping the bit
        mutated.genotype[i] = 1 - mutated.genotype[i]

    return mutated

class MutationBinomial():
  def __init__(self, pm):
    self.pm = pm

  def info(self):
    return f'mut-binom-{self.pm}'

  def apply(self, individual: Individual):
    l = binomial(individual.size(), self.pm)
    mutated = individual.copy_genotype()
    indexes_to_mutate = sample(range(0, individual.size()), l)
    for index in indexes_to_mutate:
      mutated.genotype[index] = 1 - mutated.genotype[index]

    return mutated
