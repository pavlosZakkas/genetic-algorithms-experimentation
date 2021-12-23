import numpy as np
from numpy.random import rand
from model import Pair, Individual
from typing import List

class Crossover():
  def apply(self, parents: Pair, fitnesses):
    pass

class NPointCrossover(Crossover):
  def __init__(self, pc, nc, is_dynamic=False):
    self.pc = pc
    self.nc = nc
    self.is_dynamic = is_dynamic

  def info(self):
    return f'{self.nc}-ncross-dynamic' if self.is_dynamic else f'{self.nc}-ncross-{self.pc}'

  def apply(self, parents: Pair, fitnesses) -> List[Individual]:
    if self.is_dynamic:
      if max(fitnesses) - min(fitnesses) == 0:
        self.pc = 0
      else:
        self.pc = abs(parents.parent1.fitness - parents.parent2.fitness) / (max(fitnesses) - min(fitnesses))

    p1 = parents.parent1.copy_genotype()
    p2 = parents.parent2.copy_genotype()
    c1 = np.array([], dtype=int)
    c2 = np.array([], dtype=int)
    pt = []

    if rand() < self.pc:
      # select crossover points that are not on the end of the binary string
      pt = np.sort(np.random.choice(range(1, p1.size() - 2),
                                    size=self.nc, replace=False))
      pt = np.append(pt, p1.size())
      pt = np.insert(pt, 0, 0)

    if len(pt) > 0:
      for index, val in enumerate(pt[:-1]):
        if index % 2 == 0:
          c1 = np.append(c1[:val], p1.genotype[val:pt[index + 1]])
          c2 = np.append(c2[:val], p2.genotype[val:pt[index + 1]])

        else:
          c1 = np.append(c1[:val], p2.genotype[val:pt[index + 1]])
          c2 = np.append(c2[:val], p1.genotype[val:pt[index + 1]])
    else:
      c1 = p1.genotype
      c2 = p2.genotype
    return [Individual(genotype=list(c1)), Individual(genotype=list(c2))]

class UniformCrossover(Crossover):
  def __init__(self, pc, is_dynamic=False):
    self.pc = pc
    self.is_dynamic = is_dynamic

  def info(self):
    return f'unif-dynamic' if self.is_dynamic else f'unif-{self.pc}'

  def apply(self, parents: Pair, fitnesses):
    if self.is_dynamic:
      if max(fitnesses) - min(fitnesses) == 0:
        self.pc = 0
      else:
        self.pc = abs(parents.parent1.fitness - parents.parent2.fitness) / (max(fitnesses) - min(fitnesses))

    p1 = parents.parent1.copy_genotype()
    p2 = parents.parent2.copy_genotype()
    c1 = p1.copy_genotype()
    c2 = p2.copy_genotype()
    if rand() < self.pc:
      for i in range(p1.size()):
        if rand() < 0.5:
          c1.genotype[i], c2.genotype[i] = p2.genotype[i], p1.genotype[i]

    return [c1, c2]

class BiasedUniformCrossover():
  def __init__(self, pc):
    self.pc = pc

  def info(self):
    return f'par-unif-{self.pc}'

  def apply(self, parent: Individual, offspring: Individual):
    parent = parent.copy_genotype()
    offspring = offspring.copy_genotype()
    crossovered = parent.copy_genotype()

    for i in range(parent.size()):
      if rand() < self.pc:
        crossovered.genotype[i] = offspring.genotype[i]

    return crossovered
