from activestructopt.sampler.base import BaseSampler
from activestructopt.common.registry import registry
from activestructopt.common.constraints import lj_reject
from pymatgen.core.structure import IStructure
import numpy as np

@registry.register_sampler("SingleAtomPerturbation")
class SingleAtomPerturbation(BaseSampler):
  def __init__(self, initial_structure: IStructure, perturbrmin = 0.1, 
    perturbrmax = 1.0, perturblmax = 0.0, perturbθmax = 0.0, lattice_prob = 0.5) -> None:
    self.initial_structure = initial_structure
    self.perturbrmin = perturbrmin
    self.perturbrmax = perturbrmax
    self.perturblmax = perturblmax
    self.perturbθmax = perturbθmax
    self.lattice_prob = lattice_prob

  def sample(self) -> IStructure:
    rejected = True
    while rejected:
      new_structure = self.initial_structure.copy()
      new_structure2 = self.initial_structure.copy()
      new_structure2.perturb(np.random.uniform(
        self.perturbrmin, self.perturbrmax))

      if np.random.rand() > self.lattice_prob:
        rand_index = np.random.randint(0, len(new_structure))

        new_structure.sites[rand_index].frac_coords = new_structure2.sites[
          rand_index].frac_coords
      else:
        new_structure.lattice = new_structure.lattice.from_parameters(
          max(0.0, new_structure.lattice.a * np.random.uniform(
            1 - self.perturblmax, 1 + self.perturblmax)),
          max(0.0, new_structure.lattice.b * np.random.uniform(
            1 - self.perturblmax, 1 + self.perturblmax)),
          max(0.0, new_structure.lattice.c * np.random.uniform(
            1 - self.perturblmax, 1 + self.perturblmax)), 
          min(180.0, max(0.0, new_structure.lattice.alpha + np.random.uniform(
            -self.perturbθmax, self.perturbθmax))), 
          min(180.0, max(0.0, new_structure.lattice.beta + np.random.uniform(
            -self.perturbθmax, self.perturbθmax))), 
          min(180.0, max(0.0, new_structure.lattice.gamma + np.random.uniform(
            -self.perturbθmax, self.perturbθmax)))
        )
      rejected = lj_reject(new_structure)
    return new_structure
