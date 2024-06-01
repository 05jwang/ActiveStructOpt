from abc import ABC, abstractmethod
from pymatgen.core.structure import IStructure
from activestructopt.simulation.base import BaseSimulation
from activestructopt.sampler.base import BaseSampler

class BaseDataset(ABC):
  @abstractmethod
  def __init__(self, simulation: BaseSimulation, sampler: BaseSampler, 
    initial_structure: IStructure, target, config, **kwargs):
    pass

  @abstractmethod
  def update(self, new_structure: IStructure):
    pass
