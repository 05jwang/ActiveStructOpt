import numpy as np
import copy
import math

def step(structure, latticeprob, σr, σl, σθ):
    new_struct = copy.deepcopy(structure)
    if np.random.rand() < latticeprob:
        lattice_step(new_struct, σl, σθ)
    else:
        positions_step(new_struct, σr)
    return new_struct

def lattice_step(structure, σl, σθ):
    structure.lattice = structure.lattice.from_parameters(
        np.maximum(0.0, structure.lattice.a + σl * np.random.randn()),
        np.maximum(0.0, structure.lattice.b + σl * np.random.randn()), 
        np.maximum(0.0, structure.lattice.c + σl * np.random.randn()), 
        structure.lattice.alpha + σθ * np.random.randn(), 
        structure.lattice.beta + σθ * np.random.randn(), 
        structure.lattice.gamma + σθ * np.random.randn()
    )

def positions_step(structure, σr):
    atom_i = np.random.choice(range(len(structure)))
    structure.sites[atom_i].a = (structure.sites[atom_i].a + 
        σr * np.random.rand() / structure.lattice.a) % 1
    structure.sites[atom_i].b = (structure.sites[atom_i].b + 
        σr * np.random.rand() / structure.lattice.b) % 1
    structure.sites[atom_i].c = (structure.sites[atom_i].c + 
        σr * np.random.rand() / structure.lattice.c) % 1

def 𝛘2(exp, th, σ):
    return np.mean((exp - th) ** 2) / (σ ** 2)

def reject(structure):
    dists = structure.distance_matrix.flatten()
    return np.min(dists[dists > 0]) < 1

def rmc(optfunc, args, exp, σ, structure, N, latticeprob = 0.1, σr = 0.5, σl = 0.1, σθ = 1.0):
    structures = []
    𝛘2s = []
    accepts = []
    uncertainties = []
    old_structure = structure
    old_𝛘2 = 𝛘2(exp, optfunc(old_structure, **(args)), σ)

    for _ in range(N):
        new_structure = step(old_structure, latticeprob, σr, σl, σθ)
        res, resσ = optfunc(new_structure, **(args))
        new_𝛘2 = 𝛘2(exp, res, σ)
        Δχ2 = new_𝛘2 - old_𝛘2
        accept = np.random.rand() < np.exp(-Δχ2/2) and not reject(new_structure)
        structures.append(new_structure)
        𝛘2s.append(new_𝛘2)
        accepts.append(accept)
        uncertainties.append(np.mean(resσ))
        if accept:
            old_structure = copy.deepcopy(new_structure)
            old_𝛘2 = new_𝛘2

    return structures, 𝛘2s, accepts, uncertainties

def ei(y, yhat, s):
    return (y - yhat) * (1 / 2 + 1 / 2 * math.erf((y - yhat) / (
        s * np.sqrt(2)))) + (s / np.sqrt(2 * np.pi)) * np.exp(
        (-(y - yhat) ** 2) / (2 * s ** 2))

def 𝛘2_ei(exp, th, thσ, σ, y):
    # TODO: Verify that a normal approximation is appropriate here
    yhat = np.sum((thσ ** 2) + ((exp - th) ** 2)) / (len(exp) * σ ** 2)
    s = np.sqrt(2 * np.sum((thσ ** 4) + 2 * (thσ ** 2) * (
        (exp - th) ** 2))) / (len(exp) * σ ** 2)
    return -ei(y, yhat, s)


def rmc_ei(optfunc, args, exp, σ, structure, best, N, σr = 0.5):
    structures = []
    𝛘2s = []
    accepts = []
    uncertainties = []
    old_structure = structure
    old_𝛘2 = 0

    for _ in range(N):
        new_structure = step(old_structure, 0.0, σr, 0.0, 0.0)
        res, resσ = optfunc(new_structure, **(args))
        new_𝛘2 = 𝛘2_ei(exp, res, resσ, σ, best)
        Δχ2 = new_𝛘2 - old_𝛘2
        accept = np.random.rand() < np.exp(-Δχ2/2) and not reject(new_structure)
        structures.append(new_structure)
        𝛘2s.append(new_𝛘2)
        accepts.append(accept)
        uncertainties.append(np.mean(resσ))
        if accept:
            old_structure = copy.deepcopy(new_structure)
            old_𝛘2 = new_𝛘2

    return structures, 𝛘2s, accepts, uncertainties