from mpi4py import MPI
import numpy as np
import random
from wolff import wolff
from Lattice import (
    lattice_1d,
    lattice_square,
    lattice_triangular,
    lattice_honeycomb,
)

# =================================
# ISING MODEL
# =================================
class Ising:
    def __init__(self, dims, h=0.0, beta=1.0, lattice="generic"):
        self.h = float(h)
        self.beta = float(beta)

        if lattice == "generic":
            self.dims = tuple(dims)
            self.neighbours_table = self._generic_lattice()
        elif lattice == "1d":
            self.dims, self.neighbours_table = lattice_1d(dims)
        elif lattice == "square":
            self.dims, self.neighbours_table = lattice_square(*dims)
        elif lattice == "triangular":
            self.dims, self.neighbours_table = lattice_triangular(*dims)
        elif lattice == "honeycomb":
            self.dims, self.neighbours_table = lattice_honeycomb(*dims)
        else:
            raise ValueError("Unknown lattice type")

        self.spins = np.random.choice([-1, 1], size=self.dims)
        self.cluster = np.zeros(self.dims, dtype=bool)
        self.sites = list(np.ndindex(self.dims))

    def size(self):
        return self.dims

    def length(self):
        return np.prod(self.dims)

    def _generic_lattice(self):
        D = len(self.dims)
        grid = np.indices(self.dims).transpose(*range(1, D + 1), 0)
        neighbours = []
        for axis in range(D):
            for shift in (-1, 1):
                neighbours.append(np.roll(grid, shift=shift, axis=axis))
        return neighbours


# =================================
# OBSERVABLES
# =================================
class Observables:
    def __init__(self):
        self.M = 0.0
        self.E = 0.0


# =================================
# METROPOLIS + ENERGY
# =================================
def energy_singleflip(ising, site):
    E = 0
    s = ising.spins[site]
    for nn_table in ising.neighbours_table:
        new_site = tuple(nn_table[site])
        if s == ising.spins[new_site]:
            E += 1
        else:
            E -= 1
    E -= ising.h * s
    return 2 * E


def metropolis_step(ising):
    site = random.choice(ising.sites)
    dE = energy_singleflip(ising, site)
    if dE < 0 or random.random() < np.exp(-ising.beta * dE):
        ising.spins[site] *= -1


def metropolis_sweep(ising):
    for _ in range(ising.length()):
        metropolis_step(ising)


def magnetization(ising):
    return abs(np.sum(ising.spins))


def energy(ising):
    E = 0
    for site in ising.sites:
        E += energy_singleflip(ising, site)
    return E / 2


def update_observables(ising, obs):
    obs.E += energy(ising)
    obs.M += magnetization(ising)


def run(ising, obs, N):
    for i in range(1, 2 * N + 1):
        metropolis_sweep(ising)
        wolff(ising)
        if i > N:
            update_observables(ising, obs)


# =================================
# PARALLEL MPI DRIVER
# =================================
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 50  # sweeps per process
    dims = (4, 4)
    beta = 1.0
    lattice_type = "square"

    # MPI process has its own Ising lattice
    ising = Ising(dims, beta=beta, lattice=lattice_type)
    obs = Observables()

    # simulation
    run(ising, obs, N)

    #results from all processes
    total_E = comm.reduce(obs.E, op=MPI.SUM, root=0)
    total_M = comm.reduce(obs.M, op=MPI.SUM, root=0)

    if rank == 0:
        # Average over all replicas and number of sites
        E_avg = total_E / (size * N * np.prod(dims))
        M_avg = total_M / (size * N * np.prod(dims))
        print(f"Average Energy per spin: {E_avg}")
        print(f"Average Magnetization per spin: {M_avg}")


if __name__ == "__main__":
    main()
