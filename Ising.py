import numpy as np
import random
from wolff import wolff


class Ising:
    def __init__(self, dims, h=0.0, beta=5.0):
        self.dims = tuple(dims)
        self.h = float(h)
        self.beta = float(beta)

        # spins = rand(-1:2:1, dims)
        self.spins = np.random.choice([-1, 1], size=self.dims)

        # cluster = fill(false, dims)
        self.cluster = np.zeros(self.dims, dtype=bool)

        # CartesianIndices
        self.sites = list(np.ndindex(self.dims))

        # neighbours_table (periodic boundary conditions)
        self.neighbours_table = []
        D = len(self.dims)

        index_grid = np.indices(self.dims).transpose(*range(1, D + 1), 0)

        for axis in range(D):
            for shift in (-1, 1):
                self.neighbours_table.append(
                    np.roll(index_grid, shift=shift, axis=axis)
                )

    def size(self):
        return self.dims

    def length(self):
        return np.prod(self.dims)


class Observables:
    def __init__(self):
        self.M = 0.0
        self.E = 0.0
        self.chi = 0.0
        self.Cv = 0.0


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

# ----------------------------
if __name__ == "__main__":
    N = 50
    dims = (4, 4)

    ising = Ising(dims, h=0.0, beta=1.0)
    observables = Observables()

    run(ising, observables, 1)
    run(ising, observables, N)

    print("E =", observables.E / N / np.prod(dims))
    print("M =", observables.M / N / np.prod(dims))
