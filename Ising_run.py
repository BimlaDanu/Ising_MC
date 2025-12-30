import numpy as np
import random
from wolff import wolff
from Lattice import (
    lattice_1d,
    lattice_square,
    lattice_triangular,
    lattice_honeycomb,
)


# =========================
# ISING MODEL
# =========================
class Ising:
    def __init__(
        self,
        dims,
        h=0.0,
        beta=5.0,
        lattice="generic",
    ):
        """
        lattice options:
        - "generic"    : original D-dimensional hypercubic lattice
        - "1d"         : 1D chain
        - "square"     : 2D square lattice
        - "triangular" : 2D triangular lattice
        - "honeycomb"  : 2D honeycomb lattice
        """

        self.h = float(h)
        self.beta = float(beta)

        # -------------------------
        # LATTICE SETUP
        # -------------------------
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

        # -------------------------
        # FIELDS
        # -------------------------
        self.spins = np.random.choice([-1, 1], size=self.dims)
        self.cluster = np.zeros(self.dims, dtype=bool)
        self.sites = list(np.ndindex(self.dims))

    # -------------------------
    # BASIC PROPERTIES
    # -------------------------
    def size(self):
        return self.dims

    def length(self):
        return np.prod(self.dims)

    # -------------------------
    # GENERIC D-DIM LATTICE
    # -------------------------
    def _generic_lattice(self):
        D = len(self.dims)
        grid = np.indices(self.dims).transpose(*range(1, D + 1), 0)

        neighbours = []
        for axis in range(D):
            for shift in (-1, 1):
                neighbours.append(
                    np.roll(grid, shift=shift, axis=axis)
                )
        return neighbours


# =========================
# OBSERVABLES
# =========================
class Observables:
    def __init__(self):
        self.M = 0.0
        self.E = 0.0
        self.chi = 0.0
        self.Cv = 0.0


# =========================
# METROPOLIS
# =========================
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


# =========================
# MEASUREMENTS
# =========================
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


# =========================
# MONTE CARLO 
# =========================
def run(ising, obs, N):
    for i in range(1, 2 * N + 1):
        metropolis_sweep(ising)
        wolff(ising)

        if i > N:
            update_observables(ising, obs)



# =========================
if __name__ == "__main__":
    N = 50
    # ising = Ising((100,), beta=1.0, lattice="1d")
    # ising = Ising((40, 40), beta=0.44, lattice="square")
    # ising = Ising((40, 40), beta=0.27, lattice="triangular")
    ising = Ising((4, 4), beta=1., lattice="honeycomb")

    obs = Observables()

    run(ising, obs, 1)
    run(ising, obs, N)

    print("E =", obs.E / N / np.prod(ising.dims))
    print("M =", obs.M / N / np.prod(ising.dims))
