import numpy as np

def index_grid(dims):
    """Return grid of Cartesian indices"""
    D = len(dims)
    return np.indices(dims).transpose(*range(1, D + 1), 0)


# =========================
# 1D CHAIN
# =========================
def lattice_1d(L):
    dims = (L,)
    grid = index_grid(dims)

    neighbours = []
    neighbours.append(np.roll(grid, shift=+1, axis=0))
    neighbours.append(np.roll(grid, shift=-1, axis=0))

    return dims, neighbours


# =========================
# SQUARE LATTICE 
# =========================
def lattice_square(Lx, Ly):
    dims = (Lx, Ly)
    grid = index_grid(dims)

    neighbours = []
    neighbours.append(np.roll(grid, +1, axis=0))
    neighbours.append(np.roll(grid, -1, axis=0))
    neighbours.append(np.roll(grid, +1, axis=1))
    neighbours.append(np.roll(grid, -1, axis=1))

    return dims, neighbours


# =========================
# TRIANGULAR LATTICE (2D)
# =========================
def lattice_triangular(Lx, Ly):
    dims = (Lx, Ly)
    grid = index_grid(dims)

    neighbours = []

    # square neighbors
    neighbours.append(np.roll(grid, +1, axis=0))
    neighbours.append(np.roll(grid, -1, axis=0))
    neighbours.append(np.roll(grid, +1, axis=1))
    neighbours.append(np.roll(grid, -1, axis=1))

    # diagonal neighbors
    neighbours.append(np.roll(np.roll(grid, +1, axis=0), +1, axis=1))
    neighbours.append(np.roll(np.roll(grid, -1, axis=0), -1, axis=1))

    return dims, neighbours


# =========================
# HONEYCOMB LATTICE (2D)-> Represented on brick-wall lattice
# =========================
def lattice_honeycomb(Lx, Ly):
    """
    Honeycomb lattice using brick-wall representation.
    Each site has 3 neighbors.
    """
    dims = (Lx, Ly)
    grid = index_grid(dims)

    neighbours = []

    # horizontal neighbors
    neighbours.append(np.roll(grid, +1, axis=0))
    neighbours.append(np.roll(grid, -1, axis=0))

    # vertical neighbors depend on sublattice parity
    up = np.roll(grid, +1, axis=1)
    down = np.roll(grid, -1, axis=1)

    parity = np.indices(dims).sum(axis=0) % 2

    nn3 = grid.copy()
    nn3[parity == 0] = up[parity == 0]
    nn3[parity == 1] = down[parity == 1]

    neighbours.append(nn3)

    return dims, neighbours
