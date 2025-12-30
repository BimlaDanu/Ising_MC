import numpy as np
import random


def create_cluster(ising, site, spin_val):
    # ising.cluster[site] = true
    ising.cluster[site] = True

    # ising.spins[site] = -spin_val
    ising.spins[site] = -spin_val

    for nn_table in ising.neighbours_table:
        new_site = tuple(nn_table[site])

        if (
            ising.spins[new_site] == spin_val
            and not ising.cluster[new_site]
        ):
            if random.random() < 1.0 - np.exp(-2.0 * ising.beta):
                create_cluster(ising, new_site, spin_val)


def wolff(ising):
    # fill!(ising.cluster, false)
    ising.cluster.fill(False)

    # rand_site = rand(ising.sites)
    rand_site = random.choice(ising.sites)

    spin_val = ising.spins[rand_site]

    create_cluster(ising, rand_site, spin_val)
