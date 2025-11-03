import numpy as np


def apply_mask(arrays, mask):
    return (array[mask] for array in arrays)

def calculate_interaction_features(
        particle_array,
        pad_value
):
    """
    Creates an interaction matrix of 
    (Events, Particles, Particles, N_features)

    Assumes each event is of (Pt, eta, phi, M)
    """
    n_events, n_particles, *_ = particle_array.shape

    interaction_matrix = np.full((n_events, n_particles, n_particles, 3), pad_value)
    for i in particle_array:
        for j in i:
            for k in i:
                
                p1, eta1, phi1, m1 = j[:4]
                p2, eta2, phi2, m2 = k[:4]

                
