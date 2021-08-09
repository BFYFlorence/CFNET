import numpy as np

def determine_hb(donor:np.ndarray,  # (3,)
                 donor_H:np.ndarray,  # (3,)
                 acceptor:np.ndarray  # (3,)
                 # units: Å
                 ):
    Rhb = np.sqrt(np.sum((donor-acceptor)*(donor-acceptor)))

    Rdh = np.sqrt(np.sum((donor-donor_H)*(donor-donor_H)))

    Rah = np.sqrt(np.sum((acceptor-donor_H)*(acceptor-donor_H)))

    cos_a = (Rdh+Rhb-Rah)/2*Rdh*Rhb
    theta = np.arccos(cos_a)  # radians, between [0, pi]
    if theta <= np.pi/6. and Rhb <=3.5:
        return True
    else:
        return False
