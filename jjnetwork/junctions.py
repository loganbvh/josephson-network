import numpy as np

from .em import Phi_0


def josephson_energy_power_law(
    junction_length: float, *, d0: float, I0: float
) -> float:
    E0 = Phi_0 * I0 / (2 * np.pi)
    EJ = E0 * (junction_length / d0) ** (-2)
    return EJ


def josephson_energy_exponential(
    junction_length: float, *, d0: float, I0: float
) -> float:
    E0 = Phi_0 * I0 / (2 * np.pi)
    EJ = E0 * np.exp(-junction_length / d0)
    return EJ
