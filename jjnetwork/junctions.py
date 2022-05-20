import numpy as np

from .em import ureg


def josephson_energy_power_law(
    junction_length: float, *, d0: float, I0: float
) -> float:
    I0 = I0 * ureg("A")
    d0 = d0 * ureg("m")
    E0 = ureg("Phi_0") * I0 / (2 * np.pi)
    d = junction_length * ureg("m")
    EJ = E0 * (d0 / d) ** 2
    return EJ.to("joules").magnitude


def josephson_energy_exponential(
    junction_length: float, *, d0: float, I0: float
) -> float:
    I0 = I0 * ureg("A")
    d0 = d0 * ureg("m")
    E0 = ureg("Phi_0") * I0 / (2 * np.pi)
    d = junction_length * ureg("m")
    EJ = E0 * np.exp(-d / d0)
    return EJ.to("joules").magnitude
