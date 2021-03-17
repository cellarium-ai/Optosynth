import numpy as np
from specs import *


class VoltageToFluorescenceConverter:
    """Logistic conversion."""
    def __init__(self, specs: VoltageToFluorescenceSpecs):
        a1 = np.exp(- specs.beta * specs.v1)
        a2 = np.exp(- specs.beta * specs.v2)
        z = (specs.f2 - specs.f1) / (specs.f1 * a1 - specs.f2 * a2)
        assert z > 0, "Bad input parameters!"
        self.v0 = np.log(z) / specs.beta
        self.gamma = z * (a1 - a2) / (1. / specs.f1 - 1. / specs.f2)
        self.beta = specs.beta
    
    def __call__(self, voltage: np.ndarray) -> np.ndarray:
        return self.gamma / (1. + np.exp(-self.beta * (voltage - self.v0)))
    
    def __repr__(self):
        return (f'Voltage to Fluorescence Converter Parameters:\n'
                f'v0: {self.v0:.3f} (mV),  gamma: {self.gamma:.3f},  beta: {self.beta:.5f} (1/mV)')
