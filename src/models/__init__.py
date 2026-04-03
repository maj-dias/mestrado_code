"""Modelos epidemiológicos"""
from .sir import SIRModel, SIRControlledModel
from .sidarthe import SIDARTHEModel, SIDARTHEControlledModel

__all__ = ['SIRModel', 'SIRControlledModel',
           'SIDARTHEModel', 'SIDARTHEControlledModel']
