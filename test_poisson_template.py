import pytest
import numpy as np

from poisson1 import solve_poisson, errornorm

def test_exact_solution():
    """Test that P2 element recover the quadretic solution up to rounding error.
    """
    uh, ue = solve_poisson(n=4, degree=2)
    error_H1 = errornorm(uh, ue, "H1")
    assert error_H1 <1e-12
    

def test_convergence_P1():
    """Test that
    """
    pass
    
