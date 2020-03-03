import numpy as np

from integrators.common import getsteps

def lagrange_dalembert(init, tspan, h, a):
    """
    Integrate the damped oscillator with damping factor a
    using the first order discrete Lagrange-d'Alembert 
    integrator.
    """
    steps = getsteps(tspan, h)
    t0, _ = tspan
    
    sol = np.empty([steps, 2], dtype=np.float64)
    sol[0] = np.array(init)
    for i in range(steps-1):
        p, x = sol[i]
        pnew = (p - h*x)/(1+a)
        xnew = x + h*pnew
        sol[i+1] = np.array((pnew, xnew))
    return sol
