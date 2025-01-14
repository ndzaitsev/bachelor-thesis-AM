import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def p(xi):
    return np.where((xi > -1) & (xi < 1), np.cos(xi), 0)

def integral_alpha(xi, x, y, a):
    # Define the integrand inside the second integral
    alpha = np.linspace(1e-10, 10, 100)
    integrand = (np.cos(alpha*(xi - x)) / (alpha)) * ((alpha*np.exp(-alpha * (a - y)) - alpha*np.exp(-alpha * (a + y))) / (1 + np.exp(-2 * alpha * a)))
    return np.trapz(integrand, alpha)

y = np.linspace(0, 10, 100)
xi = np.linspace(-10, 10, 100)

G = 79.3  #Steel
a = 10  
x1 = 0
x2 = a/2
x3 = a

# Compute the function values
func_values1 = []
for y_val in y:
    integral_xi = quad(lambda xi_val: p(xi_val) * integral_alpha(xi_val, x1, y_val, a), -1, 1)[0]
    func_values1.append(integral_xi / np.pi)
    
func_values2 = []
for y_val in y:
    integral_xi = quad(lambda xi_val: p(xi_val) * integral_alpha(xi_val, x2, y_val, a), -1, 1)[0]
    func_values2.append(integral_xi / np.pi)
    
func_values3 = []
for y_val in y:
    integral_xi = quad(lambda xi_val: p(xi_val) * integral_alpha(xi_val, x3, y_val, a), -1, 1)[0]
    func_values3.append(integral_xi / np.pi)

# Plot the function
plt.plot(y, func_values1, label='x=0')
plt.plot(y, func_values2, label='x=a/2')
plt.plot(y, func_values3, label='x=a')
plt.xlabel('y')
plt.ylabel(r'$\tau_{yz}$')
plt.legend()
plt.grid(True)
plt.show()