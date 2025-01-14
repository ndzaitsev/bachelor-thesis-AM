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

x = np.linspace(-10, 10, 100)
xi = np.linspace(-10, 10, 100)

G = 79.3  #Steel
a = 10  
y1 = 0
y2 = a/2
y3 = a

# Compute the function values
func_values1 = []
for x_val in x:
    integral_xi = quad(lambda xi_val: p(xi_val) * integral_alpha(xi_val, x_val, y1, a), -1, 1)[0]
    func_values1.append(integral_xi / np.pi)
    
func_values2 = []
for x_val in x:
    integral_xi = quad(lambda xi_val: p(xi_val) * integral_alpha(xi_val, x_val, y2, a), -1, 1)[0]
    func_values2.append(integral_xi / np.pi)
    
func_values3 = []
for x_val in x:
    integral_xi = quad(lambda xi_val: p(xi_val) * integral_alpha(xi_val, x_val, y3, a), -1, 1)[0]
    func_values3.append(integral_xi / np.pi)



# Plot the function
plt.plot(x, func_values1, label='y=0')
plt.plot(x, func_values2, label='y=a/2')
plt.plot(x, func_values3, label='y=a')
plt.ylabel(r'$\tau_{yz}$')
plt.legend()
plt.xlabel('x')
plt.grid(True)
plt.show()