import numpy as np
from scipy import integrate

def p(xi):
    return np.where((xi > -1) & (xi < 1), np.cos(xi), 0)

def integrand(alpha, xi, x, y, a, G):
    return (np.cos(xi - x) / (G * alpha)) * ((1 - np.exp(-2 * alpha * y)) / (1 + np.exp(-2 * alpha * a))) * np.exp(-alpha * (a - y)) * p(xi)

G = 79.3  #Steel
a = 10  

x = np.linspace(-a, a, 100)
y = np.linspace(0, a, 100)

integral_values = np.zeros((len(y), len(x)))

for i in range(len(x)):
    for j in range(len(y)):
        inner_integral = lambda alpha: integrand(alpha, x[i], x[i], y[j], a, G)
        xi_integral, _ = integrate.quad(inner_integral, 0, np.inf)
        integral_values[j, i] = xi_integral / np.pi

# Plotting the results
import matplotlib.pyplot as plt
X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, integral_values, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('w(x,y)')
plt.show()
