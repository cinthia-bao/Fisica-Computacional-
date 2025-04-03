import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema de Lorenz
sigma, rho, beta = 10.0, 28.0, 8.0/3.0

# Función del sistema de Lorenz
def lorenz(state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# Método RK4
def rk4_step(state, dt):
    k1 = lorenz(state)
    k2 = lorenz(state + 0.5 * dt * k1)
    k3 = lorenz(state + 0.5 * dt * k2)
    k4 = lorenz(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Simulación
dt = 0.01
steps = 10000
state = np.array([1.0, 1.0, 1.0])  # Condición inicial

# Almacenar resultados
trajectory = np.zeros((steps, 3))
for i in range(steps):
    trajectory[i] = state
    state = rk4_step(state, dt)

# Gráfico 3D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
x, y, z = trajectory.T
ax.plot(x, y, z, lw=0.5, color='blue')
ax.set_title("Atractor de Lorenz - RK4", fontsize=14)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
