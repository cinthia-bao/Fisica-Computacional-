import numpy as np
import matplotlib.pyplot as plt

# Parámetros del Van der Pol
mu = 2.0  # Parámetro de no linealidad (prueba con 0.1, 1.0, 5.0)

# Definición del sistema
def van_der_pol(state):
    x, y = state
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return np.array([dxdt, dydt])

# Método RK4
def rk4_step(state, dt):
    k1 = van_der_pol(state)
    k2 = van_der_pol(state + 0.5 * dt * k1)
    k3 = van_der_pol(state + 0.5 * dt * k2)
    k4 = van_der_pol(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Método Euler-Cromer adaptado
def euler_cromer_step(state, dt):
    x, y = state
    # Primero actualizamos la velocidad (y)
    new_y = y + dt * (mu * (1 - x**2) * y - x)
    # Luego la posición (x) con la nueva velocidad
    new_x = x + dt * new_y
    return np.array([new_x, new_y])

# Simulación
dt = 0.01
fin = 20
steps = int(fin/dt)
state_rk4 = np.array([2, 0])  # Condición inicial
state_ec = state_rk4.copy()

# Almacenamiento
traj_rk4 = np.zeros((steps, 2))
traj_ec = np.zeros((steps, 2))

for i in range(steps):
    traj_rk4[i] = state_rk4
    traj_ec[i] = state_ec
    state_rk4 = rk4_step(state_rk4, dt)
    state_ec = euler_cromer_step(state_ec, dt)

# Gráfico del espacio de fases
plt.figure(figsize=(15, 6))

# RK4
plt.subplot(121)
x_rk4, y_rk4 = traj_rk4[1000:].T  # Ignoramos transitorio inicial
plt.plot(x_rk4, y_rk4, 'b', lw=1, label=f'RK4 (dt={dt})')
plt.title(f'Espacio de Fases - Van der Pol (μ={mu}) - RK4')
plt.xlabel('Posición (x)')
plt.ylabel('Velocidad (y)')
plt.grid(True)

# Euler-Cromer
plt.subplot(122)
x_ec, y_ec = traj_ec[1000:].T
plt.plot(x_ec, y_ec, 'r', lw=1, label=f'Euler-Cromer (dt={dt})')
plt.title(f'Espacio de Fases - Van der Pol (μ={mu}) - Euler-Cromer')
plt.xlabel('Posición (x)')
plt.ylabel('Velocidad (y)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Diagrama de tiempo para comparar métodos
plt.figure(figsize=(12, 5))
plt.plot(np.arange(steps)*dt, traj_rk4[:, 0], 'b-', lw=1, label='RK4')
plt.plot(np.arange(steps)*dt, traj_ec[:, 0], 'r--', lw=1, label='Euler-Cromer')
plt.title('Comparación temporal (Posición x)')
plt.xlabel('Tiempo')
plt.ylabel('x(t)')
plt.legend()
plt.grid(True)
plt.show()
