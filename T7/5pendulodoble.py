import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros del sistema
l1, l2 = 7, 10  # Longitudes de los péndulos
m1, m2 = 7, 14  # Masas de los péndulos
g = 9.81  # Gravedad
dt = 0.01  # Intervalo de tiempo
t10, t20 = np.pi / 4, np.pi / 3  # Ángulos iniciales
w1_, w2_ = 0, 0  # Velocidades angulares iniciales
fin = 20  # Tiempo total de simulación

# Función para calcular las aceleraciones angulares
def accelerations(theta1, theta2, omega1, omega2):
    delta = theta1 - theta2
    denom1 = l1 * (2 * m1 + m2 - m2 * np.cos(2 * delta))
    alpha1 = (
        -g * (2 * m1 + m2) * np.sin(theta1)
        - m2 * g * np.sin(theta1 - 2 * theta2)
        - 2 * m2 * np.sin(delta) * (omega2**2 * l2 + omega1**2 * l1 * np.cos(delta))
    ) / denom1
    denom2 = l2 * (2 * m1 + m2 - m2 * np.cos(2 * delta))
    alpha2 = (
        2
        * np.sin(delta)
        * (
            omega1**2 * l1 * (m1 + m2)
            + g * (m1 + m2) * np.cos(theta1)
            + omega2**2 * l2 * m2 * np.cos(delta)
        )
    ) / denom2
    return alpha1, alpha2

# Función para calcular las posiciones
def coord(theta1, theta2):
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    return x1, x2, y1, y2

# Simulación del movimiento
def simulate(theta1_0, theta2_0, omega1_0, omega2_0, fin, dt):
    steps = int(fin / dt)
    theta1, theta2 = [theta1_0], [theta2_0]
    omega1, omega2 = omega1_0, omega2_0
    x1, x2, y1, y2 = [], [], [], []
    theta1_dot, theta2_dot = [], []  # Guardar las velocidades angulares
    for _ in range(steps):
        x1_i, x2_i, y1_i, y2_i = coord(theta1[-1], theta2[-1])
        x1.append(x1_i)
        x2.append(x2_i)
        y1.append(y1_i)
        y2.append(y2_i)
        alpha1, alpha2 = accelerations(theta1[-1], theta2[-1], omega1, omega2)
        omega1 += alpha1 * dt
        omega2 += alpha2 * dt
        theta1.append(theta1[-1] + omega1 * dt)
        theta2.append(theta2[-1] + omega2 * dt)
        theta1_dot.append(omega1)
        theta2_dot.append(omega2)
    return x1, x2, y1, y2, theta1, theta2, theta1_dot, theta2_dot

# Ejecutar la simulación
x1, x2, y1, y2, theta1, theta2, theta1_dot, theta2_dot = simulate(t10, t20, w1_, w2_, fin, dt)

# Configuración de la animación y del gráfico
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Subgráfico 1: Animación del movimiento
ax1.set_xlim(-25, 25)
ax1.set_ylim(-25, 25)
ax1.set_aspect('equal')
ax1.set_facecolor('lightblue')
line, = ax1.plot([], [], 'o-', lw=2, color='darkred')

# Función de actualización para la animación
def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    return line,

# Subgráfico 2: Espacio de fase
ax2.set_xlabel(r'$\theta$ (rad)')
ax2.set_ylabel(r'$\omega$ (rad/s)')
ax2.plot(theta1[:-1], theta1_dot, label=r'$(\theta_1, \omega_1)$', color='darkred')
ax2.plot(theta2[:-1], theta2_dot, label=r'$(\theta_2, \omega_2)$', color='darkblue')
ax2.legend()
ax2.set_title('Espacio de fase del péndulo doble')

# Crear la animación
ani = FuncAnimation(fig, update, frames=len(x1), interval=dt * 1000, blit=True)

plt.tight_layout()
plt.show()
