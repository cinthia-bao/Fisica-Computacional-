#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animación 2D de la ecuación de calor en placa cuadrada con Crank-Nicolson
Incluye gráficas de perfiles X e Y
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

# Configuración del backend para la animación
plt.switch_backend('TkAgg')  # Alternativas: Qt5Agg, MacOSX

# =============================================
# 1. PARÁMETROS Y DISCRETIZACIÓN
# =============================================

# Parámetros físicos
L = 1.0                # Longitud de la placa (m)
N = 50                 # Número de puntos por dimensión
Tmax = 10             # Tiempo máximo de simulación (s)
alpha = 0.01           # Difusividad térmica (m²/s)
source_temp = 100.0    # Temperatura de la fuente (°C)

# Discretización
dx = dy = L / (N - 1)
dt = 0.001             # Paso de tiempo
Nt = int(Tmax / dt)

# Coeficiente de Crank-Nicolson
r = alpha * dt / (2 * dx**2)

# =============================================
# 2. INICIALIZACIÓN Y MATRICES
# =============================================

# Condición inicial - fuente en el centro
u = np.zeros((N, N))
u[N//4:N//4*3, N//4:N//4*3] = source_temp  # Fuente cuadrada en el centro

# Matrices para el método ADI (Alternating Direction Implicit)
def build_matrices(size, r):
    """Construye matrices para Crank-Nicolson"""
    main_diag = (1 + 2*r) * np.ones(size)
    off_diag = -r * np.ones(size-1)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
    B = diags([r*np.ones(size-1), (1-2*r)*np.ones(size), r*np.ones(size-1)], [-1, 0, 1], format='csr')
    return A, B

A_x, B_x = build_matrices(N, r)
A_y, B_y = build_matrices(N, r)

def update_solution(u):
    """Paso temporal usando ADI"""
    u_half = np.zeros_like(u)
    
    # Primera mitad: implícito en X
    for j in range(1, N-1):
        b = B_x @ u[:,j] + r * (u[:,j+1] + u[:,j-1] - 2*u[:,j])
        u_half[1:-1,j] = spsolve(A_x[1:-1,1:-1], b[1:-1])
    
    # Segunda mitad: implícito en Y
    u_new = np.zeros_like(u)
    for i in range(1, N-1):
        b = B_y @ u_half[i,:] + r * (u_half[i+1,:] + u_half[i-1,:] - 2*u_half[i,:])
        u_new[i,1:-1] = spsolve(A_y[1:-1,1:-1], b[1:-1])
    
    # Condiciones de frontera (temperatura fija 0 en bordes)
    u_new[0,:] = u_new[-1,:] = u_new[:,0] = u_new[:,-1] = 0
    
    return u_new

# =============================================
# 3. CONFIGURACIÓN DE LA ANIMACIÓN
# =============================================

# Crear figura con diseño personalizado
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(3, 2, width_ratios=[1.5, 1], height_ratios=[1, 1, 0.2])
ax1 = fig.add_subplot(gs[0:2, 0])  # Mapa de calor
ax2 = fig.add_subplot(gs[0, 1])    # Perfil en X
ax3 = fig.add_subplot(gs[1, 1])    # Perfil en Y
ax4 = fig.add_subplot(gs[2, :])    # Barra de información

# Configuración de ejes
x = y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# Mapa de calor
img = ax1.imshow(u.T, cmap='inferno', extent=[0, L, 0, L], 
                origin='lower', vmin=0, vmax=source_temp)
plt.colorbar(img, ax=ax1, label='Temperatura (°C)')
ax1.set_title("Distribución de Temperatura 2D")
ax1.set_xlabel("Posición X (m)")
ax1.set_ylabel("Posición Y (m)")

# Perfil en X (Y = L/2)
line_x, = ax2.plot(x, u[:, N//2], 'r-', lw=2)
ax2.set_title(f"Perfil en X (Y = {L/2:.2f} m)")
ax2.set_ylim(0, source_temp*1.1)
ax2.grid(True, alpha=0.3)

# Perfil en Y (X = L/2)
line_y, = ax3.plot(y, u[N//2, :], 'b-', lw=2)
ax3.set_title(f"Perfil en Y (X = {L/2:.2f} m)")
ax3.set_ylim(0, source_temp*1.1)
ax3.grid(True, alpha=0.3)

# Información temporal
time_text = ax4.text(0.5, 0.5, "", transform=ax4.transAxes, 
                    ha='center', va='center', fontsize=12)
ax4.axis('off')

# =============================================
# 4. FUNCIÓN DE ANIMACIÓN
# =============================================

def init():
    """Inicialización de la animación"""
    img.set_array(u.T)
    line_x.set_ydata(u[:, N//2])
    line_y.set_ydata(u[N//2, :])
    time_text.set_text(f"Tiempo = {0:.3f} s")
    return img, line_x, line_y, time_text

def animate(t):
    """Actualización de la animación"""
    global u
    u = update_solution(u)
    
    # Actualizar visualizaciones
    img.set_array(u.T)
    line_x.set_ydata(u[:, N//2])
    line_y.set_ydata(u[N//2, :])
    
    # Actualizar información
    max_temp = np.max(u)
    time_text.set_text(
        f"Tiempo = {t*dt:.3f} s / {Tmax:.1f} s | "
        f"Temp máxima = {max_temp:.1f} °C | "
        f"α = {alpha} m²/s | Δx = Δy = {dx:.3f} m | Δt = {dt:.4f} s"
    )
    
    return img, line_x, line_y, time_text

# Crear animación
ani = FuncAnimation(
    fig, 
    animate, 
    frames=Nt,
    init_func=init,
    interval=50, 
    blit=True,
    repeat=False
)

plt.tight_layout()
plt.show()

# Para guardar la animación (opcional):
# ani.save('difusion_calor_2d.mp4', writer='ffmpeg', fps=15, dpi=300)