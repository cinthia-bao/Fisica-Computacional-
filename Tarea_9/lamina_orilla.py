#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solución 2D de la ecuación de calor con Crank-Nicolson
Fuente de calor en el borde izquierdo (x=0) - Versión corregida
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.sparse import diags, eye, lil_matrix
from scipy.sparse.linalg import spsolve

# Configuración de estilo
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({
    'font.size': 12,
    'figure.facecolor': 'white',
    'axes.grid': False,  # Cambiado a False para evitar warning
    'grid.alpha': 0.3,
    'axes.titlesize': 16,
    'axes.labelsize': 14
})

# =============================================
# 1. PARÁMETROS Y DISCRETIZACIÓN
# =============================================

# Parámetros físicos
L = 5.0               # Longitud de cada lado (m)
N = 50                # Número de nodos por dimensión
Tmax = 10.0           # Tiempo máximo (s)
alpha = 0.1           # Difusividad térmica (m²/s)
source_temp = 100.0   # Temperatura de la fuente (°C)

# Discretización
dx = dy = L / (N - 1)
dt = 0.05
Nt = int(Tmax / dt)

# Coeficientes Crank-Nicolson
r = alpha * dt / (2 * dx**2)

# =============================================
# 2. IMPLEMENTACIÓN NUMÉRICA CORREGIDA
# =============================================

# Inicialización - Fuente en el borde izquierdo (x=0)
u = np.zeros((N, N))
u[0, :] = source_temp  # Toda la columna x=0 a temperatura fuente

# Matrices para ADI - Versión corregida
def build_matrices(size, r):
    # Usamos lil_matrix para facilitar el slicing
    A = lil_matrix((size, size))
    B = lil_matrix((size, size))
    
    for i in range(size):
        if i > 0:
            A[i, i-1] = -r
            B[i, i-1] = r
        A[i, i] = 1 + 2*r
        B[i, i] = 1 - 2*r
        if i < size-1:
            A[i, i+1] = -r
            B[i, i+1] = r
    
    # Convertir a formato CSR para mejor eficiencia en solve
    return A.tocsr(), B.tocsr()

A_x, B_x = build_matrices(N, r)
A_y, B_y = build_matrices(N, r)

def update_solution(u):
    # Primera mitad (implícito en x)
    u_half = u.copy()
    
    for j in range(1, N-1):
        b = B_x @ u[:,j] + r * (u[:,j+1] + u[:,j-1] - 2*u[:,j])
        # Resolver solo para puntos internos (excluyendo bordes)
        u_half[1:-1,j] = spsolve(A_x[1:-1,1:-1], b[1:-1])
    
    # Segunda mitad (implícito en y)
    u_new = u_half.copy()
    
    for i in range(1, N-1):
        b = B_y @ u_half[i,:] + r * (u_half[i+1,:] + u_half[i-1,:] - 2*u_half[i,:])
        u_new[i,1:-1] = spsolve(A_y[1:-1,1:-1], b[1:-1])
    
    # Mantener condiciones de frontera:
    u_new[0, :] = source_temp  # Fuente de calor permanente
    u_new[-1,:] = 0            # Borde derecho
    u_new[:, 0] = 0            # Borde inferior
    u_new[:,-1] = 0            # Borde superior
    
    return u_new

# =============================================
# 3. ANIMACIÓN MEJORADA CON PERFILES
# =============================================

fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0])  # Mapa de calor
ax2 = fig.add_subplot(gs[0, 1])  # Perfil en x
ax3 = fig.add_subplot(gs[1, 0])  # Perfil en y
ax4 = fig.add_subplot(gs[1, 1])  # Información

# Configuración de los ejes
x = y = np.linspace(0, L, N)

# Mapa de calor
img = ax1.imshow(u.T, cmap='inferno', extent=[0, L, 0, L], origin='lower', vmin=0, vmax=source_temp)
cbar = plt.colorbar(img, ax=ax1)
cbar.set_label('Temperatura (°C)')
ax1.set_title("Mapa de Calor 2D - Fuente en borde izquierdo")
ax1.set_xlabel("Posición X (m)")
ax1.set_ylabel("Posición Y (m)")

# Perfil en X (y = L/2)
line_x, = ax2.plot(x, u[:, N//2], 'r-', lw=2, label=f'Perfil en Y = {L/2:.1f} m')
ax2.set_title("Perfil de Temperatura en dirección X")
ax2.set_xlabel("Posición X (m)")
ax2.set_ylabel("Temperatura (°C)")
ax2.set_ylim(0, source_temp*1.1)
ax2.grid(True)

# Perfil en Y (x = L/4, L/2, 3L/4)
line_y1, = ax3.plot(y, u[N//4, :], 'b-', lw=2, label=f'X = {L/4:.1f} m')
line_y2, = ax3.plot(y, u[N//2, :], 'g-', lw=2, label=f'X = {L/2:.1f} m')
line_y3, = ax3.plot(y, u[3*N//4, :], 'm-', lw=2, label=f'X = {3*L/4:.1f} m')
ax3.set_title("Perfiles de Temperatura en dirección Y")
ax3.set_xlabel("Posición Y (m)")
ax3.set_ylabel("Temperatura (°C)")
ax3.set_ylim(0, source_temp*1.1)
ax3.legend()
ax3.grid(True)

# Información
ax4.axis('off')
info_text = ax4.text(0.5, 0.5, "", transform=ax4.transAxes, ha='center', va='center')
ax4.set_title("Información de la Simulación")

def animate(t):
    global u
    u = update_solution(u)
    
    # Actualizar mapa de calor
    img.set_array(u.T)
    img.set_clim(vmin=0, vmax=np.max(u))
    
    # Actualizar perfiles
    line_x.set_ydata(u[:, N//2])
    line_y1.set_ydata(u[N//4, :])
    line_y2.set_ydata(u[N//2, :])
    line_y3.set_ydata(u[3*N//4, :])
    
    # Actualizar información
    info = f"Tiempo: {t*dt:.2f} s / {Tmax:.1f} s\n"
    info += f"Temperatura máxima: {np.max(u):.1f} °C\n"
    info += f"Parámetros:\nα = {alpha} m²/s\nΔx = Δy = {dx:.3f} m\nΔt = {dt:.3f} s"
    info_text.set_text(info)
    
    return img, line_x, line_y1, line_y2, line_y3, info_text

ani = FuncAnimation(fig, animate, frames=Nt, interval=50, blit=True)
plt.tight_layout()
plt.show()

# =============================================
# 4. GRÁFICOS ESTÁTICOS DE PERFILES (CORREGIDO)
# =============================================

# Simular para tiempos específicos
snapshot_times = [0, 0.5, 2, 5, 10]
snapshots = []
u_static = np.zeros((N, N))
u_static[0, :] = source_temp  # Condición inicial con fuente en el borde

for t in range(int(Tmax/dt)):
    u_static = update_solution(u_static)
    if t*dt in snapshot_times:
        snapshots.append((t*dt, u_static.copy()))

# Crear figura estática
fig_static, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig_static.suptitle("Evolución de los Perfiles de Temperatura - Fuente en borde izquierdo", y=1.02)

# Perfil en X (y = L/2)
colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))
for (time, temp), color in zip(snapshots, colors):
    ax1.plot(x, temp[:, N//2], color=color, lw=2, label=f't = {time:.1f} s')

ax1.set_title(f"Perfil en X (Y = {L/2:.1f} m)")
ax1.set_xlabel("Posición X (m)")
ax1.set_ylabel("Temperatura (°C)")
ax1.legend(title="Tiempo", frameon=True)
ax1.grid(True)

# Perfiles en Y para diferentes X
for (time, temp), color in zip(snapshots, colors):
    ax2.plot(y, temp[N//2, :], color=color, lw=2, linestyle='-', label=f'X={L/2:.1f}m, t={time:.1f}s')
    ax2.plot(y, temp[3*N//4, :], color=color, lw=2, linestyle='--', label=f'X={3*L/4:.1f}m, t={time:.1f}s')

ax2.set_title("Perfiles en Y para diferentes posiciones X")
ax2.set_xlabel("Posición Y (m)")
ax2.set_ylabel("Temperatura (°C)")
ax2.legend(title="Posición y Tiempo", frameon=True, ncol=2)
ax2.grid(True)

# Diagrama esquemático
ax_sketch = fig_static.add_axes([0.45, 0.6, 0.1, 0.3])
ax_sketch.set_xlim(0, 1)
ax_sketch.set_ylim(0, 1)
ax_sketch.axis('off')
ax_sketch.plot([0], [0.5], 'ro', markersize=10)  # Fuente en el borde
ax_sketch.plot([0, 1], [0.5, 0.5], 'k-', lw=2)  # Línea horizontal
ax_sketch.text(0.1, 0.7, "Fuente de calor", ha='left', fontsize=8)
ax_sketch.text(0.5, 0.3, "Línea Y", ha='center', fontsize=8)
ax_sketch.set_title("Configuración", fontsize=10)

plt.tight_layout()
plt.savefig('perfiles_temperatura_borde.png', dpi=300, bbox_inches='tight')
plt.show()