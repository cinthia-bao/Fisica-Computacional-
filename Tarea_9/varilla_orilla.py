# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 12:11:24 2025

@author: ale
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# Configuración de estilo profesional
plt.style.use('seaborn-darkgrid')
plt.rcParams.update({
    'font.size': 12,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.titlesize': 16,
    'axes.labelsize': 14
})

# Parámetros físicos
L = 10.0         # Longitud de la varilla (m)
Nx = 100         # Aumentamos la resolución espacial
Tmax = 5.0       # Tiempo máximo (s)
alpha = 0.1      # Difusividad térmica (m²/s)

# Discretización
dx = L / (Nx - 1)
dt = 0.05        # Paso de tiempo
Nt = int(Tmax / dt)

# Coeficiente de Crank-Nicolson
r = alpha * dt / (2 * dx**2)

# Inicialización de la malla de temperatura
u = np.zeros(Nx)
u[0] = 100  # Fuente de calor en el extremo izquierdo

# Configuración de matrices tridiagonales
main_diag = np.ones(Nx) * (1 + 2 * r)
off_diag = np.ones(Nx - 1) * (-r)
A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
B_main = np.ones(Nx) * (1 - 2 * r)
B = np.diag(B_main) + np.diag(off_diag * -1, k=1) + np.diag(off_diag * -1, k=-1)

# Condiciones de frontera
A[0, 0] = A[-1, -1] = 1
A[0, 1] = A[-1, -2] = -1
B[0, 0] = B[-1, -1] = 1
B[0, 1] = B[-1, -2] = -1

# =============================================
# 1. ANIMACIÓN MEJORADA
# =============================================

# Crear figura con dos subplots
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])

# Configuración del plot principal
x = np.linspace(0, L, Nx)
line, = ax1.plot(x, u, lw=3, color='darkred')
ax1.set_ylim(0, 110)
ax1.set_xlim(0, L)
ax1.set_xlabel("Posición a lo largo de la varilla (m)")
ax1.set_ylabel("Temperatura (°C)")
ax1.set_title("Evolución Temporal de la Temperatura")

# Configuración del plot inferior (gradiente)
cmap = LinearSegmentedColormap.from_list('heatmap', ['blue', 'yellow', 'red'])
gradient = np.vstack((u, u))
img = ax2.imshow(gradient, cmap=cmap, aspect='auto', extent=[0, L, 0, 1], vmin=0, vmax=100)
ax2.set_xlabel("Posición (m)")
ax2.set_yticks([])
ax2.set_title("Mapa de Calor")

# Configuración del plot lateral (leyenda de color)
cbar = fig.colorbar(img, cax=ax3)
cbar.set_label('Temperatura (°C)')
ax3.set_title("Propagación\ndel Calor", pad=20)
ax3.axis('off')

# Texto informativo
info_text = ax3.text(0.5, 0.1, 
                    f"α = {alpha} m²/s\nΔx = {dx:.2f} m\nΔt = {dt:.2f} s",
                    ha='center', va='center', 
                    bbox=dict(facecolor='white', alpha=0.8))

# Función de animación
def animate(t):
    global u
    b = B @ u
    u = np.linalg.solve(A, b)
    
    # Actualizar línea principal
    line.set_ydata(u)
    
    # Actualizar mapa de calor
    gradient = np.vstack((u, u))
    img.set_array(gradient)
    
    # Actualizar título con tiempo
    ax1.set_title(f"Evolución Temporal de la Temperatura (t = {t*dt:.2f} s)")
    
    # Actualizar información
    info_text.set_text(f"α = {alpha} m²/s\nΔx = {dx:.2f} m\nΔt = {dt:.2f} s\nt = {t*dt:.2f} s")
    
    return line, img, info_text

# Crear animación
ani = FuncAnimation(fig, animate, frames=Nt, interval=50, blit=True)

plt.tight_layout()
plt.show()

# =============================================
# 2. IMAGEN ESTÁTICA FINAL
# =============================================

# Simular para tiempos específicos
u_static = np.zeros(Nx)
u_static[0] = 100  # Misma condición inicial
snapshot_times = [0, 0.5, 1, 2, 3, 5]  # Tiempos para capturas
snapshots = []

for t in range(int(Tmax/dt)):
    b = B @ u_static
    u_static = np.linalg.solve(A, b)
    if t*dt in snapshot_times:
        snapshots.append((t*dt, u_static.copy()))

# Crear figura estática
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))

for (time, temp), color in zip(snapshots, colors):
    plt.plot(x, temp, lw=2.5, color=color, label=f't = {time:.1f} s')

plt.title("Difusión de Calor en una Varilla\n(Solución por Crank-Nicolson)", pad=20)
plt.xlabel("Posición a lo largo de la varilla (m)")
plt.ylabel("Temperatura (°C)")
plt.ylim(0, 110)
plt.grid(True, alpha=0.3)

# Leyenda mejorada
legend = plt.legend(title="Tiempo (s)", frameon=True, shadow=True, 
                   borderpad=1, loc='upper right')
legend.get_frame().set_facecolor('#f5f5f5')

# Añadir diagrama esquemático
inset_ax = plt.axes([0.6, 0.5, 0.3, 0.3])
inset_ax.set_xlim(0, L)
inset_ax.set_ylim(0, 1)
inset_ax.set_xticks([])
inset_ax.set_yticks([])

# Dibujar varilla y fuente de calor
inset_ax.plot([0, L], [0.5, 0.5], 'k-', lw=3)
inset_ax.plot(0, 0.5, 'ro', markersize=10)  # Fuente de calor en extremo izquierdo
inset_ax.text(0, 0.3, 'Fuente\nde calor', ha='center', va='top')
inset_ax.set_title("Configuración Inicial", pad=10)

# Añadir información técnica
plt.text(0.02, 0.98, 
         f"Parámetros:\nα = {alpha} m²/s\nΔx = {dx:.3f} m\nΔt = {dt:.3f} s",
         transform=plt.gca().transAxes,
         ha='left', va='top',
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('difusion_calor_varilla_estatico.png', dpi=300, bbox_inches='tight')
plt.show()