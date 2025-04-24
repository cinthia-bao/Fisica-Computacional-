#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulación 3D de la ecuación de calor en un cubo con Crank-Nicolson
Visualización optimizada con cortes 2D y perfiles de temperatura
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
from matplotlib import cm
import time

# =============================================
# 1. CONFIGURACIÓN Y PARÁMETROS
# =============================================

# Configuración para mejor rendimiento
plt.style.use('seaborn')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True

# Parámetros físicos optimizados
L = 1.0            # Longitud del cubo [m]
N = 15             # Puntos por dimensión (reducido para rendimiento)
alpha = 0.01       # Difusividad térmica [m²/s]
Tmax = 0.1         # Tiempo total de simulación [s]
source_temp = 100  # Temperatura inicial en la fuente [°C]

# Discretización espacial y temporal
dx = L / (N - 1)
dt = 0.002         # Paso de tiempo [s]
Nt = int(Tmax / dt)
print(f"Simulación 3D con {N}³ puntos y {Nt} pasos temporales")

# Coeficiente para Crank-Nicolson
r = alpha * dt / (2 * dx**2)

# =============================================
# 2. INICIALIZACIÓN Y MATRICES
# =============================================

# Campo de temperatura 3D
u = np.zeros((N, N, N))

# Fuente de calor en el centro (cubo pequeño)
u[N//4:3*N//4, N//4:3*N//4, N//4:3*N//4] = source_temp

# Construcción de matrices para ADI en 3D
def build_system_matrices(N, r):
    """Construye matrices del sistema para Crank-Nicolson 3D"""
    # Matriz 1D
    main_diag = (1 + 2*r) * np.ones(N)
    off_diag = -r * np.ones(N-1)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
    B = diags([r*np.ones(N-1), (1-2*r)*np.ones(N), r*np.ones(N-1)], [-1, 0, 1], format='csr')
    
    # Matrices identidad
    I = eye(N, format='csr')
    
    # Productos Kronecker para 3D
    Ax = kron(kron(A, I), I)
    Ay = kron(kron(I, A), I)
    Az = kron(kron(I, I), A)
    
    Bx = kron(kron(B, I), I)
    By = kron(kron(I, B), I)
    Bz = kron(kron(I, I), B)
    
    return Ax, Ay, Az, Bx, By, Bz

# Matrices del sistema (precomputadas para eficiencia)
Ax, Ay, Az, Bx, By, Bz = build_system_matrices(N, r)

# =============================================
# 3. FUNCIÓN DE ACTUALIZACIÓN
# =============================================

def update_temperature(u):
    """Actualiza el campo de temperatura usando ADI en 3D"""
    u_flat = u.reshape(-1)
    
    # Primera etapa: implícito en X
    b = Bx @ u_flat
    u_flat = spsolve(Ax, b)
    
    # Segunda etapa: implícito en Y
    b = By @ u_flat
    u_flat = spsolve(Ay, b)
    
    # Tercera etapa: implícito en Z
    b = Bz @ u_flat
    u_flat = spsolve(Az, b)
    
    # Reformar y aplicar condiciones de frontera
    u_new = u_flat.reshape((N, N, N))
    u_new[0,:,:] = u_new[-1,:,:] = u_new[:,0,:] = u_new[:,-1,:] = u_new[:,:,0] = u_new[:,:,-1] = 0
    
    return u_new

# =============================================
# 4. VISUALIZACIÓN INTERACTIVA
# =============================================

# Configuración de la figura
fig = plt.figure(figsize=(15, 8))
fig.suptitle('Difusión de Calor 3D en Cubo - Método Crank-Nicolson', fontsize=14)

# Crear subplots para cortes 2D
ax1 = fig.add_subplot(231)  # Corte XY
ax2 = fig.add_subplot(232)  # Corte XZ
ax3 = fig.add_subplot(233)  # Corte YZ
ax4 = fig.add_subplot(212)  # Perfiles

# Configuración inicial de las visualizaciones
slice_idx = N // 2  # Índice para cortes centrales
x = y = z = np.linspace(0, L, N)

# Cortes 2D iniciales
im1 = ax1.imshow(u[:,:,slice_idx].T, cmap='inferno', origin='lower', 
                extent=[0, L, 0, L], vmin=0, vmax=source_temp)
ax1.set_title(f'Corte XY en Z = {z[slice_idx]:.2f}m')
ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
plt.colorbar(im1, ax=ax1, label='Temperatura [°C]')

im2 = ax2.imshow(u[:,slice_idx,:].T, cmap='inferno', origin='lower', 
                extent=[0, L, 0, L], vmin=0, vmax=source_temp)
ax2.set_title(f'Corte XZ en Y = {y[slice_idx]:.2f}m')
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Z [m]')
plt.colorbar(im2, ax=ax2, label='Temperatura [°C]')

im3 = ax3.imshow(u[slice_idx,:,:].T, cmap='inferno', origin='lower', 
                extent=[0, L, 0, L], vmin=0, vmax=source_temp)
ax3.set_title(f'Corte YZ en X = {x[slice_idx]:.2f}m')
ax3.set_xlabel('Y [m]')
ax3.set_ylabel('Z [m]')
plt.colorbar(im3, ax=ax3, label='Temperatura [°C]')

# Perfiles de temperatura
line_x, = ax4.plot(x, u[:, N//2, N//2], 'r-', label='Perfil X (Y=Z=L/2)')
line_y, = ax4.plot(y, u[N//2, :, N//2], 'b-', label='Perfil Y (X=Z=L/2)')
line_z, = ax4.plot(z, u[N//2, N//2, :], 'g-', label='Perfil Z (X=Y=L/2)')
ax4.set_title('Perfiles de Temperatura a través del Centro')
ax4.set_xlabel('Posición [m]')
ax4.set_ylabel('Temperatura [°C]')
ax4.legend()
ax4.grid(True)
ax4.set_ylim(0, source_temp*1.1)

# Texto informativo
info_text = fig.text(0.1, 0.05, '', fontsize=10)

# =============================================
# 5. ANIMACIÓN
# =============================================

def init():
    """Inicialización de la animación"""
    im1.set_array(u[:,:,slice_idx].T)
    im2.set_array(u[:,slice_idx,:].T)
    im3.set_array(u[slice_idx,:,:].T)
    line_x.set_ydata(u[:, N//2, N//2])
    line_y.set_ydata(u[N//2, :, N//2])
    line_z.set_ydata(u[N//2, N//2, :])
    info_text.set_text(f'Tiempo = 0.000 s / {Tmax:.3f} s | Temp máxima = {np.max(u):.1f} °C')
    return im1, im2, im3, line_x, line_y, line_z, info_text

def animate(t):
    """Actualización del frame de animación"""
    global u
    start_time = time.time()
    
    # Actualizar solución
    u = update_temperature(u)
    
    # Actualizar visualizaciones
    im1.set_array(u[:,:,slice_idx].T)
    im2.set_array(u[:,slice_idx,:].T)
    im3.set_array(u[slice_idx,:,:].T)
    line_x.set_ydata(u[:, N//2, N//2])
    line_y.set_ydata(u[N//2, :, N//2])
    line_z.set_ydata(u[N//2, N//2, :])
    
    # Calcular tiempo de simulación real
    sim_time = t * dt
    comp_time = time.time() - start_time
    
    # Actualizar información
    info_text.set_text(
        f'Tiempo simulación = {sim_time:.3f} s / {Tmax:.3f} s | '
        f'Tiempo cálculo = {comp_time:.3f} s | '
        f'Temp máxima = {np.max(u):.1f} °C | '
        f'α = {alpha} m²/s | Δx = {dx:.3f} m | Δt = {dt:.4f} s'
    )
    
    return im1, im2, im3, line_x, line_y, line_z, info_text

# Configurar y mostrar animación
ani = FuncAnimation(
    fig, animate, frames=min(Nt, 100),  # Limitar a 100 frames para demo
    init_func=init, interval=100, blit=False
)

plt.tight_layout()
plt.show()