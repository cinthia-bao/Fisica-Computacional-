#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:28:29 2025

@author: ale
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec

# Configuración global de estilo
rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Parámetros físicos
n = 1.0           # Moles de gas
R = 8.314         # Constante de los gases ideales (J/mol·K)

# Rangos de valores
V = np.linspace(0.1, 10, 500)  # Volumen (m³)
T = np.linspace(100, 500, 500) # Temperatura (K)
P_kpa = np.linspace(1, 100, 10)   # Presión (kPa)

# =============================================
# 1. Diagrama P-V (Isotermas)
# =============================================
plt.figure(figsize=(12, 7), dpi=100)
temperaturas = np.linspace(100, 500, 8)  # 8 isotermas
colormap = plt.cm.plasma_r(np.linspace(0.2, 0.9, len(temperaturas)))

for temp, color in zip(temperaturas, colormap):
    P_ideal = (n * R * temp) / V  # Ley de los gases ideales (en Pa)
    plt.plot(V, P_ideal/1e3, color=color, lw=2.5, 
             label=f'{temp:.0f} K', alpha=0.8)

plt.title("Diagrama P-V para Gas Ideal\n(Curvas Isotérmicas)", pad=20)
plt.xlabel("Volumen (m³)")
plt.ylabel("Presión (kPa)")
plt.xlim(0, 5)
plt.ylim(0, 40)
plt.grid(True, linestyle='--', alpha=0.5)

# Leyenda mejorada
legend = plt.legend(title="Temperatura", frameon=True, 
                    shadow=True, borderpad=1)
legend.get_frame().set_facecolor('#f5f5f5')

# Anotación física
plt.annotate(r'$PV = nRT$', xy=(6, 400), fontsize=18,
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.5", alpha=0.9))

plt.tight_layout()
plt.show()

# =============================================
# 2. Diagrama V-T (Isobaras)
# =============================================
fig = plt.figure(figsize=(14, 6))
gs = GridSpec(1, 2, width_ratios=[3, 1])

# Gráfica principal
ax0 = plt.subplot(gs[0])
presiones = np.linspace(10, 100, 8)  # 8 isobaras
colormap = plt.cm.viridis_r(np.linspace(0.2, 0.9, len(presiones)))

for pressure, color in zip(presiones, colormap):
    V_ideal = (n * R * T) / (pressure * 1e3)  # Convertir kPa a Pa
    ax0.plot(T, V_ideal, color=color, lw=2.5, 
            label=f'{pressure:.0f} kPa', alpha=0.8)

ax0.set_title("Diagrama V-T para Gas Ideal\n(Curvas Isobáricas)", pad=15)
ax0.set_xlabel("Temperatura (K)")
ax0.set_ylabel("Volumen (m³)")
ax0.set_xlim(100, 500)
ax0.grid(True, linestyle='--', alpha=0.5)

# Leyenda como subplot
ax1 = plt.subplot(gs[1])
ax1.axis('off')
legend = ax1.legend(*ax0.get_legend_handles_labels(), 
                   title="Presión (kPa)", loc='center',
                   frameon=True, shadow=True)
legend.get_frame().set_facecolor('#f5f5f5')

# Anotación científica
ax0.annotate(r'$\frac{V}{T} = \frac{nR}{P} = \text{const.}$', 
             xy=(350, 0.8), fontsize=16,
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.5", alpha=0.9))

plt.tight_layout()
plt.show()

# =============================================
# 3. Diagrama T-P (Isocoras)
# =============================================
plt.figure(figsize=(12, 7))
volumenes = np.linspace(1, 10, 8)  # 8 isocoras
colormap = plt.cm.cool(np.linspace(0.2, 0.9, len(volumenes)))

for volume, color in zip(volumenes, colormap):
    P_ideal = (n * R * T) / volume  # en Pa
    plt.plot(T, P_ideal/1e3, color=color, lw=2.5, 
             label=f'{volume:.1f} m³', alpha=0.8)

plt.title("Diagrama T-P para Gas Ideal\n(Curvas Isocóricas)", pad=20)
plt.xlabel("Temperatura (K)")
plt.ylabel("Presión (kPa)")
plt.xlim(100, 500)
plt.ylim(0, 10)

# Leyenda con estilo
legend = plt.legend(title="Volumen", ncol=2, frameon=True,
                   shadow=True, borderpad=1, loc='upper left')
legend.get_frame().set_facecolor('#f5f5f5')

# Ecuación destacada
plt.annotate(r'$\frac{P}{T} = \frac{nR}{V} = \text{const.}$', 
             xy=(120, 400), fontsize=16,
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.5", alpha=0.9))

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()