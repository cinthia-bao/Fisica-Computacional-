#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 13:12:06 2025

@author: ale
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Definimos el sistema de ecuaciones Lotka-Volterra
def lotka_volterra(t, z, alpha, beta, delta, gamma):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Parámetros del sistema
alpha = 0.5    # Tasa de crecimiento de la presa
beta = 0.02    # Tasa de depredación
delta = 0.01   # Tasa de conversión de presas en depredadores
gamma = 0.3    # Tasa de mortalidad del depredador
# Condiciones iniciales
x0 = 40  # Población inicial de presas
y0 = 10  # Población inicial de depredadores
z0 = [x0, y0]

# Tiempo de simulación
t_span = (0, 200)  # Intervalo de tiempo
sol = solve_ivp(lotka_volterra, t_span, z0, args=(alpha, beta, delta, gamma), dense_output=True)
t = np.linspace(0, 200, 1000)
z = sol.sol(t)

# Graficamos las poblaciones en función del tiempo
plt.figure(figsize=(12, 5))
plt.plot(t, z[0], label="Presas ", color="blue")
plt.plot(t, z[1], label="Depredadores ", color="red")
plt.xlabel("Tiempo")
plt.ylabel("Población")
plt.title("Modelo de Lotka-Volterra: Poblaciones vs Tiempo")
plt.legend()
plt.grid()
plt.show()

# Graficamos el espacio fase
plt.figure(figsize=(6, 6))
plt.plot(z[0], z[1], color="purple")
plt.xlabel("Presas (x)")
plt.ylabel("Depredadores (y)")
plt.title("Espacio fase del modelo Lotka-Volterra")
plt.grid()
plt.show()
