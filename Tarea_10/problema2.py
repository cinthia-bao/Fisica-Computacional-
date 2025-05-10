import numpy as np
import matplotlib.pyplot as plt

# Constantes
q = 1.0  # carga en Coulombs
epsilon_0 = 8.854e-12  # permitividad en el vacío
r_sphere = 1.0  # radio de la esfera en metros

# Función para calcular el campo eléctrico (ley de Coulomb)
def electric_field_coulomb(r):
    R_mag = np.linalg.norm(r)  # magnitud de R
    if R_mag == 0:
        return np.array([0.0, 0.0, 0.0])  # evitar división por cero
    return (q / (4 * np.pi * epsilon_0)) * (r / R_mag**3)

# Malla esférica
N_theta = 50  # divisiones en ángulo polar
N_phi = 50  # divisiones en ángulo azimutal
theta = np.linspace(0, np.pi, N_theta)  # ángulos polares (0 a pi)
phi = np.linspace(0, 2 * np.pi, N_phi)  # ángulos azimutales (0 a 2pi)
theta_grid, phi_grid = np.meshgrid(theta, phi)

# Coordenadas de la esfera
x = r_sphere * np.sin(theta_grid) * np.cos(phi_grid)
y = r_sphere * np.sin(theta_grid) * np.sin(phi_grid)
z = r_sphere * np.cos(theta_grid)

# Elemento de área en coordenadas esféricas
dA = r_sphere**2 * np.sin(theta_grid) * (np.pi / N_theta) * (2 * np.pi / N_phi)

# Calcular flujo eléctrico
flux = 0.0
for i in range(N_theta):
    for j in range(N_phi):
        r = np.array([x[i, j], y[i, j], z[i, j]])  # vector de posición
        E = electric_field_coulomb(r)  # campo eléctrico
        n_hat = r / np.linalg.norm(r)  # vector normal unitario
        flux += np.dot(E, n_hat) * dA[i, j]

# Comparar con el flujo esperado
flux_expected = q / epsilon_0

print(f"Flujo calculado numéricamente: {flux:.2e} N·m²/C")
print(f"Flujo esperado analíticamente: {flux_expected:.2e} N·m²/C")
print(f"Error relativo: {abs(flux - flux_expected) / flux_expected:.2%}")

# Visualización de la esfera y el campo eléctrico
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(
    x, y, z,
    x, y, z,
    length=0.1, normalize=True, color='blue', alpha=0.5
)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Campo eléctrico en una superficie esférica')
plt.show()

