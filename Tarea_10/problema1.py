import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from matplotlib.colors import LogNorm
import time

class ChargedParticle:
    """Clase para representar una partícula cargada acelerada"""
    
    def __init__(self, q=1.0, a=0.05, v0=0.0):
        self.q = q  # Carga en Coulombs
        self.a = a  # Aceleración en m/s²
        self.v0 = v0  # Velocidad inicial
        self.epsilon_0 = 8.854e-12  # Permitividad del vacío
        self.c = 3e8  # Velocidad de la luz
    
    def position(self, t):
        """Posición de la partícula en el tiempo t"""
        return np.array([self.v0*t + 0.5*self.a*t**2, 0.0, 0.0])
    
    def velocity(self, t):
        """Velocidad normalizada (β = v/c)"""
        return np.array([(self.v0 + self.a*t)/self.c, 0.0, 0.0])
    
    def acceleration(self, t):
        """Aceleración normalizada (β̇ = a/c)"""
        return np.array([self.a/self.c, 0.0, 0.0])

def cross_product(a, b):
    """Producto cruz optimizado para arrays 3D"""
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])

def compute_retarded_time(particle, r_obs, t_obs, max_iter=50, tol=1e-6):
    """Calcula el tiempo retardado iterativamente"""
    def equation(t_ret):
        return t_obs - t_ret - np.linalg.norm(r_obs - particle.position(t_ret))/particle.c
    
    try:
        sol = root_scalar(equation, bracket=[0, t_obs], method='bisect', 
                         maxiter=max_iter, rtol=tol)
        return sol.root if sol.converged else None
    except:
        return None

def calculate_fields(particle, grid, t_obs):
    """Calcula campos eléctricos para una malla de puntos"""
    X, Y = grid
    Ex_total = np.zeros_like(X)
    Ey_total = np.zeros_like(Y)
    Ex_rad = np.zeros_like(X)
    Ey_rad = np.zeros_like(Y)
    
    # Vectorización parcial del cálculo
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            r_obs = np.array([X[i,j], Y[i,j], 0])
            t_ret = compute_retarded_time(particle, r_obs, t_obs)
            
            if t_ret is None:
                continue
                
            R_vec = r_obs - particle.position(t_ret)
            R = np.linalg.norm(R_vec)
            n = R_vec/R
            beta = particle.velocity(t_ret)
            beta_dot = particle.acceleration(t_ret)
            K = 1.0 - np.dot(n, beta)
            
            # Campo de velocidad (Coulomb)
            E_vel = (n - beta)*(1-np.linalg.norm(beta)**2)/(R**2 * K**3)
            
            # Campo de radiación
            E_rad = np.cross(n, np.cross((n - beta), beta_dot))/(particle.c*R*K**3)
            
            # Campos totales
            E_total = particle.q/(4*np.pi*particle.epsilon_0) * (E_vel + E_rad)
            Ex_total[i,j], Ey_total[i,j] = E_total[0], E_total[1]
            Ex_rad[i,j], Ey_rad[i,j] = E_rad[0], E_rad[1]
    
    return Ex_total, Ey_total, Ex_rad, Ey_rad

def plot_fields(grid, fields, t_obs, particle):
    """Visualización de los campos calculados"""
    X, Y = grid
    Ex, Ey, Ex_r, Ey_r = fields
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.suptitle(f'Campos Electromagnéticos (t = {t_obs} s)', y=1.02)
    
    # Campo de velocidad
    strm = axes[0,0].streamplot(X, Y, Ex, Ey, color=np.log(np.hypot(Ex, Ey)+1e-10),
                               cmap='plasma', density=1.5, linewidth=1)
    axes[0,0].set_title('Campo de Velocidad (1/r²)')
    fig.colorbar(strm.lines, ax=axes[0,0], label='log|E|')
    
    # Campo de radiación
    strm = axes[0,1].streamplot(X, Y, Ex_r, Ey_r, color=np.log(np.hypot(Ex_r, Ey_r)+1e-10),
                               cmap='viridis', density=1.5, linewidth=1)
    axes[0,1].set_title('Campo de Radiación (1/r)')
    fig.colorbar(strm.lines, ax=axes[0,1], label='log|E|')
    
    # Campo total
    strm = axes[1,0].streamplot(X, Y, Ex+Ex_r, Ey+Ey_r, color=np.log(np.hypot(Ex+Ex_r, Ey+Ey_r)+1e-10),
                               cmap='inferno', density=1.5, linewidth=1)
    axes[1,0].set_title('Campo Total')
    fig.colorbar(strm.lines, ax=axes[1,0], label='log|E|')
    
    # Magnitud del campo radiado
    E_mag = np.hypot(Ex_r, Ey_r)
    im = axes[1,1].imshow(np.log(E_mag.T+1e-10), extent=[X.min(), X.max(), Y.min(), Y.max()],
                         origin='lower', cmap='hot', aspect='auto')
    axes[1,1].set_title('Magnitud del Campo Radiado (log)')
    fig.colorbar(im, ax=axes[1,1], label='log|E|')
    
    # Posición de la partícula
    pos = particle.position(t_obs)
    for ax in axes.flat:
        ax.plot(pos[0], pos[1], 'ro', markersize=8)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.axis('equal')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'campos_t_{t_obs:.1f}.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Función principal"""
    start_time = time.time()
    
    # Crear partícula y configuración
    particle = ChargedParticle(q=1.0, a=0.05, v0=0.0)
    x_vals = np.linspace(-2, 2, 40)
    y_vals = np.linspace(-2, 2, 40)
    grid = np.meshgrid(x_vals, y_vals)
    times = [2.0, 3.0, 5.0]
    
    print("Calculando campos...")
    for t_obs in times:
        print(f"Procesando t = {t_obs} s...")
        fields = calculate_fields(particle, grid, t_obs)
        plot_fields(grid, fields, t_obs, particle)
    
    print(f"Tiempo total de ejecución: {time.time()-start_time:.2f} segundos")
    print("Resultados guardados como campos_t_*.png")

if __name__ == "__main__":
    main()
