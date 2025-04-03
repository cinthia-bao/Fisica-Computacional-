import numpy as np
import pandas as pd
from pi import *

# Parametros de la ecuacion de Duffing
alpha = -1.0
beta = 0.1
delta = 0.2
gamma = 0.5
omega = 2.4

# Parametros para la simulacion
dt = 0.001
t_max = 100
npasos = int(t_max / dt)
T = 2 * np.pi / omega


# Condiciones iniciales
x = 0.1
v = 0.0

xi = []
vi = []
vip = []
xip = []
tip = []
ti = np.linspace(0,t_max,npasos)

# Recordemos que la ecuacion de Duffing es: d2x/dt2 + delta*dx/dt + alpha*x + beta*x**3 = f0*c0s(om*t)
# Hacemos que v = dx/dt y a = dv/dt
for t in ti:
    a = -delta * v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)
    v += a * dt
    x += v * dt
    xi.append(x)
    vi.append(v)
    if np.isclose(t % T, 0, atol=dt):
        xip.append(x)
        vip.append(v)
        tip.append(t)

df = pd.DataFrame({
    'Tiempo': ti,
    'Posicion': xi,
    'Velocidad':vi
})

dfp = pd.DataFrame({
    'Tiempo': tip,
    'Posicion': xip,
    'Velocidad':vip
})


plot = (
    ggplot(df, aes(x='Tiempo',y='Posicion')) +
    geom_line(color='blue') +
    labs(title='Oscilador de Duffing', x='Tiempo (s)', y='Posicion (m)')
)

plot.show()

ef = (
    ggplot(df, aes(x='Posicion',y='Velocidad')) +
    geom_point(color='blue', size = 0.2) +
    labs(title='Espacio fase / Oscilador de Duffing', x='X', y=r'$\dot{X}$')
)

ef.show()


poincare = (
    ggplot(dfp, aes(x='Posicion',y='Velocidad')) +
    geom_point(color='blue') +
    labs(title='Mapa de Poincare / Oscilador de Duffing', x='X', y=r'$\dot{X}$')
)

poincare.show()
