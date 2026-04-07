# CLAUDE_21Paper_demo.md — Contrato: Demo completa del 21Paper

**Fecha:** Abril 2026  
**Proyecto:** 21Paper — Unified Homotopy Framework  
**Objetivo:** Implementar la demo central del paper: identificar y simular
un sistema de flujo por orificio usando el marco homotópico unificado.

---

## 1. Sistema físico: tanque con orificio

Ecuación del tanque:

```
A_t · dh/dt + Cd · A_o · sqrt(2·g·h) = Q_in(t)
```

Normalizado (A_t = 1):

```
dh/dt + f(h) = u(t)
```

donde:
- `f(h) = alpha · sqrt(h)` con `alpha = Cd · A_o · sqrt(2g) / A_t`
- `u(t) = Q_in(t) / A_t`

Parámetros reales (conocidos, para generar datos):
- `alpha = 0.5` (combina Cd, áreas, gravedad)
- `h(0) = 1.0` (nivel inicial)

---

## 2. Excitación u(t): escalón-rampa

```python
def u(t):
    if t < 2.0:
        return 0.0          # sin flujo de entrada
    elif t < 6.0:
        return 0.3 * (t - 2.0) / 4.0   # rampa de 0 a 0.3
    else:
        return 0.3           # escalón constante
```

Tiempo total: t ∈ [0, 10], pero solo tomamos **20 puntos**.

---

## 3. Generar datos "experimentales" (la verdad)

```python
from scipy.integrate import solve_ivp
import numpy as np

alpha = 0.5

def f_true(h):
    return alpha * np.sqrt(np.maximum(h, 0))

def rhs(t, h):
    return u(t) - f_true(h[0])

sol = solve_ivp(rhs, [0, 10], [1.0], t_eval=np.linspace(0, 10, 20),
                method='RK45', rtol=1e-10)

t_data = sol.t          # 20 puntos
h_data = sol.y[0]       # 20 valores de h(t)
u_data = np.array([u(ti) for ti in t_data])
```

De los datos, estimar f_i:
```python
# Derivada numérica (o usar la ecuación directamente):
# dh/dt ≈ (h_{i+1} - h_{i-1}) / (2T)  para puntos interiores
# f_i = u_i - dh/dt_i

# Mejor: usar la ODE directamente
# f_i = u_i - dh/dt_i
# Calcular dh/dt con diferencias centradas
T = t_data[1] - t_data[0]
dhdt = np.zeros(20)
dhdt[0] = (h_data[1] - h_data[0]) / T
dhdt[-1] = (h_data[-1] - h_data[-2]) / T
for i in range(1, 19):
    dhdt[i] = (h_data[i+1] - h_data[i-1]) / (2*T)

f_data = u_data - dhdt   # valores de f(h) en los 20 puntos
```

---

## 4. Nivel 1: Identificar RBF por homotopía

### 4.1 Configuración de la RBF

```python
M = 5  # 5 centros (suficiente para 20 datos)
```

RBF Gaussiana: `φ(r) = exp(-r²)`

### 4.2 Inicialización: K-means

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=M, random_state=42).fit(h_data.reshape(-1, 1))
c0 = kmeans.cluster_centers_.flatten()   # 5 centros iniciales
c0 = np.sort(c0)                          # ordenar

# Anchos iniciales: distancia al vecino más cercano
lambda0 = np.zeros(M)
for j in range(M):
    dists = np.abs(c0 - c0[j])
    dists[j] = np.inf
    lambda0[j] = np.min(dists) * 0.8
```

### 4.3 Paso 1: Pesos por mínimos cuadrados (z₁ lineal, exacto)

```python
# Construir matriz de diseño Φ
Phi = np.zeros((20, M))
for i in range(20):
    for j in range(M):
        r = np.abs(h_data[i] - c0[j]) / lambda0[j]
        Phi[i, j] = np.exp(-r**2)

# Pesos por mínimos cuadrados
w0 = np.linalg.lstsq(Phi, f_data, rcond=None)[0]

# Residuo
N0 = Phi @ w0 - f_data
print(f"Residuo inicial: ||N0|| = {np.linalg.norm(N0):.6e}")
```

### 4.4 Paso 2: Corrección z₁ (Newton) sobre centros

```python
# Jacobiano dN_i/dc_j
# dN_i/dc_j = w_j · φ'(r_ij) · (c_j - h_i) / (λ_j · |h_i - c_j|)
# Para Gaussiana: φ'(r) = -2r·exp(-r²)

Jc = np.zeros((20, M))
for i in range(20):
    for j in range(M):
        r_ij = (h_data[i] - c0[j]) / lambda0[j]
        phi_prime = -2 * r_ij * np.exp(-r_ij**2)
        Jc[i, j] = w0[j] * phi_prime * (-1.0 / lambda0[j])

# Corrección Newton sobre centros
delta_c1 = -np.linalg.lstsq(Jc, N0, rcond=None)[0]
c1 = c0 + delta_c1
```

### 4.5 Paso 3: Recalcular pesos con centros actualizados

```python
# Reconstruir Φ con c1
Phi1 = np.zeros((20, M))
for i in range(20):
    for j in range(M):
        r = np.abs(h_data[i] - c1[j]) / lambda0[j]
        Phi1[i, j] = np.exp(-r**2)

w1 = np.linalg.lstsq(Phi1, f_data, rcond=None)[0]
N1 = Phi1 @ w1 - f_data
print(f"Residuo post-z1 centros: ||N1|| = {np.linalg.norm(N1):.6e}")
```

### 4.6 Paso 4: Corrección z₂ (Halley) sobre centros

```python
# Hessiano diagonal d²N_i/dc_j²
# Para Gaussiana: φ''(r) = (4r² - 2)·exp(-r²)

Hc_diag = np.zeros(M)
for j in range(M):
    hess_sum = 0.0
    for i in range(20):
        r_ij = (h_data[i] - c1[j]) / lambda0[j]
        phi_pp = (4*r_ij**2 - 2) * np.exp(-r_ij**2)
        hess_sum += w1[j] * phi_pp / lambda0[j]**2
    Hc_diag[j] = hess_sum

# Recalcular Jacobiano en c1
Jc1 = np.zeros((20, M))
for i in range(20):
    for j in range(M):
        r_ij = (h_data[i] - c1[j]) / lambda0[j]
        phi_prime = -2 * r_ij * np.exp(-r_ij**2)
        Jc1[i, j] = w1[j] * phi_prime * (-1.0 / lambda0[j])

# z₂ Halley por componente
delta_c2 = np.zeros(M)
for j in range(M):
    Jcol = Jc1[:, j]
    gp = np.dot(Jcol, Jcol)  # (J^T J)_jj
    if abs(gp) > 1e-12 and abs(Hc_diag[j]) > 1e-15:
        delta_c2[j] = -0.5 * np.dot(N1, Jcol)**2 * Hc_diag[j] / gp**1.5

c2 = c1 + delta_c2
```

### 4.7 Paso 5: Corrección z₁ sobre anchos

```python
# Jacobiano dN_i/dλ_j
# dN_i/dλ_j = -w_j · φ'(r_ij) · r_ij / λ_j
# (porque dr/dλ = -|h-c|/λ² = -r/λ)

# Reconstruir Φ con c2, calcular w2
Phi2 = np.zeros((20, M))
for i in range(20):
    for j in range(M):
        r = np.abs(h_data[i] - c2[j]) / lambda0[j]
        Phi2[i, j] = np.exp(-r**2)

w2 = np.linalg.lstsq(Phi2, f_data, rcond=None)[0]
N2 = Phi2 @ w2 - f_data

Jlam = np.zeros((20, M))
for i in range(20):
    for j in range(M):
        r_ij = np.abs(h_data[i] - c2[j]) / lambda0[j]
        phi_prime = -2 * r_ij * np.exp(-r_ij**2)
        Jlam[i, j] = -w2[j] * phi_prime * r_ij / lambda0[j]

delta_lam1 = -np.linalg.lstsq(Jlam, N2, rcond=None)[0]
lambda1 = lambda0 + delta_lam1
lambda1 = np.maximum(lambda1, 0.01)  # evitar anchos negativos
```

### 4.8 Paso final: Recalcular pesos con parámetros finales

```python
Phi_final = np.zeros((20, M))
for i in range(20):
    for j in range(M):
        r = np.abs(h_data[i] - c2[j]) / lambda1[j]
        Phi_final[i, j] = np.exp(-r**2)

w_final = np.linalg.lstsq(Phi_final, f_data, rcond=None)[0]
N_final = Phi_final @ w_final - f_data
print(f"Residuo final: ||N_final|| = {np.linalg.norm(N_final):.6e}")
```

---

## 5. Nivel 2: Simular ODE con la RBF identificada

### 5.1 Definir RBF identificada y sus derivadas

```python
def rbf_f(h):
    """RBF identificada: f_hat(h) = Σ w_j · exp(-((h-c_j)/λ_j)²)"""
    result = 0.0
    for j in range(M):
        r = (h - c2[j]) / lambda1[j]
        result += w_final[j] * np.exp(-r**2)
    return result

def rbf_df(h):
    """Derivada: f_hat'(h)"""
    result = 0.0
    for j in range(M):
        r = (h - c2[j]) / lambda1[j]
        result += w_final[j] * (-2*r) * np.exp(-r**2) / lambda1[j]
    return result

def rbf_d2f(h):
    """Segunda derivada: f_hat''(h)"""
    result = 0.0
    for j in range(M):
        r = (h - c2[j]) / lambda1[j]
        result += w_final[j] * (4*r**2 - 2) * np.exp(-r**2) / lambda1[j]**2
    return result
```

### 5.2 Solución lineal de partida

```python
# Linealización: k = f'(h_eq) donde h_eq es el nivel de equilibrio
h_eq = np.mean(h_data)
k = rbf_df(h_eq)

# Solución lineal: dh/dt + k·h = u(t)
# h_0(t) = h(0)·exp(-kt) + ∫₀ᵗ exp(-k(t-s))·u(s) ds
```

### 5.3 Resolver con el solver integral del 20Paper

```python
# No linealidad reducida
def f_tilde(h):
    return rbf_f(h) - k * h

def df_tilde(h):
    return rbf_df(h) - k

def d2f_tilde(h):
    return rbf_d2f(h)

# Solver integral con kernel exponencial K = exp(-k(t-s))
# Discretización fina: 200 puntos (10× más que los datos)
n_sim = 200
t_sim = np.linspace(0, 10, n_sim)
T_sim = t_sim[1] - t_sim[0]
u_sim = np.array([u(ti) for ti in t_sim])

# IIR filter
alpha_iir = np.exp(-k * T_sim)
h_sim = np.zeros(n_sim)
h_sim[0] = 1.0  # h(0)
S = 0.0

for step in range(1, n_sim):
    h_sim[step] = h_sim[step-1]  # estimación inicial
    
    w_k = 1.0  # peso trapezoidal interior
    y0_term = 1.0 * np.exp(-k * t_sim[step])
    
    # z₁: Newton sobre residuo integral
    ft = f_tilde(h_sim[step])
    g = h_sim[step] + T_sim*w_k*ft - T_sim*w_k*u_sim[step] - alpha_iir*S - y0_term
    gp = 1.0 + T_sim*w_k*df_tilde(h_sim[step])
    
    z1 = -g / gp
    h_sim[step] += z1
    
    # z₂: Halley
    ft = f_tilde(h_sim[step])
    g = h_sim[step] + T_sim*w_k*ft - T_sim*w_k*u_sim[step] - alpha_iir*S - y0_term
    gp = 1.0 + T_sim*w_k*df_tilde(h_sim[step])
    gpp = T_sim*w_k*d2f_tilde(h_sim[step])
    
    z2 = -0.5 * g**2 * gpp / gp**3
    h_sim[step] += z2
    
    # Actualizar IIR
    h_k = u_sim[step] - f_tilde(h_sim[step])
    S = alpha_iir * S + T_sim * w_k * h_k
```

---

## 6. Salida esperada

### Tabla de parámetros RBF identificados

```
NIVEL 1: IDENTIFICACIÓN RBF (homotopía paramétrica)
====================================================
Datos: 20 puntos, f(h) = 0.5·sqrt(h) (verdadera, desconocida)

K-means → c₀ = [0.32, 0.51, 0.68, 0.82, 0.95]
vecinos → λ₀ = [0.15, 0.14, 0.13, 0.11, 0.10]

Paso 1: w = LS(Φ,f) .................. ||N₀|| = ???
Paso 2: z₁ Newton sobre centros ...... ||N₁|| = ???  (< N₀)
Paso 3: z₂ Halley sobre centros ...... ||N₂|| = ???  (< N₁)
Paso 4: z₁ Newton sobre anchos ....... ||N₃|| = ???  (< N₂)
Paso 5: w = LS(Φ_final, f) ........... ||N_f|| = ???  (< N₃)

Parámetros finales:
  j   w_j        c_j        λ_j
  1   ???        ???        ???
  2   ???        ???        ???
  3   ???        ???        ???
  4   ???        ???        ???
  5   ???        ???        ???
```

### Tabla comparativa de simulación

```
NIVEL 2: SIMULACIÓN ODE (homotopía integral)
=============================================
k = f'(h_eq) = ???, kernel K = exp(-k(t-s))

  t      h_real    h_RBF     h_lineal    error_RBF
  0.0    1.000     1.000     1.000       0.0e+00
  0.5    ???       ???       ???         ???
  1.0    ???       ???       ???         ???
  ...
  10.0   ???       ???       ???         ???

Error máx RBF vs real:  ???
Error máx lineal vs real: ???
Mejora RBF/lineal: ???×
```

### Figuras a generar

1. `fig_identification.png` — f_real vs f_RBF vs datos, mostrando la calidad del ajuste
2. `fig_simulation.png` — h(t) real vs RBF vs lineal, con u(t) superpuesto
3. `fig_residuals.png` — ||N|| por paso de homotopía (debe decrecer)
4. `fig_rbf_components.png` — las 5 Gaussianas individuales y su suma

---

## 7. Archivo a crear: demo_21paper.py

Un solo archivo Python (~300 líneas) que:
1. Define el sistema físico (tanque+orificio)
2. Genera 20 datos con RK45
3. Identifica la RBF por homotopía (Nivel 1)
4. Simula la ODE con la RBF identificada (Nivel 2)
5. Compara con la solución lineal de partida
6. Imprime tablas y genera 4 figuras

Ubicación: `/home/rodo/21Paper/demo_21paper.py`

### Dependencias

- numpy
- scipy (solo para solve_ivp como referencia)
- matplotlib (para figuras)
- sklearn (solo para KMeans)

### Ejecución

```bash
cd /home/rodo/21Paper
python demo_21paper.py
```

---

## 8. Criterios de éxito

| Criterio | Valor |
|---|---|
| ||N_final|| / ||N_0|| | < 0.1 (90% reducción de residuo) |
| Error máx RBF vs real | < 0.05 (5% del rango de h) |
| Error máx lineal vs real | > 0.1 (la lineal no basta) |
| Mejora RBF/lineal | > 3× |
| f_RBF(h) vs f_real(h) | Visualmente indistinguibles en el rango de datos |
| Convergencia Nivel 1 | ||N|| decrece monótonamente en cada paso |
| Simulación estable | Sin NaN, sin divergencia en t ∈ [0,10] |

---

## 9. Lo que demuestra esta demo

1. **K-means + LS + Newton + Halley** (4 pasos) identifican una RBF
   que aproxima sqrt(h) con 5 centros y 20 datos.

2. **La solución lineal dh/dt + kh = u** es el punto de partida analítico
   conocido. La RBF corrige la no linealidad sqrt(h) - kh.

3. **El solver integral** con kernel K=exp(-kt) resuelve la ODE
   sin derivadas numéricas, usando las correcciones z₁, z₂.

4. **Todo es homotopía**: identificación = homotopía sobre θ,
   simulación = homotopía sobre y_k. Mismas fórmulas.

---

## 10. Reglas

1. **NO importar** nada del regressor/ ni del 20Paper/src/.
   La demo es self-contained.
2. **NO usar scipy** en el solver (solo para generar referencia RK45).
3. **sklearn** solo para KMeans. Si no está disponible, implementar
   KMeans simple (20 puntos, 5 centros, converge en 10 iteraciones).
4. **Imprimir todo**: parámetros, residuos, errores. La demo es
   pedagógica, no optimizada.
5. **Las figuras** van a `/home/rodo/21Paper/figures/` con 300 DPI.
6. **float64 siempre.**

---

*Contrato demo 21Paper — Abril 2026*
*Rodolfo H. Rodrigo / INAUT-UNSJ-CONICET*
