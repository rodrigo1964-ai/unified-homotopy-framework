# CLAUDE_21Paper_mlp_demo.md — Contrato: Demo MLP del 21Paper

**Fecha:** Abril 2026  
**Proyecto:** 21Paper — Unified Homotopy Framework  
**Objetivo:** Repetir la demo del orificio con MLP en vez de RBF,
demostrando que el framework es independiente de la arquitectura.

---

## 1. Arquitectura MLP

```
entrada (h) ──→ [σ(W₁h + b₁)] ──→ [W₂·hidden + b₂] ──→ salida (f̂)
   R¹              R⁸ sigmoide        R¹ lineal
```

Parámetros:
- **No lineales:** W₁ ∈ R⁸, b₁ ∈ R⁸ (16 parámetros)
- **Lineales:** W₂ ∈ R¹ˣ⁸, b₂ ∈ R (9 parámetros)
- **Total:** 25 parámetros

Función sigmoide: σ(x) = 1/(1 + exp(-x))

```python
def mlp_f(h, W1, b1, W2, b2):
    """MLP: h (escalar) -> f_hat (escalar)"""
    hidden = sigmoid(W1 * h + b1)   # R^8
    return W2 @ hidden + b2          # R^1
```

---

## 2. Sistema físico (idéntico a la demo RBF)

```
dh/dt + f(h) = u(t)
f(h) = 0.5 * sqrt(h)   (real, desconocida)
h(0) = 1.0
u(t) = escalón-rampa: 0 → rampa → 0.3
t ∈ [0, 10], 20 puntos de datos
```

Generar datos con RK45 (rtol=1e-10) idéntico a demo_21paper.py.

---

## 3. Nivel 1: Identificación MLP por homotopía

### 3.1 Estructura bilineal

El residuo de la MLP es:
```
N_i(θ) = W₂ · σ(W₁·h_i + b₁) + b₂ - f_i
```

Esto es LINEAL en (W₂, b₂) y NO LINEAL en (W₁, b₁).
Es la misma estructura bilineal que la RBF.

### 3.2 Paso 1: Pesos lineales por z₁ (exacto)

Fijar (W₁, b₁). Construir la matriz de diseño:
```python
# Φ[i,j] = σ(W1[j] * h_data[i] + b1[j])  para j=0..7
Phi = sigmoid(np.outer(h_data, W1) + b1)  # 20 x 8

# Agregar columna de 1s para b2
Phi_ext = np.column_stack([Phi, np.ones(20)])  # 20 x 9

# Resolver [W2, b2] por mínimos cuadrados
wb = np.linalg.lstsq(Phi_ext, f_data, rcond=None)[0]
W2 = wb[:8]
b2 = wb[8]
```

Esto es z₁ sobre (W₂, b₂), exacto en un paso.
Porque ∂²N/∂W₂² = 0 → z₂ = z₃ = 0 para los pesos lineales.

### 3.3 Inicialización de (W₁, b₁)

```python
# Distribución uniforme de las sigmoides sobre el rango de h
h_min, h_max = h_data.min(), h_data.max()

# Los centros de las sigmoides: equiespaciados
centers = np.linspace(h_min, h_max, 8)

# W1[j] controla la pendiente, b1[j] controla el centro
# σ(W1*h + b1) = 0.5 cuando W1*h + b1 = 0 → h = -b1/W1
# Elegimos W1 = 5.0 (pendiente moderada) para todas
W1_init = np.full(8, 5.0)
b1_init = -W1_init * centers   # centros de las sigmoides
```

### 3.4 Paso 2: Corrección z₁ (Newton) sobre (W₁, b₁)

El Jacobiano de N respecto a W₁[j]:
```
∂N_i/∂W1[j] = W2[j] · σ'(W1[j]*h_i + b1[j]) · h_i
```

El Jacobiano respecto a b₁[j]:
```
∂N_i/∂b1[j] = W2[j] · σ'(W1[j]*h_i + b1[j])
```

donde σ'(x) = σ(x)(1-σ(x)).

Vector de parámetros no lineales: θ_nl = [W1; b1] ∈ R¹⁶
Jacobiano: J_nl ∈ R^{20×16}

```python
def compute_jacobian_nl(h_data, W1, b1, W2):
    """Jacobiano de N respecto a (W1, b1)"""
    N = len(h_data)
    M = len(W1)
    J = np.zeros((N, 2*M))
    
    for i in range(N):
        for j in range(M):
            z = W1[j] * h_data[i] + b1[j]
            s = sigmoid(z)
            sp = s * (1 - s)   # σ'
            
            # ∂N_i/∂W1[j]
            J[i, j] = W2[j] * sp * h_data[i]
            
            # ∂N_i/∂b1[j]
            J[i, M+j] = W2[j] * sp
    
    return J

# Newton sobre (W1, b1)
J_nl = compute_jacobian_nl(h_data, W1, b1, W2)
N0 = mlp_predict(h_data, W1, b1, W2, b2) - f_data
delta_nl = -np.linalg.lstsq(J_nl, N0, rcond=None)[0]

W1 += delta_nl[:8]
b1 += delta_nl[8:]
```

### 3.5 Paso 3: Recalcular pesos lineales

```python
Phi = sigmoid(np.outer(h_data, W1) + b1)
Phi_ext = np.column_stack([Phi, np.ones(20)])
wb = np.linalg.lstsq(Phi_ext, f_data, rcond=None)[0]
W2, b2 = wb[:8], wb[8]
```

### 3.6 Paso 4: Corrección z₂ (Halley) sobre (W₁, b₁)

El Hessiano de N respecto a W₁[j]:
```
∂²N_i/∂W1[j]² = W2[j] · σ''(z_ij) · h_i²
```

donde σ''(x) = σ'(x)(1 - 2σ(x)).

```python
def compute_halley_correction(h_data, W1, b1, W2, N1, J_nl):
    """Corrección z₂ de Halley por componente"""
    M = len(W1)
    delta2 = np.zeros(2*M)
    
    for j in range(M):
        # Hessiano diagonal para W1[j]
        H_jj = 0.0
        for i in range(len(h_data)):
            z = W1[j] * h_data[i] + b1[j]
            s = sigmoid(z)
            sp = s * (1 - s)
            spp = sp * (1 - 2*s)
            H_jj += W2[j] * spp * h_data[i]**2
        
        # z₂ Halley
        Jcol = J_nl[:, j]
        gp = np.dot(Jcol, Jcol)
        if abs(gp) > 1e-12 and abs(H_jj) > 1e-15:
            delta2[j] = -0.5 * np.dot(N1, Jcol)**2 * H_jj / gp**1.5
        
        # Análogo para b1[j]
        H_bj = 0.0
        for i in range(len(h_data)):
            z = W1[j] * h_data[i] + b1[j]
            s = sigmoid(z)
            sp = s * (1 - s)
            spp = sp * (1 - 2*s)
            H_bj += W2[j] * spp
        
        Jcol_b = J_nl[:, M+j]
        gp_b = np.dot(Jcol_b, Jcol_b)
        if abs(gp_b) > 1e-12 and abs(H_bj) > 1e-15:
            delta2[M+j] = -0.5 * np.dot(N1, Jcol_b)**2 * H_bj / gp_b**1.5
    
    return delta2
```

### 3.7 Iteración completa (3-5 outer iterations)

```python
for iteration in range(5):
    # Paso 1: pesos lineales (z₁ exacto)
    Phi = sigmoid(np.outer(h_data, W1) + b1)
    Phi_ext = np.column_stack([Phi, np.ones(20)])
    wb = np.linalg.lstsq(Phi_ext, f_data, rcond=None)[0]
    W2, b2 = wb[:8], wb[8]
    
    # Residuo
    N0 = mlp_predict(h_data, W1, b1, W2, b2) - f_data
    print(f"Iter {iteration}: ||N|| = {np.linalg.norm(N0):.6e}")
    if np.linalg.norm(N0) < 1e-6:
        break
    
    # Paso 2: z₁ Newton sobre (W1, b1)
    J_nl = compute_jacobian_nl(h_data, W1, b1, W2)
    delta1 = -np.linalg.lstsq(J_nl, N0, rcond=None)[0]
    W1 += delta1[:8]
    b1 += delta1[8:]
    
    # Recalcular pesos lineales
    Phi = sigmoid(np.outer(h_data, W1) + b1)
    Phi_ext = np.column_stack([Phi, np.ones(20)])
    wb = np.linalg.lstsq(Phi_ext, f_data, rcond=None)[0]
    W2, b2 = wb[:8], wb[8]
    
    # Paso 3: z₂ Halley sobre (W1, b1)
    N1 = mlp_predict(h_data, W1, b1, W2, b2) - f_data
    J_nl = compute_jacobian_nl(h_data, W1, b1, W2)
    delta2 = compute_halley_correction(h_data, W1, b1, W2, N1, J_nl)
    W1 += delta2[:8]
    b1 += delta2[8:]
```

---

## 4. Nivel 2: Simulación ODE con MLP

Idéntico al caso RBF pero con las derivadas de la MLP:

```python
def mlp_df(h, W1, b1, W2):
    """f̂'(h) = W₂ · diag(σ') · W₁"""
    z = W1 * h + b1
    s = sigmoid(z)
    sp = s * (1 - s)
    return np.dot(W2, sp * W1)

def mlp_d2f(h, W1, b1, W2):
    """f̂''(h) = W₂ · diag(σ'') · W₁²"""
    z = W1 * h + b1
    s = sigmoid(z)
    sp = s * (1 - s)
    spp = sp * (1 - 2*s)
    return np.dot(W2, spp * W1**2)
```

Linealización: k = mlp_df(h_eq), kernel K = exp(-k(t-s)).
Residuo integral, IIR filter, correcciones z₁+z₂ sobre y_k.
200 puntos de simulación.

---

## 5. Salida esperada

### Tabla comparativa MLP vs RBF

```
COMPARACIÓN MLP vs RBF (mismo problema, mismos datos)
======================================================

                        RBF (5 centros)    MLP (8 sigm.)
Parámetros totales      15 (5w+5c+5λ)     25 (8W1+8b1+8W2+1b2)
Parámetros lineales     5                  9
Parámetros no lineales  10                 16
Iteraciones externas    1                  3-5
||N_final||             0.031              ???
Error simulación máx    2.3%               ???
Mejora vs lineal        35×                ???
```

### Figuras a generar

1. `fig_mlp_identification.png` — f_real vs f_MLP vs datos
2. `fig_mlp_simulation.png` — h(t) real vs MLP vs lineal
3. `fig_mlp_residuals.png` — ||N|| por iteración (debe decrecer)
4. `fig_mlp_sigmoids.png` — las 8 sigmoides individuales y su combinación
5. `fig_mlp_vs_rbf.png` — comparación directa MLP vs RBF

---

## 6. Archivo: demo_21paper_mlp.py

Un solo archivo Python (~350 líneas) autocontenido.
Ubicación: `/home/rodo/21Paper/demo_21paper_mlp.py`

### Dependencias

- numpy
- scipy (solo solve_ivp como referencia)
- matplotlib

NO usar PyTorch, TensorFlow, sklearn. Todo desde cero con numpy.

---

## 7. Criterios de éxito

| Criterio | Valor |
|---|---|
| ||N_final|| < ||N_0|| | Reducción > 5× |
| Error simulación máx MLP | < 5% |
| Mejora MLP vs lineal | > 3× |
| Convergencia | ||N|| decrece monótonamente |
| Sin NaN/divergencia | En todo t ∈ [0,10] |

---

## 8. Lo que demuestra esta demo

1. **Mismas fórmulas z₁, z₂, z₃** aplican a MLP, no solo a RBF.
2. **Misma estructura bilineal**: W₂ lineal (LS exacto), W₁,b₁ no lineales (Newton+Halley).
3. **Misma cadena ε → M → N** con tasa de Barron en vez de tasa RBF.
4. **Sin backpropagation, sin learning rate, sin PyTorch.**
5. Los Jacobianos y Hessianos de la MLP son **analíticos** (σ, σ', σ'' en forma cerrada).

---

## 9. Reglas

1. **Self-contained.** No importar nada de demo_21paper.py ni del regressor/.
2. **Solo numpy** para la MLP. Sin frameworks de deep learning.
3. **Derivadas analíticas** de σ, σ', σ''. No autodiff, no diferencias finitas.
4. **Imprimir** parámetros, residuos, errores en cada paso.
5. **Figuras** en `/home/rodo/21Paper/figures/` con prefijo `fig_mlp_`.
6. **float64 siempre.**
7. **Comparar** con resultados de la demo RBF al final.

---

*Contrato demo MLP 21Paper — Abril 2026*
*Rodolfo H. Rodrigo / INAUT-UNSJ-CONICET*
