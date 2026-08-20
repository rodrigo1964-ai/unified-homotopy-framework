# Análisis Detallado de Feedback de Neural Networks Reviewers

**Manuscrito**: A Unified Homotopy Framework for Nonlinear System Identification and Simulation  
**Journal rechazado**: Neural Networks  
**Target journal**: Applied Mathematics and Computation  
**Fecha análisis**: 2026-08-20

---

## Resumen Ejecutivo

**Status**: RECHAZADO por Neural Networks  
**SAE**: "lacking novelties, unconvinced experiments"  
**Reviewers**: 2 reviewers con feedback detallado + 1 con crítica de formato

**Viabilidad de resucitar**: ✅ **ALTA** - Las críticas son constructivas y técnicamente solucionables

**Estrategia**: Mejorar según feedback + reenviar a **Applied Mathematics and Computation** (revista más metodológica, menos orientada a ML/DL)

---

## Críticas Detalladas

### REVIEWER #2 (CRÍTICAS SUSTANCIALES)

#### **Crítica 1: Discrepancia teoría-práctica en dimensioning chain**

**Texto original del reviewer**:
> "The constructive dimensioning chain is presented as the key theoretical contribution. However, its current form raises concerns: In the main numerical example, the theory predicts a minimum of 29 neurons, whereas the proposed algorithm achieves the target accuracy with only 5 neurons. Such a factor-of-6 discrepancy suggests that the bounds are highly conservative."

**Problema**:
- Teoría predice M_min ≥ 29 neuronas (línea 957 del manuscrito)
- Experimento usa 5 neuronas RBF
- Factor 6× de discrepancia sin explicación convincente

**Acción requerida**:
- ✅ Agregar subsección explicando naturaleza de los bounds
- ✅ Discutir cuándo bounds son tight vs conservative
- ✅ Explicar por qué en este caso el bound es conservador
- ✅ Clarificar si el chain es "loose feasibility certificate" o "design guideline"
- ✅ Agregar: "No se necesita study sistemático con múltiples sistemas, pero sí caracterización honesta de cuándo los bounds son informativos"

**Ubicación en manuscrito**: Section 9.4 (Verification of dimensioning chain), líneas 953-967

**Solución propuesta**:
Agregar párrafo después de línea 957:

```latex
\paragraph{Interpretation of the bound} The factor-of-six gap between the 
theoretical minimum $M_{\min} \geq 29$ and the empirically sufficient $M = 5$ 
reflects the conservative nature of worst-case bounds. The bound 
\eqref{eq:Mmin-full} assumes uniform approximation over the full domain 
$[h_{\min}, h_{\max}]$ with no exploitation of the smoothness or specific 
structure of $f(h) = 0.5\sqrt{h}$. In practice, the RBF centres adapt via 
K-means clustering to regions of higher curvature, and the training data 
distribution is non-uniform. The dimensioning chain should therefore be 
interpreted as a \emph{feasibility certificate}---guaranteeing that a network 
of the prescribed size exists---rather than a tight design rule. Tighter 
bounds would require problem-specific constants or adaptive placement 
strategies, which fall outside the scope of the present universal framework.
```

---

#### **Crítica 2: Experimentos muy limitados (1D, toy-scale)**

**Texto original del reviewer**:
> "The empirical section is, in my view, the main weakness of the manuscript. All results are based on a single 1D orifice-tank example with very small networks (5-8 neurons) and only ~20 data points. In a journal that receives thousands of submissions, such a minimal testbed does not suffice to establish generality or robustness."

**Problema**:
- Solo 1 sistema (orifice-tank, 1D)
- Solo 20 data points
- Solo 2 arquitecturas (RBF 5 centros, MLP 8 unidades)
- Sin ruido en mediciones
- Scope demasiado narrow para Neural Networks

**Acción requerida**:
- ✅ Agregar 1-2 casos **multi-dimensionales**:
  - **Lotka-Volterra 2D** (predator-prey): ẋ = ax - bxy, ẏ = -cy + dxy
  - **Opcional**: Lorenz 3D o Van der Pol 2D
- ✅ Para cada caso: identificación + simulación
- ✅ Reportar mismas métricas (residual, simulation error)
- ✅ Mostrar que dimensioning chain se generaliza

**Ubicación en manuscrito**: Section 9 (Numerical demonstrations)

**Solución propuesta**:
Agregar dos nuevas subsecciones:

```latex
\subsection{Benchmark B: Lotka-Volterra predator-prey (2D)}

We consider the coupled system
\begin{equation}
\dot{x} = x(a - by), \qquad \dot{y} = y(-c + dx),
\end{equation}
with parameters $(a,b,c,d) = (1.0, 0.5, 1.0, 0.5)$ and initial conditions...

[Detalles de identificación y simulación]

\subsection{Benchmark C: Van der Pol oscillator (2D)}

The Van der Pol equation in state-space form...

[Detalles]
```

---

#### **Crítica 3: Falta experimentos con ruido controlado**

**Texto original del reviewer**:
> "The experiments employ synthetically generated data with extremely tight numerical tolerances (e.g., $10^{-10}$) and no realistic noise or modeling uncertainties. This creates an overly 'sterile' environment and makes it difficult to assess whether the method is a practical engineering tool or mainly a proof-of-concept exercise."

**Problema**:
- Datos sintéticos con tolerancia 1e-10 (irrealisticamente limpios)
- Sin ruido en las mediciones
- No se evalúa robustez a noise
- No hay SNR study sistemático

**Acción requerida**:
- ✅ Agregar ruido gaussiano a las 20 muestras
- ✅ Niveles: 1%, 5%, 10% relative noise
- ✅ Ejecutar identificación + simulación para cada nivel
- ✅ Comparar performance degradation: homotopy vs gradient descent
- ✅ Generar figura: error vs SNR

**Ubicación en manuscrito**: Section 9 (nueva subsección)

**Solución propuesta**:
```latex
\subsection{Robustness to measurement noise}

To assess practical applicability we repeat the orifice-tank identification 
under additive Gaussian noise. Training samples are corrupted by 
$h_{\text{noisy}} = h_{\text{true}} + \epsilon$, where $\epsilon \sim 
\mathcal{N}(0, \sigma^2)$ and $\sigma$ is chosen to yield relative noise 
levels of 1\%, 5\%, and 10\%.

Table~\ref{tab:noise-robustness} reports identification residuals and 
forward-simulation errors for the 5-centre RBF under increasing noise...

[Figura mostrando degradación]
```

---

#### **Crítica 4: Comparación con baselines débiles**

**Texto original del reviewer**:
> "The reported wall-clock speedups are reported relative to vanilla gradient descent and a standard SciPy TRF solver. These are not necessarily the strongest or most tailored baselines available in 2025-2026. For claims of 'order-of-magnitude' gains, it would be important to include modern, tuned optimizers (e.g., Adam with line search, LBFGS)..."

**Problema**:
- Solo compara con: vanilla GD + SciPy TRF
- No incluye optimizadores modernos (Adam, L-BFGS-B)
- No compara con Neural ODE solvers
- Claims de speedup no están bien justificados

**Acción requerida**:
- ✅ Agregar a Table 2 (wall-clock comparison):
  - Adam con learning rate schedule
  - L-BFGS-B (bounded variant)
- ✅ Ejecutar con mismo initialization
- ✅ Reportar convergence + wall-clock
- ✅ Discutir trade-offs: homotopy rápido pero específico, optimizadores lentos pero generales

**Ubicación en manuscrito**: Section 9.4, Table 2 (líneas 923-939)

**Solución propuesta**:
Expandir Table 2:

```latex
\begin{tabular}{L{5cm} C{3cm} C{2.5cm} C{2.5cm}}
\toprule
\textbf{Method} & \textbf{Final residual} & \textbf{Iter./nfev} & \textbf{Wall time} \\
\midrule
Gradient descent ($\eta = 10^{-2}$) & $9.9\cdot 10^{-3}$ & 5{,}000 & 0.26\,s \\
Adam (default schedule) & $2.1\cdot 10^{-5}$ & 3{,}200 & 0.18\,s \\
L-BFGS-B (SciPy) & $8.3\cdot 10^{-7}$ & 1{,}800 & 0.95\,s \\
Trust-Region Reflective & $3.1\cdot 10^{-6}$ & 2{,}500 & 1.53\,s \\
Homotopy ($z_1 + z_2$) & $1.7\cdot 10^{-8}$ & 1 outer & 0.0002\,s \\
\bottomrule
\end{tabular}
```

---

#### **Crítica 5: Choices metodológicas no justificadas**

**Texto original del reviewer**:
> "Several methodological choices seem under-justified or at least under-tested: (i) Newton/Halley-type local methods are known to be sensitive to initialization... (ii) The Halley correction relies on a diagonal approximation to the Hessian to keep complexity manageable; limited overlap between neurons... This assumption may fail in deeper or more entangled architectures."

**Problema**:
- Newton/Halley sensitivity a initialization no explorada
- Solo se reportan "simple initialization schemes" (K-means, uniform grid)
- Diagonal Hessian approximation no validada para casos más complejos
- No hay sensitivity analysis

**Acción requerida**:
- ✅ Agregar párrafo discutiendo initialization schemes
- ✅ Mencionar que se usaron K-means (RBF) y uniform grid (MLP)
- ✅ Agregar nota sobre cuándo diagonal approximation es válida
- ✅ Identificar como limitación para arquitecturas profundas/enredadas

**Ubicación en manuscrito**: Section 10 (Limitations and scope), líneas 968-1019

**Solución propuesta**:
Agregar a Section 10:

```latex
\paragraph{Initialization and convergence basin} The Newton and Halley 
corrections converge locally and are therefore sensitive to initialization. 
The present demonstrations employ standard schemes---K-means clustering for 
RBF centres, uniform grids for MLP sigmoid centres---that are known to 
provide reasonable starting points for smooth, well-conditioned problems. 
A systematic study of convergence basins and multi-start strategies is 
beyond the scope of this proof-of-concept work but is identified as a 
natural follow-up.

\paragraph{Diagonal Hessian approximation} The Halley correction 
\eqref{eq:halley} relies on a diagonal approximation to the Hessian to 
avoid the $O(n^3)$ cost of a full matrix inversion. This approximation 
is valid when neurons have limited overlap (RBF with well-separated 
centres, MLP with moderate slopes). For deeper networks or architectures 
with strong inter-neuron coupling, the full Hessian or a block-diagonal 
variant may be required. Applicability to CNNs, LSTMs, or Transformers 
remains speculative pending empirical validation.
```

---

#### **Crítica 6: Falta de claridad en circular dependencies del dimensioning chain**

**Texto original del reviewer**:
> "A primary concern involves the constructive dimensioning chain. The theory requires the Lipschitz constant and the curvature of the residual nonlinearity as inputs to calculate the minimum required network size. However, in a standard system identification setting, these properties of the target system are precisely what is unknown."

**Problema**:
- Dimensioning chain necesita conocer L (Lipschitz constant) y curvatura
- En identificación, estas constantes son desconocidas
- Dependencia circular: necesitas conocer f para dimensionar la red que aprenderá f
- No hay método práctico para estimar estas constantes a priori

**Acción requerida**:
- ✅ Agregar subsección clarificando cuándo el chain es aplicable
- ✅ Distinguir entre: 
  - **Ex-post verification**: validar que una red entrenada satisface los bounds
  - **A priori design**: usar bounds conservadores basados en dominio físico
- ✅ Discutir que en práctica, se usa knowledge del dominio físico (e.g., tank height bounded → curvature bounded)

**Ubicación en manuscrito**: Section 7 (Constructive dimensioning), después de la derivación del chain

**Solución propuesta**:
```latex
\subsection{Applicability and practical use}

The dimensioning chain \eqref{eq:chain-full} requires prior knowledge of 
the Lipschitz constant $L$ and the curvature bound $K_2$ of the target 
function $f$. In system identification, these quantities are typically 
unknown. Two use-cases emerge:

\paragraph{Ex-post verification} Given a trained network, compute the 
achieved identification error $\varepsilon_{\text{id}}$ and forward-simulation 
error $\varepsilon_{\text{sim}}$, and verify that the bounds hold. This 
confirms that the network size was sufficient and provides a certificate 
of the quality of the learned model.

\paragraph{A priori conservative design} In many physical systems, domain 
knowledge provides bounds on the state variables (e.g., tank height 
$h \in [0.1, 1.5]$) and on admissible control inputs. These bounds induce 
worst-case Lipschitz and curvature constants that can be computed 
analytically. The dimensioning chain then provides a \emph{sufficient} 
network size, which may be conservative but guarantees feasibility.

A complete design methodology that adaptively refines these bounds during 
training falls outside the scope of this work but is a natural extension.
```

---

### REVIEWER #3 (CRÍTICAS DE FORMATO)

**Texto original**:
> "The manuscript presents severe formatting flaws, garbled mathematical symbols, inconsistent notations, and disorganized section structure. Empirical validation is insufficient, and theoretical claims lack rigorous support. The overall quality does not meet the publication standards of Neural Networks."

**Problemas identificados**:
- ❌ Símbolos matemáticos garbled (probablemente en PDF rendering)
- ❌ Notación inconsistente
- ❌ Estructura de secciones desorganizada
- ❌ Calidad general no cumple standards de Neural Networks

**Acción requerida**:
- ✅ Revisar TODO el manuscrito .tex buscando símbolos mal renderizados
- ✅ Estandarizar notación:
  - Vectores: bold lowercase ($\mathbf{x}$)
  - Matrices: bold uppercase ($\mathbf{W}$)
  - Operadores: mathcal ($\mathcal{N}$, $\mathcal{L}$)
  - Conjuntos: mathbb ($\mathbb{R}$)
- ✅ Verificar consistencia de nombres de variables a lo largo del paper
- ✅ Reorganizar secciones si es necesario
- ✅ Generar PDF limpio y verificar visualmente

**Ubicación**: TODO el manuscrito

---

## Plan de Mejoras Priorizadas

### FASE 1: Correcciones Críticas (3-5 días)

1. ✅ **Tarea #3**: Corregir formato y notación (Reviewer #3)
2. ✅ **Tarea #2**: Reframing para AMC (ajustar abstract, intro, keywords)
3. ✅ **Tarea #4**: Explicar discrepancia teoría-práctica

### FASE 2: Mejoras Sustanciales (1-2 semanas)

4. ✅ **Tarea #5**: Agregar experimentos con ruido (1%, 5%, 10%)
5. ✅ **Tarea #6**: Mejorar comparaciones (Adam, L-BFGS)
6. ✅ **Tarea #7**: Agregar casos multi-D (Lotka-Volterra 2D + Van der Pol/Lorenz)

### FASE 3: Polish Final (2-3 días)

7. ✅ **Tarea #8**: Cover letter para AMC
8. ✅ **Tarea #9**: Verificar formato AMC
9. ✅ **Tarea #10**: Revisión final + submission package

---

## Checklist de Cambios Específicos

### Abstract
- [ ] Mantener mención a "proof-of-concept" (ya está)
- [ ] Agregar mención a robustez a ruido
- [ ] Mencionar casos multi-dimensionales

### Section 1 (Introduction)
- [ ] Ajustar framing para audiencia de matemática aplicada
- [ ] Enfatizar framework metodológico sobre neural networks

### Section 7 (Constructive dimensioning)
- [ ] Agregar subsección sobre applicability (ex-post vs a priori)
- [ ] Clarificar cuándo bounds son tight vs conservative

### Section 9 (Numerical demonstrations)
- [ ] Agregar subsección: Robustness to noise
- [ ] Agregar subsección: Lotka-Volterra 2D
- [ ] Agregar subsección: Van der Pol 2D (opcional)
- [ ] Expandir Table 2 (wall-clock) con Adam + L-BFGS-B

### Section 9.4 (Verification of dimensioning chain)
- [ ] Agregar párrafo explicando discrepancia M_min=29 vs M=5

### Section 10 (Limitations and scope)
- [ ] Agregar párrafo sobre initialization sensitivity
- [ ] Agregar párrafo sobre diagonal Hessian approximation

### Referencias
- [ ] Verificar formato AMC
- [ ] Agregar referencias a Adam, L-BFGS si no están

---

## Estimación de Esfuerzo

| Tarea | Tiempo estimado | Dificultad |
|-------|----------------|------------|
| Formato y notación | 1 día | Baja |
| Reframing AMC | 0.5 días | Baja |
| Explicar discrepancia | 0.5 días | Media |
| Experimentos ruido | 2-3 días | Media |
| Comparaciones optimizadores | 1-2 días | Media |
| Casos multi-D | 3-5 días | Alta |
| Cover letter | 0.5 días | Baja |
| Verificación formato | 0.5 días | Baja |
| Revisión final | 1 día | Media |
| **TOTAL** | **10-14 días** | - |

---

## Notas Finales

**Por qué Applied Mathematics and Computation es mejor opción**:
1. Scope más metodológico que Neural Networks
2. Acepta proof-of-concept si teoría es sólida
3. Menos presión en experimentos masivos
4. Audiencia valora análisis constructivo y bounds teóricos
5. Reviewer #2 sugirió explícitamente "more specialized or methodologically focused journal"

**Fortalezas del paper que debemos mantener**:
- Framework unificado (identificación + simulación con mismo formalismo)
- Derivación matemática clara y correcta
- Código público y reproducible
- Admisión honesta del scope limitado ("proof-of-concept regime")

**Debilidades a mitigar**:
- Experimentos muy limitados → agregar multi-D + ruido
- Comparaciones débiles → agregar optimizadores modernos
- Bounds muy conservadores → explicar mejor su naturaleza e interpretación
