# CLAUDE.md — 21Paper: Unified Homotopy Framework

## Proyecto
**21Paper:** A Unified Homotopy Framework for Nonlinear System Identification and Simulation: Constructive Dimensioning of Bilinear Neural Architectures

## Autores
Rodolfo H. Rodrigo (INAUT/UNSJ-CONICET), H. Daniel Patino (INAUT/UNSJ-CONICET)

## Target venue
Journal submission (venue pendiente de confirmacion)

## Estado
Codigo implementado y funcional, demos publicados en Figshare (DOI 10.6084/m9.figshare.31955865)

---

## CONTEXTO ACADEMICO

### Idea central
Marco algoritmico unificado derivado de la ecuacion de deformacion homotoica de Liao que resuelve dos problemas tradicionalmente tratados por metodos separados:

1. **Identificacion** de aproximador neuronal (RBF, MLP) desde datos
2. **Simulacion** de la ODE no lineal resultante

Ambos problemas se reducen a correcciones Newton (z1) y Halley (z2) en forma cerrada, explotando la estructura bilineal compartida: ultima capa lineal en pesos, parametros internos no lineales.

### Cadena de dimensionamiento constructivo

```
ε_sim → ε_id → M_min → N_min → T_max
```

Mapea precision objetivo de simulacion al numero minimo de neuronas, numero minimo de muestras de entrenamiento, y maximo paso de integracion admisible.

### Resultado experimental central
Para la ODE de descarga-flujo `dh/dt + 0.5·sqrt(h) = u(t)` con 20 datos sinteticos:

- **RBF (5 Gaussianas):** error de simulacion maximo 2.3% en 200 pasos, residual de identificacion 3.1e-2, 1 iteracion externa, mejora 35x sobre baseline linearizado
- **MLP (8 sigmoides):** error de simulacion 0.96%, residual 1.7e-8 (precision de maquina), 1 iteracion externa, mejora 8.1x sobre baseline

---

## ESTRUCTURA DE ARCHIVOS

```
/home/rodo/21Paper/
├── CLAUDE.md                          # Este contrato
├── README.md                          # Documentacion publica
├── LICENSE                            # MIT
├── CITATION.cff                       # Metadata de citacion
├── requirements.txt                   # numpy, scipy, matplotlib
│
├── demo_21paper.py                    # Demo RBF
├── demo_21paper_mlp.py                # Demo MLP
│
├── benchmark_wallclock.py             # Comparacion wall-clock vs otros metodos
├── bench_wallclock.log                # Resultados timing
├── mlp_run.log                        # Log de ejecucion MLP
├── RESULTS_mlp_demo.md                # Resultados documentados
│
├── docs/                              # Documentacion adicional
├── figures/                           # Figuras generadas
├── tests/                             # Tests unitarios
├── dev/                               # Desarrollo (no publicado)
└── _dev/                              # Development artifacts
    ├── CLAUDE_demo.md
    └── CLAUDE_mlp_demo.md
```

---

## METODO

### Residual bilineal

Para ambas arquitecturas el residual de identificacion tiene la forma:

```
N(theta) = Phi(eta) · w - f
```

donde:
- `w`: parametros lineales (pesos ultima capa)
- `eta`: parametros no lineales (centros y anchos RBF, o pesos/bias ocultos MLP)

### Correcciones homotoicas

- **z1 (Newton):** correccion primer orden usando Jacobiano del residual
- **z2 (Halley):** correccion segundo orden usando diagonal del Hessiano de parametros
- **z3 (opcional):** tercer orden, no usado en demos

El subproblema lineal se resuelve exactamente por minimos cuadrados (z1 exacto, z2=z3=0). El subproblema no lineal se corrige con z1+z2. No hay learning rate escalar ni gradiente descendente estocastico; la magnitud de cada correccion esta determinada por la geometria local del residual.

---

## BENCHMARKS

### Comparacion wall-clock (MLP, 20 datos, 1 core CPU)

Mediana sobre 5 corridas, inicializacion identica, sin warm start.

| Metodo | Residual final | Iter/nfev | Wall time |
|--------|----------------|-----------|-----------|
| Homotopy z1+z2 | 1.7e-8 | 1/1 | ~10 ms |
| scipy.optimize.least_squares | 2.1e-8 | ~15/~60 | ~50 ms |
| scipy.optimize.minimize (L-BFGS-B) | 3.4e-5 | ~30/~90 | ~80 ms |

(Datos aproximados, ver `bench_wallclock.log` para resultados exactos)

---

## DEPENDENCIAS

```
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.3
```

Python 3.8, 3.10, 3.12 testeados.

---

## DEMOS

### demo_21paper.py — RBF

Identifica `f(h) = 0.5·sqrt(h)` con 5 centros Gaussianos desde 20 datos ruidosos. Luego forward-simula la ODE sobre 200 pasos.

**Ejecutar:**
```bash
python demo_21paper.py
```

**Salida esperada:**
- Figura con identificacion y simulacion
- Metricas: error maximo, residual, iteraciones

### demo_21paper_mlp.py — MLP

Identifica `f(h) = 0.5·sqrt(h)` con MLP de 8 sigmoides. Alcanza precision de maquina en 1 iteracion.

**Ejecutar:**
```bash
python demo_21paper_mlp.py
```

**Salida esperada:**
- Figura con identificacion y simulacion
- Convergencia a 1.7e-8 residual

---

## RELACION CON OTROS PAPERS

### Papers previos (fundamento teorico)
- **10Paper:** HAM continuo de Liao, bases teoricas
- **20Paper:** Regresores integrales, formulacion de Volterra

### Papers posteriores (aplicaciones)
- **22Paper:** Mixed-model homotopy regressors, bases neuronales multiples
- **23Paper:** Bases holonomicas via hipergeometrica confluente 1F1
- **40Paper:** dHAM (Discrete HAM), formulacion BDF2 con z1+z2+z3

Este paper establece el puente entre la teoria HAM continua y las implementaciones discretas practicas para identificacion neuronal.

---

## FIGSHARE

Dataset publicado en Figshare:
**DOI:** [10.6084/m9.figshare.31955865](https://doi.org/10.6084/m9.figshare.31955865)

Contiene:
- Codigo fuente de demos
- Resultados numericos
- Figuras reproducibles
- Metadata ORCID

---

## TESTING

```bash
cd tests/
pytest -v
```

Tests cubren:
- Convergencia de z1+z2 para RBF y MLP
- Verificacion contra scipy.integrate.solve_ivp (RK45)
- Estabilidad numerica de las correcciones

---

## ORCID

**Rodolfo H. Rodrigo:** [0000-0002-8787-0038](https://orcid.org/0000-0002-8787-0038)

---

## ESTADO ACTUAL

- [x] Implementacion de framework unificado
- [x] Demo RBF funcional
- [x] Demo MLP funcional
- [x] Benchmarks wall-clock vs scipy
- [x] Publicacion en Figshare
- [x] Tests unitarios
- [ ] Manuscrito (en preparacion)
- [ ] Submission a journal

---

## NOTAS TECNICAS

### Bilinealidad

La estructura bilineal es clave: el residual es lineal en `w` dado `eta`, y no lineal en `eta`. Esto permite:

1. Resolver exactamente el subproblema lineal en cada iteracion (sin iteracion interna)
2. Aplicar correcciones geometricas (Halley) solo sobre parametros no lineales
3. Convergencia en pocas iteraciones externas (tipicamente 1-3)

### Operador lineal auxiliar

La eleccion del operador `L` actua como sesgo inductivo explicito y cuantificable. En este framework:
- Para identificacion: `L = I` (identidad)
- Para simulacion: `L = d/dt` (derivada temporal)

La eleccion de `L` determina las condiciones de convergencia y el numero minimo de neuronas `M_min`.

---

*Contrato v1.0 — Junio 2026*
*Rodolfo H. Rodrigo / INAUT-UNSJ-CONICET*
