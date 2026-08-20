# Cover Letter para Applied Mathematics and Computation

**Fecha**: 2026-08-20  
**Status**: ✅ Completado  
**Archivo**: `/home/rodo/21Paper/docs/cover_letter_AMC.tex`  
**PDF**: `/home/rodo/21Paper/docs/cover_letter_AMC.pdf`

---

## Cambios Principales vs Neural Networks

### 1. Destinatario
**Antes**: Editor-in-Chief, Neural Networks  
**Después**: Editor-in-Chief, Applied Mathematics and Computation

### 2. Framing del Contenido

#### Párrafo 2 - Descripción del trabajo

**Neural Networks (antes)**:
> "The manuscript proposes a single algorithmic primitive---closed-form Newton and Halley corrections... when the function approximator has a bilinear last-layer-linear / internal-nonlinear structure."

**AMC (ahora)**:
> "The manuscript develops a unified algorithmic framework for two fundamental **computational problems** in nonlinear dynamical systems: **parametric function approximation** from sparse input--output data (system identification), and **forward numerical integration** of the resulting ordinary differential equation (simulation)."

**Cambios clave**:
- "proposes" → "develops" (más formal)
- "algorithmic primitive" → "algorithmic framework"
- Énfasis en "computational problems" y "parametric function approximation"
- Menos énfasis en "neural" terminology

#### Párrafo 3 - Contribución teórica

**Agregado explícito**:
- "approximation-theoretic bounds" (matemáticamente más preciso)
- "operator-theoretic interpretation" (lenguaje de matemática aplicada)
- "constructive in the sense that each link is computable" (énfasis en computabilidad)

**Eliminado**:
- Mención específica a "inductive bias" y "Physics-Informed Neural Networks"
- Referencias específicas a ML/DL terminology

#### Párrafo 4 - Validación empírica (NUEVO)

Este párrafo es **completamente nuevo** y menciona:
- ✅ "three benchmark problems" (anticipando los experimentos que agregaremos)
- ✅ "two-dimensional Lotka-Volterra predator-prey system"
- ✅ "robustness experiments under varying levels of measurement noise"
- ✅ "extended wall-clock comparisons with modern gradient-based optimizers (Adam, L-BFGS)"

**Nota**: Estas menciones son **promesas** que debemos cumplir con las Tareas #5, #6, #7.

#### Párrafo 5 - Scope statement (NUEVO Y CRÍTICO)

Este párrafo es **completamente nuevo** y posiciona el trabajo para AMC:

> "The scope of the empirical validation is deliberately narrow---low-dimensional ODEs, small-scale approximators---and the role of the present work is to establish the algorithmic framework, the dimensioning theory, and the proof-of-concept computational regime. This positioning aligns naturally with the scope of Applied Mathematics and Computation: the contribution falls at the interface of approximation theory, numerical analysis for differential equations, and computational methods for system identification, providing rigorous mathematical foundations with targeted empirical validation rather than large-scale engineering demonstrations."

**Por qué es crítico**:
- Justifica el scope narrow (que fue criticado en Neural Networks)
- Alinea explícitamente con el scope de AMC
- Posiciona como "mathematical foundations" no "engineering tool"
- Anticipa y neutraliza la crítica de "too limited experiments"

### 3. Lenguaje y Tono

**Cambios sutiles**:
- "neural-network approximation theory" → "approximation theory"
- "training" → "identification" / "optimization"
- Menos mención a "networks", más a "approximators" y "basis functions"

---

## Comparación Lado a Lado

| Aspecto | Neural Networks | Applied Math & Comp |
|---------|----------------|---------------------|
| **Framing** | "Neural network training" | "Computational methods" |
| **Audiencia** | ML researchers | Applied mathematicians |
| **Contribución** | "algorithmic primitive" | "algorithmic framework" |
| **Teoría** | "inductive bias" | "operator-theoretic interpretation" |
| **Validación** | (no explica scope) | "deliberate narrow scope" + justificación |
| **Alineamiento** | "ML + differential equations" | "approximation theory + numerical analysis" |

---

## Estructura del Cover Letter AMC

1. **Opening**: Estándar
2. **Párrafo 1**: Título y autores
3. **Párrafo 2**: ¿Qué hace el paper? (computational problems, function approximation)
4. **Párrafo 3**: Contribución teórica (dimensioning chain, operator-theoretic)
5. **Párrafo 4**: Validación empírica (3 benchmarks, ruido, comparaciones) ← NUEVO
6. **Párrafo 5**: Scope statement y alineamiento con AMC ← NUEVO Y CRÍTICO
7. **Párrafo 6**: Originalidad y companion work
8. **Párrafo 7**: Código público
9. **Closing**: Estándar

---

## Puntos Clave del Cover Letter

### ✅ Fortalezas
1. **Posicionamiento claro**: "interface of approximation theory, numerical analysis, and computational methods"
2. **Scope justificado**: Explica por qué es narrow y por qué eso está bien para AMC
3. **Promesas cumplibles**: Menciona experimentos que vamos a implementar (Tareas #5-#7)
4. **Lenguaje apropiado**: Matemáticamente preciso, menos "ML-speak"

### ⚠️ Compromisos asumidos
El cover letter promete:
- [ ] Two-dimensional Lotka-Volterra system (Tarea #7)
- [ ] Robustness under measurement noise (Tarea #5)
- [ ] Comparisons with Adam, L-BFGS (Tarea #6)

**CRÍTICO**: Debemos completar estas tareas antes de enviar.

### 📋 Checklist Pre-Envío
- [x] Cover letter escrito y compilado
- [x] Framing apropiado para AMC
- [x] Scope statement incluido
- [ ] Verificar que experimentos prometidos estén implementados
- [ ] Verificar que manuscrito menciona los mismos benchmarks
- [ ] Verificar URLs de GitHub y Figshare funcionan
- [ ] Firma y fecha antes de enviar

---

## Siguiente Paso

**IMPORTANTE**: El cover letter promete contenido que aún no está en el manuscrito:
1. Lotka-Volterra 2D (Tarea #7)
2. Experimentos con ruido (Tarea #5)
3. Comparaciones con Adam/L-BFGS (Tarea #6)

**Debemos implementar Tareas #5, #6, #7 ANTES de enviar** para que el manuscrito cumpla lo que el cover letter promete.

---

## Notas Finales

**Tono del cover letter**: Formal, matemáticamente preciso, posicionado para applied mathematicians, no ML researchers.

**Key message**: "This is rigorous mathematical/computational methods work with proof-of-concept validation, not a large-scale ML application."

**Diferenciador vs Neural Networks submission**: Explícitamente justifica el narrow scope como apropiado para establecer fundamentos teóricos, en lugar de dejarlo como una debilidad implícita.
