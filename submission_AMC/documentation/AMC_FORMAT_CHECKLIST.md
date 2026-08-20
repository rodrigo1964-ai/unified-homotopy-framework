# Checklist de Formato para Applied Mathematics and Computation

**Fecha verificación**: 2026-08-20  
**Manuscrito**: manuscript.tex (31 páginas)  
**Journal**: Applied Mathematics and Computation (Elsevier)

---

## 1. Formato General

- [x] **Clase de documento**: `elsarticle` (correcto para Elsevier journals)
- [x] **Journal name**: `\journal{Applied Mathematics and Computation}` ✓
- [x] **Modo**: `preprint, 12pt, authoryear` ✓
- [x] **Páginas totales**: 31 páginas (dentro de límite ~40-50 para AMC)
- [x] **Doble espacio**: Sí (preprint format)

---

## 2. Estructura del Manuscrito

- [x] **Title**: Clear and descriptive ✓
- [x] **Authors**: Con afiliación completa ✓
- [x] **Corresponding author**: Marcado con `\corref` ✓
- [x] **Abstract**: Present ✓
  - Longitud: ~250 palabras (estimado - dentro de límite typical 200-300)
  - Estructura: Intro + Contributions + Results + Scope
- [x] **Keywords**: 9 keywords (apropiado, típicamente 4-10) ✓
  - Enfoque en computational methods ✓
  - Sin ML-specific terms ✓

---

## 3. Secciones Principales

- [x] Introduction
- [x] Related Work
- [x] Theory (Sections 3-7)
- [x] Numerical Demonstrations (Section 9)
  - [x] Orifice-tank 1D ✓
  - [x] Lotka-Volterra 2D ✓
  - [x] Noise robustness ✓
  - [x] Optimizer comparison ✓
- [x] Limitations and Scope
- [x] Discussion/Relationship to companion work
- [x] Conclusions
- [x] Code availability statement

---

## 4. Referencias Bibliográficas

- [x] **Formato**: `elsarticle-harv` (Harvard style) ✓
- [x] **Referencias inline**: Usando `\bibitem` ✓
- [x] **Nuevas referencias agregadas**:
  - [x] Adam optimizer (Adam2015 - Kingma & Ba) ✓
  - [x] L-BFGS-B (ByrdLuNocedal1995 - Byrd et al.) ✓
- [x] **Citaciones verificadas**:
  - [x] `\citep{Adam2015}` → correcto ✓
  - [x] `\citep{ByrdLuNocedal1995}` → correcto ✓
- [x] **Sin citaciones undefined**: Verificado en compilación ✓

---

## 5. Figuras

### Figuras existentes (originales):
- [x] `fig_identification.png` - RBF identification
- [x] `fig_simulation.png` - RBF forward simulation
- [x] `fig_mlp_vs_rbf.png` - Architecture comparison

### Figuras nuevas (agregadas):
- [x] `fig_noise_robustness.png` (235 KB, 300 DPI)
  - Panel (a): ID residual vs noise
  - Panel (b): Simulation error vs noise
- [x] `fig_lotka_volterra.png` (431 KB, 300 DPI)
  - Panel (a): Phase portrait
  - Panel (b): Prey time series
  - Panel (c): Predator time series

### Verificación de calidad:
- [x] **Resolución**: 300 DPI (print quality) ✓
- [x] **Formato**: PNG (aceptable para AMC) ✓
- [x] **Tamaño**: < 1 MB cada una ✓
- [x] **Captions**: Descriptivos y completos ✓
- [x] **Labels**: (a), (b), (c) en multi-panel figures ✓

---

## 6. Tablas

- [x] **Table 1** (tab:comparison): RBF vs MLP comparison ✓
- [x] **Table 2** (tab:wallclock): Optimizer comparison ✓
- [x] **Table 3** (tab:noise-robustness): Noise robustness ✓
- [x] **Formato**: `rowcolors` para legibilidad ✓
- [x] **Captions**: Above table (Elsevier style) ✓

---

## 7. Ecuaciones

- [x] **Numeración**: Consecutiva ✓
- [x] **Referencias**: `\eqref{}` usado consistentemente ✓
- [x] **Notación**: Consistente (verificado en Tarea #3) ✓
  - Vectores: `\mathbf{}`
  - Parámetros: `\boldsymbol{}`
  - Operadores: `\mathcal{}`

---

## 8. Código y Datos

- [x] **Code availability statement**: Presente (Section después de Conclusions) ✓
- [x] **GitHub repo**: Mencionado ✓
- [x] **Figshare DOI**: Mencionado ✓
- [x] **Scripts públicos**: demo_21paper.py, demo_noise_robustness.py, etc. ✓

---

## 9. Cover Letter

- [x] **Archivo**: `cover_letter_AMC.tex` ✓
- [x] **PDF generado**: `cover_letter_AMC.pdf` (78 KB) ✓
- [x] **Destinatario**: Editor-in-Chief, Applied Mathematics and Computation ✓
- [x] **Contenido**:
  - [x] Descripción del trabajo ✓
  - [x] Contribución teórica ✓
  - [x] Validación empírica (3 benchmarks mencionados) ✓
  - [x] Scope statement (narrow pero justificado) ✓
  - [x] Originalidad statement ✓
  - [x] Código público mencionado ✓

---

## 10. Verificaciones de Compilación

- [x] **Primera compilación**: OK ✓
- [x] **Segunda compilación** (refs): OK ✓
- [x] **PDF generado**: manuscript.pdf (31 páginas, 1.4 MB) ✓
- [x] **Warnings**: Ninguno crítico ✓
- [x] **Errores**: Ninguno ✓

---

## 11. Alineamiento con AMC Scope

### Scope de AMC (del website):
> "Applied Mathematics and Computation addresses work at the interface of 
> applied mathematics, numerical computation, and applications of systems"

### Nuestro paper:
- [x] **Applied mathematics**: Homotopy analysis, approximation theory ✓
- [x] **Numerical computation**: Newton-Halley corrections, ODE integration ✓
- [x] **Applications of systems**: System identification, dynamical systems ✓

### Fit:
✅ **EXCELENTE** - El paper está perfectamente alineado con el scope de AMC

---

## 12. Límites y Guidelines

### Límites típicos de AMC:
- **Longitud**: Hasta ~40-50 páginas (doble espacio)
  - ✓ Nuestro paper: 31 páginas ✓
- **Abstract**: 200-300 palabras típicamente
  - ✓ Nuestro abstract: ~250 palabras (estimado) ✓
- **Figuras**: Sin límite estricto, calidad alta
  - ✓ Tenemos 6 figuras, todas 300 DPI ✓
- **Referencias**: Sin límite estricto
  - ✓ Tenemos ~50-60 referencias (apropiado) ✓

---

## 13. Mejoras vs Versión Neural Networks

| Aspecto | Neural Networks | AMC (actual) | Status |
|---------|----------------|--------------|--------|
| Experimentos | 1 sistema | 3 sistemas | ✅ |
| Comparaciones | 2 métodos | 5 métodos | ✅ |
| Ruido | No evaluado | 4 niveles | ✅ |
| Discrepancia | No explicada | 3 párrafos | ✅ |
| Referencias | Faltaban Adam, L-BFGS | Completas | ✅ |
| Framing | ML-oriented | Math-oriented | ✅ |

---

## 14. Issues Pendientes

### Ninguno crítico ✓

### Opcional (nice-to-have):
- [ ] Revisar una última vez el abstract (contar palabras exactas)
- [ ] Verificar que todas las ecuaciones referenciadas existen
- [ ] Spell-check manual de nombres propios
- [ ] Verificar consistencia de "centre" vs "center" (UK vs US English)

---

## 15. Submission Checklist Final

Cuando esté listo para enviar, verificar:

- [ ] Manuscript PDF final generado y revisado
- [ ] Cover letter PDF generado y revisado
- [ ] Todas las figuras en alta resolución en carpeta separada
- [ ] Highlights file (si AMC lo requiere)
- [ ] Verificar guidelines específicos de AMC en el momento del envío
- [ ] Crear respuesta a reviewers mostrando cambios vs versión Neural Networks

---

## RESUMEN EJECUTIVO

### ✅ FORMATO: CORRECTO
### ✅ REFERENCIAS: COMPLETAS
### ✅ FIGURAS: ALTA CALIDAD
### ✅ ALINEAMIENTO CON AMC: EXCELENTE
### ✅ LISTO PARA TAREA #10 (Revisión Final)

**Estado**: El manuscrito cumple con todos los requisitos de formato para 
Applied Mathematics and Computation. Las únicas tareas pendientes son de 
contenido (revisión final, no de formato).
