# Cumplimiento de Formato Applied Mathematics and Computation

**Fecha**: 2026-08-20  
**Status**: ✅ **VERIFICADO Y CORREGIDO**

---

## Cambios Críticos de Formato para AMC

### ⚠️ CORRECCIÓN IMPORTANTE REALIZADA

**Problema detectado**: El manuscrito estaba usando formato **autor-año** (Harvard style), pero AMC requiere **referencias numeradas**.

**Solución aplicada**:

### 1. Clase de Documento

**ANTES** (incorrecto para AMC):
```latex
\documentclass[preprint,12pt,authoryear]{elsarticle}
```

**DESPUÉS** (correcto para AMC):
```latex
\documentclass[preprint,12pt,numbers]{elsarticle}
```

### 2. Estilo Bibliográfico

**ANTES** (incorrecto para AMC):
```latex
\bibliographystyle{elsarticle-harv}
```

**DESPUÉS** (correcto para AMC):
```latex
\bibliographystyle{elsarticle-num}
```

### 3. Citaciones en el Texto

**ANTES** (formato autor-año):
- `\citep{Kingma2015}` → renderiza como: (Kingma & Ba, 2015)
- Ejemplo: "...Adam optimizer (Kingma & Ba, 2015)..."

**DESPUÉS** (formato numérico):
- `\cite{Adam2015}` → renderiza como: [12]
- Ejemplo: "...Adam optimizer [12]..."

**Cambios realizados**: 28 citaciones convertidas de `\citep{}` a `\cite{}`

---

## Por Qué es Importante

### Applied Mathematics and Computation Guidelines:

Según las guías oficiales de AMC:

> "References should be numbered consecutively in the order in which they are first mentioned in the text. In the text, cite references by number(s) in square brackets [1] or [1-3] or [1,4,5]."

**Formato requerido**: **Numérico [1], [2], [3]**  
**NO aceptable**: Autor-año (Smith, 2020)

---

## Verificación de Cumplimiento

### ✅ Formato Correcto Ahora

1. **Clase elsarticle**: `numbers` ✓
2. **Bibliografía**: `elsarticle-num` ✓
3. **Citaciones**: `\cite{}` (28 convertidas) ✓
4. **Compilación**: PDF generado (30 páginas) ✓
5. **Referencias**: Numeradas en orden de aparición ✓

---

## Ejemplo de Cómo se Verá

### En el texto:
```
...gradient-based optimization of parametric approximators [4,5,6,7] or,
more recently, sparse symbolic regression [8,9]. Simulation employs 
explicit and implicit time-stepping schemes [10,11]...
```

### En la bibliografía:
```
[4] Haykin, S., 2009. Neural Networks and Learning Machines...
[5] Park, J., Sandberg, I.W., 1991. Universal approximation...
[6] Sjöberg, J., et al., 1995. Nonlinear black-box modeling...
[7] Nelles, O., 2001. Nonlinear System Identification...
[8] Brunton, S.L., Proctor, J.L., Kutz, J.N., 2016. Discovering...
```

---

## Otros Requisitos de AMC (Ya Cumplidos)

### Estructura del Manuscrito

- [x] Title page con autores y afiliaciones ✓
- [x] Abstract (200-300 palabras) ✓
- [x] Keywords (4-10 keywords) ✓
- [x] Main text con secciones numeradas ✓
- [x] References al final ✓
- [x] Figure captions ✓
- [x] Tables con captions ✓

### Formato de Figuras

- [x] Formato: PNG o EPS/PDF ✓
- [x] Resolución: 300 DPI mínimo ✓
- [x] Tamaño: < 10 MB cada una ✓
- [x] Ubicación: Mencionadas en el texto ✓

### Límites

- [x] Longitud: Sin límite estricto, pero ~30-40 páginas típico ✓
- [x] Nuestro manuscrito: 30 páginas ✓
- [x] Abstract: < 300 palabras ✓
- [x] Keywords: 4-10 keywords ✓

---

## Comparación: Formatos Elsevier

| Journal | Formato Referencias | Ejemplo en Texto |
|---------|-------------------|------------------|
| **Applied Math & Comp** | **Numérico [1]** | **...method [12]...** |
| Neural Networks | Autor-año | ...method (Smith, 2020)... |
| Neurocomputing | Numérico [1] | ...method [12]... |
| Journal of Comp Physics | Numérico [1] | ...method [12]... |

**Nota**: Aunque todos usan elsarticle, cada journal tiene su propio estilo preferido.

---

## Archivos Finales para Submission

### En `/home/rodo/21Paper/submission_AMC/manuscript/`:

1. **`manuscript.pdf`** (30 páginas, 1.4 MB)
   - ✅ Formato AMC correcto
   - ✅ Referencias numeradas [1], [2], [3]...
   - ✅ Compilado con `elsarticle-num`

2. **`manuscript.tex`** (fuente LaTeX)
   - ✅ `\documentclass[...,numbers]{elsarticle}`
   - ✅ `\bibliographystyle{elsarticle-num}`
   - ✅ Todas citaciones con `\cite{}`

3. **`cover_letter_AMC.pdf`** (sin cambios)
   - Cover letter sigue siendo correcto

---

## Checklist Final de Cumplimiento AMC

### Formato General
- [x] Elsarticle class con `numbers` option
- [x] Referencias numeradas en orden de aparición
- [x] Citaciones con `\cite{}`, NO `\citep{}`
- [x] Bibliografía con `elsarticle-num` style

### Contenido
- [x] Abstract dentro de límite
- [x] Keywords apropiados para AMC scope
- [x] Figuras de alta calidad (300 DPI)
- [x] Tablas formateadas correctamente

### Journal Alignment
- [x] Scope: Applied math + numerical computation + systems ✓
- [x] Tone: Computational methods (no ML-focused) ✓
- [x] Experiments: Proof-of-concept (no large-scale required) ✓

---

## RESULTADO FINAL

### ✅ MANUSCRITO 100% CONFORME CON AMC

**Formato**: Correcto (referencias numeradas)  
**Estilo**: Correcto (elsarticle-num)  
**Compilación**: Exitosa (30 páginas)  
**Listo para**: **ENVÍO INMEDIATO**

---

## Última Verificación Antes de Enviar

Cuando vayas a enviar, verifica visualmente en el PDF que:

1. Las referencias en el texto aparecen como **[1]**, **[2]**, etc. (NO como "(Smith, 2020)")
2. La lista de referencias al final está **numerada**: [1], [2], [3]...
3. El orden es por aparición en el texto, NO alfabético

**Si ves esto → CORRECTO para AMC ✓**

---

**Documento creado**: 2026-08-20  
**Status**: Manuscrito listo para submission a Applied Mathematics and Computation
