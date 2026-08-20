# Submission Package for Applied Mathematics and Computation

**Date**: August 20, 2026  
**Journal**: Applied Mathematics and Computation (Elsevier)  
**Submission Type**: **NEW SUBMISSION** (not a revision)

---

## Important Note on Submission History

This is a **completely new submission** to a different journal:

- **Previous submission**: Neural Networks (Elsevier) - **REJECTED** (NEUNET-D-26-03166)
- **Current submission**: Applied Mathematics and Computation (Elsevier) - **NEW**

This is NOT a revision to Neural Networks. It is a fresh submission to Applied Mathematics and Computation after incorporating the feedback from Neural Networks reviewers and reframing the work for a more appropriate venue.

---

## Package Contents

### 1. `/manuscript/` - Main Submission Files

- **`manuscript.pdf`** (30 pages, 1.4 MB) - Final PDF for submission
- **`manuscript.tex`** (LaTeX source) - For editorial use if requested
- **`cover_letter_AMC.pdf`** (2 pages) - Cover letter to Editor-in-Chief
- **`cover_letter_AMC.tex`** (LaTeX source)

**These 2 PDFs are the primary submission files.**

### 2. `/figures/` - All Figures (High Resolution)

- `fig_identification.png` - RBF identification (original)
- `fig_simulation.png` - RBF forward simulation (original)
- `fig_mlp_vs_rbf.png` - Architecture comparison (original)
- **`fig_noise_robustness.png`** (NEW) - Robustness to measurement noise
- **`fig_lotka_volterra.png`** (NEW) - Lotka-Volterra 2D system

All figures are 300 DPI, print-quality PNG format.

### 3. `/supplementary/` - Code and Scripts

- `demo_noise_robustness.py` - Noise robustness experiments
- `demo_optimizer_comparison.py` - Optimizer benchmarks (Adam, L-BFGS-B, etc.)
- `demo_lotka_volterra.py` - Lotka-Volterra 2D system

These scripts are referenced in the manuscript and available on GitHub.

### 4. `/documentation/` - Supporting Documents

- **`RESPONSE_TO_REVIEWERS.md`** - Detailed response showing how Neural Networks feedback was addressed
- `AMC_FORMAT_CHECKLIST.md` - Verification that manuscript meets AMC guidelines
- `REVIEWER_FEEDBACK_ANALYSIS.md` - Analysis of Neural Networks rejection
- `COVER_LETTER_AMC_SUMMARY.md` - Summary of cover letter strategy

---

## Submission Checklist

### Required for AMC Submission:

- [x] **Manuscript PDF** (`manuscript.pdf`)
- [x] **Cover Letter PDF** (`cover_letter_AMC.pdf`)
- [x] **Figures** (all in `/figures/`, high resolution)
- [ ] **Highlights** (create if AMC requires - check during submission)
- [ ] **Graphical Abstract** (check if AMC requires)

### Optional but Recommended:

- [x] **Response to Reviewers** (explains improvements vs Neural Networks version)
- [x] **Code availability** (mentioned in manuscript, publicly available)

---

## Key Changes vs Neural Networks Submission

| Aspect | Neural Networks | AMC (This Submission) |
|--------|----------------|----------------------|
| **Status** | Rejected | New submission |
| **Journal** | Neural Networks | Applied Math & Comp |
| **Experiments** | 1 system (1D) | 3 systems (1D, 2D, noise) |
| **Comparisons** | 2 methods | 5 methods (+ Adam, L-BFGS-B) |
| **Pages** | 26 | 30 |
| **Figures** | 3 | 5 |
| **Framing** | ML-oriented | Math/computational methods |

---

## Submission Instructions

### Step 1: Go to AMC Editorial Manager
https://www.editorialmanager.com/amc/

### Step 2: Create New Submission

- Select: "New Submission"
- Article Type: "Original Research Article" or "Full Length Article"

### Step 3: Upload Files

**Required files**:
1. Main manuscript PDF: `submission_AMC/manuscript/manuscript.pdf`
2. Cover letter PDF: `submission_AMC/manuscript/cover_letter_AMC.pdf`
3. Figures (upload individually from `submission_AMC/figures/`):
   - fig_identification.png
   - fig_simulation.png
   - fig_mlp_vs_rbf.png
   - fig_noise_robustness.png
   - fig_lotka_volterra.png

**Optional** (upload if system requests):
- LaTeX source: `manuscript.tex`
- Supplementary material: Scripts from `/supplementary/`

### Step 4: Metadata

**Title**:
> A Unified Homotopy Framework for Nonlinear System Identification and Simulation: Constructive Dimensioning of Bilinear Neural Architectures

**Authors**:
- Rodolfo H. Rodrigo (corresponding)
- H. Daniel Patiño

**Keywords** (copy from manuscript):
> Homotopy analysis method, nonlinear system identification, ordinary differential equations, radial basis functions, Newton-Halley corrections, constructive dimensioning, bilinear approximation, function approximation theory, computational methods

**Highlights** (create if requested):
1. Unified homotopy framework for identification and ODE simulation
2. Constructive dimensioning chain from target accuracy to network size
3. Validated on 1D and 2D systems with noise robustness experiments
4. Faster than Adam and L-BFGS-B on bilinear structures
5. Proof-of-concept for embedded/real-time applications

### Step 5: Cover Letter

Paste content from `cover_letter_AMC.pdf` or upload the PDF directly.

### Step 6: Suggested Reviewers

**Suggestion**: Provide 3-5 names of experts in:
- Homotopy methods for ODEs
- Function approximation theory
- Computational methods for system identification
- Numerical analysis

**Avoid**: Reviewers who are:
- Co-authors or collaborators
- From the same institution
- Competing on similar methods

### Step 7: Additional Comments (Optional)

You may mention:
> "This manuscript was previously submitted to Neural Networks and was rejected with feedback that the scope was too narrow and experiments insufficient. We have substantially expanded the experimental validation (3 systems vs 1, including 2D coupled system and noise robustness) and reframed the contribution for the computational methods community. A detailed response to the Neural Networks reviewers is included as supplementary documentation."

---

## After Submission

1. **Save the submission ID** (AMC will assign one, e.g., AMC-D-26-XXXXX)
2. **Track status** via Editorial Manager
3. **Expected timeline**:
   - Initial decision: 4-8 weeks
   - If minor/major revision: address and resubmit
   - If accepted: production ~4-6 weeks

---

## Contact Information

**Corresponding Author**:
- Name: Rodolfo H. Rodrigo
- Email: rrodrigo@inaut.unsj.edu.ar
- Institution: Instituto de Automática (INAUT), UNSJ-CONICET, Argentina

---

## Notes

- This is a **NEW submission**, not a transfer or appeal
- Neural Networks rejection is mentioned only for context in optional comments
- The manuscript has been significantly improved based on that feedback
- Applied Mathematics and Computation is a better fit for the methodological scope

**Good luck with the submission! 🎯**
