# CLL Multi-Omics Analysis Tutorial

## 📓 MultiOmicsBind_CLL_data_tutorial.ipynb

### Overview

This tutorial demonstrates MultiOmicsBind on **real clinical data** from 200 Chronic Lymphocytic Leukemia (CLL) patients, replicating and extending MOFA-style factor analysis.

### Goals

1. **Predict IGHV status** (unmutated vs mutated) - Key prognostic marker
2. **Predict Trisomy12** (absent vs present) - Chromosomal abnormality  
3. **Discover 4 patient subgroups** based on integrated multi-omics
4. **Identify biomarkers** from each omics layer

### Data

- **4 Omics Modalities**: mRNA (5,000 genes), Methylation (4,248 CpG sites), Drugs (310 responses), Mutations (69 alterations)
- **200 CLL Patients** from `/Users/shivaprasad/Documents/PROJECTS/GitHub/MO/CLL_data/`
- **Clinical Variables**: IGHV status, Trisomy12, age, gender, survival outcomes

### Notebook Structure

1. **Cell 1-3**: Imports and setup
2. **Cell 4-5**: Load data and explore clinical variables (IGHV, Trisomy12 distributions)
3. **Cell 6-7**: Load multi-omics data (mRNA, methylation, drugs, mutations)
4. **Cell 8-9**: Preprocessing (filter patients with complete data)
5. **Cell 10-13**: Task 1 - Predict IGHV status (train, evaluate, metrics)
6. **Cell 14-15**: Task 2 - Predict Trisomy12 (train, evaluate, metrics)  
7. **Cell 16-17**: Feature importance analysis (biomarker discovery)
8. **Cell 18-19**: MOFA-style visualization (4 patient subgroups)
9. **Cell 20**: Summary and conclusions

### Key Features

✅ **Binary Classification**: IGHV (0 vs 1) and Trisomy12 (0 vs 1)  
✅ **Multi-Omics Integration**: All 4 modalities with binding modality approach  
✅ **Biomarker Discovery**: Gradient-based feature importance per modality  
✅ **Patient Stratification**: PCA visualization showing 4 distinct subgroups  
✅ **MOFA Comparison**: Replicates Factor1/Factor3 patient separation

### Expected Outputs

1. `CLL_metadata_processed.csv` - Filtered patient data
2. `feature_importance_IGHV.csv` - Top biomarkers for IGHV prediction
3. `patient_subgroups_MOFAstyle.png` - 4-subgroup visualization
4. `clinical_stratification.png` - IGHV/Trisomy12 distributions

### How to Run

```bash
cd /Users/shivaprasad/Documents/PROJECTS/GitHub/MO/MultiOmicsBind/examples
jupyter notebook MultiOmicsBind_CLL_data_tutorial.ipynb
```

Or open in VS Code and run all cells.

### Expected Results

- **IGHV Classification**: ~70-85% test accuracy (depends on train/test split)
- **Trisomy12 Classification**: ~70-80% test accuracy
- **4 Patient Subgroups**: Clear separation in Factor1 vs Factor2 space
- **Top Biomarkers**: Genes, CpG sites, drugs, and mutations ranked by importance

### Comparison to MOFA

| Feature | MOFA | MultiOmicsBind |
|---------|------|----------------|
| Method | Matrix factorization | Deep learning with binding modality |
| Interpretability | Factors (linear combinations) | Gradient-based feature importance |
| Patient stratification | ✅ 4 subgroups | ✅ 4 subgroups (replicated) |
| Biomarker discovery | Factor loadings | Feature importance scores |
| Predictive modeling | Limited | ✅ Direct classification |
| Drug associations | ✅ Shown | ✅ Integrated in embeddings |

### Clinical Significance

- **IGHV status**: Most important prognostic marker in CLL
  - Unmutated (IGHV=0): Aggressive, worse survival
  - Mutated (IGHV=1): Indolent, better survival
  
- **Trisomy 12**: Chromosomal abnormality
  - Present in ~15-20% of CLL patients
  - Associated with intermediate prognosis
  - Often occurs with IGHV-unmutated disease

### Next Steps

1. Run the notebook to generate all results
2. Examine top biomarkers in `feature_importance_IGHV.csv`
3. Correlate subgroups with clinical outcomes (TTT, TTD)
4. Explore drug-gene interactions from cross-modal similarity
5. Validate findings in independent CLL cohorts

---

**Created**: October 2025  
**Author**: Shivaprasad Patil  
**Framework**: MultiOmicsBind v0.1.3
