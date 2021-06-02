# Prognostic value of polygenic risk scores for adults with psychosis

**Aim**: determine whether polygenic risk scores (PRS) significantly contribute to the prediction of clinical outcomes 
in adults with psychosis.

Available datatsets can be found in `./data`. If needed, file names should be changed in the code. 
Edit the code with file names and paths before running it. Folder organization is as follows:
 
```
../prs_psychosis --> Project folder
./model --> Python scripts
./Rcode --> R scripts
./data --> Input data
./out --> Output results
```

Clone github project folder:

```
clone https://github.com/landiisotta/prs_psychosis
```

Provided a Python3 environment (`myenv`), within terminal run:

```
source myenv/bin/activate
cd ./prs_psychosis
pip install -r ./model/requirements.txt
```

## Datasets

Datasets are organized as follow:

### BioMe

Selected patients with SCZ diagnosis (i.e., `scz.icd.binary`==1).

Outcome variables:

- `NLP.agressive` aggressive behavior: 0 not agressive, 1 aggressive; 
   
- `biomeOrd.PsychAdmit` number of psychiatric hospitalizations: 
   recoded binary =0 (no admissions), >=1 (at least one admission).
   
Features:

- 38 clinical features (feature `biomeDem.relationship` was binarized to 
  dummy variable and class 0 was dropped);
  
- 20 ancestry PCs;

- 1 PRS.
   
**Remark**: among features we recoded `biomeDem.relationship` 
            (0=single, 1=widowed, 2=divorced/separated, 3=partnered); 
            and `biomeQ.aut|scz|dep.grandpar` (0=None, 1=one or more).

### GPC

Outcome variables:

- `OPCRIT.90` course of disorder w/ and w/o deterioration: =0 (w/o deterioration), 
   =1 (w/ deterioration).
   
Features:
   
- 79 clinical features (dropped 11 with >70% missing information);
- 20 ancestry PCs;
- 1 PRS.

**Remark**: among features `opcrit.01`, `opcrit.40`, and `opcrit.52` were 
            binarized to dummy variables and class 0 was dropped.

## Methods

Pipeline:

1. Create datsets as described in "Datasets" section. Each resulting dataset will include different features 
   (i.e., clinical, clinical+PCs, clinical+PCs+PRS, PCs+PRS) and the outcome variable in last position 
   and it will be split into training and test. 
   (1) Stratifiers for GPC: `ancestry`, `OPCRIT.90`; 
   (2) Stratifiers for BioMe: `gill.ContinentalGrouping`, outcome variables (`NLP.agressive`, `biomeOrd.PsychAdmit`).
 
2. Rescaling continuous variables with min/max scaler, i.e., (x-min(x))/(max(x)-min(x)). Min/max from the training set 
   are then used to scale test set.

3. Train logistic regression with 10x3 repeated cross validation and grid-search 
   (varying regularization parameters), select best model based on highest F2 score, and compare F2 score 
   distributions for nested models. Run pairwise t-test with FDR correction for comparisons.

4. Evaluate best model on test set and report F2 and AUPRC (compare scores with 100 bootstrap iterations).

**Remark**: Remember to edit configuration files at `./model/configuration/config_biome|gpc.json`.

Pipeline was replicated separately for EUR/AFR/AMR ancestries and with the "binarized" version of PRS 
(create binarized PRS with `binarize_prs.R`).

### Code description

Edit configuration files (folder `./model/configuration`) and initialize logger (folder `./model/logger`).
 
`R` (in folder `./Rscripts`):

> `biome_preprocessing.R`; `gpc_preprocessing.R`: Data preprocessing for BioME 
   and Genomic Psychiatry Cohort (GPC) datasets.

>  `binarize_prs.R`: Binarize PRS feature.

> `comparison_f2val.R`: Visualize boxplots for F2 validation score comparisons with p-values from two-sided 
   pairwise t-test comparisons (FDR correction). Perform ANOVA testing.

> `bootstrap_auprc.R`: Generate bootstrap AUPRC estimates. Perform ANOVA and two-sided pairwise t-tests (FDR correction).

`Python` (in folder `./model`):

> `scaling_biome.py`; `scaling_gpc.py`: Scale BioMe and GPC datasets.

> `gridsearch.py`: Grid-search within repeated CV framework.

> `traineval.py`: Train best model on the entire dataset and evaluate it on test set.

> `visualization.py`: Plot precision-recall curves (PRC) and grid-search performance.

> `utils.py`: Functions.

> `gs_results_viz.ipynb`: Jupyter notebook for grid-search and PRC visualization.
 
### Step-by-step modeling

1. Run `biome_preprocessing.R` and `gpc_preprocessing.R`.
   
   - `biome_preprocessing.R`: 
      
      **Input**: `biome_dataset.txt`;
       
      **Output**: it saves preprocessed datasets to `./out` with different sets of features and outcomes 
      and split into traininig and test sets. 
      In particular, `biome_clinical|genetic|all_agressive|psych_admit_train|test.txt`. 
      File `rescale_feature_biome.txt` lists names of the features to scale and `categorical_feature_biome_idx.txt` 
      lists categorical feature indices.
      
   - `gpc_preprocessing.R`:
      
      **Input**: `gpc.tsv` (GPC data); `opcrit_helper_2020.txt` (feature types); `gpc_coltypes.tsv` (feature groups).
      
      **Output**: as above, it saves preprocessed datasets to `./out` with different sets of features and split into 
      training and test sets. In particular, `gpc_all|clinical|genetic_opcrit90_train|test.txt`. 
      File `rescale_feature_opcrit90.txt` lists the names of the features to scale and `categorical_feature_gpc_idx.txt` 
      lists categorical feature indices.
      
2. Run `scaling_biome.py` and `scaling_gpc.py`:

   ```
   cd ./model
   python scaling_biome|gpc.py
   ```

   **Input**: preprocessed datasets and feature lists output in 1. 
   
   **Output**: scaled datasets. For BioMe dataset, only genetic features, i.e., PRS and ancestry PCs are scaled.
   Files can be found in `./out` folder with names marked as "scaled" (except for BioMe clinical only dataset that 
   remains unchanged).
   
3. Run `gridsearch.py` for grid-search repeated CV.

   ```
    python -m gridsearch \
   --config ./configuration/config_biome|gpc.json \
   --training_set ../out/data_training.txt \
   --out ../out \
   --oversampling True|False \
   --cat_feat ../out/categorical_feature_idx_biome|gpc.txt 
   ```
   
   **Output**: `gridsearch_LRscores.txt` grid-search performance for best hyperparameters selection.
   
4. Run `traineval.py` to fit best model to the entire training set and evaluate it on the test set.

   ```
    python -m traineval \
   --config ./configuration/config_biome|gpc.json \
   --grid_search ../out/gridsearch_LRscores.txt \
   --training_set ../out/data_training.txt \
   --test_set ../out/data_test.txt \
   --scoring "F2" \
   --out ../out
   ```

    **Output**: `gridsearch_LRbestestimator_retrained.pkl` (best model estimator); `best_model_eval.pkl` tuple with: 
    true labels, predicted labels, predicted probabilities, prediction score (e.g., F2, AUPRC), precision, recall; 
    `predicted_prob.txt` (predicted labels and probabilities).

5. Code for test set comparisons of bootstrap AUPRC estimates can be found in `bootstrap_auprc.R`.
  
6. Visualize results: code and functions for results visualization and score comparisons, as displayed/reported 
   in the manuscript, can be found in: `visualization.py`, `gs_results_viz.ipynb`; `comparison_f2val.R`.
