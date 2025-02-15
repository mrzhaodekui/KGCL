# KGCL
## Title
KGCL: Knowledge-Enhanced Graph Contrastive Learning for Retrosynthesis Prediction Based on Molecular Graph Editing
## Environment Requirements  
- python = 3.11.8
- pytorch = 2.2.2
- numpy = 1.26.4
- rdkit = 2024.03.4

## Data and Checkpoints
The original datasets used in this paper are from:

USPTO-50K: [https://github.com/Hanjun-Dai/GLN](https://github.com/Hanjun-Dai/GLN) (schneider50k)

USPTO-FULL:[https://github.com/Hanjun-Dai/GLN](https://github.com/Hanjun-Dai/GLN) (uspto_multi)

The raw data, processed data can be accessed via [link](https://drive.google.com/drive/folders/11YMNrm7St-GgVF278orHSXk-EKM3ltqH?usp=sharing). The directory structure should be as follows:

```
KGCL
├───data
|   ├───uspto_50K
|   │       ├───canonicalized_test.csv
|   │       ├───canonicalized_train.csv
|   │       ├───canonicalized_val.csv
|   │       ├───raw_test.csv
|   │       ├───raw_train.csv
|   │       └───raw_val.csv
|   │       
|   │       
|   └───uspto_full
|           ├───canonicalized_test.csv
|           ├───canonicalized_train.csv
|           ├───canonicalized_val.csv
|           ├───raw_test.csv
|           ├───raw_train.csv
|           └───raw_val.csv
```
- Data
    - The raw data of the USPTO-50K dataset and USPTO-FULL dataset is stored in the corresponding folders in the files `raw_train.csv`, `raw_val.csv`, and `raw_test.csv`.
    - All the processed data are named `canonicalized_train.csv` , `canonicalized_val.csv` and `canonicalized_test.csv` and are put in the corresponding folders respectively.
## Data preprocessing
- generate the edit labels and the edits sequence for reaction 
```
python preprocess.py --mode train --dataset USPTO_50k \
python preprocess.py --mode valid --dataset USPTO_50k \
python preprocess.py --mode test --dataset USPTO_50k \ 
or
python preprocess.py --mode train --dataset uspto_full \
python preprocess.py --mode valid --dataset uspto_full \
python preprocess.py --mode test --dataset uspto_full \ 
```
-   Prepare the data for training without using reaction classes as a condition
```
python prepare_data.py --dataset USPTO_50k or uspto_full
```
- Prepare the data for training using reaction classes as a condition
```
python prepare_data.py --dataset USPTO_50k --use_rxn_class
```
## Train KGCL model

- Run the following to train the model with specified dataset without using reaction classes as a condition
```
python train.py --dataset uspto_50k or uspto_full
```
The trained model will be saved at KGCL/experiments/uspto_50k/without_rxn_class/

- Run the following to train the model with USPTO-50K dataset using reaction classes as a condition
```
python train.py --dataset uspto_50k --use_rxn_class
```
The trained model will be saved at KGCL/experiments/uspto_50k/with_rxn_class/
# Test
To evaluate the trained model, run
```
python eval.py or
python eval.py --use_rxn_class
```
The raw prediction file saved at KGCL/experiments/.../pred_results.txt
## Reproducing our results
- To reproduce our exact accuracy and MaxFrag accuracy results on USPTO-50K dataset, run

```
python eval.py --dataset uspto_50k \
python eval.py --dataset uspto_50k --use_rxn_class \
```
This will display the exact accuracy and MaxFrag accuracy results for reaction class unknown and known setting
- To reproduce our round-trip accuracy results on USPTO-50K dataset, run
```
python eval-rtacc.py
```
This will display the round-trip accuracy results for reaction class unknown setting
- To reproduce our exact accuracy results on USPTO-FULL dataset, run
```
python eval-full.py
```
This will display the exact accuracy results for reaction class unknown setting