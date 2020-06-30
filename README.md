# Detecting-Bias-In-German-Parliamentiary-Proceedings
Detect ethno-religious biases within German parliamentary proceedings reaching from 1867 - 2020 based on word embeddings trained on them. The bias dimensions are:

* judaism
* christianity
* catholicism
* protestantism
-----------

## Extract parliamentary protocols and pre-process them

### Reichstag
1) Extract Reichstag proceedings from each of the original files (e.g. '1942.corr.seg'):
```
python extract_proceedings.py [file] (will be saved in ./data)
```
2) After all protocols from all 4 files are extracted, run python create_corpora.py to create sorted and balanced slices:
```
python create_corpora.py (will be saved in ./data)
```
3) Process the resulting balanced slices by applying a cascade of text processing functions on each extracted protocol; specify protocol_type as either RT (Reichstagsprotokolle) or BRD (Bundestagsprotokolle) as the processing steps slightly vary between them:
```
python process_protocols [slice_folder] [protocols_type]
```
4) Train an embedding space on the processed slices (model will be stored ```./models``` , vocab in ```./data/vocab```)
```
bash train_embeddings.sh 
```

### Bundestag
Bundestag protocols are already provided in a handy format, with separate folders for each legislatory period and protocols already separated into distinct text files.
Thus, steps 1) and 2) are omitteded and the folders containing protocols of each legislatory period can be directly pre-processed

## Evaluation
A number of bias tests can be run on the trained embedding spaces, the scripts to run them are contained in the ```./evaluation``` folder

Explicit bias tests:

* WEAT --> ```bash run_weat.sh```
* BAT / ECT --> ```bash run_bat_ect.sh (specify test type)```

Implicit bias tests:

* K-Means -> ```bash run_kmeans.sh```

Semantic evaluation:
* To run the Simlex test, run the script ```run_simlex.sh```


## Harmonic Function Label Propagation

1) First, create the Positive Pointwise Mutual Information (PPMI) matrix for each processed slice:
```
bash ppmi.sh (matrix is stored in ./matrices, index in ./tok2indx)
```

2) After, the label propagation can be executed by running the script ```propagate.sh```, which will save the position scores as an array in the folder ```./fu_scores```
and writes the mean position scores of each target specification to disk. Use the opposing attribute specifications of your choice as labels (so far, only *sentiment* is implemented) to propagate from.



 
