# Detecting-Bias-In-German-Parliamentiary-Proceedings
Detect ethno-religious biases within German parliamentary proceedings reaching from 1867 - 2020, based on word embeddings trained from scratch on the corpora. The target dimensions are:

* judaism
* christianity
* catholicism
* protestantism
-----------
On the provided USB stick, four original collections of unprocessed protocols (1895.corr.seg, 1918.corr.seg, 1933.corr.seg, 1942.corr.seg) are provided. The following steps need to be taken to vectorize the data, and then evaluate the sub-corpora with regard to ethno-religious bias.

All word2vec models used for the bias experiments of the thesis are provided in the ```./models folder```, with the vocab in the ```./vocab``` folder, and should be reproducible. TWEC models are provided under ```./models/rt_twec``` and ```./models/bundestag_twec``` respectively. PPMI matrices can be produced with the pre-set parameters, the subsequent propagation results should be reproducible, since harmonic function label propagation is deterministic.

The folders ```./data/reichstag``` and ```./data/bundestag``` contain the preprocessed slices.

## Extract parliamentary protocols and pre-process them

### Reichstag
1) Extract Reichstag proceedings from each of the original files (e.g. '1895.corr.seg'):
```
python extract_proceedings.py [file] (output will be saved in ./data/protocols_1895)
```
2) After all protocols from all 4 files are extracted, run python create_corpora.py to create historically aligned slices or slices that are balanced in number of documents per slice:
```
python create_corpora.py -s [historic|balanced] (historic output will be saved in ./data/kaiserreich_1, ./data/kaiserreich_2, ./data/weimar, ./data/ns)
```
3) Process the resulting slices by applying a cascade of text processing functions on each extracted protocol; specify protocol_type as either RT (Reichstagsprotokolle) or BRD (Bundestagsprotokolle) as the processing steps slightly vary according to the time period:
```
python process_protocols [slice_folder] [protocol_type] (output will be saved in ./data/kaiserreich_1_processed, etc.)
```
4) Train an embedding space on the processed slices (model will be stored in ```./models``` , vocab in ```./data/vocab```)
```
bash train_embeddings.sh (--model_architecture [word2vec], --protocols [the processed slice, e.g. ./data/reichstag/kaiserreich_1_processed)
```

### Bundestag
Bundestag protocols are already provided in a handy format, with separate folders for each legislatory period and protocols already separated into distinct text files.
Thus, steps 1) and 2) are omitteded and the folders containing protocols of each legislatory period can be directly pre-processed. On the USB stick, a sample of Bundestag protocols is provided with folder ```./data/wp_7```

## Evaluation
A number of bias tests can be run on the trained embedding spaces, the scripts to run them are contained in the ```./evaluation``` folder

Explicit bias tests:

* WEAT --> ```bash run_weat.sh```.
* BAT / ECT --> ```bash run_bat_ect.sh (specify test type)```.

Implicit bias tests:

* K-Means -> ```bash run_kmeans.sh```.

Semantic evaluation:
* To run the Simlex test, run the script ```run_simlex.sh```.

Subspace Projections:
* To compute subspace projections, run the sript ```ripa.sh```. Provide a semantic domain to plot with the ```--sem_domain``` argument and set ```plot_projections``` to True.

The scripts on the provided USB stick should work with the pre-set arguments, to run each bias test.

## Harmonic Function Label Propagation

1) First, create the Positive Pointwise Mutual Information (PPMI) matrix for each processed slice:
```
bash ppmi.sh (--protocols  [the processed slice, e.g. ./data/reichstag/kaiserreich_1_processed] -> matrix is stored in ./matrices/ppmi_kaiserreich_1.npz, index in ./ppmi_vocab/kaiserreich_1.json))
```

2) After, the label propagation can be executed by running the script ```propagate.sh```. Provide a semantic domain to propagate from (```--semantic_domain``` [dom])

# Visualize Semantic Shift

1) First, temporally aligned embedding spaces must be trained. For this, change to the ```./twec``` folder and run:

```
python train_temporal_embeddings.py  [protocol_type]
```
You might have to change the folder arguments inside the script to the paths where your processed data lies.

2) Now, semantic shifts can be plotted with the script:
```
python closest_over_time_with_anns.py -w [word1 word2 etc.] -n [number of neighbors] --protocol_type [RT/BRD] --model_folder [folder in which TWEC models reside]
```

# Publication Plots

To reproduce the "Corpus Statistics" and "average RIPA over slices" plots, open the Jupyter notebook ```reproduce_plots.ipynb```


 
