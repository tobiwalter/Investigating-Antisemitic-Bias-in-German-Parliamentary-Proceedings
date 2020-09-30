# Investigating Antisemitic Bias in German Parliamentary Proceedings
Explore ethno-religious biases within German parliamentary proceedings reaching from 1867 - 2020, based on word embeddings that are trained from scratch on eight different slices of proceedings, with each slice corresponding to a known historical period.

|       | Kaiserreich I | Kaiserreich II | Weimar | NS   | CDU I  | SPD I  | CDU II | SPD II | CDU III |
|-------|---------------|----------------|--------|------|--------|--------|--------|--------|---------|
| start | 1867          | 1890           | 1918   | 1933 | 1949   | 1969   | 1982   | 1998   | 2005    |
| end   | 1890          | 1918           | 1933   | 1942 | 1969   | 1982   | 1998   | 2005   | 2020    |

The four target sets to investigate bias are:
* Judaism
* Christianity
* Catholicism
* Protestantism

Based on those targets sets, four explicit bias specifications of the form *B<sub>E</sub> = (T<sub>1</sub>, T<sub>2</sub>, A<sub>1</sub>, A<sub>2</sub>)* as per the definition of [Lauscher et al.](https://arxiv.org/pdf/1909.06092.pdf) get tested for the presence of anti-semitic bias, namely:

* B<sub>E1</sub> = (T<sub>Christian</sub>, T<sub>Jewish</sub>, A<sub>pos</sub>, A<sub>neg</sub>)
* B<sub>E2</sub> = (T<sub>Protestant</sub>, T<sub>Catholic</sub>, A<sub>pos</sub>, A<sub>neg</sub>)
* B<sub>E3</sub> = (T<sub>Protestant</sub>, T<sub>Jewish</sub>, A<sub>pos</sub>, A<sub>neg</sub>)
* B<sub>E4</sub> = (T<sub>Catholic</sub>, T<sub>Jewish</sub>, A<sub>pos</sub>, A<sub>neg</sub>)

*A<sub>pos</sub>* respectively *A<sub>neg</sub>* are placeholders for the opposing attribute sets of each of the six antisemitic streams *s* in *S* and the general sentiment dimension of pleasant/unpleasant words. 

*S* covers six anti-semitic streams, namely religious, economic, patriotic, racial, conspiratorial and ethic streams. Each stream is semantically linked to antisemitic tendencies that subsume commonly-held stereotypes towards Jews.

-----------
All word2vec models used for the bias experiments in the thesis are provided in the ```./models folder```, with the corresponding vocab in the ```./vocab``` folder. Hence, the results should be reproducible. The temporally aligned TWEC models for plotting semantic shifts are provided under ```./models/aligned``` and ```./models/aligned_brd``` respectively. The PPMI matrices for running label propagation are to be found in the folder ```./matrices```. They can also be reproduced with the pre-set min_count and window_size parameters of the ```ppmi.sh``` script. Since harmonic function label propagation is deterministic, subsequent propagation results on those PPMI are reproducible.

The folders ```./data/reichstag``` and ```./data/bundestag``` contain the preprocessed proceedings corresponding to each historic slice.

-----------
Originally, four original collections of OCR-ed and sentence-tokenized, but otherwise unprocessed parliamentary proceedings (1895.corr.seg, 1918.corr.seg, 1933.corr.seg, 1942.corr.seg) were available. Check ```1942.corr.seg``` in the ```./data``` folder for a sample of the original data.

The following steps need to be taken to first preprocess and vectorize the data, and to then apply several bias evaluation methods on the data representation of each historic slice.

### Reichstag
1) Extract Reichstag proceedings from each of the original files (e.g. '1942.corr.seg'):
```
python extract_proceedings.py [file] (output will be saved in ./data/protocols_1942)
```
2) After the protocols from all 4 original files are extracted, either create historically aligned slices or slices that are balanced in number of documents per slice:
```
python create_corpora.py -s [historic|balanced] (historic output will be saved in ./data/kaiserreich_1, ./data/kaiserreich_2, ./data/weimar, ./data/ns, ./data/cdu_1, etc.)
```
3) Process the resulting slices by applying a cascade of text processing functions on each extracted protocol; specify protocol_type as either RT (Reichstagsprotokolle) or BRD (Bundestagsprotokolle) as the processing steps slightly vary between Reichstag and Bundestag proceedings:
```
python process_protocols [slice_folder] [protocol_type] (output will be saved in ./data/kaiserreich_1_processed, etc.)
```
4) Train an embedding space on the processed slices (model will be stored in ```./models``` , vocab in ```./data/vocab```)
```
bash train_embeddings.sh (--model_architecture [word2vec], --protocols [the processed slice, e.g. ./data/reichstag/kaiserreich_1_processed)
```

### Bundestag
Bundestag protocols are already provided in a handy format, with separate folders for each legislatory period and protocols already separated into distinct text files.
Thus, steps 1) and 2) are omitteded and the folders containing protocols of each legislatory period can be directly pre-processed. A sample of the original Bundestag protocols is provided under folder ```./data/slice_7```

## Evaluation
A range of bias evaluation tests can be run on the trained word2vec embedding spaces. All bash scripts for the bias experiments are contained in the ```./evaluation``` folder

Explicit bias tests under DEBIE:

* WEAT --> ```bash run_weat.sh```.
* BAT / ECT --> ```bash run_bat_ect.sh (specify test type as either --BAT/ECT)```.

Implicit bias tests under DEBIE:

* K-Means --> ```bash run_kmeans.sh```.

Semantic evaluation:
* Simlex --> ```bash run_simlex.sh```.

Subspace Projections:
* To compute the subspace projections onto the Christian-Jewish bias subspace, run the sript ```ripa.sh```. In order to plot the projections, provide a semantic domain to plot with the ```--sem_domain``` argument and set ```plot_projections``` to True.

The evaluation scripts on the provided USB stick should work with the pre-set arguments, to reproduce the results of each bias test.

## Harmonic Function Label Propagation

1) For HFLP, input representations based on Positive Pointwise Mutual Information (PPMI) are employed. The following command creates a PPMI matrix for the processed slice provided as an argument:
```
bash ppmi.sh (--protocols  [the processed slice, e.g. ./data/reichstag/kaiserreich_1_processed] -> matrix is stored in ./matrices/ppmi_kaiserreich_1.npz, index in ./ppmi_vocab/kaiserreich_1.json))
```

2) Afterwards, the label propagation algorithm can be executed on the PPMI matrix by running the script ```propagate.sh```. Provide a semantic domain to propagate from with the ```--semantic_domain``` argument.

# Visualize Semantic Shift

1) First, temporally aligned embedding spaces must be trained. For this, change to the ```./twec``` folder and run:

```
python train_temporal_embeddings.py  [protocol_type]
```
You might have to change the provided folder arguments of the ```train_slice``` function inside the ```train_temporal_embeddings.py``` script to the paths where your processed data lies.

2) Now, semantic shifts can be plotted with the script:
```
python closest_over_time_with_anns.py -w [word1 word2 etc.] -n [number of neighbors] --protocol_type [RT/BRD] --model_folder [folder in which TWEC models reside]
```

# Publication Plots

To reproduce the "Corpus Statistics" and "average RIPA over slices" plots, open the Jupyter notebook ```reproduce_plots.ipynb```
