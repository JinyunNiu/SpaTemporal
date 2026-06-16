# SpaTemporal
![image](https://github.com/JinyunNiu/SpaTemporal/blob/main/Overview.jpg)

## Overview
Spatiotemporal transcriptomics enables the characterization of gene expression dynamics within a spatial context over time, offering unprecedented opportunities to study complex biological processes such as development, tissue remodeling, and disease progression. However, significant morphological expansion and non-linear spatiotemporal evolution during developmental stages pose substantial challenges for the precise alignment and integration of multi-slice data across discrete time points.
Here, we present SpaTemporal, a deep learning framework designed to integrate spatiotemporal transcriptomics data across multiple species and states. SpaTemporal employs a standardized pre-alignment strategy to correct systematic spatial discrepancies between different time points. By constructing a three-dimensional spatiotemporal graph that incorporates both temporal progression and spatial information, SpaTemporal leverages spatiotemporal graph neural networks to jointly model spatial and temporal dependencies. Furthermore, a cross-temporal triplet constraint is introduced to ensure the temporal smoothness and consistency of representations across similar tissue regions.
Comprehensive benchmarking across various platforms, resolutions, and species demonstrates that SpaTemporal significantly outperforms existing methods in identifying dynamic spatial domains. Moreover, by constructing a pseudo-spatiotemporal maps of spatial domains, SpaTemporal elucidates complex lineage dynamics during mouse embryonic development and precisely identifies key spatially variable and temporally associated genes. In a longitudinal mouse model of Alzheimer’s disease, SpaTemporal accurately delineates the evolutionary trajectory of pathological domains driven by Aβ deposition, successfully uncovering critical biomarkers associated with plaque progression.
Overall, SpaTemporal provides a powerful computational solution for deciphering the development of complex biological tissues within a spatiotemporal transcriptomics framework.

## Datasets
All data used in this work are available at: https://zenodo.org/records/19591185.<br><br>
The mouse embryo and mouse brain datasets can be downloaded from here: [https://db.cngb.org/stomics/datasets/STDS0000058](https://db.cngb.org/stomics/datasets/STDS0000058).<br><br>
The American axolotl telencephalon dataset can be downloaded here: [https://db.cngb.org/stomics/datasets/STDS0000056/summary](https://db.cngb.org/stomics/datasets/STDS0000056/summary).<br><br>
The Alzheimer's disease mouse dataset can be downloaded here: https://singlecell.broadinstitute.org/single_cell/study/SCP1375/integrative-in-situ-mapping-of-single-cell-transcriptional-states-and-tissue-histopathology-in-an-alzheimer-disease-model#study-download.

## Repository Structure

```text
SpaTemporal/
├── graph_3D.py/     # Constructing spatiotemporal graph
├── mnn.py/          # Constructing triples across time points
├── module.py/       # core network and mask data augmentation
├── process.py/      # Data preprocessing
├── train.py         # Model training
└── utils.py         # Clustering and visualization
```

## Installations
- NVIDIA GPU (Nvidia GeForce RTX 4090).
- `pip install -r requiremnt.txt`
  
## Running demo
TThis step-by-step tutorial (7 chapters) covers the complete workflow for spatial-temporal data alignment and integration, leading directly to downstream domain identification.

Quick Start: https://spatemporal-tutorials.readthedocs.io/en/latest/.

- Tutorial 1: Alignment of developmental mouse embryo data
- Tutorial 2: Alignment of developmental axolotl telencephalon data
- Tutorial 3: Aligning different states and time periods of Alzheimer’s disease mice data
- Tutorial 4: Integrating developmental mouse embryo data
- Tutorial 5: Integrating developmental mouse brain data
- Tutorial 6: Integrating developmental axolotl telencephalon data
- Tutorial 7: Integrating Alzheimer’s mouse brain data

## Contact details
If you have any questions, please contact niujinyun@aliyun.com.



