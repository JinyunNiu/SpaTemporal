# SpaTemporal
![image](https://github.com/JinyunNiu/SpaTemporal/blob/main/Overview.jpg)

## Overview
Spatiotemporal transcriptomics enables the characterization of gene expression dynamics within a spatial context over time, offering unprecedented opportunities to study complex biological processes such as development, tissue remodeling, and disease progression. However, significant morphological expansion and non-linear spatiotemporal evolution during developmental stages pose substantial challenges for the precise alignment and integration of multi-slice data across discrete time points.
Here, we present SpaTemporal, a deep learning framework designed to integrate spatiotemporal transcriptomics data across multiple species and states. SpaTemporal employs a standardized pre-alignment strategy to correct systematic spatial discrepancies between different time points. By constructing a three-dimensional spatiotemporal graph that incorporates both temporal progression and spatial information, SpaTemporal leverages spatiotemporal graph neural networks to jointly model spatial and temporal dependencies. Furthermore, a cross-temporal triplet constraint is introduced to ensure the temporal smoothness and consistency of representations across similar tissue regions.
Comprehensive benchmarking across various platforms, resolutions, and species demonstrates that SpaTemporal significantly outperforms existing methods in identifying dynamic spatial domains. Moreover, by constructing a pseudo-spatiotemporal maps of spatial domains, SpaTemporal elucidates complex lineage dynamics during mouse embryonic development and precisely identifies key spatially variable and temporally associated genes. In a longitudinal mouse model of Alzheimer’s disease, SpaTemporal accurately delineates the evolutionary trajectory of pathological domains driven by Aβ deposition, successfully uncovering critical biomarkers associated with plaque progression.
Overall, SpaTemporal provides a powerful computational solution for deciphering the development of complex biological tissues within a spatiotemporal transcriptomics framework.
## Datasets
All data used in this work are available at: 

## Installations
- NVIDIA GPU (a single Nvidia GeForce RTX 4090).
- `pip install -r requiremnt.txt`
  
## Running demo

## Contact details
If you have any questions, please contact niujinyun@aliyun.com.



