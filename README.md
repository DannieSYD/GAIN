# Molecular Conformer Ensemble Learning with Geometry-Aware Interaction Networks

## Overview
The dynamic nature of molecules in their environment necessitates modeling them as conformational ensembles rather than static structures. While recent advances in geometric deep learning have demonstrated remarkable success in molecular property prediction, most approaches process only single conformers, limiting their ability to capture the full complexity of molecular systems. Previous methods for handling conformer ensembles either rely on structural averaging techniques that can generate unphysical structures or require rigid molecular alignment that restricts their applicability. To address these limitations, we present GAIN (Geometry-Aware Interaction Networks), which introduces two key innovations: selective information processing through specialized geometry-aware expert networks and gated aggregation mechanisms for guided cross-conformer information integration.
GAIN maintains joint equivariance to both permutation of the conformer ensemble and geometric transformations (e.g., rotations and translations) of individual conformers. Comprehensive experiments across diverse molecular property prediction tasks demonstrate that GAIN consistently outperforms existing conformer ensemble methods and state-of-the-art structural aggregation models.

## Dataset preparation

1. download the dataset.zip from https://drive.google.com/file/d/132gumuh-wSpLf4yd0TOrKuL1jHbQyUQe/view?usp=sharing 
2. unzip the dataset.zip and put the data in the datasets folder.

## Run

```bash
bash run_drugs_gain.bash  # Run GAIN on drugs dataset
bash run_kraken_gain.bash  # Run GAIN on kraken dataset
bash run_cov2_gain.bash  # Run GAIN on cov2 dataset
bash run_cov3cl_gain.bash  # Run GAIN on cov23cl dataset
```
