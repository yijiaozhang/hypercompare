# Introduction

This repository hosts the code for paper "Systematic comparison of graph embedding methods in practical tasks. (2021). *Physical Review E, 104(4), 044315.* Yi-Jiao Zhang, Kai-Cheng Yang, Filippo Radicchi". [[DOI]](https://doi.org/10.1103/PhysRevE.104.044315) [[arXiv]](https://arxiv.org/abs/2106.10198).
We provide a general framework to compare the performance of different network embedding methods on downstream tasks.

The network embedding methods considered in our paper include Node2vec, HOPE, Laplacian Eigenmap, Isomap, HyperMap, HyperLink, Mercator, Poincaré maps, Hydra and community embedding.
The downstream tasks used to measure the performance of different embedding methods are mapping accuracy (correlation between the pairwise distance in the embedding space and in the original graph), greedy routing, and link prediction.

# Quick start

## Installation

### `hypercomparison` package

The main functionality of the framework is implemented as a Python package `hypercomparison`.
We recommend installing the package locally.
To do so, clone the repository and go to the directory `code/lib`.
Please make sure your virtual environment for this project is activated, then run

```bash
pip install -e ./
```

Then you can try

```python
import hypercomparison.utils
import hypercomparison.node2vec_HOPE
```

to check whether the package is installed properly.

### Requirements

The code is implemented with Python 3.7.4.
The following packages are required:

```bash
numpy==1.17.2
networkx==2.3
scikit-learn==0.21.3
gensim==3.8.0
scipy==1.3.1
infomap==1.3.0
python-louvain==0.14
snakemake==5.5.4
torch==1.3.1
```

### Demos

Please check out the demo workflows under [`code/workflow`](code/workflow).
We use [`snakemake`](https://snakemake.readthedocs.io/en/stable/) to organize our workflows.

For example, you can run `snakemake -s demo_snakefile.smk run_node2vec_link_prediction_all -j1` in folder `code/workflow/node2vec` to get the link prediction result of the example network `karate` using Node2vec.

# Repo structure and important files

| Folder                                                             | Note                                                |
| ------------------------------------------------------------------ | --------------------------------------------------- |
| [`code/lib/hypercomparison`](code/lib/hypercomparison)             | The `hypercomparison` package                       |
| [`code/workflow`](code/workflow)                                   | Demo workflows for running the tests                |
| [`code/lib/hypercomparison/data`](code/lib/hypercomparison/data)   | The folder to store the network data                |
| [`data`](data)                                                     | The folder to store the outputs                     |
| [`code/workflow/network_list.csv`](code/workflow/network_list.csv) | Descriptive stats of and links to the networks used in this paper  |

**[Note]**: The input networks should go to `code/lib/hypercomparison/data` with the file name format `{networkname}_edges.txt`. The output data will be stored in folder `data`.

# Implemented embedding methods

## node2vec

**Ref**: Grover, Aditya, and Jure Leskovec. "[node2vec: Scalable feature learning for networks](https://dl.acm.org/doi/abs/10.1145/2939672.2939754)." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.

An implementation of node2vec can be found in [`code/lib/hypercomparison/node2vec_HOPE.py`](code/lib/hypercomparison/node2vec_HOPE.py)

To use node2vec on downstream tasks, go to the folder `code/workflow/node2vec`, and run

```bash
snakemake -s demo_snakefile.smk run_node2vec_corr_routing_all -j1
```

for mapping accuracy and greedy routing.

For link prediction, run

```bash
snakemake -s demo_snakefile.smk run_node2vec_link_prediction_all -j1
```



## HOPE

**Ref**: Ou, Mingdong, et al. "[Asymmetric transitivity preserving graph embedding](https://dl.acm.org/doi/abs/10.1145/2939672.2939751)." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.

An implementation of HOPE can be found in the file [`code/lib/hypercomparison/node2vec_HOPE.py`](code/lib/hypercomparison/node2vec_HOPE.py)

To use HOPE on downstream tasks, go to the folder `code/workflow/hope` and run

```bash
snakemake -s demo_snakefile.smk run_hope_corr_routing_all -j1
snakemake -s demo_snakefile.smk run_hope_link_prediction_all -j1
```

## Laplacian Eigenmap (LE):

**Ref**: Belkin, Mikhail, and Partha Niyogi. "[Laplacian eigenmaps and spectral techniques for embedding and clustering](https://dl.acm.org/doi/abs/10.5555/2980539.2980616)." Nips. Vol. 14. No. 14. 2001.

We use the function `SpectralEmbedding` from `sklearn.manifold` to implement LE.

To use LE on downstream tasks, go to the folder [`code/workflow/le`](code/workflow/le) and run:

```bash
snakemake -s demo_snakefile.smk run_le_corr_routing_all -j1
snakemake -s demo_snakefile.smk run_le_link_prediction_all -j1
```

## Isomap:

**Ref**: Tenenbaum, Joshua B., Vin De Silva, and John C. Langford. "[A global geometric framework for nonlinear dimensionality reduction](https://science.sciencemag.org/content/290/5500/2319.abstract)." science 290.5500 (2000): 2319-2323.

Isomap is an embedding method aims at preserving geodesic distances between pairwise nodes. It is also and extension of Multidimensional scaling (MDS) by incorporating the geodesic distances matrix of a network.
In our work, we use function `MDS` in `sklearn.manifold` with the shortest path length matrix of the network as the input to implement Isomap.

To use Isomap on downstream tasks, go to the folder `code/workflow/isomap` and run

```bash
snakemake -s demo_snakefile.smk run_isomap_corr_routing_all -j1
snakemake -s demo_snakefile.smk run_isomap_link_prediction_all -j1
```

## HyperMap

**Ref**: Papadopoulos, Fragkiskos, Rodrigo Aldecoa, and Dmitri Krioukov. "[Network geometry inference using common neighbors.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.92.022807)" Physical Review E 92.2 (2015): 022807.

We use the original C++ implementation provided by the authors of the paper.
To obtain Hypermap, visit [the official repo](https://bitbucket.org/dk-lab/2015_code_hypermap/src/master/).

The steps to download and use the code are:

1. Go the the `code/workflow` folder, clone the C++ code of HyperMap by running

```bash
git clone https://bitbucket.org/dk-lab/2015_code_hypermap.git
```

2. To compile the code, simply run (you will need a C++ complier)

```bash
make
```

3. Move the demonstration files in `code/workflow/hypermap` to the directory of the HyperMap C++ code. Then you can try

```bash
snakemake -s demo_snakefile.smk run_hypermap_corr_routing_all -j1
snakemake -s demo_snakefile.smk run_hypermap_link_prediction_all -j1
```

## HyperLink

**Ref**: Kitsak, Maksim, Ivan Voitalov, and Dmitri Krioukov. "[Link prediction with hyperbolic geometry](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043113)." Physical Review Research 2.4 (2020): 043113.

Similar to HyperMap, HyperLink is also implemented by C++, see [here](https://bitbucket.org/dk-lab/2020_code_hyperlink/src/master/). To integrate it into our framework, follow the instructions:

1. Go to `code/workflow` folder, clone the C++ code of HyperLink by running

```bash
git clone https://bitbucket.org/dk-lab/2020_code_hyperlink.git
```

2. To compile the code, run (you will need a C++ complier)

```bash
g++ hyperlink.cpp auxiliary/network.cpp auxiliary/global.cpp auxiliary/mle.cpp -o hyperlink.exe -O3
```

3. Move the demonstration files in `code/workflow/hyperlink` to the same dictionary with the HyperLink C++ code. Then you can try

```bash
snakemake -s demo_snakefile.smk run_hyperlink_corr_routing_all -j1
snakemake -s demo_snakefile.smk run_hyperlink_link_prediction_all -j1
```

**[Note 1]**: HyperLink requires the node index of the input networks to start from 0, otherwise some unexpected outcomes may occur.

**[Note 2]**: HyperLink first tries to estimate a `kp_max` value for the input network but may fail to do so in some cases. To make it work, we set `xleft = 2` at line 348 in `auxiliary/network.cpp` in the HyperLink source code. But for some rare cases, `kp_max` still cannot be resolved. We skip these networks in the evaluation.

## Mercator

**Ref**: García-Pérez, Guillermo, et al. "[Mercator: uncovering faithful hyperbolic embeddings of complex networks](https://iopscience.iop.org/article/10.1088/1367-2630/ab57d2/meta)." New Journal of Physics 21.12 (2019): 123033.

The implementation of Mercator can be found [here](https://github.com/networkgeometry/mercator).
We use it as a local python package.

To apply Mercator to downstream tasks, go to  [`code/workflow/mercator`](code/workflow/mercator) and run

```bash
snakemake -s demo_snakefile.smk run_mercator_corr_routing_all -j1
snakemake -s demo_snakefile.smk run_mercator_link_prediction_all -j1
```

## Poincar\'e maps

**Ref**: Klimovskaia, Anna, et al. "[Poincaré maps for analyzing complex hierarchies in single-cell data](https://www.nature.com/articles/s41467-020-16822-4)." Nature communications 11.1 (2020): 1-9.

The implementation of Poincaré maps can be found [here](https://github.com/facebookresearch/PoincareMaps). Put all the files with suffix `.py` of [Poincaré maps](https://github.com/facebookresearch/PoincareMaps) in the folder `code/workflow/poincare_maps` and use

```bash
snakemake -s demo_snakefile.smk run_poinmap_corr_routing_all -j1
snakemake -s demo_snakefile.smk run_poinmap_link_prediction_all -j1
```

to test the performance of Poincaré maps of downstream tasks on networks.

## Hydra

**Ref**: Keller-Ressel, Martin, and Stephanie Nargang. "[Hydra: a method for strain-minimizing hyperbolic embedding of network-and distance-based data.](https://academic.oup.com/comnet/article-abstract/8/1/cnaa002/5741150)" Journal of Complex Networks 8.1 (2020): cnaa002.

The authors implement it as a R-package [hydra](https://cran.r-project.org/web/packages/hydra/index.html).
Here we import the R-package hydra in Python and, see [`code/workflow/hydra`](code/workflow/hydra/) for examples.

You will need to install the following packages

```
hydra (R-package)
rpy2==2.9.4
```

To use Hydra on downstream tasks, go to the folder `code/workflow/hydra` and run

```bash
snakemake -s demo_snakefile.smk run_hydra_corr_routing_all -j1
snakemake -s demo_snakefile.smk run_hydra_link_prediction_all -j1
```

## Community embedding

**Ref**: Faqeeh, Ali, Saeed Osat, and Filippo Radicchi. "[Characterizing the analogy between hyperbolic embedding and community structure of complex networks](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.098301)." Physical review letters 121.9 (2018): 098301.

We implement the community embedding method in [`code/lib/hypercomparison/community_embedding.py`](code/lib/hypercomparison/community_embedding.py).

To use community embedding on downstream tasks, go to the folder `code/workflow/community_embedding` and run

```bash
snakemake -s demo_snakefile.smk run_community_corr_routing_all -j1
snakemake -s demo_snakefile.smk run_community_link_prediction_all -j1
```

# Citation

```bib
@article{PhysRevE.104.044315,
  title = {Systematic comparison of graph embedding methods in practical tasks},
  author = {Zhang, Yi-Jiao and Yang, Kai-Cheng and Radicchi, Filippo},
  journal = {Phys. Rev. E},
  volume = {104},
  issue = {4},
  pages = {044315},
  numpages = {13},
  year = {2021},
  month = {Oct},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.104.044315},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.104.044315}
}

```
