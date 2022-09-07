# Analyzing Transformers in Embedding Space!
**code still requires some refactoring and documentation**

This code encompasses all the experiments presented in the [paper](https://arxiv.org/abs/2209.02535).


## Setup (Linux)
First create a directory named `artifacts` here. Run in shell:
```
mkdir artifacts
```


To use the notebook `parameter-alignment.ipynb` you must also download models from multiBERTs. Run in shell:
```
./load_multiberts.sh
```

## Cite Us
If you want to cite us:
```
@misc{transformers_in_embedding_space,
  doi = {10.48550/ARXIV.2209.02535},
  url = {https://arxiv.org/abs/2209.02535},
  author = {Dar, Guy and Geva, Mor and Gupta, Ankit and Berant, Jonathan},
  title = {Analyzing Transformers in Embedding Space},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```