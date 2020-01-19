## Pytorch implementation of DNN-regressor using Entity Embedding[1]

A Pytorch implementation of Entity Embedding DNN model.  
And its general application to an experimental flow for a real-world tabular data.  
See the files ```model.py``` and ```demo.ipynb``` for more detail.  
  
> [1] Guo, Cheng, and Felix Berkhahn. "Entity embeddings of categorical variables." arXiv preprint arXiv:1604.06737 (2016).  

### model.py

```model.py``` shows an implementation of DNN for regression with input of factorized (*) categorical variables (first half) and numerical variables (second half).  
Preparing a unique Embedding Layer for each categorical variable and perform mapping to the corresponding dense vector space (Entity Embedding).  
- *Example*:
  - Mapping a categorical variable with 7 categories which elements are Mon.(0), Tue.(1), Wed.(2), Thu.(3), Fri.(4), Sat.(5), Sun.(6) to 2D space by an Embedding Layer)  
  
Regression is performed by combining the embedded categorical features and numerical variables and inputting them to Fully Connected Layers.  
  
> (*) Factorization: The process of converting each element of a categorical variable into a corresponding positive index.  

### demo.ipynb

```demo.ipynb``` shows a experimental flow for solving a real-world problem using ```model.py```.  
By cooperating with ```model.py``` with [```df2numpy.TransformDF2Numpy```](https://github.com/kitayama1234/TransformDF2Numpy),
a conversion tool from pandas.DaraFrame to numpy.array, it realizes generality applicable to various tabler dataset.

