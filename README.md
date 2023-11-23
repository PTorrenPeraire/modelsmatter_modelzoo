# Benchmarking single step retrosynthesis models in a multi-step setting
Package to interchangeably use single-step retrosynthesis models as shown in Models Matter: The Impact of Single-Step Models on Synthesis Planning [1] as an extension to AiZynthFinder[2].

## Overview

The SingleStepModelZoo allows the fast extension of AiZynthFinder[2] to interchageably use alternative single-step retrosynthesis models. The package includes Chemformer[3], LocalRetro[4] and MHNreact[5] as shown in [1], however additional single-step models can easily be incorporated. To use the models within the multi-step setting please refer to [ModelsMatter].

Due to minor changes within the repositries, we fork the mentioned models which are included as submodules within the repository.

## Installation
1. ``` git clone --recurse-submodules https://github.com/PTorrenPeraire/modelsmatter_modelzoo ```
2. ```cd modelsmatter_modelzoo```
3. a) If installing from scratch ```conda env create -f ssbenchmark.yaml```
   b) If installing into AiZynthFinder instance ```conda env update -n [AiZynthFinder Environment Name] --file ssbenchmark.yaml```
5. ```conda activate ssbenchmark``` or ```conda activate [AiZynthFinder Environment Name]```
6. ```poetry install```

By default, the environment includes all packages required for the use of the default models.

## Using ModelZoo

### Running default models
```python
from ssbenchmark.model_zoo import ModelZoo
model = ModelZoo(single_step_model, single_step_module_path, single_step_use_gpu, single_step_settings)
```
The relvant variables are as follows:
- `single_step_model`: Each model is selected using their respetive (string) identifer, i.e. 'chemformer', 'mhnreact', 'localretro'. 
- `single_step_module_path`: Path to the local installation of the single-step model package. If the selected model is prepared as a package this can be ignored, if the path is sent within the single_step_model class then this is not necessary.
- `single_step_use_gpu`: Boolean whether to carry out single-step model inference on gpu
- `single_step_settings`: Additional settings passed to the single-step model


Inference is carried out using the setup model using the standard `model_call` function. This requires a list of one or more smiles and returns the predicted reactants along with their respective probabilities as predicted by the single-step model.
```
reactants, probabilities = model.model_call(smiles)
```
All model classes can be found in `ssbenchmark/ssmodels/`.

### Introducing new models
New models follow the same standard template, as shown in `ssbenchmark/ssmodels/model_example.py`. Importantly, each model class must include a `model_setup` function and a `model_call` function. Any string identifier can be used to identify the model, which is shown just before the start of the class and set as `example_model` in the example. The models can then be called using the same format as the default models, simply introducing the relevant model identifier and variables.

## License
The software is licensed under the MIT license.

## Acknowledgments




## References
1.  P. Torren-Peraire, A. K. Hassen, S. Genheden, J. Verhoeven, D. Clevert, M. Preuss, and I. Tetko,
“Models Matter: The Impact of Single-Step Retrosynthesis on Synthesis Planning,” arxiv, 2023. [Online]. Available:
https://arxiv.org/abs/2308.05522
2.  S. Genheden, A. Thakkar, V. Chadimová, J.-L. Reymond, O. Engkvist, and E. Bjerrum,
“AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic
planning,” Journal of Cheminformatics, vol. 12, no. 1, p. 70, 2020. [Online]. Available:
https://doi.org/10.1186/s13321-020-00472-1
3. R. Irwin, S. Dimitriadis, J. He, and E. J. Bjerrum, “Chemformer: a pre-trained
transformer for computational chemistry,” Machine Learning: Science and Technology,
vol. 3, no. 1, p. 015022, 2022, publisher: IOP Publishing. [Online]. Available:
https://doi.org/10.1088/2632-2153/ac3ffb
4. S. Chen and Y. Jung, “Deep Retrosynthetic Reaction Prediction using Local Reactivity and
Global Attention,” JACS Au, vol. 1, no. 10, pp. 1612–1620, 2021, publisher: American
Chemical Society. [Online]. Available: https://doi.org/10.1021/jacsau.1c00246
5. P. Seidl, P. Renz, N. Dyubankova, P. Neves, J. Verhoeven, J. K. Wegner, M. Segler,
S. Hochreiter, and G. Klambauer, “Improving Few- and Zero-Shot Reaction Template
Prediction Using Modern Hopfield Networks,” Journal of Chemical Information and
Modeling, vol. 62, no. 9, pp. 2111–2120, 2022, publisher: American Chemical Society.
[Online]. Available: https://doi.org/10.1021/acs.jcim.1c01065
