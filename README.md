

## MarioNette | [Webpage](https://people.csail.mit.edu/smirnov/marionette/) | [Paper](https://arxiv.org/abs/2104.14553) | [Video](https://youtu.be/KMrdh8RQCJk)


### Modifications for time-consistency

- Using the ML framework developed by the authors, that requires integrating into their callback structure (e.g. modifying training in `interfaces.py` and adding a callback for rtpt (for monitoring on the dgx instances))
- I currently only copied over the evaluation for the object representation evaluation, i.e. object consistency related topics and no motion loss. That means there is:
  - an implementation of dataset and training using stacks of frames#
  - Computation and metrics storing for adjusted mutual information and the few shot accuracy
  - Adapted object consistency that runs, though I can't guarantee it to be bug free. while porting the code some dimensions had to swapped and adapted (mario nette contains one dimension additionally for number of layers (NL)). I could imagine that e.g. Width and Height in the motion data does not align with data in here, maybe dimension swapping or scaling is necessary. 
  - `multi_train.py` replica for a similar `multi_train` command like in SPACE-Time (MOC) that trains multiple configuration of hyperparameters and seeds.
  - A number of comments to keep track of dimensions throughout the code
  - Empirically finding solid default parameters (see `experiment-sets` in `multi_train.py`) for space invaders, as the documentation in the original paper is confusing to non-existing.


#### Running with time-consistency

`python multi_train.py --data <path_to_folder_with_training_images>`

multi_train.py will then iterate through each configuration of hyperparameters and e.g. overwrite `seed`, `layer_size`, `num_layers`, `num_classes` and especially important `aow`
All other paths e.g. motion data or ground truth for evaluation are defined as relative paths to `--data` so if the data is generated like for SPACE-Time (MOC) that should be all. 

There is probably also help in `python multi-train.py --help`, that uses the library of the authors called `ttools` to hopefully give out some information. But you can also just read `multi_train.py`.


<img src="https://people.csail.mit.edu/smirnov/marionette/im.png" width="75%" alt="MarioNette" />

**MarioNette: Self-Supervised Sprite Learning**<br>
Dmitriy Smirnov, Michaël Gharbi, Matthew Fisher, Vitor Guizilini, Alexei A. Efros, Justin Solomon<br>
[NeurIPS 2021](https://neurips.cc/Conferences/2021)

### Set-up
To install the neecssary dependencies, run:
```
conda env create -f environment.yml
conda activate MarioNette
```
Also, be sure to execute `export PYTHONPATH=:$PYTHONPATH` prior to running any of the scripts.

### Training
To train a MarioNette model, run:
```
python scripts/train.py --checkpoint_dir out_dir --data data_dir
```
Your dataset should be stored in `data_dir`, with each
input frame named `#.png`. If the images are not 128x128 pixels, specify the
resolution using the `--canvas_size` flag.
Optionally, pass a `--layer_size` flag to specify the
anchor grid resolution, `--num_layers` to specify the number of layers, or
`--num_classes` to specify the size of the spirte dictionary.

To monitor the training, launch a TensorBoard instance with `--logdir out_dir`.

### BibTeX
```
@article{smirnov2021marionette,
  title={{MarioNette}: Self-Supervised Sprite Learning},
  author={Smirnov, Dmitriy and Gharbi, Michael and Fisher, Matthew and Guizilini, Vitor and Efros, Alexei A. and Solomon, Justin},
  year={2021},
  journal={Conference on Neural Information Processing Systems}
}
```
