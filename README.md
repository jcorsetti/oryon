# Oryon: Open-Vocabulary Object 6D Pose Estimation

This is the repository that contains source code for the [Oryon website](https://jcorsetti.github.io/oryon/) and its implementation.
This work is to be presented at CVPR'24.


## Roadmap

- Code release: 26 March 2024 
- Added test and train splits: 7 Dec 2023
- Website and arxiv released: 4 Dec 2023

## Installation

First of all, download `oryon_data.zip` and `pretrained_models.zip` from the release of this repository.
The first contains the ground-truth information and the specification of the image pairs used, the second contains the third-party checkpoint used in Oryon (i.e, the tokenizer and PointDSC).

Run `setup.sh` to install the environment and download the external checkpoints.

## Running Oryon

By default all experiments folder are created in `exp_data/`.
This can be modified in the config file.
Training with default settings:

`
python run_train.py exp_name=baseline
`

Run the following to obtain results with the basic 4 configurations. By default, the last checkpoint is used.

`python run_test.py -cp exp_data/baseline/ dataset.test.name=nocs test.mask=predicted
`

`python run_test.py -cp exp_data/baseline/ dataset.test.name=nocs test.mask=oracle
`

`python run_test.py -cp exp_data/baseline/ dataset.test.name=toyl test.mask=predicted
`

`python run_test.py -cp exp_data/baseline/ dataset.test.name=toyl test.mask=oracle
`


## Dataset preparation

Our data is based on three publicly available datasets:
- [REAL275](https://github.com/hughw19/NOCS_CVPR2019), used for test. We sample from the real test partition.
- [Toyota-Light](https://bop.felk.cvut.cz/datasets/) (TOYL), used for test. We sample from real the test partition from the BOP challenge. 
- [ShapeNet6D](https://github.com/ethnhe/FS6D-PyTorch) (SN6D), used for training. Note that SN6D itself does not provide textual annotations, but it uses object models from [ShapeNetSem](https://shapenet.org/), which do provide object names and synsets for each object model.

We sample scenes from each dataset to build the training and testing partition (20000 image pairs for SN6D and 2000 for REAL275 and TOYL), and report in the following folder the scene ids and image ids used for each partition.


### REAL275 (referred as NOCS)

From the [repository](https://github.com/hughw19/NOCS_CVPR2019) download the test ground-truth, the object models and the data of the `real_test` partition. This should result in three files: `obj_models.zip`, `gts.zip` and `real_test.zip`

Run the `prepare_nocs.sh` script to unzip and run the preprocessing.

By default this will create the `nocs` folder in `data`, and can be changed by modifying the above script.


### Toyota-Light

Download the object models and the test partition from the official BOP website:

`
wget https://bop.felk.cvut.cz/media/data/bop_datasets/tyol_models.zip
`

`
wget https://bop.felk.cvut.cz/media/data/bop_datasets/tyol_test_bop19.zip
`

Run the `prepare_toyl.sh` script to unzip and run the preprocessing.

By default this will create the `toyl` folder in `data`, and can be changed by modifying the above script.


### ShapeNet6D

Download the images from the official repository of [ShapeNet6D](https://github.com/ethnhe/FS6D-PyTorch), and the object models of ShapeNet from [HuggingFace](https://huggingface.co/datasets/ShapeNet/ShapeNetSem-archive).

Run the `prepare_sn6d.sh` script to unzip and run the preprocessing.

Note that each image of ShapeNet6D shows a different random background, so that we consider each image as being part of a different scene.
ShapeNet6D provides a map from their object ids to the object ids of the original ShapeNetSem: we use this map to associated the object name and synonym sets of ShapeNetSem to each object model in ShapeNet6D.

NB: ShapeNet6D is not currently supported for evaluation (i.e., the symmetry annotations needed by the BOP toolkit are missing).

## Acknowledgements

This work was supported by the European Union’s Horizon Europe research and innovation programme under grant agreement No 101058589 (AI-PRISM), and made use of time on Tier 2 HPC facility JADE2, funded by EPSRC (EP/T022205/1).

We thank the authors of the following repositories for open-sourcing the code, on which we relied for this project:
- [CATSeg](https://github.com/KU-CVLAB/CAT-Seg)
- [PointDSC](https://github.com/XuyangBai/PointDSC)
- [BOP Toolkit](https://github.com/thodan/bop_toolkit)


## Citing Oryon

```BibTeX
@inproceedings{corsetti2024oryon,
  title= {Open-vocabulary object 6D pose estimation}, 
  author = {Corsetti, Jaime and Boscaini, Davide and Oh, Changjae and Cavallaro, Andrea and Poiesi, Fabio},
  journal = {IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year = {2024}
}
```

## Website License
The website template is from [Nerfies](https://github.com/nerfies/nerfies.github.io).

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
