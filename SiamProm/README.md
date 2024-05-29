# SiamProm: a framework for cyanobacterial promoter identification

SiamProm originated from the paper **Recognition of Cyanobacteria Promoters via Siamese Network-based Contrastive Learning under Novel Non-promoter Generation**.

## The architecture of SiamProm

![SiamProm](./figs/fig2.webp)


## Dependency(Mindspore Version)

| Main Package 	| Version 	|
| ------------	| -------:	|
| Python       	| 3.9.18  	|
| Mindspore    	| 2.2.10  	|
| CUDA         	| 11.6.1   	|
| Scikit-learn  | 1.3.2   	|
| Pandas      	| 2.1.4   	|
| Hydra        	| 1.3.2   	|
| Pyyaml      	| 6.0.1   	|

Build your environment manually or through a yaml file.

For the installation of mindspore GPU version, you can refer to mindspore official documentation.

### YAML file

```bash
conda env create -f env_SiamProm_mindspore.yaml
conda activate SiamProm_mindspore
```

## Usage

```bash
python train.py sampling=phantom
```

Optional values for the parameter `sampling` are `phantom`(default), `random`, `cds`, and `partial`.

> For more detailed instructions on parameter management and configuration usage, please refer to the [Hydra](https://hydra.cc/docs/1.3/intro/) documentation.

## Citation

If you find this work useful in your research, please consider citing:
```bibtex
@article{yang2024recognition,
  title={Recognition of cyanobacteria promoters via Siamese network-based contrastive learning under novel non-promoter generation},
  author={Yang, Guang and Li, Jianing and Hu, Jinlu and Shi, Jian-Yu},
  journal={Briefings in Bioinformatics},
  volume={25},
  number={3},
  pages={bbae193},
  year={2024}
}
```
