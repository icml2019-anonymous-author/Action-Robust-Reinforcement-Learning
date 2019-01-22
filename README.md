# Action-Robust-Reinforcement-Learning
Code accompanying the anonymous submission to ICML 2019 of the paper "Action Robust Reinforcement Learning and Applications in Continuous Control"

## Requirements:
* [MuJoCo](http://mujoco.org)
* Python 3 (it might work with Python 2, not tested)
* [PyTorch](http://pytorch.org/)
* [OpenAI Gym](https://github.com/openai/gym)
* [tdqm](https://github.com/tqdm/tqdm)
* numpy
* matplotlib

## Howto train:

```bash
python3.6 main.py --updates_per_step 10 --env-name "Hopper-v2" --alpha 0.1 --method pr_mdp
```
Where method can take 3 values `mdp` `pr_mdp` or `nr_mdp`, where `pr/nr` are the probabilistic robust and noisy robust as defined in the paper.

*All results are saved in the models folder.*

## Howto evaluate:
See `models_example` for our pre-trained data.

Once a model has been trained, run:
```bash
python3.6 test.py --eval_type model
```
where `--eval_type model` will evaluate for model (mass) uncertainty and `--eval_type model_noise` will create the 2d visualizations.

## Howto visualize:
See `visualize_data.ipynb` for an example of how to access and visualize your models.


*Our approach is built upon the DDPG implementation by https://github.com/ikostrikov/pytorch-ddpg-naf*
