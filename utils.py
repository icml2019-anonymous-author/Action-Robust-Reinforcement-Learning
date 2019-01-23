import os
import torch


def save_model(actor, adversary, basedir=None):
    if not os.path.exists('models/'):
        os.makedirs('models/')

    actor_path = "{}/ddpg_actor".format(basedir)
    adversary_path = "{}/ddpg_adversary".format(basedir)

    # print('Saving models to {} {}'.format(actor_path, adversary_path))
    torch.save(actor.state_dict(), actor_path)
    torch.save(adversary.state_dict(), adversary_path)


def load_model(actor, adversary, basedir=None):
    actor_path = "{}/ddpg_actor".format(basedir)
    adversary_path = "{}/ddpg_adversary".format(basedir)

    print('Loading models from {} {}'.format(actor_path, adversary_path))
    actor.load_state_dict(torch.load(actor_path))
    adversary.load_state_dict(torch.load(adversary_path))
