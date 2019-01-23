import argparse
import os
import gym
import numpy as np
import pickle
from tqdm import trange

import torch
from torch.distributions import uniform

from ddpg import DDPG
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from replay_memory import ReplayMemory, Transition
from utils import save_model

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='name of the environment to run')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--no_ou_noise', default=False, action='store_true')
parser.add_argument('--param_noise', default=False, action='store_true')
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of neurons in the hidden layers (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--method', default='mdp', choices=['mdp', 'pr_mdp', 'nr_mdp'])
parser.add_argument('--flip_ratio', default=False, action='store_true')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='control given to adversary (default: 0.2)')
parser.add_argument('--exploration_method', default=None, choices=['mdp', 'nr_mdp', 'pr_mdp'])
args = parser.parse_args()

if args.exploration_method is None:
    args.exploration_method = args.method

args.ou_noise = not args.no_ou_noise

env = NormalizedActions(gym.make(args.env_name))
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = DDPG(args.gamma, args.tau, args.hidden_size, env.observation_space.shape[0], env.action_space,
             True, args.alpha)

noise = uniform.Uniform(agent.Tensor([-1.0]), agent.Tensor([1.0]))

results_dict = {'eval_rewards': [],
                'value_losses': [],
                'policy_losses': [],
                'adversary_losses': [],
                'train_rewards': []
                }
value_losses = []
policy_losses =[]
adversary_losses = []

base_dir = os.getcwd() + '/models/' + args.env_name + '/'

if args.param_noise:
    base_dir += 'param_noise/'
elif args.ou_noise:
    base_dir += 'ou_noise/'
else:
    base_dir += 'no_noise/'

if args.exploration_method == args.method:
    if args.method != 'mdp':
        if args.flip_ratio:
            base_dir += 'flip_ratio_'
        base_dir += args.method + '_' + str(args.alpha) + '_' + str(args.updates_per_step) + '/'
    else:
        base_dir += 'non_robust/'
else:
    if args.method != 'mdp':
        if args.flip_ratio:
            base_dir += 'flip_ratio_'
        base_dir += 'alternative_' + args.method + '_' + str(args.alpha) + '_' + str(args.updates_per_step) + '/'
    else:
        base_dir += 'alternative_non_robust_' + args.exploration_method + '_' + str(args.alpha) + '_' + str(args.updates_per_step) + '/'

run_number = 0
while os.path.exists(base_dir + str(run_number)):
    run_number += 1
base_dir = base_dir + str(run_number)
os.makedirs(base_dir)

memory = ReplayMemory(args.replay_size)

ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05,
                                     desired_action_stddev=args.noise_scale,
                                     adaptation_coefficient=1.05) if args.param_noise else None

total_steps = 0

print(base_dir)
for i_episode in trange(args.num_episodes):
    state = agent.Tensor([env.reset()])

    if args.ou_noise:
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * \
                        max(0, args.exploration_end - i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()

    if args.param_noise:
        agent.perturb_actor_parameters(param_noise)

    episode_reward = 0
    while True:
        agent.eval()
        action = agent.select_action(state, ounoise, param_noise, mdp_type=args.exploration_method)
        agent.train()
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        total_steps += 1
        episode_reward += reward

        action = agent.Tensor(action)
        mask = agent.Tensor([not done])
        next_state = agent.Tensor([next_state])
        reward = agent.Tensor([reward])

        memory.push(state, action, mask, next_state, reward)

        state = next_state

        if len(memory) > args.batch_size:
            for idx in range(args.updates_per_step):
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))

                adversary_update = (idx == 0 and not args.flip_ratio) or (idx != 0 and args.flip_ratio)

                value_loss, policy_loss, adversary_loss = agent.update_parameters(batch,
                                                                                  mdp_type=args.method,
                                                                                  adversary_update=adversary_update,
                                                                                  exploration_method=args.exploration_method)
                value_losses.append(value_loss)
                policy_losses.append(policy_loss)
                adversary_losses.append(adversary_loss)

            if total_steps % 500 == 0:
                results_dict['value_losses'].append((total_steps, np.mean(value_losses)))
                results_dict['policy_losses'].append((total_steps, np.mean(policy_losses)))
                results_dict['adversary_losses'].append((total_steps, np.mean(adversary_losses)))
                del value_losses[:]
                del policy_losses[:]
                del adversary_losses[:]

        if done:
            break

    results_dict['train_rewards'].append((total_steps, np.mean(episode_reward)))

    # Update param_noise based on distance metric
    if args.param_noise and len(memory) > args.batch_size:
        episode_transitions = memory.sample(args.batch_size)
        states = torch.cat([transition[0] for transition in episode_transitions], 0)
        unperturbed_actions = agent.select_action(states, None, None)
        perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

        ddpg_dist = ddpg_distance_metric(perturbed_actions.cpu().numpy(), unperturbed_actions.cpu().numpy())
        param_noise.adapt(ddpg_dist)

    if i_episode % 50 == 0:
        episode_reward = 0
        agent.eval()
        for _ in range(10):
            state = agent.Tensor([env.reset()])
            with torch.no_grad():
                while True:
                    action = agent.actor(state)

                    next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                    episode_reward += reward

                    next_state = agent.Tensor([next_state])

                    state = next_state
                    if done:
                        break
        agent.train()
        episode_reward = episode_reward * 1.0 / 10
        results_dict['eval_rewards'].append((total_steps, episode_reward))

    save_model(actor=agent.actor, adversary=agent.adversary, basedir=base_dir)
    with open(base_dir + '/results', 'wb') as f:
        pickle.dump(results_dict, f)
save_model(actor=agent.actor, adversary=agent.adversary, basedir=base_dir)

env.close()
