import os
import time
import argparse
import gym
import numpy as np

from baselines.common.misc_util import (set_global_seeds, boolean_flag)
from baselines import logger, bench
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *
import baselines.ddpg.training as training

import tensorflow as tf
from mpi4py import MPI


def run(env_id, seed, noise_type, layer_norm, evaluation, memory_limit, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    print("rank: %d" % (rank))
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    if evaluation and rank == 0:
        eval_env = gym.make(env_id)
        enal_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), "gym_eval"))
        env = bench.Monitor(env, None)
    else:
        eval_env = None

    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1]

    for current_noise_type in noise_type.split(","):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == "none":
            pass
        elif "adaptive-param" in current_noise_type:
            _, stddev = current_noise_type.split("_")
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev),
                                                 desired_action_stddev=float(stddev))
        elif "normal" in current_noise_type:
            _, stddev = current_noise_type.split("_")
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions),
                                             sigma=float(stddev)*np.ones(nb_actions))
        elif "ou" in current_noise_type.split("_"):
            _, stddev = current_noise_type.split("_")
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                        sigma=float(stddev)*np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    memory = Memory(limit=memory_limit, action_shape=env.action_space.shape,
                    observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    seed = seed + 1000000 * rank
    logger.info("rank {} : seed={}, logdir={}".format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
                   action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info("total runtime: {}s".format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--env-id", type=str, default="HalfCheetah-v2")
    boolean_flag(parser, "render-eval", default=False)
    boolean_flag(parser, "layer-norm", default=True)
    boolean_flag(parser, "render", default=False)
    boolean_flag(parser, "normalize-returns", default=False)
    boolean_flag(parser, "normalize-observations", default=True)
    parser.add_argument("--seed", help="RNG seed", type=int, default=0)
    parser.add_argument("--critic-l2-reg", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    boolean_flag(parser, "popart", default=False)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--reward-scale", type=float, default=1.)
    parser.add_argument("--clip-norm", type=float, default=None)
    parser.add_argument("--nb-epochs", type=int, default=500)
    parser.add_argument("--nb-epoch-cycles", type=int, default=20)
    parser.add_argument("--nb-train-steps", type=int, default=50)
    parser.add_argument("--nb-eval-steps", type=int, default=100)
    parser.add_argument("--nb-rollout-steps", type=int, default=100)
    parser.add_argument("--noise-type", type=str, default="adaptive-param_0.2")
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--memory-limit", type=int, default=1e6)
    boolean_flag(parser, "evaluation", default=False)
    args = parser.parse_args()
    if args.num_timesteps is not None:
        assert args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps

    dict_args = vars(args)
    del dict_args["num_timesteps"]
    return dict_args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    for key, value in args.items():
        print(key, value)
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure()
    run(**args)


















