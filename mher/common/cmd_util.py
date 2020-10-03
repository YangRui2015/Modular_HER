"""
Helpers for command line
"""
import os
import gym
import argparse
from mher.common import logger 

def common_arg_parser():
    """
    Create common used argparses for training
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='FetchReach-v1')
    parser.add_argument('--seed', help='set seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='her')
    parser.add_argument('--random_init', help='Random init epochs before training',default=0, type=int)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--num_timesteps', type=float, default=1e6)
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default='mlp', type=str)
    parser.add_argument('--num_env', help='Number of environment being run in parallel. Default set to 1', default=1, type=int)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--policy_save_interval', default=10, type=int)
    parser.add_argument('--load_path', help='Path to load trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play_episodes', help='Number of episodes to play after training', default=1, type=int)
    parser.add_argument('--play_no_training', default=False, action='store_true')
    return parser

def parse_unknown_args(args):
    """
    Parse arguments not consumed by arg parser into a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):
        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v
    return {k: parse(v) for k,v in parse_unknown_args(args).items()}

def preprocess_kwargs(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    return args, extra_args