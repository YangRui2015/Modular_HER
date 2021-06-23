import os
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from numpy.core.fromnumeric import size
from numpy.lib.function_base import i0
from numpy.lib.npyio import save
from numpy.ma.core import right_shift
import seaborn as sns; sns.set()
import glob2
import argparse
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


smooth = True

def smooth_reward_curve(x, y):
    halfwidth = 2
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    try:
        data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    except:
        import pdb; pdb.set_trace()
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


# Load all data.
def load_data(dir, key='test/success_rate', filename='progress.csv'):
    data = []
    # find all */progress.csv under dir
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(dir, '**', filename))]
    for curr_path in paths:
        if not os.path.isdir(curr_path):
            continue
        results = load_results(os.path.join(curr_path, filename))
        if not results:
            print('skipping {}'.format(curr_path))
            continue
        print('loading {} ({})'.format(curr_path, len(results['epoch'])))

        success_rate = np.array(results[key])[:50]
        epoch = np.array(results['epoch'])[:50] + 1

        # Process and smooth data.
        assert success_rate.shape == epoch.shape
        x = epoch
        y = success_rate
        if smooth:
            x, y = smooth_reward_curve(epoch, success_rate)
        assert x.shape == y.shape
        data.append((x, y))
    return data

def load_datas(dirs, key='test/success_rate', filename='progress.csv'):
    datas = []
    for dir in dirs:
        data = load_data(dir, key, filename)
        datas.append(data)
    return datas

# Plot datas
def plot_datas(datas, labels, info, fontsize=15, i=0, j=0):
    title, xlabel, ylabel = info
    for data, label in zip(datas, labels):
        try:
            xs, ys = zip(*data)
        except:
            import pdb; pdb.set_trace()
        xs, ys = pad(xs), pad(ys)
        assert xs.shape == ys.shape

        plt.plot(xs[0], np.nanmedian(ys, axis=0), label=label)
        plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)
    plt.title(title, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.legend(fontsize=fontsize-3, loc=4, bbox_to_anchor=(0.5, 0.06, 0.5, 0.5))
    plt.xticks(fontsize=fontsize-3)
    plt.yticks(fontsize=fontsize-4)

def plot_main(dirs, labels, info, key='test/success_rate', filename='progress.csv', save_dir='./test.png'):
    plt.figure(dpi=300, figsize=(5,4))
    datas = load_datas(dirs, key, filename)

    plot_datas(datas, labels, info)
    plt.subplots_adjust(left=0.14, right=0.98, bottom=0.15, top=0.92, hspace=0.3, wspace=0.15)
    plt.savefig(save_dir)


if __name__ == '__main__':
    data_dirs = ['', '']
    save_dir = ''
    legend = ['HER', 'CHER']
    infos = ['title', 'Epoch', 'Median success rate'] 
    plot_main(data_dirs, legend, infos, key='test/mean_Q', save_dir=save_dir)