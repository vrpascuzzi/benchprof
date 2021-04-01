#
# CSV header
#
# "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",
# "Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value" [11],
# "Rule Name","Rule Type","Rule Description"
#
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

import csv
import sys


def main():
    inp_files = sys.argv[1:]

    data = {'buffer':   {'seed':
                         {'duration': [], 'sm': [], 'occupancy': []},
                         'gen':
                         {'duration': [], 'sm': [], 'occupancy': []},
                         'transform':
                         {'duration': [], 'sm': [], 'occupancy': []}
                         },
            'usm':      {'seed':
                         {'duration': [], 'sm': [], 'occupancy': []},
                         'gen':
                         {'duration': [], 'sm': [], 'occupancy': []},
                         'transform':
                         {'duration': [], 'sm': [], 'occupancy': []}
                         },
            'cuda':     {'seed':
                         {'duration': [], 'sm': [], 'occupancy': []},
                         'gen':
                         {'duration': [], 'sm': [], 'occupancy': []},
                         'transform':
                         {'duration': [], 'sm': [], 'occupancy': []}
                         }
            }
    data_sum = {'buffer': [], 'usm': [], 'cuda': []}

    # Loop input files
    for f in inp_files:
        # Status
        print('-- Doing', f)

        # Get backend
        backend = None
        if 'buffer' in f:
            backend = 'buffer'
        elif 'usm' in f:
            backend = 'usm'
        elif 'cuda' in f:
            backend = 'cuda'

        # Get kernel
        kernel = None
        if 'seed' in f:
            kernel = 'seed'
        elif 'gen_' in f:
            kernel = 'gen'
        elif 'transform' in f:
            kernel = 'transform'

        # Open file as csv
        with open(f) as csvfile:
            reader = csv.reader(csvfile)

            # Temporary lists to hold input
            _dur = []
            _sm = []
            _occ = []

            # Get metric value
            for row in reader:
                val = row[11]
                if 'Duration' in row:
                    if ',' in val:
                        val = float(val.replace(',', '')) * 10e-9
                    _dur.append(val)
                elif 'SM [%]' in row:
                    val = float(val) / 100.0
                    _sm.append(val)
                elif 'Occupancy' in row:
                    val = float(val) / 100.0
                    _occ.append(val)
            dur = np.mean(_dur)
            sm = np.mean(_sm)
            occ = np.mean(_occ)
            data[backend][kernel]['duration'].append(dur)
            data[backend][kernel]['sm'].append(sm)
            data[backend][kernel]['occupancy'].append(occ)

    # For each backend, calculate the duration sum of all kernels
    backends = [backend for backend in data.keys()]
    for backend in backends:
        for kernel in data[backend].keys():
            _sum = 0.0
            for i in range(0, len(data[backend][kernel])):
                _sum += data[backend][kernel]['duration'][i]
                data_sum[backend].append(_sum)

    # Three plots, one for each kernel: seed, gen, transform.
    # Each plot will consist of the backends -- buffer, usm and cuda -- each
    # having curves for "Duration [s]", "SM Fraction" and "Occupancy".
    x_data = [r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$',
              r'$10^{4}$', r'$10^{5}$', r'$10^{6}$', r'$10^{7}$', r'$10^{8}$']

    linestyle_seed = '-.'
    linestyle_gen = '--'
    linestyle_transform = ':'

    marker_buffer = '^'
    marker_usm = 'v'
    marker_cuda = 's'

    color_buffer = 'orange'
    color_usm = 'seagreen'
    color_cuda = 'royalblue'

    alpha_buffer = 1.0
    alpha_usm = 0.5
    alpha_cuda = 0.3

    for backend in data.keys():
        for kernel in data[backend].keys():
            data[backend][kernel]['duration'].reverse()

    # Seed, duration
    kernels = ['Seed', 'Generate', 'Transform']
    labels = ['Buffer API', 'USM API', 'Native CUDA']

    x = np.arange(len(labels))
    bar_width = 0.3
    x = np.arange(len(x_data))
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 4), dpi=250)
    rectsbuf1 = ax.bar(x - bar_width, data['buffer']['seed']['duration'],
                    bar_width, label='Seed (%s)' % labels[0], edgecolor='white', linewidth=0.25, color='mistyrose')
    rectsbuf2 = ax.bar(x - bar_width, data['buffer']['gen']['duration'],
                    bar_width, label='Generate (%s)' % labels[0], edgecolor='white', linewidth=0.25, color='salmon')
    rectsbuf3 = ax.bar(x - bar_width, data['buffer']['transform']['duration'],
                    bar_width, label='Transform (%s)' % labels[0], edgecolor='white', linewidth=0.25, color='tomato')

    rectsusm1 = ax.bar(x, data['usm']['seed']['duration'],
                    bar_width, label='Seed (%s)' % labels[1], edgecolor='white', linewidth=0.25, color='darkseagreen')
    rectsusm2 = ax.bar(x, data['usm']['gen']['duration'],
                    bar_width, label='Generate (%s)' % labels[1], edgecolor='white', linewidth=0.25, color='seagreen')
    rectsusm3 = ax.bar(x, data['usm']['transform']['duration'],
                    bar_width, label='Transform (%s)' % labels[1], edgecolor='white', linewidth=0.25, color='darkgreen')
    
    rectscuda1 = ax.bar(x + bar_width, data['cuda']['seed']['duration'],
                    bar_width, label='Seed (%s)' % labels[2], edgecolor='white', linewidth=0.25, color='lightgray')
    rectscuda2 = ax.bar(x + bar_width, data['cuda']['gen']['duration'],
                    bar_width, label='Generate (%s)' % labels[2], edgecolor='white', linewidth=0.25, color='gray')
    rectscuda3 = ax.bar(x + bar_width, data['cuda']['transform']['duration'],
                    bar_width, label='Transform (%s)' % labels[2], edgecolor='white', linewidth=0.25, color='black')

    ax.set_ylabel('Time [s]', fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xlabel('Batch size', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_data)
    plt.xticks(fontsize=10)

    leg = ax.legend(prop={'size': 10})
    leg_buf = ax.legend([rectsbuf1,rectsbuf2,rectsbuf3], kernels, loc='upper center', bbox_to_anchor=(0.35, 1.0), title='Buffer API', prop={'size': 7})
    leg_usm = ax.legend([rectsusm1,rectsusm2,rectsusm3], kernels, loc='upper center', bbox_to_anchor=(0.5, 1.0), title='USM API', prop={'size': 7})
    leg_cuda = ax.legend([rectscuda1,rectscuda2,rectscuda3], kernels, loc='upper center', bbox_to_anchor=(0.65, 1.0), title='Native CUDA', prop={'size': 7})

    ax.add_artist(leg_buf)
    ax.add_artist(leg_usm)
    
    fig.tight_layout()
    plt.yscale('log')
    plt.ylim(10e-6, 10e-1)
    plt.gcf().subplots_adjust(left=0.12)
    plt.savefig('onemkl_cuda_bar.png')
    plt.show()
    # plt.clf()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=250)
    rects1 = ax.bar(
        x - bar_width, data['buffer']['seed']['duration'], bar_width, label=labels[0])
    rects2 = ax.bar(x, data['usm']['seed']['duration'],
                    bar_width, label=labels[1])
    rects3 = ax.bar(
        x + bar_width, data['cuda']['seed']['duration'], bar_width, label=labels[2])

    ax.set_ylabel('Time [s]', fontsize=16)
    plt.yticks(fontsize=12)
    ax.set_xlabel('Batch size', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(x_data)
    plt.xticks(fontsize=12)
    ax.legend(prop={'size': 12})
    fig.tight_layout()
    plt.yscale('log')
    plt.ylim(10e-6, 10e-1)
    plt.gcf().subplots_adjust(left=0.12)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=250)
    rects1 = ax.bar(
        x - bar_width, data['buffer']['gen']['duration'], bar_width, label=labels[0], color='tomato')
    rects2 = ax.bar(x, data['usm']['gen']['duration'],
                    bar_width, label=labels[1], color='darkgreen')
    rects3 = ax.bar(
        x + bar_width, data['cuda']['gen']['duration'], bar_width, label=labels[2], color='black')

    ax.set_ylabel('Time [s]', fontsize=16)
    plt.yticks(fontsize=12)
    ax.set_xlabel('Batch size', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(x_data)
    plt.xticks(fontsize=12)
    ax.legend(prop={'size': 12})
    fig.tight_layout()
    plt.yscale('log')
    plt.ylim(10e-6, 10e-1)
    plt.gcf().subplots_adjust(left=0.12)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=250)
    rects1 = ax.bar(
        x - bar_width, data['buffer']['transform']['duration'], bar_width, label=labels[0])
    rects2 = ax.bar(x, data['usm']['transform']
                    ['duration'], bar_width, label=labels[1])
    rects3 = ax.bar(
        x + bar_width, data['cuda']['transform']['duration'], bar_width, label=labels[2])

    ax.set_ylabel('Time [s]', fontsize=16)
    plt.yticks(fontsize=12)
    ax.set_xlabel('Batch size', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(x_data)
    plt.xticks(fontsize=12)
    ax.legend(prop={'size': 12})
    fig.tight_layout()
    plt.yscale('log')
    plt.ylim(10e-6, 10e-1)
    plt.gcf().subplots_adjust(left=0.12)
    plt.show()


if __name__ == '__main__':
    main()
