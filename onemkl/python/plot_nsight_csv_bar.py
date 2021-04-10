#
# CSV header
#
# "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",
# "Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value" [11],
# "Rule Name","Rule Type","Rule Description"
#
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
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
                        val = val.replace(',', '')
                    val = float(val) * 10e-9
                    _dur.append(val)
                elif 'SM [%]' in row:
                    val = float(val)
                    _sm.append(val)
                elif 'Occupancy' in row:
                    val = float(val)
                    _occ.append(val)
            dur = np.mean(_dur)
            sm = np.mean(_sm)
            occ = np.mean(_occ)
            data[backend][kernel]['duration'].append(dur)
            data[backend][kernel]['sm'].append(sm)
            data[backend][kernel]['occupancy'].append(occ)

    # x_data = [r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$',
    #           r'$10^{4}$', r'$10^{5}$', r'$10^{6}$', r'$10^{7}$', r'$10^{8}$']
    x_data = [r'$10^{0}$', r'$10^{1}$', r'$10^{2}$',
              r'$10^{4}$', r'$10^{5}$', r'$10^{6}$', r'$10^{7}$', r'$10^{8}$']

    linestyle_seed = '-.'
    linestyle_gen = '--'
    linestyle_transform = ':'

    marker_buffer = '^'
    marker_usm = 'v'
    marker_cuda = 's'

    colors = ['thistle', 'plum', 'mediumslateblue', 'orchid', 'mediumorchid',
        'blueviolet', 'mediumpurple', 'darkviolet', 'rebeccapurple']

    alpha_buffer = 1.0
    alpha_usm = 0.5
    alpha_cuda = 0.3

    for backend in data.keys():
        for kernel in data[backend].keys():
            data[backend][kernel]['duration'].reverse()

    # Seed, duration
    kernels = ['Seed', 'Generate', 'Transform']
    labels = ['Buffer API', 'USM API', 'Native CUDA']

    bar_width = 0.3
    x = np.arange(len(x_data))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 4), dpi=250)
    ax.grid(zorder=0, color='black', linestyle='--',
            linewidth=0.25, axis='y', alpha=0.25)
    rectsbuf1 = ax.bar(x - bar_width, data['buffer']['seed']['duration'],
                    bar_width, label='Seed (%s)' % labels[0], edgecolor='white', linewidth=0.25, color=colors[0], zorder=5)
    rectsbuf2 = ax.bar(x - bar_width, data['buffer']['gen']['duration'],
                    bar_width, label='Generate (%s)' % labels[0], edgecolor='white', linewidth=0.25, color=colors[1], zorder=5, bottom=data['buffer']['seed']['duration'])
    rectsbuf3 = ax.bar(x - bar_width, data['buffer']['transform']['duration'],
                    bar_width, label='Transform (%s)' % labels[0], edgecolor='white', linewidth=0.25, color=colors[2], zorder=5,
                    bottom=np.array(data['buffer']['seed']['duration']) +
                                    np.array(data['buffer']['gen']['duration']))

    rectsusm1=ax.bar(x, data['usm']['seed']['duration'],
                    bar_width, label='Seed (%s)' % labels[1], edgecolor='white', linewidth=0.25, color=colors[3], zorder=5)
    rectsusm2=ax.bar(x, data['usm']['gen']['duration'],
                    bar_width, label='Generate (%s)' % labels[1], edgecolor='white', linewidth=0.25, color=colors[4], zorder=5, bottom=data['usm']['seed']['duration'])
    rectsusm3=ax.bar(x, data['usm']['transform']['duration'],
                    bar_width, label='Transform (%s)' % labels[1], edgecolor='white', linewidth=0.25, color=colors[5], zorder=5,
                    bottom=np.array(data['usm']['seed']['duration']) +
                                    np.array(data['usm']['gen']['duration']))

    rectscuda1=ax.bar(x + bar_width, data['cuda']['seed']['duration'],
                    bar_width, label='Seed (%s)' % labels[2], edgecolor='white', linewidth=0.25, color=colors[6], zorder=5)
    rectscuda2=ax.bar(x + bar_width, data['cuda']['gen']['duration'],
                    bar_width, label='Generate (%s)' % labels[2], edgecolor='white', linewidth=0.25, color=colors[7], zorder=5, bottom=data['cuda']['seed']['duration'])
    rectscuda3=ax.bar(x + bar_width, data['cuda']['transform']['duration'],
                    bar_width, label='Transform (%s)' % labels[2], edgecolor='white', linewidth=0.25, color=colors[8], zorder=5,
                    bottom=np.array(data['cuda']['seed']['duration']) +
                                    np.array(data['cuda']['gen']['duration']))

    ax.set_ylabel('Time [s]', fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xlabel('Batch size', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_data)
    plt.xticks(fontsize=10)

    leg=ax.legend(prop={'size': 12})
    leg_buf=ax.legend([rectsbuf1, rectsbuf2, rectsbuf3], kernels, loc='upper left', bbox_to_anchor=(
        0.05, 0.98), title='Buffer API', prop={'size': 8}, framealpha=1.0)
    leg_usm=ax.legend([rectsusm1, rectsusm2, rectsusm3], kernels, loc='upper left', bbox_to_anchor=(
        0.20, 0.98), title='USM API', prop={'size': 8}, framealpha=1.0)
    leg_cuda=ax.legend([rectscuda1, rectscuda2, rectscuda3], kernels, loc='upper left', bbox_to_anchor=(
        0.35, 0.98), title='Native CUDA', prop={'size': 8}, framealpha=1.0)

    ax.add_artist(leg_buf)
    ax.add_artist(leg_usm)

    fig.tight_layout()
    plt.yscale('log')
    plt.ylim(10e-6, 10e-1)
    plt.gcf().subplots_adjust(left=0.12)
    plt.savefig('compare_kernel_dur.png')
    plt.show()
    # plt.clf()


    for backend in data.keys():
        for kernel in data[backend].keys():
            data[backend][kernel]['occupancy'].reverse()
    
    buf_seed_occ = data['buffer']['seed']['occupancy']
    buf_gen_occ = data['buffer']['gen']['occupancy']
    buf_trans_occ = data['buffer']['transform']['occupancy']

    usm_seed_occ = data['usm']['seed']['occupancy']
    usm_gen_occ = data['usm']['gen']['occupancy']
    usm_trans_occ = data['usm']['transform']['occupancy']

    cuda_seed_occ = data['cuda']['seed']['occupancy']
    cuda_gen_occ = data['cuda']['gen']['occupancy']
    cuda_trans_occ = data['cuda']['transform']['occupancy']

    print(buf_seed_occ,     usm_seed_occ,   cuda_seed_occ)
    print(buf_gen_occ,      usm_gen_occ,    cuda_gen_occ)
    print(buf_trans_occ,    usm_trans_occ,  cuda_trans_occ)

    sum_buf_seed_occ    = np.sum(buf_seed_occ)
    sum_buf_gen_occ     = np.sum(buf_gen_occ)
    sum_buf_trans_occ   = np.sum(buf_trans_occ)

    sum_usm_seed_occ    = np.sum(usm_seed_occ)
    sum_usm_gen_occ     = np.sum(usm_gen_occ)
    sum_usm_trans_occ   = np.sum(usm_trans_occ)

    sum_cuda_seed_occ   = np.sum(cuda_seed_occ)
    sum_cuda_gen_occ    = np.sum(cuda_gen_occ)
    sum_cuda_trans_occ  = np.sum(cuda_trans_occ)

    sum_buf_all         = np.sum(np.array(sum_buf_seed_occ)+np.array(sum_buf_gen_occ)+np.array(sum_buf_trans_occ))
    sum_usm_all         = np.sum(np.array(sum_usm_seed_occ)+np.array(sum_usm_gen_occ)+np.array(sum_usm_trans_occ))
    sum_cuda_all        = np.sum(np.array(sum_cuda_seed_occ)+np.array(sum_cuda_gen_occ)+np.array(sum_cuda_trans_occ))

    norm_buf_seed_occ   = np.array(buf_seed_occ)    / sum_buf_all * 4.10
    norm_buf_gen_occ    = np.array(buf_gen_occ)     / sum_buf_all * 4.10
    norm_buf_trans_occ  = np.array(buf_trans_occ)   / sum_buf_all * 4.10
    norm_usm_seed_occ   = np.array(usm_seed_occ)    / sum_usm_all * 4.10
    norm_usm_gen_occ    = np.array(usm_gen_occ)     / sum_usm_all * 4.10
    norm_usm_trans_occ  = np.array(usm_trans_occ)   / sum_usm_all * 4.10
    norm_cuda_seed_occ  = np.array(cuda_seed_occ)   / sum_cuda_all * 4.10
    norm_cuda_gen_occ   = np.array(cuda_gen_occ)    / sum_cuda_all * 4.10
    norm_cuda_trans_occ = np.array(cuda_trans_occ)  / sum_cuda_all * 4.10

    fig, ax=plt.subplots(figsize=(8, 4), dpi=250)
    ax.grid(zorder=0, color='black', linestyle='--',
            linewidth=0.25, axis='y', alpha=0.25)
    rectsbuf1=ax.bar(x - bar_width, norm_buf_seed_occ,
                    bar_width, label='Seed (%s)' % labels[0], edgecolor='white', linewidth=0.25, color=colors[0], zorder=5)
    rectsbuf2=ax.bar(x - bar_width, norm_buf_gen_occ,
                    bar_width, label='Generate (%s)' % labels[0], edgecolor='white', linewidth=0.25, color=colors[1], zorder=5, bottom=norm_buf_seed_occ)
    rectsbuf3=ax.bar(x - bar_width, norm_buf_trans_occ,
                    bar_width, label='Transform (%s)' % labels[0], edgecolor='white', linewidth=0.25, color=colors[2], zorder=5,
                    bottom=norm_buf_seed_occ + norm_buf_gen_occ)

    rectsusm1=ax.bar(x, norm_usm_seed_occ,
                    bar_width, label='Seed (%s)' % labels[1], edgecolor='white', linewidth=0.25, color=colors[3], zorder=5)
    rectsusm2=ax.bar(x, norm_usm_gen_occ,
                    bar_width, label='Generate (%s)' % labels[1], edgecolor='white', linewidth=0.25, color=colors[4], zorder=5, bottom=norm_usm_seed_occ)
    rectsusm3=ax.bar(x, norm_usm_trans_occ,
                    bar_width, label='Transform (%s)' % labels[1], edgecolor='white', linewidth=0.25, color=colors[5], zorder=5,
                    bottom=norm_usm_seed_occ + norm_usm_gen_occ)

    rectscuda1=ax.bar(x + bar_width, norm_cuda_seed_occ,
                    bar_width, label='Seed (%s)' % labels[2], edgecolor='white', linewidth=0.25, color=colors[6], zorder=5)
    rectscuda2=ax.bar(x + bar_width, norm_cuda_gen_occ,
                    bar_width, label='Generate (%s)' % labels[2], edgecolor='white', linewidth=0.25, color=colors[7], zorder=5, bottom=norm_cuda_seed_occ)
    rectscuda3=ax.bar(x + bar_width, norm_cuda_trans_occ,
                    bar_width, label='Transform (%s)' % labels[2], edgecolor='white', linewidth=0.25, color=colors[8], zorder=5,
                    bottom=norm_cuda_seed_occ + norm_cuda_gen_occ)

    ax.set_ylabel('Relative Occupancy [A.U.]', fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xlabel('Batch size', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_data)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.xticks(fontsize=10)

    leg=ax.legend(prop={'size': 12})
    leg_buf=ax.legend([rectsbuf1, rectsbuf2, rectsbuf3], kernels, loc='upper left', bbox_to_anchor=(
        0.05, 0.98), title='Buffer API', prop={'size': 8}, framealpha=1.0)
    leg_usm=ax.legend([rectsusm1, rectsusm2, rectsusm3], kernels, loc='upper left', bbox_to_anchor=(
        0.20, 0.98), title='USM API', prop={'size': 8}, framealpha=1.0)
    leg_cuda=ax.legend([rectscuda1, rectscuda2, rectscuda3], kernels, loc='upper left', bbox_to_anchor=(
        0.35, 0.98), title='Native CUDA', prop={'size': 8}, framealpha=1.0)

    ax.add_artist(leg_buf)
    ax.add_artist(leg_usm)

    fig.tight_layout()
    # plt.yscale('log')
    plt.ylim(0.0, 1.0)
    plt.gcf().subplots_adjust(left=0.12)
    plt.savefig('compare_kernel_occ.png')
    plt.show()


if __name__ == '__main__':
    main()
