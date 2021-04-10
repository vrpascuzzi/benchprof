#
#
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np

import csv
import sys

distributions = ['uniform_float', 'uniform_double',
                 'gaussian_float', 'gaussian_double', 'lognormal_float', 'bits_int']
# x_data = [r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$',
#           r'$10^{4}$', r'$10^{5}$', r'$10^{6}$', r'$10^{7}$', r'$10^{8}$']


def read_data(inp_files):
    data_dict = {
        'amdcpu':      {
            'type': 'cpu',
            'name': 'AMD Rome 7742',
            'data': {
                'philox':   {
                    'buffer': {},
                    'usm': {},
                    'native': {},
                },
                'mrg':      {
                    'buffer': {},
                    'usm': {},
                    'native': {},
                }
            }
        },
        'intelcpu':    {
            'type': 'cpu',
            'name': 'Intel(R) Core(TM) i7-1080H',
            'data': {
                'philox':   {
                    'buffer': {},
                    'usm': {},
                    'native': {},
                },
                'mrg':      {
                    'buffer': {},
                    'usm': {},
                    'native': {},
                }
            }
        },
        'intelgpu':    {
            'type': 'gpu',
            'name': 'Intel(R) UHD Graphics 630',
            'data': {
                'philox':   {
                    'buffer': {},
                    'usm': {},
                    'native': {},
                },
                'mrg':      {
                    'buffer': {},
                    'usm': {},
                    'native': {},
                }
            }
        },
        'vega56':      {
            'type': 'gpu',
            'name': 'MSI Radeon RX Vega 56',
            'data': {
                'philox':   {
                    'buffer': {},
                    'usm': {},
                    'native': {},
                },
                'mrg':      {
                    'buffer': {},
                    'usm': {},
                    'native': {},
                }
            }
        },
        'a100':        {
            'name': 'NVIDIA A100',
            'type': 'gpu',
            'data': {
                'philox':   {
                    'buffer': {},
                    'usm': {},
                    'native': {},
                },
                'mrg':      {
                    'buffer': {},
                    'usm': {},
                    'native': {},
                }
            }
        }
    }

    # Loop input files
    for f in inp_files:
        # Primary key is the device name
        _pk = f.split('/')[1].split('_')[0]
        _gen_key = f.split('/')[1].split('_')[-1].split('.')[0]

        # Check if we have a oneMKL input or native
        _api_key = ''
        if 'mkl' in f:
            _api_key = f.split('/')[1].split('_')[5]
        else:
            _api_key = 'native'

        # Start parsing data; open file as csv
        with open(f) as csvfile:
            reader = csv.reader(csvfile)

            # Local variable to hold clock data
            clock_dict = {'uniform_float':
                          {'total': [], 'kernel': []},
                          'uniform_double':
                          {'total': [], 'kernel': []},
                          'gaussian_float':
                          {'total': [], 'kernel': []},
                          'gaussian_double':
                          {'total': [], 'kernel': []},
                          'lognormal_float':
                          {'total': [], 'kernel': []},
                          'bits_int':
                          {'total': [], 'kernel': []},
                          }

            # Get metric value
            for row in reader:
                _distr_key = str(row[0])
                clock_total = float(row[4])
                clock_kernel = float(row[7])
                clock_dict[_distr_key]['total'].append(clock_total)
                clock_dict[_distr_key]['kernel'].append(clock_kernel)

            # Add data to the data dictionary
            data_dict[_pk]['data'][_gen_key][_api_key] = clock_dict

    return data_dict


# def plot_data(data_as_dict, gen_key='philox', distr_key='uniform_float', api_key='buffer', clock_key='total', show_plot=False, platform=None):
def plot_data(data_as_dict):
    """Plot data.

    Args:
        data_as_dict
            dictionary of data
        gen_key
            key for generator type; 'philox' or 'mrg'
        distr_key
            distribution to pull clock data from
        platform
            'cpu', 'gpu' or NoneType; if NoneType, plot all platforms
    """
    x_data = [r'$10^{0}$', r'$10^{1}$', r'$10^{2}$',
              r'$10^{4}$', r'$10^{5}$', r'$10^{6}$', r'$10^{7}$', r'$10^{8}$']
    x = np.arange(len(x_data))
    bar_width = 0.3
    labels = ['Buffer API', 'USM API', 'Native CUDA']

    # Loop over gens
    for gen_key in ['philox', 'mrg']:
        # Loop over distributions
        for distr_key in distributions:
            for api_key in ['buffer', 'usm']:
                fig, ax = plt.subplots(figsize=(8, 4), dpi=250)
                # Each x_data point (batch size) will include the platforms specified
                colors = ['tomato', 'blue', 'seagreen', 'darkred', 'gray', 'black']
                rects = []
                space_index = 1
                color_index = 0
                for pk in data_as_dict.keys():

                    if pk in ['amdcpu', 'intelcpu', 'intelgpu']:
                        # print(pk)
                        name = data_as_dict[pk]['name']
                        total_clock_data = data_as_dict[pk]['data'][gen_key][api_key][distr_key]['total']
                        rect = ax.bar(x - space_index * bar_width, total_clock_data,
                                             bar_width, label=name, edgecolor='white', linewidth=0.25, color=colors[color_index], alpha=0.8)
                        rects.append(rect)
                        space_index -= 1
                        color_index += 1

                ax.set_ylabel('Time [s]', fontsize=12)
                plt.yticks(fontsize=12)
                ax.set_xlabel('Batch size', fontsize=12)
                ax.set_xticks(x)
                ax.set_xticklabels(x_data)
                plt.xticks(fontsize=10)
                leg = ax.legend(loc='upper left', prop={'size': 10})
                fig.tight_layout()
                plt.yscale('log')
                plt.ylim(10e-7, 10e0)
                plt.gcf().subplots_adjust(left=0.12)
                ofname = 'mkl_rng_%s_%s_%s.png' % (gen_key, distr_key, api_key)
                plt.savefig('plots/%s' % ofname)

    # ###########################################
    # # AMD GPU COMPARISONS
    # ###########################################

    pk = 'vega56'
    name = data_as_dict[pk]['name']

    # Loop over gens
    for gen_key in ['philox', 'mrg']:
        for distr_key in distributions:

            fig, ax = plt.subplots(figsize=(8, 4), dpi=250)
            colors = ['steelblue', 'dodgerblue', 'darkblue']
            rects = []

            clock_total_buffer_data = data_as_dict[pk]['data'][gen_key]['buffer'][distr_key]['total']
            clock_total_usm_data = data_as_dict[pk]['data'][gen_key]['usm'][distr_key]['total']
            clock_total_native_data = data_as_dict[pk]['data'][gen_key]['native'][distr_key]['total']

            clock_kernel_buffer_data = data_as_dict[pk]['data'][gen_key]['buffer'][distr_key]['kernel']
            clock_kernel_usm_data = data_as_dict[pk]['data'][gen_key]['usm'][distr_key]['kernel']
            clock_kernel_native_data = data_as_dict[pk]['data'][gen_key]['native'][distr_key]['kernel']

            rectbuf = ax.bar(x -  bar_width, clock_total_buffer_data,
                                 bar_width, label='Buffer API', edgecolor='white', linewidth=0.25, color=colors[0], alpha=0.8)
            # rect = ax.bar(x -  bar_width, clock_kernel_buffer_data,
            #                      bar_width, label=name, edgecolor='white', linewidth=0.25, color=colors[0])
            rectusm = ax.bar(x, clock_total_usm_data,
                                 bar_width, label='USM API', edgecolor='white', linewidth=0.25, color=colors[1], alpha=0.8)
            # rect = ax.bar(x, clock_kernel_usm_data,
            #                      bar_width, label=name, edgecolor='white', linewidth=0.25, color=colors[1])
            rectnative = ax.bar(x +  bar_width, clock_total_native_data,
                                 bar_width, label='Native', edgecolor='white', linewidth=0.25, color=colors[2], alpha=0.8)
            # rect = ax.bar(x +  bar_width, clock_kernel_native_data,
            #                      bar_width, label=name, edgecolor='white', linewidth=0.25, color=colors[2])

            ax.set_ylabel('Time [s]', fontsize=12)
            plt.yticks(fontsize=12)
            ax.set_xlabel('Batch size', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(x_data)
            plt.xticks(fontsize=10)

            leg = ax.legend(loc='upper left', prop={'size': 10})

            fig.tight_layout()
            plt.yscale('log')
            plt.ylim(10e-7, 10e0)
            plt.gcf().subplots_adjust(left=0.12)
            ofname = 'compare_all_%s_%s_%s.png' % (pk, gen_key, distr_key)
            plt.savefig('plots/%s' % ofname)

    # # ###########################################
    # # # CUDA GPU COMPARISONS
    # # ###########################################

    pk = 'a100'
    name = data_as_dict[pk]['name']

    # Loop over gens
    for gen_key in ['philox', 'mrg']:
        # Loop over distributions
        for distr_key in distributions:

            fig, ax = plt.subplots(figsize=(8, 4), dpi=250)
            colors = ['plum', 'darkviolet', 'indigo']
            rects = []

            clock_total_buffer_data = data_as_dict[pk]['data'][gen_key]['buffer'][distr_key]['total']
            clock_total_usm_data = data_as_dict[pk]['data'][gen_key]['usm'][distr_key]['total']
            clock_total_native_data = data_as_dict[pk]['data'][gen_key]['native'][distr_key]['total']

            clock_kernel_buffer_data = data_as_dict[pk]['data'][gen_key]['buffer'][distr_key]['kernel']
            clock_kernel_usm_data = data_as_dict[pk]['data'][gen_key]['usm'][distr_key]['kernel']
            clock_kernel_native_data = data_as_dict[pk]['data'][gen_key]['native'][distr_key]['kernel']

            rectbuf = ax.bar(x -  bar_width, clock_total_buffer_data,
                                 bar_width, label='Buffer API', edgecolor='white', linewidth=0.25, color=colors[0], alpha=0.8)
            # rect = ax.bar(x -  bar_width, clock_kernel_buffer_data,
            #                      bar_width, label=name, edgecolor='white', linewidth=0.25, color=colors[0])
            rectusm = ax.bar(x, clock_total_usm_data,
                                 bar_width, label='USM API', edgecolor='white', linewidth=0.25, color=colors[1], alpha=0.8)
            # rect = ax.bar(x, clock_kernel_usm_data,
            #                      bar_width, label=name, edgecolor='white', linewidth=0.25, color=colors[1])
            rectnative = ax.bar(x +  bar_width, clock_total_native_data,
                                 bar_width, label='Native', edgecolor='white', linewidth=0.25, color=colors[2], alpha=0.8)
            # rect = ax.bar(x +  bar_width, clock_kernel_native_data,
            #                      bar_width, label=name, edgecolor='white', linewidth=0.25, color=colors[2])

            ax.set_ylabel('Time [s]', fontsize=12)
            plt.yticks(fontsize=12)
            ax.set_xlabel('Batch size', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(x_data)
            plt.xticks(fontsize=10)

            leg = ax.legend(loc='upper left', prop={'size': 10})

            fig.tight_layout()
            plt.yscale('log')
            plt.ylim(10e-7, 10e0)
            plt.gcf().subplots_adjust(left=0.12)
            ofname = 'compare_all_%s_%s_%s.png' % (pk, gen_key, distr_key)
            plt.savefig('plots/%s' % ofname)

    ###########################################
    # FastCaloSim
    ###########################################
    x_data = ['AMD Rome 7742', 'Intel Core i7-1080H', 'MSI Radeon RX Vega 56', 'NVIDIA A100']
    type = ['SYCL', 'Native']
    colors = ['tomato', 'salmon', 'dodgerblue', 'lightsteelblue', 'maroon', 'seagreen', 'mediumaquamarine']
    x = np.arange(len(x_data))
    bar_width = 0.3

    # AMD CPU, Intel CPU, Vega56, A100
    fcs_singlee_data = [17.5883, 14.0385, 4.10102, 2.74803372]
    fcs_ttbar_data = [29.1012, 27.8678, 30.2191, 26.4782631]
    fcs_singlee_data_native = [18.4631, 13.5987, 2.8499618]
    fcs_ttbar_data_native = [32.611713, 24.6261, 26.5545094]

    # Two plots, one for single-electron and another for ttbar
    fig, ax = plt.subplots(figsize=(8, 4), dpi=250)

    ax.grid(zorder=0, color='black', linestyle='--', linewidth=0.25, axis='y', alpha=0.25)
    # Put data into rects
    rect_amdcpu = ax.bar(x[0] - 0.5 * bar_width, fcs_singlee_data[0],
                         bar_width, label='%s (SYCL)' % x_data[0], edgecolor='white', linewidth=0.25, color=colors[0], zorder=5)
    rect_amdcpu_native = ax.bar(x[0] + 0.5*bar_width, fcs_singlee_data_native[0],
                                bar_width, label='%s (Native)' % x_data[0], edgecolor='white', linewidth=0.25, color=colors[1], zorder=5)

    rect_intelcpu = ax.bar(x[1] - 0.5 * bar_width, fcs_singlee_data[1],
                           bar_width, label='%s (SYCL)' % x_data[1], edgecolor='white', linewidth=0.25, color=colors[2], zorder=5)
    rect_intelcpu_native = ax.bar(x[1] + 0.5 * bar_width, fcs_singlee_data_native[1],
                                  bar_width, label='%s (Native)' % x_data[1], edgecolor='white', linewidth=0.25, color=colors[3], zorder=5)

    rect_vega = ax.bar(x[2], fcs_singlee_data[2],
                       bar_width, label='%s (SYCL)' % x_data[2], edgecolor='white', linewidth=0.25, color=colors[4], zorder=5)

    rect_a100 = ax.bar(x[3] - 0.5 * bar_width, fcs_singlee_data[3],
                       bar_width, label='%s (SYCL)' % x_data[3], edgecolor='white', linewidth=0.25, color=colors[5], zorder=5)
    rect_a100_native = ax.bar(x[3] + 0.5 * bar_width, fcs_singlee_data_native[2],
                       bar_width, label='%s (Native)' % x_data[3], edgecolor='white', linewidth=0.25, color=colors[6], zorder=5)

    ax.set_ylabel('Simulation Time [s]', fontsize=12)
    plt.yticks(fontsize=10)
    ax.set_xlabel('Platform', fontsize=12)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_xticklabels('')
    ax.yaxis.set_major_locator(MultipleLocator(5))
    plt.xticks(fontsize=8, rotation=45)

    leg = ax.legend(prop={'size': 8})
    leg_amdcpu = ax.legend([rect_amdcpu, rect_amdcpu_native], type, loc='upper left', bbox_to_anchor=(0.033, 0.84), title=x_data[0], prop={'size': 7}, facecolor='white', framealpha=1.0)
    leg_amdcpu.get_title().set_fontsize('9')
    leg_intelcpu = ax.legend([rect_intelcpu, rect_intelcpu_native], type, loc='upper left', bbox_to_anchor=(0.28, 0.84), title=x_data[1], prop={'size': 7}, facecolor='white', framealpha=1.0)
    leg_intelcpu.get_title().set_fontsize('9')
    leg_vega = ax.legend([rect_vega], type, loc='upper left', bbox_to_anchor=(0.51, 0.82), title=x_data[2], prop={'size': 7}, facecolor='white', framealpha=1.0)
    leg_vega.get_title().set_fontsize('9')
    leg_a100 = ax.legend([rect_a100, rect_a100_native], type, loc='upper left', bbox_to_anchor=(0.81, 0.84), title=x_data[3], prop={'size': 7}, facecolor='white', framealpha=1.0)
    leg_a100.get_title().set_fontsize('9')
    ax.add_artist(leg_amdcpu)
    ax.add_artist(leg_intelcpu)
    ax.add_artist(leg_vega)
    
    fig.tight_layout()
    # plt.yscale('log')
    plt.ylim(0.0, 60.0)
    plt.gcf().subplots_adjust(left=0.12)
    ofname = 'compare_all_fcs_singlee.png'
    plt.savefig('plots/%s' % ofname)
    plt.show()
    # plt.clf()

    # ttbar
    fig, ax = plt.subplots(figsize=(8, 4), dpi=250)

    ax.grid(zorder=0, color='black', linestyle='--', linewidth=0.25, axis='y', alpha=0.25)
    # Put data into rects
    rect_amdcpu = ax.bar(x[0] - 0.5 * bar_width, fcs_ttbar_data[0],
                         bar_width, label='%s (SYCL)' % x_data[0], edgecolor='white', linewidth=0.25, color=colors[0], zorder=5)
    rect_amdcpu_native = ax.bar(x[0] + 0.5*bar_width, fcs_ttbar_data_native[0],
                                bar_width, label='%s (Native)' % x_data[0], edgecolor='white', linewidth=0.25, color=colors[1], zorder=5)

    rect_intelcpu = ax.bar(x[1] - 0.5 * bar_width, fcs_ttbar_data[1],
                           bar_width, label='%s (SYCL)' % x_data[1], edgecolor='white', linewidth=0.25, color=colors[2], zorder=5)
    rect_intelcpu_native = ax.bar(x[1] + 0.5 * bar_width, fcs_ttbar_data_native[1],
                                  bar_width, label='%s (Native)' % x_data[1], edgecolor='white', linewidth=0.25, color=colors[3], zorder=5)

    rect_vega = ax.bar(x[2], fcs_ttbar_data[2],
                       bar_width, label='%s (SYCL)' % x_data[2], edgecolor='white', linewidth=0.25, color=colors[4], zorder=5)

    rect_a100 = ax.bar(x[3] - 0.5 * bar_width, fcs_ttbar_data[3],
                       bar_width, label='%s (SYCL)' % x_data[3], edgecolor='white', linewidth=0.25, color=colors[5], zorder=5)
    rect_a100_native = ax.bar(x[3] + 0.5 * bar_width, fcs_ttbar_data_native[2],
                       bar_width, label='%s (Native)' % x_data[3], edgecolor='white', linewidth=0.25, color=colors[6], zorder=5)

    ax.set_ylabel('Simulation Time [s]', fontsize=12)
    plt.yticks(fontsize=10)
    ax.set_xlabel('Platform', fontsize=12)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.set_xticklabels('')
    ax.yaxis.set_major_locator(MultipleLocator(5))
    plt.xticks(fontsize=8, rotation=45)

    leg = ax.legend(prop={'size': 8})
    leg_amdcpu = ax.legend([rect_amdcpu, rect_amdcpu_native], type, loc='upper left', bbox_to_anchor=(0.033, 0.84), title=x_data[0], prop={'size': 7}, facecolor='white', framealpha=1.0)
    leg_amdcpu.get_title().set_fontsize('9')
    leg_intelcpu = ax.legend([rect_intelcpu, rect_intelcpu_native], type, loc='upper left', bbox_to_anchor=(0.28, 0.84), title=x_data[1], prop={'size': 7}, framealpha=1.0)
    leg_intelcpu.get_title().set_fontsize('9')
    leg_vega = ax.legend([rect_vega], type, loc='upper left', bbox_to_anchor=(0.51, 0.82), title=x_data[2], prop={'size': 7}, framealpha=1.0)
    leg_vega.get_title().set_fontsize('9')
    leg_a100 = ax.legend([rect_a100, rect_a100_native], type, loc='upper left', bbox_to_anchor=(0.81, 0.84), title=x_data[3], prop={'size': 7}, framealpha=1.0)
    leg_a100.get_title().set_fontsize('9')
    ax.add_artist(leg_amdcpu)
    ax.add_artist(leg_intelcpu)
    ax.add_artist(leg_vega)
    
    fig.tight_layout()
    # plt.yscale('log')
    plt.ylim(0.0, 60.0)
    plt.gcf().subplots_adjust(left=0.12)
    ofname = 'compare_all_fcs_ttbar.png'
    plt.savefig('plots/%s' % ofname)
    plt.show()


if __name__ == '__main__':
    # Parse args
    inp_files = sys.argv[1:]
    # Read data
    data_dict = read_data(inp_files)
    # Plot data
    plot_data(data_dict)
