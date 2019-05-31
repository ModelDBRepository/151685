#!/usr/bin/env python
""" plot.py - plotting routines for the basic figures
    Copyright (C) 2013 Shane Lee and Stephanie Jones

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import os
import fileio
import specfn
import spikefn

class FigTemplate():
    """ Figure template class for basic figure
    """
    def __init__(self):
        mpl.rc('font', size=8)
        self.f = plt.figure(figsize=(7.5, 10))

        self.gs = {
            'all': gs.GridSpec(8, 100, left=0.15, right=0.95, bottom=0.1, top=0.95),
        }

        self.ax = {
            'raster_L2': self.f.add_subplot(self.gs['all'][:2, :80]),
            'raster_L5': self.f.add_subplot(self.gs['all'][2:4, :80]),
            'dipole_L2': self.f.add_subplot(self.gs['all'][4:5, :80]),
            'dipole_L5': self.f.add_subplot(self.gs['all'][5:6, :80]),
            'spec_dipole': self.f.add_subplot(self.gs['all'][6:, :]),
        }
        self.labels()

    def labels(self):
        self.ax['raster_L2'].set_ylabel('L2 pyramidal (black), fs (red)')
        self.ax['raster_L2'].set_xticklabels([])

        self.ax['raster_L5'].set_ylabel('L5 pyramidal (black), fs (red)')
        self.ax['raster_L5'].set_xticklabels([])

        self.ax['dipole_L2'].set_ylabel('L2 dipole (nAm)')
        self.ax['dipole_L2'].set_xticklabels([])

        self.ax['dipole_L5'].set_ylabel('L5 dipole (nAm)')
        self.ax['dipole_L5'].set_xticklabels([])

        self.ax['spec_dipole'].set_ylabel('Frequency (Hz)')
        self.ax['spec_dipole'].set_xlabel('Time (ms)')

    def close(self):
        plt.close(self.f)

    def save(self, fpng):
        self.f.savefig(fpng, dpi=250)

def plot_simulation(dsub):
    d = os.path.join(os.getcwd(), 'data', dsub)
    f = os.path.join(d, "data.pkl")
    fspikes = os.path.join(d, "spikes.txt")
    fpng = os.path.join(d, "spec.png")

    x = fileio.pkl_load(f)

    # dt is given here in ms
    fs = 1000. / x['p']['dt']

    spikes = spikefn.spikes_from_file(fspikes, x['gid_dict'])

    n = dict.fromkeys(spikes)
    for celltype in spikes:
        n[celltype] = len(spikes[celltype])

    # get total counts
    N_L2 = n['L2_basket'] + n['L2_pyramidal']
    N_L5 = n['L5_basket'] + n['L5_pyramidal']

    yticks = {
        'L2': np.linspace(0, 1, N_L2 + 2),
        'L5': np.linspace(0, 1, N_L5 + 2),
    }

    ind_L2_pyr = np.arange(0, N_L2, 1)[:n['L2_pyramidal']]
    ind_L2_inh = np.arange(0, N_L2, 1)[n['L2_pyramidal']:]

    ind_L5_pyr = np.arange(0, N_L5, 1)[:n['L5_pyramidal']]
    ind_L5_inh = np.arange(0, N_L5, 1)[n['L5_pyramidal']:]

    fig = FigTemplate()

    # L2 spikes
    for i, spk_cell in zip(ind_L2_pyr, spikes['L2_pyramidal']):
        y = yticks['L2'][i] * np.ones(len(spk_cell))
        fig.ax['raster_L2'].scatter(spk_cell, y, marker='|', s=2, color='k')

    for i, spk_cell in zip(ind_L2_inh, spikes['L2_basket']):
        y = yticks['L2'][i] * np.ones(len(spk_cell))
        fig.ax['raster_L2'].scatter(spk_cell, y, marker='|', s=2, color='r')

    # L5 spikes
    for i, spk_cell in zip(ind_L5_pyr, spikes['L5_pyramidal']):
        y = yticks['L5'][i] * np.ones(len(spk_cell))
        fig.ax['raster_L5'].scatter(spk_cell, y, marker='|', s=2, color='k')

    for i, spk_cell in zip(ind_L5_inh, spikes['L5_basket']):
        y = yticks['L5'][i] * np.ones(len(spk_cell))
        fig.ax['raster_L5'].scatter(spk_cell, y, marker='|', s=2, color='r')

    fig.ax['raster_L2'].set_ylim((yticks['L2'][0], yticks['L2'][-1]))
    fig.ax['raster_L5'].set_ylim((yticks['L5'][0], yticks['L5'][-1]))

    # dipole
    fig.ax['dipole_L2'].plot(x['t'], x['dipole_L2'])
    fig.ax['dipole_L5'].plot(x['t'], x['dipole_L5'])

    ylims = fig.ax['dipole_L5'].get_ylim()
    fig.ax['dipole_L2'].set_ylim(ylims)

    pc = specfn.pspec_ax(fig.ax['spec_dipole'], x['fspec'], x['spec'], (x['t'][0], x['t'][-1]))
    fig.f.colorbar(pc, ax=fig.ax['spec_dipole'])

    for axh in fig.ax:
        fig.ax[axh].set_xlim((x['t'][0], x['t'][-1]))

    fig.save(fpng)
    print("Saved file {}".format(fpng))
    fig.close()

if __name__ == '__main__':
    # fig = FigTemplate()
    # fig.save('testing.png')
    d = 'gamma_L5weak_L2weak'
    plot_simulation(d)
