""" run.py - runtime for the cortical dipole simulation
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

import numpy as np
import os
import shutil
import sys
import time

from mpi4py import MPI
from neuron import h as nrn
nrn.load_file("stdrun.hoc")

import fileio as fio
import network
import params_default
import plot
import specfn

# spike write function
def spikes_write(net, filename_spikes):
    pc = nrn.ParallelContext()

    for rank in range(int(pc.nhost())):
        # guarantees node order and no competition
        pc.barrier()

        if rank == int(pc.id()):
            # net.spiketimes and net.spikegids are type nrn.Vector()
            L = int(net.spikegids.size())
            with open(filename_spikes, 'a') as file_spikes:
                for i in range(L):
                    file_spikes.write('%3.2f\t%d\n' % (net.spiketimes.x[i], net.spikegids.x[i]))

    # let all nodes iterate through loop in which only one rank writes
    pc.barrier()

# compares params against a default dict
def compare_params(f, rank):
    p = params_default.get_params_default()

    lines = fio.clean_lines(f)

    # ignore comments
    lines = [line for line in lines if line[0] is not '#']

    params_bad = []

    for line in lines:
        # splits line by ':'
        param, val = line.split(": ")

        if param in p:
            try:
                p[param] = float(val)

            except ValueError:
                p[param] = str(val)

        else:
            params_bad.append(param)

    # with the added bonus of saving this time to the indiv params
    for param, val in p.items():
        if param.startswith('tstop_'):
            try:
                if val == -1:
                    p[param] = p['tstop']

            except:
                pass

    if rank == 0:
        if params_bad:
            print(params_bad)

    return p

# All units for time: ms
def exec_runsim(fparam):
    pc = nrn.ParallelContext()
    rank = int(pc.id())

    # just a simple timing
    if rank == 0:
        t_start = time.time()

    # absolute path for param file
    fparam = os.path.join(os.getcwd(), fparam)
    simname = os.path.basename(fparam).split('.')[0]

    # creates param dict from default
    p = compare_params(fparam, rank)

    ddata = {
        'base': os.path.join(os.getcwd(), 'data'),
    }
    ddata['sim'] = os.path.join(ddata['base'], simname)

    # spike file needs to be known by all nodes
    file_spikes_tmp = os.path.join(ddata['sim'], 'spikes_tmp.txt')

    # create the data directory
    if rank == 0:
        # create directories that do not exist
        if not os.path.isdir(ddata['base']):
            os.mkdir(ddata['base'])

        if not os.path.isdir(ddata['sim']):
            os.mkdir(ddata['sim'])

        # copy the param file over
        fparam_new = os.path.join(ddata['sim'], 'params.txt')
        shutil.copy(fparam, fparam_new)

        fdata = os.path.join(ddata['sim'], 'data.pkl')
        fspikes = os.path.join(ddata['sim'], 'spikes.txt')

    # get all nodes to this place before continuing
    # tries to ensure we're all running the same params at the same time
    pc.barrier()
    pc.gid_clear()

    # global variable for all nodes
    nrn("dp_total_L2 = 0.")
    nrn("dp_total_L5 = 0.")

    # Set tstop before instantiating any classes
    nrn.tstop = p['tstop']
    nrn.dt = p['dt']

    # Create network from network's Network class
    net = network.NetworkOnNode(p)

    # set t vec to record
    t_vec = nrn.Vector()
    t_vec.record(nrn._ref_t)

    # set dipoles to record
    dp_rec_L2 = nrn.Vector()
    dp_rec_L2.record(nrn._ref_dp_total_L2)

    # L5 dipole
    dp_rec_L5 = nrn.Vector()
    dp_rec_L5.record(nrn._ref_dp_total_L5)

    # sets the default max solver step in ms (purposefully large)
    pc.set_maxstep(10)

    # initialize cells after all the NetCon delays have been specified
    nrn.finitialize()
    nrn.fcurrent()

    # set state variables if they have been changed since nrn.finitialize
    nrn.frecord_init()

    # actual simulation
    pc.psolve(nrn.tstop)

    # combine dp_rec, this combines on every proc
    # 1 refers to adding the contributions together
    pc.allreduce(dp_rec_L2, 1)
    pc.allreduce(dp_rec_L5, 1)

    # aggregate the currents independently on each proc
    net.aggregate_currents()

    # combine the net.current{} variables on each proc
    pc.allreduce(net.current['L5Pyr_soma'], 1)
    pc.allreduce(net.current['L2Pyr_soma'], 1)

    # write output spikes
    spikes_write(net, file_spikes_tmp)

    # write time and calculated dipole to data file only if on the first proc
    # only execute this statement on one proc
    if rank == 0:
        # dt is in ms, fs in Hz
        fs = 1000. / p['dt']

        # also run the spec here. convert dipole from fAm to nAm
        t = np.array(t_vec.x)
        dpl_L2 = 1e-6 * np.array(dp_rec_L2.x)
        dpl_L5 = 1e-6 * np.array(dp_rec_L5.x)
        dpl_agg = dpl_L2 + dpl_L5

        # calculate the morlet spec
        morlet = specfn.MorletSpec(dpl_agg, fs)

        # package and save data
        data = {
            't': t,
            'fs': fs,
            'p': p,
            'gid_dict': net.gid_dict,
            'fspec': morlet.f,
            'spec': morlet.TFR,

            # conversion from fAm to nAm, see mod file
            'dipole_L2': dpl_L2,
            'dipole_L5': dpl_L5,

            'current_L2Pyr_soma': np.array(net.current['L2Pyr_soma'].x),
            'current_L5Pyr_soma': np.array(net.current['L5Pyr_soma'].x),
        }
        fio.pkl_save(fdata, data)

        # move the spike file to the spike dir
        shutil.move(file_spikes_tmp, fspikes)
        tsim = time.time() - t_start
        plot.plot_simulation(simname)
        print("... finished in: {:4.4f} s".format(tsim))

    nrn.quit()

    return simname

if __name__ == "__main__":
    # reads the specified param file
    try:
        fparam = sys.argv[1]

    except (IndexError):
        print("Usage: {} param_input".format(sys.argv[0]))
        sys.exit(1)

    simname = exec_runsim(fparam)
