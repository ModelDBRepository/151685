""" spikefn.py - parsing spikes from the file written by the sim
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
import fileio as fio

# returns lists of spiketimes for each cell of the simulation
def spikes_from_file(fspikes, gid_dict):
    src_list = []
    src_extinput_list = []
    src_unique_list = []

    # fill in 2 lists from the keys
    for key in gid_dict:
        if key.startswith('L2_') or key.startswith('L5_'):
            src_list.append(key)

        elif key == 'extinput':
            src_extinput_list.append(key)

        else:
            src_unique_list.append(key)

    # check to see if there are spikes in here, otherwise return an empty array
    if os.stat(fspikes).st_size:
        s = np.loadtxt(open(fspikes, 'rb'))

    else:
        s = np.array([], dtype='float64')

    # sorted here
    s_sorted = {}

    for t, gid_float in s[:]:
        gid = int(gid_float)
        if gid not in s_sorted:
            s_sorted[gid] = []
        s_sorted[gid].append(t)

    gids = [gid for gid in s_sorted]
    gids.sort()

    s_by_type = {}

    for key in src_list:
        if key not in s_by_type:
            s_by_type[key] = []

        for gid in gid_dict[key]:
            if gid in s_sorted:
                s_sorted[gid].sort()
                s_by_type[key].append(np.array(s_sorted[gid]))

            else:
                s_by_type[key].append(np.array([]))

    return s_by_type

if __name__ == '__main__':
    f = 'data/gammaweak/data.pkl'
    fspk = 'data/gammaweak/spikes.txt'
    x = fio.pkl_load(f)
    s = spikes_from_file(fspk, x['gid_dict'])
