""" class_net.py - establishes the Network class and related methods
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

import itertools as it
from neuron import h as nrn
import numpy as np

import feed
from L2_pyramidal import L2Pyr
from L5_pyramidal import L5Pyr
from L2_basket import L2Basket
from L5_basket import L5Basket

# creates the external feed params based on individual simulation params p
def create_pext(p, tstop):
    # indexable py list of param dicts for parallel
    # turn off individual feeds by commenting out relevant line here.
    # always valid, no matter the length
    p_ext = []

    # p_unique is a dict of input param types that end up going to each cell uniquely
    p_unique = {}

    # default params
    feed_prox = {
        'f_input': p['f_input_prox'],
        't0': p['t0_input_prox'],
        'tstop': p['tstop_input_prox'],
        'stdev': p['f_stdev_prox'],
        'L2Pyr_ampa': (p['input_prox_A_weight_L2Pyr_ampa'], p['input_prox_A_delay_L2']),
        'L2Pyr_nmda': (p['input_prox_A_weight_L2Pyr_nmda'], p['input_prox_A_delay_L2']),
        'L5Pyr_ampa': (p['input_prox_A_weight_L5Pyr_ampa'], p['input_prox_A_delay_L5']),
        'L5Pyr_nmda': (p['input_prox_A_weight_L5Pyr_nmda'], p['input_prox_A_delay_L5']),
        'L2Basket_ampa': (p['input_prox_A_weight_inh_ampa'], p['input_prox_A_delay_L2']),
        'L2Basket_nmda': (p['input_prox_A_weight_inh_nmda'], p['input_prox_A_delay_L2']),
        'L5Basket_ampa': (p['input_prox_A_weight_inh_ampa'], p['input_prox_A_delay_L5']),
        'L5Basket_nmda': (p['input_prox_A_weight_inh_nmda'], p['input_prox_A_delay_L5']),
        'events_per_cycle': p['events_per_cycle_prox'],
        'distribution': p['distribution_prox'],
        'lamtha': 100.,
        'loc': 'proximal',
    }

    # ensures time interval makes sense
    p_ext = feed_validate(p_ext, feed_prox, tstop)

    feed_dist = {
        'f_input': p['f_input_dist'],
        't0': p['t0_input_dist'],
        'tstop': p['tstop_input_dist'],
        'stdev': p['f_stdev_dist'],
        'L2Pyr_ampa': (p['input_dist_A_weight_L2Pyr_ampa'], p['input_dist_A_delay_L2']),
        'L2Pyr_nmda': (p['input_dist_A_weight_L2Pyr_nmda'], p['input_dist_A_delay_L2']),
        'L5Pyr_ampa': (p['input_dist_A_weight_L5Pyr_ampa'], p['input_dist_A_delay_L5']),
        'L5Pyr_nmda': (p['input_dist_A_weight_L5Pyr_nmda'], p['input_dist_A_delay_L5']),
        'L2Basket_ampa': (p['input_dist_A_weight_inh_ampa'], p['input_dist_A_delay_L2']),
        'L2Basket_nmda': (p['input_dist_A_weight_inh_nmda'], p['input_dist_A_delay_L2']),
        'events_per_cycle': p['events_per_cycle_dist'],
        'distribution': p['distribution_dist'],
        'lamtha': 100.,
        'loc': 'distal',
    }

    p_ext = feed_validate(p_ext, feed_dist, tstop)

    # Create evoked response parameters
    # f_input needs to be defined as 0
    # these vals correspond to non-perceived max
    # conductance threshold in uS (Jones et al. 2007)
    p_unique['evprox0'] = {
        't0': p['t_evprox_early'],
        'L2_pyramidal': (p['gbar_evprox_early_L2Pyr'], 0.1, p['sigma_t_evprox_early']),
        'L2_basket': (p['gbar_evprox_early_L2Basket'], 0.1, p['sigma_t_evprox_early']),
        'L5_pyramidal': (p['gbar_evprox_early_L5Pyr'], 1., p['sigma_t_evprox_early']),
        'L5_basket': (p['gbar_evprox_early_L5Basket'], 1., p['sigma_t_evprox_early']),
        'lamtha_space': 3.,
        'loc': 'proximal',
    }

    # see if relative start time is defined
    if p['dt_evprox0_evdist'] == -1:
        # if dt is -1, assign the input time based on the input param
        t0_evdist = p['t_evdist']
    else:
        # use dt to set the relative timing
        t0_evdist = p_unique['evprox0']['t0'] + p['dt_evprox0_evdist']

    # relative timing between evprox0 and evprox1
    # not defined by distal time
    if p['dt_evprox0_evprox1'] == -1:
        t0_evprox1 = p['t_evprox_late']

    else:
        t0_evprox1 = p_unique['evprox0']['t0'] + p['dt_evprox0_evprox1']

    # next evoked input is distal
    p_unique['evdist'] = {
        't0': t0_evdist,
        'L2_pyramidal': (p['gbar_evdist_L2Pyr'], 0.1, p['sigma_t_evdist']),
        'L5_pyramidal': (p['gbar_evdist_L5Pyr'], 0.1, p['sigma_t_evdist']),
        'L2_basket': (p['gbar_evdist_L2Basket'], 0.1, p['sigma_t_evdist']),
        'lamtha_space': 3.,
        'loc': 'distal',
    }

    # next evoked input is proximal also
    p_unique['evprox1'] = {
        't0': t0_evprox1,
        'L2_pyramidal': (p['gbar_evprox_late_L2Pyr'], 0.1, p['sigma_t_evprox_late']),
        'L2_basket': (p['gbar_evprox_late_L2Basket'], 0.1, p['sigma_t_evprox_late']),
        'L5_pyramidal': (p['gbar_evprox_late_L5Pyr'], 5., p['sigma_t_evprox_late']),
        'L5_basket': (p['gbar_evprox_late_L5Basket'], 5., p['sigma_t_evprox_late']),
        'lamtha_space': 3.,
        'loc': 'proximal',
    }

    # this needs to create many feeds
    # (amplitude, delay, mu, sigma). ordered this way to preserve compatibility
    p_unique['extgauss'] = {
        'stim': 'gaussian',
        'L2_basket': (p['L2Basket_Gauss_A_weight'], 1., p['L2Basket_Gauss_mu'], p['L2Basket_Gauss_sigma']),
        'L2_pyramidal': (p['L2Pyr_Gauss_A_weight'], 0.1, p['L2Pyr_Gauss_mu'], p['L2Pyr_Gauss_sigma']),
        'L5_basket': (p['L5Basket_Gauss_A_weight'], 1., p['L5Basket_Gauss_mu'], p['L5Basket_Gauss_sigma']),
        'L5_pyramidal': (p['L5Pyr_Gauss_A_weight'], 1., p['L5Pyr_Gauss_mu'], p['L5Pyr_Gauss_sigma']),
        'lamtha': 100.,
        'loc': 'proximal'
    }

    # define T_pois as 0 or -1 to reset automatically to tstop
    if p['T_pois'] in (0, -1):
        p['T_pois'] = tstop

    # Poisson distributed inputs to proximal
    p_unique['extpois'] = {
        'stim': 'poisson',
        'L2_basket': (p['L2Basket_Pois_A_weight'], 1., p['L2Basket_Pois_lamtha']),
        'L2_pyramidal': (p['L2Pyr_Pois_A_weight'], 0.1, p['L2Pyr_Pois_lamtha']),
        'L5_basket': (p['L5Basket_Pois_A_weight'], 1., p['L5Basket_Pois_lamtha']),
        'L5_pyramidal': (p['L5Pyr_Pois_A_weight'], 1., p['L5Pyr_Pois_lamtha']),
        'lamtha_space': 100.,
        't_interval': (p['t0_pois'], p['T_pois']),
        'loc': 'proximal'
    }

    return p_ext, p_unique

# qnd function to add feeds if they are sensible
def feed_validate(p_ext, d, tstop):
    # only append if t0 is less than simulation tstop
    if tstop > d['t0']:
        if d['tstop'] > tstop:
            d['tstop'] = tstop

        # if stdev is zero, increase synaptic weights 5 fold to make
        # single input equivalent to 5 simultaneous input to prevent spiking
        if not d['stdev'] and d['distribution'] != 'uniform':
            for key in d.keys():
                if key.endswith('Pyr'):
                    d[key] = (d[key][0] * 5., d[key][1])

                elif key.endswith('Basket'):
                    d[key] = (d[key][0] * 5., d[key][1])

        # if L5 delay is -1, use same delays as L2 unless
        # L2 delay is 0.1 in which case use 1.
        if d['L5Pyr_ampa'][1] == -1:
            for key in d.keys():
                if key.startswith('L5'):
                    if d['L2Pyr'][1] != 0.1:
                        d[key] = (d[key][0], d['L2Pyr'][1])
                    else:
                        d[key] = (d[key][0], 1.)

        p_ext.append(d)

    return p_ext

# create Network class on each node of the sim
class NetworkOnNode():
    def __init__(self, p):
        # set the params internally for this net
        # better than passing it around
        self.p = p

        # Number of time points
        # Originally used to create the empty vec for synaptic currents,
        # ensuring that they exist on this node irrespective of whether
        # or not cells of relevant type actually do
        self.N_t = np.arange(0., nrn.tstop, self.p['dt']).size + 1

        # Create a nrn.Vector() with size 1xself.N_t, zero'd
        self.current = {
            'L5Pyr_soma': nrn.Vector(self.N_t, 0),
            'L2Pyr_soma': nrn.Vector(self.N_t, 0),
        }

        # int variables for grid of pyramidal cells (for now in both L2 and L5)
        self.gridpyr = {
            'x': self.p['N_pyr_x'],
            'y': self.p['N_pyr_y'],
        }

        # Parallel stuff
        self.pc = nrn.ParallelContext()
        self.n_hosts = int(self.pc.nhost())
        self.rank = int(self.pc.id())
        self.N_src = 0

        # numbers of sources
        self.N = {}

        # init self.N_cells
        self.N_cells = 0

        # zdiff is expressed as a positive DEPTH of L5 relative to L2
        # this is a deviation from the original, where L5 was defined at 0
        # this should not change interlaminar weight/delay calculations
        self.zdiff = 1307.4

        # params of external inputs in p_ext
        # Global number of external inputs ... automatic counting makes more sense
        # p_unique represent ext inputs that are going to go to each cell
        self.p_ext, self.p_unique = create_pext(self.p, nrn.tstop)
        self.N_extinput = len(self.p_ext)

        # Source list of names
        # in particular order (cells, extinput, alpha names of unique inputs)
        self.src_list_new = self.__create_src_list()

        # cell position lists, also will give counts: must be known by ALL nodes
        # extinput positions are all located at origin.
        # sort of a hack bc of redundancy
        self.pos_dict = dict.fromkeys(self.src_list_new)

        # create coords in pos_dict for all cells first
        self.__create_coords_pyr()
        self.__create_coords_basket()
        self.__count_cells()

        # create coords for all other sources
        self.__create_coords_extinput()

        # count external sources
        self.__count_extsrcs()

        # create dictionary of GIDs according to cell type
        # global dictionary of gid and cell type
        self.gid_dict = {}
        self.__create_gid_dict()

        # assign gid to hosts, creates list of gids for this node in __gid_list
        # __gid_list length is number of cells assigned to this id()
        self.__gid_list = []
        self.__gid_assign()

        # create cells (and create self.origin in create_cells_pyr())
        self.cells_list = []
        self.extinput_list = []

        # external unique input list dictionary
        self.ext_list = dict.fromkeys(self.p_unique)

        # initialize the lists in the dict
        for key in self.ext_list.keys():
            self.ext_list[key] = []

        # create sources and init
        self.__create_all_src()
        self.__state_init()

        # parallel network connector
        self.__parnet_connect()

        # set to record spikes
        self.spiketimes = nrn.Vector()
        self.spikegids = nrn.Vector()
        self.__record_spikes()

    # aggregate recording all the somatic voltages for pyr
    def aggregate_currents(self):
        """ this method must be run post-integration
        """
        for cell in self.cells_list:
            # check for celltype
            if cell.celltype == 'L5_pyramidal':
                # iterate over somatic currents, assumes this list exists
                # is guaranteed in L5Pyr()
                for key, I_soma in cell.dict_currents.items():
                    # self.current_L5Pyr_soma was created upon
                    # in parallel, each node has its own Net()
                    self.current['L5Pyr_soma'].add(I_soma)

            elif cell.celltype == 'L2_pyramidal':
                for key, I_soma in cell.dict_currents.items():
                    # self.current_L5Pyr_soma was created upon
                    # in parallel, each node has its own Net()
                    self.current['L2Pyr_soma'].add(I_soma)

    # reverse lookup of gid to type
    def gid_to_type(self, gid):
        for gidtype, gids in self.gid_dict.items():
            if gid in gids:
                return gidtype

    # cell counting routine
    def __count_cells(self):
        # cellname list is used *only* for this purpose for now
        for src in self.cellname_list:
            # if it's a cell, then add the number to total number of cells
            self.N[src] = len(self.pos_dict[src])
            self.N_cells += self.N[src]

    # count all of the external sources
    def __count_extsrcs(self):
        """ general counting method requires pos_dict is correct for each source
            and that all sources are represented
        """
        # all src numbers are based off of length of pos_dict entry
        # generally done here in lieu of upstream changes

        for src in self.extname_list:
            self.N[src] = len(self.pos_dict[src])

    # parallel create cells AND external inputs (feeds) - see note
    def __create_all_src(self):
        """ these are spike SOURCES but cells are also targets
            external inputs are not targets
        """
        # loop through gids on this node
        for gid in self.__gid_list:
            # check existence of gid with Neuron
            if self.pc.gid_exists(gid):
                # get type of cell and pos via gid
                # now should be valid for ext inputs
                type = self.gid_to_type(gid)
                type_pos_ind = gid - self.gid_dict[type][0]
                pos = self.pos_dict[type][type_pos_ind]

                # figure out which cell type is assoc with the gid
                # create cells based on loc property
                # creates a NetCon object internally to Neuron
                if type == 'L2_pyramidal':
                    self.cells_list.append(L2Pyr(pos, self.p))
                    self.pc.cell(gid, self.cells_list[-1].connect_to_target(None))

                    # run the IClamp function here
                    # create_all_IClamp() is defined in L2Pyr (etc)
                    self.cells_list[-1].create_all_IClamp(self.p)

                elif type == 'L5_pyramidal':
                    self.cells_list.append(L5Pyr(pos, self.p))
                    self.pc.cell(gid, self.cells_list[-1].connect_to_target(None))

                    # run the IClamp function here
                    self.cells_list[-1].create_all_IClamp(self.p)

                elif type == 'L2_basket':
                    self.cells_list.append(L2Basket(pos))
                    self.pc.cell(gid, self.cells_list[-1].connect_to_target(None))

                    # also run the IClamp for L2_basket
                    self.cells_list[-1].create_all_IClamp(self.p)

                elif type == 'L5_basket':
                    self.cells_list.append(L5Basket(pos))
                    self.pc.cell(gid, self.cells_list[-1].connect_to_target(None))

                    # run the IClamp function here
                    self.cells_list[-1].create_all_IClamp(self.p)

                elif type == 'extinput':
                    # to find param index, take difference between REAL gid
                    # here and gid start point of the items
                    p_ind = gid - self.gid_dict['extinput'][0]

                    # now use the param index in the params and create
                    # the cell and artificial NetCon
                    self.extinput_list.append(feed.ParFeedAll(type, None, self.p_ext[p_ind], gid))
                    self.pc.cell(gid, self.extinput_list[-1].connect_to_target())

                elif type in self.p_unique.keys():
                    gid_post = gid - self.gid_dict[type][0]
                    cell_type = self.gid_to_type(gid_post)

                    # create dictionary entry, append to list
                    self.ext_list[type].append(feed.ParFeedAll(type, cell_type, self.p_unique[type], gid))
                    self.pc.cell(gid, self.ext_list[type][-1].connect_to_target())

                else:
                    print("None of these types in Net()")
                    exit()

            else:
                print("GID does not exist. See Cell()")
                exit()

    # Creates cells and grid
    def __create_coords_pyr(self):
        """ pyr grid is the immutable grid, origin now calculated in relation to feed
        """
        xrange = np.arange(self.gridpyr['x'])
        yrange = np.arange(self.gridpyr['y'])

        # create list of tuples/coords, (x, y, z)
        self.pos_dict['L2_pyramidal'] = [pos for pos in it.product(xrange, yrange, [0])]
        self.pos_dict['L5_pyramidal'] = [pos for pos in it.product(xrange, yrange, [self.zdiff])]

    # create basket cell coords based on pyr grid
    def __create_coords_basket(self):
        # define relevant x spacings for basket cells
        xzero = np.arange(0, self.gridpyr['x'], 3)
        xone = np.arange(1, self.gridpyr['x'], 3)

        # split even and odd y vals
        yeven = np.arange(0, self.gridpyr['y'], 2)
        yodd = np.arange(1, self.gridpyr['y'], 2)

        # create general list of x,y coords and sort it
        coords = [pos for pos in it.product(xzero, yeven)] + [pos for pos in it.product(xone, yodd)]
        coords_sorted = sorted(coords, key=lambda pos: pos[1])

        # append the z value for position for L2 and L5
        self.pos_dict['L2_basket'] = [pos_xy + (0,) for pos_xy in coords_sorted]
        self.pos_dict['L5_basket'] = [pos_xy + (self.zdiff,) for pos_xy in coords_sorted]

    # creates origin AND creates external input coords
    def __create_coords_extinput(self):
        xrange = np.arange(self.gridpyr['x'])
        yrange = np.arange(self.gridpyr['y'])

        # origin's z component isn't really used in calculating
        # distance functions from origin
        # these must be ints!
        origin_x = xrange[int((len(xrange)-1)/2)]
        origin_y = yrange[int((len(yrange)-1)/2)]
        origin_z = np.floor(self.zdiff/2)
        self.origin = (origin_x, origin_y, origin_z)

        self.pos_dict['extinput'] = [self.origin for i in range(self.N_extinput)]

        # at this time, each of the unique inputs is per cell
        for key in self.p_unique.keys():
            # create the pos_dict for all the sources
            self.pos_dict[key] = [self.origin for i in range(self.N_cells)]

    # creates the immutable source list along with corresponding numbers of cells
    def __create_src_list(self):
        # base source list of tuples, name and number, in this order
        self.cellname_list = [
            'L2_basket',
            'L2_pyramidal',
            'L5_basket',
            'L5_pyramidal',
        ]

        # add the legacy extinput here
        self.extname_list = []
        self.extname_list.append('extinput')

        # grab the keys for the unique set of inputs and sort the names
        # append them to the src list along with the number of cells
        unique_keys = sorted(self.p_unique.keys())
        self.extname_list += unique_keys

        # return one final source list
        src_list = self.cellname_list + self.extname_list
        return src_list

    # creates gid dicts and pos_lists
    def __create_gid_dict(self):
        # initialize gid index gid_ind to start at 0
        gid_ind = [0]

        # append a new gid_ind based on previous and next cell count
        # order is guaranteed by self.src_list_new
        for i in range(len(self.src_list_new)):
            # N = self.src_list_new[i][1]
            # grab the src name in ordered list src_list_new
            src = self.src_list_new[i]

            # query the N dict for that number and append here to gid_ind, based on previous entry
            gid_ind.append(gid_ind[i]+self.N[src])

            # accumulate total source count
            self.N_src += self.N[src]

        # now actually assign the ranges
        for i in range(len(self.src_list_new)):
            src = self.src_list_new[i]
            self.gid_dict[src] = np.arange(gid_ind[i], gid_ind[i+1])

    # this happens on EACH node
    def __gid_assign(self):
        """ creates self.__gid_list for THIS node
        """
        # round robin assignment of gids
        for gid in range(self.rank, self.N_cells, self.n_hosts):
            # set the cell gid
            self.pc.set_gid2node(gid, self.rank)
            self.__gid_list.append(gid)

            # now to do the cell-specific external input gids on the same proc
            # these are guaranteed to exist because all of these inputs were created
            # for each cell
            for key in self.p_unique.keys():
                gid_input = gid + self.gid_dict[key][0]
                self.pc.set_gid2node(gid_input, self.rank)
                self.__gid_list.append(gid_input)

        # legacy handling of the external inputs
        # NOT perfectly balanced
        for gid_base in range(self.rank, self.N_extinput, self.n_hosts):
            # shift the gid_base to the extinput gid
            gid = gid_base + self.gid_dict['extinput'][0]

            # set as usual
            self.pc.set_gid2node(gid, self.rank)
            self.__gid_list.append(gid)

        # extremely important to get the gids in the right order
        self.__gid_list.sort()

    # connections in parallel - search for gids on nodes
    def __parnet_connect(self):
        """ this NODE is aware of its cells as targets
            for each syn, return list of source GIDs.
            for each item in the list, do a:
            nc = pc.gid_connect(source_gid, target_syn), weight,delay
            Both for synapses AND for external inputs
        """
        # loop over target zipped gids and cells
        # cells_list has NO extinputs anyway. also no extgausses
        for gid, cell in zip(self.__gid_list, self.cells_list):
            # ignore iteration over inputs, since they are NOT targets
            if self.pc.gid_exists(gid) and self.gid_to_type(gid) is not 'extinput':
                # print "rank:", self.rank, "gid:", gid, cell, self.gid_to_type(gid)

                # for each gid, find all the other cells connected to it, based on gid
                # this MUST be defined in EACH class of cell in self.cells_list
                # parconnect receives connections from other cells
                # parreceive receives connections from external inputs
                cell.parconnect(gid, self.gid_dict, self.pos_dict, self.p)
                cell.parreceive(gid, self.gid_dict, self.pos_dict, self.p_ext)

                # now do the unique inputs specific to these cells
                # parreceive_ext receives connections from UNIQUE external inputs
                for type in self.p_unique.keys():
                    p_type = self.p_unique[type]
                    cell.parreceive_ext(type, gid, self.gid_dict, self.pos_dict, p_type)

    # setup spike recording for this node
    def __record_spikes(self):
        # iterate through gids on this node and
        # set to record spikes in spike time vec and id vec
        # agnostic to type of source, will sort that out later
        for gid in self.__gid_list:
            if self.pc.gid_exists(gid):
                self.pc.spike_record(gid, self.spiketimes, self.spikegids)

    # initializes the state closer to baseline
    def __state_init(self):
        for cell in self.cells_list:
            seclist = nrn.SectionList()
            seclist.wholetree(sec=cell.soma)

            for sect in seclist:
                for seg in sect:
                    if cell.celltype == 'L2_pyramidal':
                        seg.v = -71.46

                    elif cell.celltype == 'L5_pyramidal':
                        if sect.name() == 'L5Pyr_apical_1':
                            seg.v = -71.32

                        elif sect.name() == 'L5Pyr_apical_2':
                            seg.v = -69.08

                        elif sect.name() == 'L5Pyr_apical_tuft':
                            seg.v = -67.30

                        else:
                            seg.v = -72.

                    elif cell.celltype == 'L2_basket':
                        seg.v = -64.9737

                    elif cell.celltype == 'L5_basket':
                        seg.v = -64.9737
