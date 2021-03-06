""" L2_pyramidal.py - class def for layer 2 pyramidal cells
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
from neuron import h as nrn
from cell import Pyr
import params_default as p_default

# Layer 2 pyramidal cell class
class L2Pyr(Pyr):
    """ Units for e: mV
        Units for gbar: S/cm^2 unless otherwise noted
    """
    def __init__(self, pos, p={}):
        # Get default L2Pyr params and update them with any corresponding params in p
        self.p_all = p_default.get_L2Pyr_params_default()

        # Get somatic, dendritic, and synapse properties
        p_soma = self.__get_soma_props(pos)
        p_dend = self.__get_dend_props()
        p_syn = self.__get_syn_props()

        # usage: Pyr.__init__(self, soma_props)
        Pyr.__init__(self, p_soma)
        self.celltype = 'L2_pyramidal'

        # geometry
        # creates dict of dends: self.dends, method from Cell()
        self.create_dends_new(p_dend)
        self.__connect_sections()

        # biophysics
        self.__biophys_soma()
        self.__biophys_dends()

        # dipole_insert() comes from Cell()
        self.yscale = self.get_sectnames()
        self.dipole_insert(self.yscale)

        # create synapses
        self.__synapse_create(p_syn)

        # run record_current_soma(), defined in Cell()
        self.record_current_soma()

    # insert IClamps in all situations
    def create_all_IClamp(self, p):
        """ temporarily an external function taking the p dict
        """
        # list of sections for this celltype
        sect_list_IClamp = [
            'soma',
        ]

        # some parameters
        t_delay = p['Itonic_t0_L2Pyr_soma']

        # T = -1 means use nrn.tstop
        if p['Itonic_T_L2Pyr_soma'] == -1:
            # t_delay = 50.
            t_dur = nrn.tstop - t_delay

        else:
            t_dur = p['Itonic_T_L2Pyr_soma'] - t_delay

        # t_dur must be nonnegative, I imagine
        if t_dur < 0.:
            t_dur = 0.

        # properties of the IClamp
        props_IClamp = {
            'loc': 0.5,
            'delay': t_delay,
            'dur': t_dur,
            'amp': p['Itonic_A_L2Pyr_soma']
        }

        # iterate through list of sect_list_IClamp to create a persistent IClamp object
        # the insert_IClamp procedure is in Cell() and checks on names
        # so names must be actual section names, or else it will fail silently
        self.list_IClamp = [self.insert_IClamp(sect_name, props_IClamp) for sect_name in sect_list_IClamp]

    # Returns hardcoded somatic properties
    def __get_soma_props(self, pos):
        return {
            'pos': pos,
            'L': self.p_all['L2Pyr_soma_L'],
            'diam': self.p_all['L2Pyr_soma_diam'],
            'cm': self.p_all['L2Pyr_soma_cm'],
            'Ra': self.p_all['L2Pyr_soma_Ra'],
            'name': 'L2Pyr',
        }

    # Returns hardcoded dendritic properties
    def __get_dend_props(self):
        return {
            'apical_trunk': {
                'L': self.p_all['L2Pyr_apicaltrunk_L'] ,
                'diam': self.p_all['L2Pyr_apicaltrunk_diam'],
                'cm': self.p_all['L2Pyr_dend_cm'],
                'Ra': self.p_all['L2Pyr_dend_Ra'],
            },
            'apical_1': {
                'L': self.p_all['L2Pyr_apical1_L'],
                'diam': self.p_all['L2Pyr_apical1_diam'],
                'cm': self.p_all['L2Pyr_dend_cm'],
                'Ra': self.p_all['L2Pyr_dend_Ra'],
            },
            'apical_tuft': {
                'L': self.p_all['L2Pyr_apicaltuft_L'],
                'diam': self.p_all['L2Pyr_apicaltuft_diam'],
                'cm': self.p_all['L2Pyr_dend_cm'],
                'Ra': self.p_all['L2Pyr_dend_Ra'],
            },
            'apical_oblique': {
                'L': self.p_all['L2Pyr_apicaloblique_L'],
                'diam': self.p_all['L2Pyr_apicaloblique_diam'],
                'cm': self.p_all['L2Pyr_dend_cm'],
                'Ra': self.p_all['L2Pyr_dend_Ra'],
            },
            'basal_1': {
                'L': self.p_all['L2Pyr_basal1_L'],
                'diam': self.p_all['L2Pyr_basal1_diam'],
                'cm': self.p_all['L2Pyr_dend_cm'],
                'Ra': self.p_all['L2Pyr_dend_Ra'],
            },
            'basal_2': {
                'L': self.p_all['L2Pyr_basal2_L'],
                'diam': self.p_all['L2Pyr_basal2_diam'],
                'cm': self.p_all['L2Pyr_dend_cm'],
                'Ra': self.p_all['L2Pyr_dend_Ra'],
            },
            'basal_3': {
                'L': self.p_all['L2Pyr_basal3_L'],
                'diam': self.p_all['L2Pyr_basal3_diam'],
                'cm': self.p_all['L2Pyr_dend_cm'],
                'Ra': self.p_all['L2Pyr_dend_Ra'],
            },
        }

    # get some synaptic properties
    def __get_syn_props(self):
        return {
            'ampa': {
                'e': self.p_all['L2Pyr_ampa_e'],
                'tau1': self.p_all['L2Pyr_ampa_tau1'],
                'tau2': self.p_all['L2Pyr_ampa_tau2'],
            },
            'nmda': {
                'e': self.p_all['L2Pyr_nmda_e'],
                'tau1': self.p_all['L2Pyr_nmda_tau1'],
                'tau2': self.p_all['L2Pyr_nmda_tau2'],
            },
            'gabaa': {
                'e': self.p_all['L2Pyr_gabaa_e'],
                'tau1': self.p_all['L2Pyr_gabaa_tau1'],
                'tau2': self.p_all['L2Pyr_gabaa_tau2'],
            },
            'gabab': {
                'e': self.p_all['L2Pyr_gabab_e'],
                'tau1': self.p_all['L2Pyr_gabab_tau1'],
                'tau2': self.p_all['L2Pyr_gabab_tau2'],
            }
        }

    # Connects sections of THIS cell together
    def __connect_sections(self):
        # child.connect(parent, parent_end, {child_start=0})
        # Distal (Apical)
        self.dends['apical_trunk'].connect(self.soma, 1, 0)
        self.dends['apical_1'].connect(self.dends['apical_trunk'], 1, 0)
        self.dends['apical_tuft'].connect(self.dends['apical_1'], 1, 0)

        # apical_oblique comes off distal end of apical_trunk
        self.dends['apical_oblique'].connect(self.dends['apical_trunk'], 1, 0)

        # Proximal (basal)
        self.dends['basal_1'].connect(self.soma, 0, 0)
        self.dends['basal_2'].connect(self.dends['basal_1'], 1, 0)
        self.dends['basal_3'].connect(self.dends['basal_1'], 1, 0)

    # Adds biophysics to soma
    def __biophys_soma(self):
        # Insert 'hh' mechanism
        self.soma.insert('hh')
        self.soma.gkbar_hh = self.p_all['L2Pyr_soma_gkbar_hh']
        self.soma.gl_hh = self.p_all['L2Pyr_soma_gl_hh']
        self.soma.el_hh = self.p_all['L2Pyr_soma_el_hh']
        self.soma.gnabar_hh = self.p_all['L2Pyr_soma_gnabar_hh']

        # Insert 'km' mechanism
        # Units: pS/um^2
        self.soma.insert('km')
        self.soma.gbar_km = self.p_all['L2Pyr_soma_gbar_km']

    # Defining biophysics for dendrites
    def __biophys_dends(self):
        # set dend biophysics
        # iterate over keys in self.dends and set biophysics for each dend
        for key in self.dends:
            # neuron syntax is used to set values for mechanisms
            # sec.gbar_mech = x sets value of gbar for mech to x for all segs
            # in a section. This method is faster than using
            # a for loop to iterate over all segments to set mech values

            # Insert 'hh' mechanism
            self.dends[key].insert('hh')
            self.dends[key].gkbar_hh = self.p_all['L2Pyr_dend_gkbar_hh']
            self.dends[key].gl_hh = self.p_all['L2Pyr_dend_gl_hh']
            self.dends[key].gnabar_hh = self.p_all['L2Pyr_dend_gnabar_hh']
            self.dends[key].el_hh = self.p_all['L2Pyr_dend_el_hh']

            # Insert 'km' mechanism
            # Units: pS/um^2
            self.dends[key].insert('km')
            self.dends[key].gbar_km = self.p_all['L2Pyr_dend_gbar_km']

    # create synapses
    def __synapse_create(self, p_syn):
        # creates synapses onto this cell
        # Somatic synapses
        self.synapses = {
            'soma_gabaa': self.syn_create(self.soma(0.5), p_syn['gabaa']),
            'soma_gabab': self.syn_create(self.soma(0.5), p_syn['gabab']),
        }

        # Dendritic synapses
        self.apicaloblique_ampa = self.syn_create(self.dends['apical_oblique'](0.5), p_syn['ampa'])
        self.apicaloblique_nmda = self.syn_create(self.dends['apical_oblique'](0.5), p_syn['nmda'])

        self.basal2_ampa = self.syn_create(self.dends['basal_2'](0.5), p_syn['ampa'])
        self.basal2_nmda = self.syn_create(self.dends['basal_2'](0.5), p_syn['nmda'])

        self.basal3_ampa = self.syn_create(self.dends['basal_3'](0.5), p_syn['ampa'])
        self.basal3_nmda = self.syn_create(self.dends['basal_3'](0.5), p_syn['nmda'])

        self.apicaltuft_ampa = self.syn_create(self.dends['apical_tuft'](0.5), p_syn['ampa'])
        self.apicaltuft_nmda = self.syn_create(self.dends['apical_tuft'](0.5), p_syn['nmda'])

    # collect receptor-type-based connections here
    def parconnect(self, gid, gid_dict, pos_dict, p):
        # init dict of dicts
        # nc_dict for ampa and nmda may be the same for this cell type
        nc_dict = {
            'ampa': None,
            'nmda': None,
        }

        # Connections FROM all other L2 Pyramidal cells to this one
        for gid_src, pos in zip(gid_dict['L2_pyramidal'], pos_dict['L2_pyramidal']):
            # don't be redundant, this is only possible for LIKE cells, but it might not hurt to check
            if gid_src != gid:
                nc_dict['ampa'] = {
                    'pos_src': pos,
                    'A_weight': p['gbar_L2Pyr_L2Pyr_ampa'],
                    'A_delay': 1.,
                    'lamtha': 3.,
                }

                # parconnect_from_src(gid_presyn, nc_dict, postsyn)
                # ampa connections
                self.ncfrom_L2Pyr.append(self.parconnect_from_src(gid_src, nc_dict['ampa'], self.apicaloblique_ampa))
                self.ncfrom_L2Pyr.append(self.parconnect_from_src(gid_src, nc_dict['ampa'], self.basal2_ampa))
                self.ncfrom_L2Pyr.append(self.parconnect_from_src(gid_src, nc_dict['ampa'], self.basal3_ampa))

                nc_dict['nmda'] = {
                    'pos_src': pos,
                    'A_weight': p['gbar_L2Pyr_L2Pyr_nmda'],
                    'A_delay': 1.,
                    'lamtha': 3.,
                }

                # parconnect_from_src(gid_presyn, nc_dict, postsyn)
                # nmda connections
                self.ncfrom_L2Pyr.append(self.parconnect_from_src(gid_src, nc_dict['nmda'], self.apicaloblique_nmda))
                self.ncfrom_L2Pyr.append(self.parconnect_from_src(gid_src, nc_dict['nmda'], self.basal2_nmda))
                self.ncfrom_L2Pyr.append(self.parconnect_from_src(gid_src, nc_dict['nmda'], self.basal3_nmda))

        # connections FROM L2 basket cells TO this L2Pyr cell
        for gid_src, pos in zip(gid_dict['L2_basket'], pos_dict['L2_basket']):
            nc_dict['gabaa'] = {
                'pos_src': pos,
                'A_weight': p['gbar_L2Basket_L2Pyr_gabaa'],
                'A_delay': 1.,
                'lamtha': 50.,
            }

            nc_dict['gabab'] = {
                'pos_src': pos,
                'A_weight': p['gbar_L2Basket_L2Pyr_gabab'],
                'A_delay': 1.,
                'lamtha': 50.,
            }

            self.ncfrom_L2Basket.append(self.parconnect_from_src(gid_src, nc_dict['gabaa'], self.synapses['soma_gabaa']))
            self.ncfrom_L2Basket.append(self.parconnect_from_src(gid_src, nc_dict['gabab'], self.synapses['soma_gabab']))

    def parreceive(self, gid, gid_dict, pos_dict, p_ext):
        for gid_src, p_src, pos in zip(gid_dict['extinput'], p_ext, pos_dict['extinput']):
            # Check if AMPA params defined in p_src
            if 'L2Pyr_ampa' in p_src.keys():
                nc_dict_ampa = {
                    'pos_src': pos,
                    'A_weight': p_src['L2Pyr_ampa'][0],
                    'A_delay': p_src['L2Pyr_ampa'][1],
                    'lamtha': p_src['lamtha']
                }

                # Proximal feed AMPA synapses
                if p_src['loc'] is 'proximal':
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_ampa, self.basal2_ampa))
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_ampa, self.basal3_ampa))
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_ampa, self.apicaloblique_ampa))

                # Distal feed AMPA synapses
                elif p_src['loc'] is 'distal':
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_ampa, self.apicaltuft_ampa))

            # Check is NMDA params defined in p_src
            if 'L2Pyr_nmda' in p_src.keys():
                nc_dict_nmda = {
                    'pos_src': pos,
                    'A_weight': p_src['L2Pyr_nmda'][0],
                    'A_delay': p_src['L2Pyr_nmda'][1],
                    'lamtha': p_src['lamtha']
                }

                # Proximal feed NMDA synapses
                if p_src['loc'] is 'proximal':
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_nmda, self.basal2_nmda))
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_nmda, self.basal3_nmda))
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_nmda, self.apicaloblique_nmda))

                # Distal feed NMDA synapses
                elif p_src['loc'] is 'distal':
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_nmda, self.apicaltuft_nmda))

    # one parreceive function to handle all types of external parreceives
    def parreceive_ext(self, type, gid, gid_dict, pos_dict, p_ext):
        """ types must be defined explicitly here
        """
        if type.startswith(('evprox', 'evdist')):
            if self.celltype in p_ext.keys():
                gid_ev = gid + gid_dict[type][0]

                nc_dict = {
                    'pos_src': pos_dict[type][gid],
                    'A_weight': p_ext[self.celltype][0],
                    'A_delay': p_ext[self.celltype][1],
                    'lamtha': p_ext['lamtha_space']
                }

                if p_ext['loc'] is 'proximal':
                    self.ncfrom_ev.append(self.parconnect_from_src(gid_ev, nc_dict, self.basal2_ampa))
                    self.ncfrom_ev.append(self.parconnect_from_src(gid_ev, nc_dict, self.basal3_ampa))
                    self.ncfrom_ev.append(self.parconnect_from_src(gid_ev, nc_dict, self.apicaloblique_ampa))

                elif p_ext['loc'] is 'distal':
                    self.ncfrom_ev.append(self.parconnect_from_src(gid_ev, nc_dict, self.apicaltuft_ampa))
                    self.ncfrom_ev.append(self.parconnect_from_src(gid_ev, nc_dict, self.apicaltuft_nmda))

        elif type == 'extgauss':
            # gid is this cell's gid
            # gid_dict is the whole dictionary, including the gids of the extgauss
            # pos_list is also the pos of the extgauss (net origin)
            # p_ext_gauss are the params (strength, etc.)

            # gid shift is based on L2_pyramidal cells NOT L5
            # I recognize this is ugly (hack)
            # gid_shift = gid_dict['extgauss'][0] - gid_dict['L2_pyramidal'][0]
            if 'L2_pyramidal' in p_ext.keys():
                gid_extgauss = gid + gid_dict['extgauss'][0]

                nc_dict = {
                    'pos_src': pos_dict['extgauss'][gid],
                    'A_weight': p_ext['L2_pyramidal'][0],
                    'A_delay': p_ext['L2_pyramidal'][1],
                    'lamtha': p_ext['lamtha']
                }

                self.ncfrom_extgauss.append(self.parconnect_from_src(gid_extgauss, nc_dict, self.basal2_ampa))
                self.ncfrom_extgauss.append(self.parconnect_from_src(gid_extgauss, nc_dict, self.basal3_ampa))
                self.ncfrom_extgauss.append(self.parconnect_from_src(gid_extgauss, nc_dict, self.apicaloblique_ampa))

        elif type == 'extpois':
            if self.celltype in p_ext.keys():
                gid_extpois = gid + gid_dict['extpois'][0]

                nc_dict = {
                    'pos_src': pos_dict['extpois'][gid],
                    'A_weight': p_ext[self.celltype][0],
                    'A_delay': p_ext[self.celltype][1],
                    'lamtha': p_ext['lamtha_space']
                }


                self.ncfrom_extpois.append(self.parconnect_from_src(gid_extpois, nc_dict, self.basal2_ampa))
                self.ncfrom_extpois.append(self.parconnect_from_src(gid_extpois, nc_dict, self.basal3_ampa))
                self.ncfrom_extpois.append(self.parconnect_from_src(gid_extpois, nc_dict, self.apicaloblique_ampa))

        else:
            print("Warning, ext type def does not exist in L2Pyr")

    # Define 3D shape and position of cell.
    def __set_3Dshape(self):
        """ By default neuron uses xy plane for
            height and xz plane for depth. This is opposite for model as a whole, but
            convention is followed in this function for ease use of gui.
            This function is not used actively but is included for some visualization.
            Not maintained.
        """
        # set 3d shape of soma by calling shape_soma from class Cell
        self.shape_soma()

        # soma proximal coords
        x_prox = 0
        y_prox = 0

        # soma distal coords
        x_distal = 0
        y_distal = self.soma.L

        # dend 0-2 are major axis, dend 3 is branch
        # deal with distal first along major cable axis
        for i in range(0, 3):
            nrn.pt3dclear(sec=self.list_dend[i])

            # x_distal and y_distal are the starting points for each segment
            # these are updated at the end of the loop
            nrn.pt3dadd(0, y_distal, 0, self.dend_diam[i], sec=self.list_dend[i])

            # update x_distal and y_distal after setting them
            # x_distal += dend_dx[i]
            y_distal += self.dend_L[i]

            # add next point
            nrn.pt3dadd(0, y_distal, 0, self.dend_diam[i], sec=self.list_dend[i])

        # now deal with dend 3
        # dend 3 will ALWAYS be positioned at the end of dend[0]
        nrn.pt3dclear(sec=self.list_dend[3])

        # activate this section with 'sec =' notation
        # self.list_dend[0].push()
        x_start = nrn.x3d(1, sec = self.list_dend[0])
        y_start = nrn.y3d(1, sec = self.list_dend[0])
        # nrn.pop_section()

        nrn.pt3dadd(x_start, y_start, 0, self.dend_diam[3], sec=self.list_dend[3])
        # self.dend_L[3] is subtracted because lengths always positive,
        # and this goes to negative x
        nrn.pt3dadd(x_start-self.dend_L[3], y_start, 0, self.dend_diam[3], sec=self.list_dend[3])

        # now deal with proximal dends
        for i in range(4, 7):
            nrn.pt3dclear(sec=self.list_dend[i])

        nrn.pt3dadd(x_prox, y_prox, 0, self.dend_diam[i], sec=self.list_dend[4])
        y_prox += -self.dend_L[4]

        nrn.pt3dadd(x_prox, y_prox, 0, self.dend_diam[4], sec=self.list_dend[4])

        # x_prox, y_prox are now the starting points for BOTH last 2 sections

        # dend 5
        # Calculate x-coordinate for end of dend
        dend5_x = -self.dend_L[5] * np.sqrt(2)/2
        nrn.pt3dadd(x_prox, y_prox, 0, self.dend_diam[5], sec=self.list_dend[5])
        nrn.pt3dadd(dend5_x, y_prox-self.dend_L[5] * np.sqrt(2)/2,
                    0, self.dend_diam[5], sec=self.list_dend[5])

        # dend 6
        # Calculate x-coordinate for end of dend
        dend6_x = self.dend_L[6] * np.sqrt(2)/2
        nrn.pt3dadd(x_prox, y_prox, 0, self.dend_diam[6], sec=self.list_dend[6])
        nrn.pt3dadd(dend6_x, y_prox-self.dend_L[6] * np.sqrt(2)/2,
                    0, self.dend_diam[6], sec=self.list_dend[6])

        # set 3D position
        # z grid position used as y coordinate in nrn.pt3dchange() to satisfy
        # gui convention that y is height and z is depth. In nrn.pt3dchange()
        # x and z components are scaled by 100 for visualization clarity
        self.soma.push()
        for i in range(0, int(nrn.n3d())):
            nrn.pt3dchange(i, self.pos[0]*100 + nrn.x3d(i), self.pos[2] +
                           nrn.y3d(i), self.pos[1] * 100 + nrn.z3d(i),
                           nrn.diam3d(i))

        nrn.pop_section()
