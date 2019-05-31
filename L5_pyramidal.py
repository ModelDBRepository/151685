""" L5_pyramidal.py - class def for layer 5 pyramidal cells
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

class L5Pyr(Pyr):
    """ Units for e: mV
        Units for gbar: S/cm^2 unless otherwise noted
        units for taur: ms
    """
    def __init__(self, pos, p={}):
        # Get default L5Pyr params and update them with corresponding params in p
        self.p_all = p_default.get_L5Pyr_params_default()

        # Get somatic, dendirtic, and synapse properties
        p_soma = self.__get_soma_props(pos)
        p_dend = self.__get_dend_props()
        p_syn = self.__get_syn_props()

        # Pyr.__init__(self, soma_props)
        Pyr.__init__(self, p_soma)
        self.celltype = 'L5_pyramidal'

        # Geometry
        # dend Cm and dend Ra set using soma Cm and soma Ra
        self.create_dends_new(p_dend)
        self.__connect_sections()

        # biophysics
        self.__biophys_soma()
        self.__biophys_dends()

        # Dictionary of length scales to calculate dipole without 3d shape.
        # Comes from Pyr().
        # dipole_insert() comes from Cell()
        self.yscale = self.get_sectnames()
        self.dipole_insert(self.yscale)

        # create synapses
        self.__synapse_create(p_syn)

        # insert iclamp
        self.list_IClamp = []

        # run record current soma, defined in Cell()
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
        t_delay = p['Itonic_t0_L5Pyr_soma']

        # T = -1 means use nrn.tstop
        if p['Itonic_T_L5Pyr_soma'] == -1:
            # t_delay = 50.
            t_dur = nrn.tstop - t_delay
        else:
            t_dur = p['Itonic_T_L5Pyr_soma'] - t_delay

        # t_dur must be nonnegative, I imagine
        if t_dur < 0.:
            t_dur = 0.

        # properties of the IClamp
        props_IClamp = {
            'loc': 0.5,
            'delay': t_delay,
            'dur': t_dur,
            'amp': p['Itonic_A_L5Pyr_soma']
        }

        # iterate through list of sect_list_IClamp to create a persistent IClamp object
        # the insert_IClamp procedure is in Cell() and checks on names
        # so names must be actual section names, or else it will fail silently
        self.list_IClamp = [self.insert_IClamp(sect_name, props_IClamp) for sect_name in sect_list_IClamp]

    # parallel connection function FROM all cell types TO here
    def parconnect(self, gid, gid_dict, pos_dict, p):
        # init dict of dicts
        # nc_dict for ampa and nmda may be the same for this cell type
        nc_dict = {
            'ampa': None,
            'nmda': None,
        }

        # connections FROM L5Pyr TO here
        for gid_src, pos in zip(gid_dict['L5_pyramidal'], pos_dict['L5_pyramidal']):
            # no autapses
            if gid_src != gid:
                nc_dict['ampa'] = {
                    'pos_src': pos,
                    'A_weight': p['gbar_L5Pyr_L5Pyr_ampa'],
                    'A_delay': 1.,
                    'lamtha': 3.,
                }

                # ampa connections
                self.ncfrom_L5Pyr.append(self.parconnect_from_src(gid_src, nc_dict['ampa'], self.apicaloblique_ampa))
                self.ncfrom_L5Pyr.append(self.parconnect_from_src(gid_src, nc_dict['ampa'], self.basal2_ampa))
                self.ncfrom_L5Pyr.append(self.parconnect_from_src(gid_src, nc_dict['ampa'], self.basal3_ampa))

                nc_dict['nmda'] = {
                    'pos_src': pos,
                    'A_weight': p['gbar_L5Pyr_L5Pyr_nmda'],
                    'A_delay': 1.,
                    'lamtha': 3.,
                }

                # nmda connections
                self.ncfrom_L5Pyr.append(self.parconnect_from_src(gid_src, nc_dict['nmda'], self.apicaloblique_nmda))
                self.ncfrom_L5Pyr.append(self.parconnect_from_src(gid_src, nc_dict['nmda'], self.basal2_nmda))
                self.ncfrom_L5Pyr.append(self.parconnect_from_src(gid_src, nc_dict['nmda'], self.basal3_nmda))

        # connections FROM L5Basket TO here
        for gid_src, pos in zip(gid_dict['L5_basket'], pos_dict['L5_basket']):
            nc_dict['gabaa'] = {
                'pos_src': pos,
                'A_weight': p['gbar_L5Basket_L5Pyr_gabaa'],
                'A_delay': 1.,
                'lamtha': 70.,
            }

            nc_dict['gabab'] = {
                'pos_src': pos,
                'A_weight': p['gbar_L5Basket_L5Pyr_gabab'],
                'A_delay': 1.,
                'lamtha': 70.,
            }

            # soma synapses are defined in Pyr()
            self.ncfrom_L5Basket.append(self.parconnect_from_src(gid_src, nc_dict['gabaa'], self.synapses['soma_gabaa']))
            self.ncfrom_L5Basket.append(self.parconnect_from_src(gid_src, nc_dict['gabab'], self.synapses['soma_gabab']))

        # connections FROM L2Pyr TO here
        for gid_src, pos in zip(gid_dict['L2_pyramidal'], pos_dict['L2_pyramidal']):
            # this delay is longer than most
            nc_dict = {
                'pos_src': pos,
                'A_weight': p['gbar_L2Pyr_L5Pyr'],
                'A_delay': 1.,
                'lamtha': 3.,
            }

            self.ncfrom_L2Pyr.append(self.parconnect_from_src(gid_src, nc_dict, self.basal2_ampa))
            self.ncfrom_L2Pyr.append(self.parconnect_from_src(gid_src, nc_dict, self.basal3_ampa))
            self.ncfrom_L2Pyr.append(self.parconnect_from_src(gid_src, nc_dict, self.apicaltuft_ampa))
            self.ncfrom_L2Pyr.append(self.parconnect_from_src(gid_src, nc_dict, self.apicaloblique_ampa))

        # connections FROM L2Basket TO here
        for gid_src, pos in zip(gid_dict['L2_basket'], pos_dict['L2_basket']):
            nc_dict = {
                'pos_src': pos,
                'A_weight': p['gbar_L2Basket_L5Pyr'],
                'A_delay': 1.,
                'lamtha': 50.,
            }

            self.ncfrom_L2Basket.append(self.parconnect_from_src(gid_src, nc_dict, self.apicaltuft_gabaa))

    # receive from external inputs
    def parreceive(self, gid, gid_dict, pos_dict, p_ext):
        for gid_src, p_src, pos in zip(gid_dict['extinput'], p_ext, pos_dict['extinput']):
            # Check if AMPA params defined in p_src
            if 'L5Pyr_ampa' in p_src.keys():
                nc_dict_ampa = {
                    'pos_src': pos,
                    'A_weight': p_src['L5Pyr_ampa'][0],
                    'A_delay': p_src['L5Pyr_ampa'][1],
                    'lamtha': p_src['lamtha']
                }

                # Proximal feed AMPA synapses
                if p_src['loc'] is 'proximal':
                    # basal2_ampa, basal3_ampa, apicaloblique_ampa
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_ampa, self.basal2_ampa))
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_ampa, self.basal3_ampa))
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_ampa, self.apicaloblique_ampa))

                # Distal feed AMPA synsapes
                elif p_src['loc'] is 'distal':
                    # apical tuft
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_ampa, self.apicaltuft_ampa))

            # Check if NMDA params defined in p_src
            if 'L5Pyr_nmda' in p_src.keys():
                nc_dict_nmda = {
                    'pos_src': pos,
                    'A_weight': p_src['L5Pyr_nmda'][0],
                    'A_delay': p_src['L5Pyr_nmda'][1],
                    'lamtha': p_src['lamtha']
                }

                # Proximal feed NMDA synapses
                if p_src['loc'] is 'proximal':
                    # basal2_nmda, basal3_nmda, apicaloblique_nmda
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_nmda, self.basal2_nmda))
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_nmda, self.basal3_nmda))
                    self.ncfrom_extinput.append(self.parconnect_from_src(gid_src, nc_dict_nmda, self.apicaloblique_nmda))

                # Distal feed NMDA synsapes
                elif p_src['loc'] is 'distal':
                    # apical tuft
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
                    # apical tuft
                    self.ncfrom_ev.append(self.parconnect_from_src(gid_ev, nc_dict, self.apicaltuft_ampa))
                    self.ncfrom_ev.append(self.parconnect_from_src(gid_ev, nc_dict, self.apicaltuft_nmda))

        elif type == 'extgauss':
            # gid is this cell's gid
            # gid_dict is the whole dictionary, including the gids of the extgauss
            # pos_dict is also the pos of the extgauss (net origin)
            # p_ext_gauss are the params (strength, etc.)
            # doesn't matter if this doesn't do anything

            # gid shift is based on L2_pyramidal cells NOT L5
            # I recognize this is ugly (hack)
            # gid_shift = gid_dict['extgauss'][0] - gid_dict['L2_pyramidal'][0]
            if 'L5_pyramidal' in p_ext.keys():
                gid_extgauss = gid + gid_dict['extgauss'][0]

                nc_dict = {
                    'pos_src': pos_dict['extgauss'][gid],
                    'A_weight': p_ext['L5_pyramidal'][0],
                    'A_delay': p_ext['L5_pyramidal'][1],
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

    # get dendritic biophysics
    def __biophys_dends(self):
        # set dend biophysics specified in Pyr()
        # self.pyr_biophys_dends()

        # set dend biophysics not specified in Pyr()
        for key in self.dends:
            # Insert 'hh' mechanism
            self.dends[key].insert('hh')
            self.dends[key].gkbar_hh = self.p_all['L5Pyr_dend_gkbar_hh']
            self.dends[key].gl_hh = self.p_all['L5Pyr_dend_gl_hh']
            self.dends[key].gnabar_hh = self.p_all['L5Pyr_dend_gnabar_hh']
            self.dends[key].el_hh = self.p_all['L5Pyr_dend_el_hh']

            # Insert 'ca' mechanims
            # Units: pS/um^2
            self.dends[key].insert('ca')
            self.dends[key].gbar_ca = self.p_all['L5Pyr_dend_gbar_ca']

            # Insert 'cad' mechanism
            self.dends[key].insert('cad')
            self.dends[key].taur_cad = self.p_all['L5Pyr_dend_taur_cad']

            # Insert 'kca' mechanism
            self.dends[key].insert('kca')
            self.dends[key].gbar_kca = self.p_all['L5Pyr_dend_gbar_kca']

            # Insert 'km' mechansim
            # Units: pS/um^2
            self.dends[key].insert('km')
            self.dends[key].gbar_km = self.p_all['L5Pyr_dend_gbar_km']

            # insert 'cat' mechanism
            self.dends[key].insert('cat')
            self.dends[key].gbar_cat = self.p_all['L5Pyr_dend_gbar_cat']

            # insert 'ar' mechanism
            self.dends[key].insert('ar')

        # set gbar_ar
        # Value depends on distance from the soma. Soma is set as
        # origin by passing self.soma as a sec argument to nrn.distance()
        # Then iterate over segment nodes of dendritic sections
        # and set gbar_ar depending on nrn.distance(seg.x), which returns
        # distance from the soma to this point on the CURRENTLY ACCESSED
        # SECTION!!!
        nrn.distance(sec=self.soma)

        for key in self.dends:
            self.dends[key].push()
            for seg in self.dends[key]:
                seg.gbar_ar = 1e-6 * np.exp(3e-3 * nrn.distance(seg.x))

            nrn.pop_section()

    # adds biophysics to soma
    def __biophys_soma(self):
        # set soma biophysics specified in Pyr
        # self.pyr_biophys_soma()

        # Insert 'hh' mechanism
        self.soma.insert('hh')
        self.soma.gkbar_hh = self.p_all['L5Pyr_soma_gkbar_hh']
        self.soma.gnabar_hh = self.p_all['L5Pyr_soma_gnabar_hh']
        self.soma.gl_hh = self.p_all['L5Pyr_soma_gl_hh']
        self.soma.el_hh = self.p_all['L5Pyr_soma_el_hh']

        # insert 'ca' mechanism
        # Units: pS/um^2
        self.soma.insert('ca')
        self.soma.gbar_ca = self.p_all['L5Pyr_soma_gbar_ca']

        # insert 'cad' mechanism
        # units of tau are ms
        self.soma.insert('cad')
        self.soma.taur_cad = self.p_all['L5Pyr_soma_taur_cad']

        # insert 'kca' mechanism
        # units are S/cm^2?
        self.soma.insert('kca')
        self.soma.gbar_kca = self.p_all['L5Pyr_soma_gbar_kca']

        # Insert 'km' mechanism
        # Units: pS/um^2
        self.soma.insert('km')
        self.soma.gbar_km = self.p_all['L5Pyr_soma_gbar_km']

        # insert 'cat' mechanism
        self.soma.insert('cat')
        self.soma.gbar_cat = self.p_all['L5Pyr_soma_gbar_cat']

        # insert 'ar' mechanism
        self.soma.insert('ar')
        self.soma.gbar_ar = self.p_all['L5Pyr_soma_gbar_ar']

    # connects sections of this cell together
    def __connect_sections(self):
        # child.connect(parent, parent_end, {child_start=0})
        # Distal (apical)
        self.dends['apical_trunk'].connect(self.soma, 1, 0)
        self.dends['apical_1'].connect(self.dends['apical_trunk'], 1, 0)
        self.dends['apical_2'].connect(self.dends['apical_1'], 1, 0)
        self.dends['apical_tuft'].connect(self.dends['apical_2'], 1, 0)

        # apical_oblique comes off distal end of apical_trunk
        self.dends['apical_oblique'].connect(self.dends['apical_trunk'], 1, 0)

        # Proximal (basal)
        self.dends['basal_1'].connect(self.soma, 0, 0)
        self.dends['basal_2'].connect(self.dends['basal_1'], 1, 0)
        self.dends['basal_3'].connect(self.dends['basal_1'], 1, 0)

    # Returns dictionary of dendritic properties and list of dendrite names
    def __get_dend_props(self):
        return {
            'apical_trunk': {
                'L': self.p_all['L5Pyr_apicaltrunk_L'] ,
                'diam': self.p_all['L5Pyr_apicaltrunk_diam'],
                'cm': self.p_all['L5Pyr_dend_cm'],
                'Ra': self.p_all['L5Pyr_dend_Ra'],
            },
            'apical_1': {
                'L': self.p_all['L5Pyr_apical1_L'],
                'diam': self.p_all['L5Pyr_apical1_diam'],
                'cm': self.p_all['L5Pyr_dend_cm'],
                'Ra': self.p_all['L5Pyr_dend_Ra'],
            },
            'apical_2': {
                'L': self.p_all['L5Pyr_apical2_L'],
                'diam': self.p_all['L5Pyr_apical2_diam'],
                'cm': self.p_all['L5Pyr_dend_cm'],
                'Ra': self.p_all['L5Pyr_dend_Ra'],
            },
            'apical_tuft': {
                'L': self.p_all['L5Pyr_apicaltuft_L'],
                'diam': self.p_all['L5Pyr_apicaltuft_diam'],
                'cm': self.p_all['L5Pyr_dend_cm'],
                'Ra': self.p_all['L5Pyr_dend_Ra'],
            },
            'apical_oblique': {
                'L': self.p_all['L5Pyr_apicaloblique_L'],
                'diam': self.p_all['L5Pyr_apicaloblique_diam'],
                'cm': self.p_all['L5Pyr_dend_cm'],
                'Ra': self.p_all['L5Pyr_dend_Ra'],
            },
            'basal_1': {
                'L': self.p_all['L5Pyr_basal1_L'],
                'diam': self.p_all['L5Pyr_basal1_diam'],
                'cm': self.p_all['L5Pyr_dend_cm'],
                'Ra': self.p_all['L5Pyr_dend_Ra'],
            },
            'basal_2': {
                'L': self.p_all['L5Pyr_basal2_L'],
                'diam': self.p_all['L5Pyr_basal2_diam'],
                'cm': self.p_all['L5Pyr_dend_cm'],
                'Ra': self.p_all['L5Pyr_dend_Ra'],
            },
            'basal_3': {
                'L': self.p_all['L5Pyr_basal3_L'],
                'diam': self.p_all['L5Pyr_basal3_diam'],
                'cm': self.p_all['L5Pyr_dend_cm'],
                'Ra': self.p_all['L5Pyr_dend_Ra'],
            },
        }

    # Sets somatic properties. Returns dictionary.
    def __get_soma_props(self, pos):
         return {
            'pos': pos,
            'L': self.p_all['L5Pyr_soma_L'],
            'diam': self.p_all['L5Pyr_soma_diam'],
            'cm': self.p_all['L5Pyr_soma_cm'],
            'Ra': self.p_all['L5Pyr_soma_Ra'],
            'name': 'L5Pyr',
        }

    # returns synaptic properties
    def __get_syn_props(self):
        return {
            'ampa': {
                'e': self.p_all['L5Pyr_ampa_e'],
                'tau1': self.p_all['L5Pyr_ampa_tau1'],
                'tau2': self.p_all['L5Pyr_ampa_tau2'],
            },
            'nmda': {
                'e': self.p_all['L5Pyr_nmda_e'],
                'tau1': self.p_all['L5Pyr_nmda_tau1'],
                'tau2': self.p_all['L5Pyr_nmda_tau2'],
            },
            'gabaa': {
                'e': self.p_all['L5Pyr_gabaa_e'],
                'tau1': self.p_all['L5Pyr_gabaa_tau1'],
                'tau2': self.p_all['L5Pyr_gabaa_tau2'],
            },
            'gabab': {
                'e': self.p_all['L5Pyr_gabab_e'],
                'tau1': self.p_all['L5Pyr_gabab_tau1'],
                'tau2': self.p_all['L5Pyr_gabab_tau2'],
            }
        }

    # create synapses
    def __synapse_create(self, p_syn):
        # creates synapses onto this cell
        # Somatic synapses
        self.synapses = {
            'soma_gabaa': self.syn_create(self.soma(0.5), p_syn['gabaa']),
            'soma_gabab': self.syn_create(self.soma(0.5), p_syn['gabab']),
        }

        # Dendritic synapses
        self.apicaltuft_gabaa = self.syn_create(self.dends['apical_tuft'](0.5), p_syn['gabaa'])
        self.apicaltuft_ampa = self.syn_create(self.dends['apical_tuft'](0.5), p_syn['ampa'])
        self.apicaltuft_nmda = self.syn_create(self.dends['apical_tuft'](0.5), p_syn['nmda'])

        self.apicaloblique_ampa = self.syn_create(self.dends['apical_oblique'](0.5), p_syn['ampa'])
        self.apicaloblique_nmda = self.syn_create(self.dends['apical_oblique'](0.5), p_syn['nmda'])

        self.basal2_ampa = self.syn_create(self.dends['basal_2'](0.5), p_syn['ampa'])
        self.basal2_nmda = self.syn_create(self.dends['basal_2'](0.5), p_syn['nmda'])

        self.basal3_ampa = self.syn_create(self.dends['basal_3'](0.5), p_syn['ampa'])
        self.basal3_nmda = self.syn_create(self.dends['basal_3'](0.5), p_syn['nmda'])

    # Define 3D shape and position of cell
    def __set_3Dshape(self):
        """ By default neuron uses xy plane for
            height and xz plane for depth. This is opposite for model as a whole, but
            convention is followed in this function for ease use of gui.
            Unused function for visualization only. Not maintained.
        """
        self.shape_soma()

        # soma proximal coords
        x_prox = 0
        y_prox = 0

        # soma distal coords
        x_distal = 0
        y_distal = self.soma.L

        # dend 0-3 are major axis, dend 4 is branch
        # deal with distal first along major cable axis
        for i in range(0, 4):
            nrn.pt3dclear(sec=self.list_dend[i])

            # x_distal and y_distal are the starting points for each segment
            # these are updated at the end of the loop
            nrn.pt3dadd(0, y_distal, 0, self.dend_diam[i], sec=self.list_dend[i])

            # update x_distal and y_distal after setting them
            # x_distal += dend_dx[i]
            y_distal += self.dend_L[i]

            # add next point
            nrn.pt3dadd(0, y_distal, 0, self.dend_diam[i], sec=self.list_dend[i])

        # now deal with dend 4
        # dend 4 will ALWAYS be positioned at the end of dend[0]
        nrn.pt3dclear(sec=self.list_dend[4])

        # activate this section with 'sec=self.list_dend[i]' notation
        x_start = nrn.x3d(1, sec=self.list_dend[0])
        y_start = nrn.y3d(1, sec=self.list_dend[0])

        nrn.pt3dadd(x_start, y_start, 0, self.dend_diam[4], sec=self.list_dend[4])
        # self.dend_L[4] is subtracted because lengths always positive,
        # and this goes to negative x
        nrn.pt3dadd(x_start-self.dend_L[4], y_start, 0, self.dend_diam[4], sec=self.list_dend[4])

        # now deal with proximal dends
        for i in range(5, 8):
            nrn.pt3dclear(sec=self.list_dend[i])

        nrn.pt3dadd(x_prox, y_prox, 0, self.dend_diam[i], sec=self.list_dend[5])
        y_prox += -self.dend_L[5]

        nrn.pt3dadd(x_prox, y_prox, 0, self.dend_diam[5],sec=self.list_dend[5])

        # x_prox, y_prox are now the starting points for BOTH of last 2 sections
        # dend 6
        # Calculate x-coordinate for end of dend
        dend6_x = -self.dend_L[6] * np.sqrt(2)/2
        nrn.pt3dadd(x_prox, y_prox, 0, self.dend_diam[6], sec=self.list_dend[6])
        nrn.pt3dadd(dend6_x, y_prox-self.dend_L[6] * np.sqrt(2)/2,
                    0, self.dend_diam[6], sec=self.list_dend[6])

        # dend 7
        # Calculate x-coordinate for end of dend
        dend7_x = self.dend_L[7] * np.sqrt(2)/2
        nrn.pt3dadd(x_prox, y_prox, 0, self.dend_diam[7], sec=self.list_dend[7])
        nrn.pt3dadd(dend7_x, y_prox-self.dend_L[7] * np.sqrt(2)/2,
                    0, self.dend_diam[7], sec=self.list_dend[7])

        # set 3D position
        # z grid position used as y coordinate in nrn.pt3dchange() to satisfy
        # gui convention that y is height and z is depth. In nrn.pt3dchange()
        # x and z components are scaled by 100 for visualization clarity
        self.soma.push()
        for i in range(0, int(nrn.n3d())):
            nrn.pt3dchange(i, self.pos[0]*100 + nrn.x3d(i), -self.pos[2] + nrn.y3d(i),
                           self.pos[1] * 100 + nrn.z3d(i), nrn.diam3d(i))

        nrn.pop_section()
