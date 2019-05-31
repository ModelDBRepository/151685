<html><pre>

    This is the model associated with Jones et al 2009 and Lee and
    Jones (2013). (See reference below).  The archive was downloaded
    from the github repository:

    <a href="https://bitbucket.org/jonescompneurolab/corticaldipole">https://bitbucket.org/jonescompneurolab/corticaldipole</a>

    on July 26th, 2017.

    It runs with  Neuron (7.3+ +parallel +python) + Python (3.5.x), parallel (mpi4py, mpich2/openmpi)

Notes and runtime
================================================
This is primarily a model of neocortical L2 and L5, featuring simplified neuronal geometries, excitatory pyramidal cells, and single compartmental fast spiking inhibitory interneurons. The defining feature of the model is the current dipole output consisting of the intracellular current contributions of the pyramidal cells in L2 and L5.

The present version was tested to run on Python 3.5 on both Mac OS X and Linux-based platforms. Please get installation help from the software vendors. There is no support available for Windows. Please report any issues you may have with Python 2.7.x.

To run the model, you must have a working Neuron with MPI and Python support.

You must run nrnivmodl to compile the mod files in the mod/ directory.

`$ nrnivmodl mod/*`

Alternatively, you may be able to run GNU make on the included Makefile:

`$ make`

If that completes successfully, then run:

`$ mpiexec -n 4 python run.py param/gamma_L5weak_L2weak.param`

where "4" is the number of cores you wish to use.

This saves data and a plot in the `data` directory.

Approximate expected output is seen in png files in this directory.

.param files
================================================
The model uses a flat text file (.param) that holds a "key: value" string per line. The keys are valid param names, and the values have to be values that are loosely valid. The model will fail to run without checks if this file is not well formed. Param values that are not explicitly given by the .param file will default to a value in params_default.py. Editable params are also in the params_default.py file. See debug.param or any of the param files as examples.

Undefined params will be ignored.

Known Issues
================================================
On some platforms, there is sometimes output such as:

stty: 'standard input': Bad file descriptor

that is multiplied by the number of cores run. This can be ignored.

License and citation
================================================
This code is released under the GNU GPLv3, except for some files that do not include a license and were taken from other sources (noted in headers, see mod files). Please cite usage of this model as:

Lee, Shane, and Stephanie R. Jones. "Distinguishing mechanisms of gamma frequency oscillations in human current source signals using a computational model of a laminar neocortical network." Frontiers in human neuroscience 7 (2013).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.
