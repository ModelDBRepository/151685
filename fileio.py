""" fileio.py - general file input/output functions
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

import re
import os
import pickle

# Cleans input files
def clean_lines(file):
    with open(file) as f_in:
        lines = (line.rstrip() for line in f_in)
        lines = [line for line in lines if line]

    return lines

# load pkl objects
def pkl_load(f):
    # remove any kind of file handlers or extra whitespace
    f = f.strip()
    f = re.sub('file://', '', f)

    # if this is a file
    if os.path.isfile(f):
        x = pickle.load(open(f, "rb"))

    else:
        print("Cannot understand file {}".format(f))
        x = None

    return x

# file saver for objects
def pkl_save(fsave, data):
    pickle.dump(data, open(fsave, "wb"), pickle.HIGHEST_PROTOCOL)
