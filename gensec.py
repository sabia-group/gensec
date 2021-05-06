"""Generate and Search

Attributes:
    parser (TYPE): Description
    protocol (TYPE): Description
"""

__author__ = "Dmitrii Maksimov"
__copyright__ = "Copyright (C) 2021 GenSec"
__license__ = "GNU General Public License v2 or later (GPLv2+)"
__version__ = "0.1.5"

from gensec.known import *
from gensec.outputs import *
from gensec.structure import *
from gensec.relaxation import *
from gensec.modules import measure_torsion_of_last
from gensec.general import *
from gensec.protocols import *
import numpy as np
import sys
import os
import shutil
from random import randint, random, uniform
from ase.io.trajectory import Trajectory
from ase.io import write
from optparse import OptionParser

parser = OptionParser()
parser.add_option(
    "-p",
    "--parameters",
    dest="parameters",
    help="File with parameters",
    metavar="FILE",
    default=None,
)

(options, args) = parser.parse_args()

""" Load the parameters from parameter file """
if len(sys.argv) > 0:
    parameters = load_parameters(sys.argv[1])
else:
    parameters = load_parameters("parameters.json")

protocol = Protocol()
protocol.run(parameters)
sys.exit(0)
