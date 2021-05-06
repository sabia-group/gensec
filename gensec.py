"""Generate and Search

Attributes:
    parser (TYPE): Description
    protocol (TYPE): Description
"""

__author__ = "Dmitrii Maksimov"
__copyright__ = "Copyright (C) 2021 GenSec"
__license__ = "GNU General Public License v2 or later (GPLv2+)"
__version__ = "0.1.5"

from gensec.protocols import Protocol
import sys
import os
import json
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


def load_parameters(parameters_file):
    """Load the parameters from parameter file"""

    if parameters_file is not None:
        print("Loading parameters")
        with open(os.path.join(os.getcwd(), parameters_file)) as f:
            parameters = json.load(f)
    else:
        print("No parameter file, using default settings")
        parameters = None
    return parameters


if len(sys.argv) > 0:
    parameters = load_parameters(sys.argv[1])
else:
    parameters = load_parameters("parameters.json")

protocol = Protocol()
protocol.run(parameters)
sys.exit(0)
