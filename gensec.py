""" Generate and Search"""

__author__ = "GenSec"
__copyright__ = "Copyright (C) 2020 Dmitrii Maksimov"
__license__ = "Public Domain"
__version__ = "0.1.1"

from optparse import OptionParser

from gensec.blacklist import *
from gensec.outputs import *
from gensec.structure import *
from gensec.relaxation import *

import numpy as np
import sys
from random import randint, random, uniform

parser = OptionParser()
parser.add_option("-t", "--test"); 
(options, args) = parser.parse_args()

""" Prefase"""
dirs = Directories()
output = Output("report.out")

if len(sys.argv)>0:
    print(sys.argv[1])
    parameters = load_parameters(sys.argv[1])
else:
    parameters = load_parameters("parameters.json")
    
workflow = Workflow()
structure = Structure(parameters)
fixed_frame = Fixed_frame(parameters)
blacklist = Blacklist(structure)
calculator = Calculator(parameters)

from ase.io import write
# import random
# random.seed(3)
# estimate_mu
# while workflow.trials < parameters["trials"]:
#     workflow.trials += 1
#     print("New Trial", workflow.trials)
#     configuration = structure.create_configuration(parameters)
#     structure.apply_configuration(configuration)
#     if all_right(structure, fixed_frame):
#         structure.mu = calculator.estimate_mu(structure, fixed_frame, parameters)
#         structure.A = 1
#         print(structure.mu)
#         break

# 
# for i in [0,0.1, 0.2,0.3,0.5,0.7,1.0,2.0,3,4,5,7,10,12,13,20,25,50,100]:
if parameters["calculator"]["optimize"] == "single":
    dirs.create_directory()
    structure.mu = np.abs(calculator.estimate_mu(structure, fixed_frame, parameters))
    calculator.relax(structure, fixed_frame, parameters, dirs.current_dir())
    sys.exit(0)


if parameters["calculator"]["optimize"] == "generate":
    while workflow.trials < parameters["trials"]:
        while workflow.success < parameters["success"]:
            workflow.trials += 1
            print("New Trial", workflow.trials)
            # output.write("Start the new Trial {}\n".format(workflow.trials))
            # Generate the vector in internal degrees of freedom
            configuration = structure.create_configuration(parameters)
            structure.apply_configuration(configuration)
            if all_right(structure, fixed_frame):
                workflow.success +=1
                print("Structuers is ok")
                dirs.create_directory()
                ensemble = merge_together(structure, fixed_frame)
                dirs.save_to_directory(ensemble, parameters)
            else:
                ensemble = merge_together(structure, fixed_frame)
                write("bad_luck.xyz", ensemble, format="xyz")
                pass
        else:
        	print("{} structures were generated".format(parameters["success"]))
        	sys.exit(0)





while workflow.trials < parameters["trials"]:
    workflow.trials += 1
    print("New Trial", workflow.trials)
    # output.write("Start the new Trial {}\n".format(workflow.trials))
    # Generate the vector in internal degrees of freedom
    configuration = structure.create_configuration(parameters)
    # print(configuration)
    # print(len(configuration))
    # if blacklist.not_in_blacklist(configuration):
    #     blacklist.add_to_blacklist(configuration)
    structure.apply_configuration(configuration)
    if all_right(structure, fixed_frame):
        print("Structuers is ok")
        dirs.create_directory()
        ensemble = merge_together(structure, fixed_frame)
        dirs.save_to_directory(ensemble, parameters)
        structure.mu = np.abs(calculator.estimate_mu(structure, fixed_frame, parameters))
        # print(structure.mu)
        calculator.relax(structure, fixed_frame, parameters, dirs.current_dir())
    else:
        ensemble = merge_together(structure, fixed_frame)
        write("bad_luck.xyz", ensemble, format="xyz")
        pass

        # for i in range(len(structure.molecules)):
        #     print(len(structure.molecules[i]))
        # print(len(fixed_frame.fixed_frame))
        # print(configuration)
        # print("something not cool")

        # if len(structure.molecules) > 1:
        #     a0 = structure.molecules[0]
        #     for i in range(1, len(structure.molecules)):
        #         a0+=structure.molecules[i]
        # else:
        #     a0 = structure.molecules[0]

        # all_atoms = a0 + fixed_frame.fixed_frame
        # write("bad_configuration.in", all_atoms,format="aims" )

    # else:
    #     # output.write("Next trial, found in blacklist")
    #     continue

# Write the enesemble into file 
# add the fixed frame also
# Extend method can be used instead this:





## Identify the periodic boundary conditions (PBC)

# Start generating of the ensembles
## Check for intermolecular clashes
## Check for intramolecular clashes
## Check for PBC clashes
# 
# Optional Preexploration of the conformational space
## RMSD blacklisting
## Internal degrees of freedom blacklisting
## SOAP blacklisting

# Potential evaluation
## Blacklist check
## Run Minimization
## Blacklist check

# Next Trial
