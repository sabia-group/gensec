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
import os
import glob
import shutil
from random import randint, random, uniform

parser = OptionParser()
parser.add_option("-t", "--test"); 
(options, args) = parser.parse_args()

""" Prefase"""


if len(sys.argv)>0:
    print(sys.argv[1])
    parameters = load_parameters(sys.argv[1])
else:
    parameters = load_parameters("parameters.json")



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

if parameters["calculator"]["optimize"] == "single":
    for directory in os.listdir(os.path.join(os.getcwd(), "generate")):
        current_dir = os.path.join(os.getcwd(), "generate", directory)
        # dirs.create_directory()
        structure.mu = np.abs(calculator.estimate_mu(structure, fixed_frame, parameters))
        for i in ["ID", "Lindh", "Lindh_RMSD"]:
            parameters["name"] = i
            if "Lindh" in i:
                parameters["calculator"]["preconditioner"]["mol"] = "Lindh"
                if i == "Lindh_RMSD":
                    parameters["calculator"]["preconditioner"]["rmsd_update"] = 0.05
            else:
                parameters["calculator"]["preconditioner"]["mol"] = "ID"
            calculator.relax(structure, fixed_frame, parameters, current_dir)
        # os.system("rm /tmp/ipi_*")
    sys.exit(0)

if parameters["calculator"]["optimize"] == "generate_and_relax_Diff":
    os.system("rm /tmp/ipi_*")
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
                for i in ["Lindh", "Lindh_RMSD"]:
                # for i in ["ID", "Lindh", "    ", "    _RMSD", "Lindh_ID_    ", "Lindh_ID_    _RMSD", "Lindh_vdW_    ", "Lindh_vdW_    _RMSD",]:
                    parameters["name"] = i   
                    if i == "ID":
                        parameters["calculator"]["preconditioner"]["mol"] = "ID"
                        parameters["calculator"]["preconditioner"]["fixed_frame"] = "ID"
                        parameters["calculator"]["preconditioner"]["mol-mol"] = "ID"
                        parameters["calculator"]["preconditioner"]["mol-fixed_frame"] = "ID"

                    elif i == "Lindh" or i =="Lindh_RMSD":
                        parameters["calculator"]["preconditioner"]["mol"] = "Lindh"
                        parameters["calculator"]["preconditioner"]["fixed_frame"] = "Lindh"
                        parameters["calculator"]["preconditioner"]["mol-mol"] = "Lindh"
                        parameters["calculator"]["preconditioner"]["mol-fixed_frame"] = "Lindh"

                    elif i == "    " or i =="    _RMSD":
                        parameters["calculator"]["preconditioner"]["mol"] = "    "
                        parameters["calculator"]["preconditioner"]["fixed_frame"] = "    "
                        parameters["calculator"]["preconditioner"]["mol-mol"] = "    "
                        parameters["calculator"]["preconditioner"]["mol-fixed_frame"] = "    "

                    elif i == "Lindh_ID_    " or i =="Lindh_ID_    _RMSD":
                        parameters["calculator"]["preconditioner"]["mol"] = "Lindh"
                        parameters["calculator"]["preconditioner"]["fixed_frame"] = "    "
                        parameters["calculator"]["preconditioner"]["mol-mol"] = "ID"
                        parameters["calculator"]["preconditioner"]["mol-fixed_frame"] = "ID"

                    elif i == "Lindh_vdW_    " or i =="Lindh_vdW_    _RMSD":
                        parameters["calculator"]["preconditioner"]["mol"] = "Lindh"
                        parameters["calculator"]["preconditioner"]["fixed_frame"] = "    "
                        parameters["calculator"]["preconditioner"]["mol-mol"] = "vdW"
                        parameters["calculator"]["preconditioner"]["mol-fixed_frame"] = "vdW"


                    if "RMSD" in i:
                        parameters["calculator"]["preconditioner"]["rmsd_update"] = 0.025
                    else:
                        parameters["calculator"]["preconditioner"]["rmsd_update"] = 100.0

                    calculator.relax(structure, fixed_frame, parameters, dirs.current_dir())
            else:
                ensemble = merge_together(structure, fixed_frame)
                write("bad_luck.xyz", ensemble, format="xyz")
                pass
        else:
            print("{} structures were generated".format(parameters["success"]))
            sys.exit(0)


# if parameters["calculator"]["optimize"] == "generate":
#     os.system("rm /tmp/ipi_*")
#     while workflow.trials < parameters["trials"]:
#         while workflow.success < parameters["success"]:
#             workflow.trials += 1
#             print("New Trial", workflow.trials)
#             # output.write("Start the new Trial {}\n".format(workflow.trials))
#             # Generate the vector in internal degrees of freedom
#             configuration = structure.create_configuration(parameters)
#             structure.apply_configuration(configuration)
#             if all_right(structure, fixed_frame):
#                 workflow.success +=1
#                 print("Structuers is ok")
#                 dirs.create_directory()
#                 ensemble = merge_together(structure, fixed_frame)
#                 dirs.save_to_directory(ensemble, parameters)
#             else:
#                 ensemble = merge_together(structure, fixed_frame)
#                 write("bad_luck.xyz", ensemble, format="xyz")
#                 pass
#         else:
#             print("{} structures were generated".format(parameters["success"]))
#             sys.exit(0)

def try_relax(structure, fixed_frame, parameters, dir):
    relaxed = True
    while True:
        try:
            calculator.relax(structure, fixed_frame, parameters, dirs.current_dir(parameters))
        except:
            relaxed = False
    return relaxed


if parameters["calculator"]["optimize"] == "search":
    dirs = Directories(parameters)
    output = Output("report.out")    
    workflow = Workflow()
    structure = Structure(parameters)
    fixed_frame = Fixed_frame(parameters)
    calculator = Calculator(parameters)
    blacklist = Blacklist(structure)
    dirs.find_last_dir(parameters)
    blacklist.check_calculated(dirs, parameters)
    blacklist.analyze_calculated(structure, fixed_frame, parameters)
    for f in glob.glob("/tmp/ipi_*"):
        os.remove(f)
    workflow.success = dirs.dir_num
    structure.mu = np.abs(calculator.estimate_mu(structure, fixed_frame, parameters))
    generated_dirs = os.listdir(os.path.join(os.getcwd(), "generate"))
    blacklist.torsional_diff_degree = 5
    if len(generated_dirs)>0:
        for generated in generated_dirs:
            gen = os.path.join(os.getcwd(), "generate", generated, generated+".in")
            configuration = structure.read_configuration(structure, fixed_frame, gen)
            print(gen)
            found = blacklist.find_in_blacklist(structure.torsions_from_conf(configuration))
            if not found:
                dirs.create_directory(parameters)
                dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
                calculator.relax(structure, fixed_frame, parameters, dirs.current_dir(parameters))
                dirs.finished(parameters)
                blacklist.check_calculated(dirs, parameters)
                blacklist.add_to_blacklist_traj(structure, fixed_frame, dirs.current_dir(parameters))
                workflow.success += 1
                workflow.trials = 0
                shutil.rmtree(os.path.join(os.getcwd(), "generate", generated))
                # else:
                #     print("Error while relaxing")
                #     shutil.rmtree(os.path.join(os.getcwd(), "generate", generated))
                #     continue
            else:
                print("Already known")
                shutil.rmtree(os.path.join(os.getcwd(), "generate", generated))


    # when run out structures 
    while workflow.trials < parameters["trials"]:
        print("New Trial", workflow.trials)
        while workflow.success < parameters["success"]:
            # output.write("Start the new Trial {}\n".format(workflow.trials))
            # Generate the vector in internal degrees of freedom
            configuration = structure.create_configuration(parameters)
            structure.apply_configuration(configuration)
            if all_right(structure, fixed_frame):
                print("Structuers is ok")
                found = blacklist.find_in_blacklist(structure.torsions_from_conf(configuration))
                if not found:
                    dirs.create_directory(parameters)
                    dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
                    calculator.relax(structure, fixed_frame, parameters, dirs.current_dir(parameters))
                    dirs.finished(parameters)
                    blacklist.check_calculated(dirs, parameters)
                    blacklist.add_to_blacklist(structure.torsions_from_conf(configuration))
                    workflow.success += 1
                    workflow.trials = 0
                else:
                    print("Found in blacklist!!!")
                    workflow.trials += 1 
                    if workflow.trials == parameters["trials"]:
                        if blacklist.torsional_diff_degree > 5:
                            blacklist.torsional_diff_degree -= 5
                            print("\n\n\n\n\n")
                            print(blacklist.torsional_diff_degree)
                            print("\n\n\n\n\n")
                            workflow.trials = 0
                            pass
                        else:
                            print("{} The space is sampled quite properly".format(parameters["trials"]))
                            sys.exit(0)
                    else:
                        pass
            else:
                print("Structuers is not ok")
                write("bad_luck.xyz", merge_together(structure, fixed_frame), format="xyz")
                pass
        else:
            print("{} number of structures was successfully relaxed!".format(parameters["success"]))
            print("Terminating algorithm")
            sys.exit(0)

if parameters["calculator"]["optimize"] == "generate":
    dirs = Directories(parameters)
    output = Output("report.out")    
    workflow = Workflow()
    structure = Structure(parameters)
    fixed_frame = Fixed_frame(parameters)
    calculator = Calculator(parameters)
    blacklist = Blacklist(structure)
    dirs.find_last_dir(parameters)
    blacklist.check_calculated(dirs, parameters)
    blacklist.analyze_calculated(structure, fixed_frame, parameters)
    dirs.find_last_generated_dir(parameters)
    for f in glob.glob("/tmp/ipi_*"):
        os.remove(f)
    workflow.success = dirs.dir_num
    while workflow.trials < parameters["trials"]:
        # print("New Trial", workflow.trials)
        while workflow.success < parameters["success"]:
            # output.write("Start the new Trial {}\n".format(workflow.trials))
            # Generate the vector in internal degrees of freedom
            configuration = structure.create_configuration(parameters)
            structure.apply_configuration(configuration)
            if all_right(structure, fixed_frame):
                # print("Structuers is ok")
                # print(len(blacklist.blacklist))
                found = blacklist.find_in_blacklist(structure.torsions_from_conf(configuration))
                if not found:
                    dirs.create_directory(parameters)
                    dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
                    blacklist.add_to_blacklist(structure.torsions_from_conf(configuration))
                    workflow.success += 1
                    workflow.trials = 0
                else:
                    # print("Found in blacklist!!!")
                    workflow.trials += 1 
                    if workflow.trials == parameters["trials"]:
                        if blacklist.torsional_diff_degree > 60:
                            blacklist.torsional_diff_degree -= 5
                            # print("\n\n\n\n\n")
                            print("Swithing to ", blacklist.torsional_diff_degree)
                            # print("\n\n\n\n\n")
                            workflow.trials = 0
                            pass
                        else:
                            print("{} The space is sampled quite properly".format(parameters["trials"]))
                            sys.exit(0)
                    else:
                        pass
            else:
                write("bad_luck.xyz", merge_together(structure, fixed_frame), format="xyz")
                pass
        else:
            print("{} number of structures was successfully generated!".format(parameters["success"]))
            print("Terminating algorithm")
            sys.exit(0)



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
# Optional Pre    loration of the conformational space
## RMSD blacklisting
## Internal degrees of freedom blacklisting
## SOAP blacklisting

# Potential evaluation
## Blacklist check
## Run Minimization
## Blacklist check

# Next Trial
