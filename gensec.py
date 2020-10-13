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
from gensec.modules import measure_torsion_of_last

import numpy as np
import sys
import os
import glob
import shutil
from random import randint, random, uniform
from ase.io.trajectory import Trajectory
from ase.io import write

parser = OptionParser()
parser.add_option("-t", "--test"); 
(options, args) = parser.parse_args()

""" Prefase"""
if len(sys.argv)>0:
    print(sys.argv[1])
    parameters = load_parameters(sys.argv[1])
else:
    parameters = load_parameters("parameters.json")
print(parameters["calculator"]["optimize"])

if "search" in parameters["calculator"]["optimize"]:
    dirs = Directories(parameters)
    output = Output("report_{}.out".format(parameters["calculator"]["optimize"]))    
    workflow = Workflow()
    structure = Structure(parameters)
    fixed_frame = Fixed_frame(parameters)
    calculator = Calculator(parameters)
    blacklist = Blacklist(structure)
    dirs.find_last_dir(parameters)
    blacklist.check_calculated(dirs, parameters)
    blacklist.analyze_calculated(structure, fixed_frame, parameters)
    output.write_parameters(parameters, structure, blacklist, dirs)
    for f in glob.glob("/tmp/ipi_*"):
        if os.path.exists(os.path.join("/tmp/", f)):
            os.remove(os.path.join("/tmp/", f))
    workflow.success = dirs.dir_num
    structure.mu = np.abs(calculator.estimate_mu(structure, fixed_frame, parameters))
    generated_dirs = os.listdir(os.path.join(os.getcwd(), "generate"))
    # while len(generated_dirs)>0:
    #     output.write_to_report("\nThere are {} candidate structures to relax\n".format(len(generated_dirs)))
    #     for generated in sorted(generated_dirs):
    #         d = os.path.join(os.getcwd(), "generate", generated)
    #         output.write_to_report("\nTaking structure from folder {}\n".format(d))
    #         gen = os.path.join(d, generated+".in")
    #         configuration = structure.read_configuration(structure, fixed_frame, gen)
    #         found = blacklist.find_in_blacklist(structure.torsions_from_conf(configuration), criteria="loose", t=10)
    #         if not found:
    #             dirs.create_directory(parameters)
    #             dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
    #             calculator.relax(structure, fixed_frame, parameters, dirs.current_dir(parameters), blacklist)
    #             dirs.finished(parameters)
    #             blacklist.check_calculated(dirs, parameters)
    #             blacklist.add_to_blacklist_traj(structure, fixed_frame, dirs.current_dir(parameters))
    #             t = blacklist.find_traj(os.path.join(dirs.current_dir(parameters)))
    #             conf = measure_torsion_of_last(Trajectory(os.path.join(dirs.current_dir(parameters), t))[-1], structure.list_of_torsions)
    #             output.write_to_report("found in blacklist {}".format(blacklist.find_in_blacklist(conf, criteria="loose", t=10)))
    #             workflow.success += 1
    #             workflow.trials = 0
    #             output.write_successfull_relax(parameters, structure, blacklist, dirs)
    #             shutil.rmtree(d)
    #             output.write_to_report("\nGenerated structure in folder {} is deleted\n".format(d))
    #             generated_dirs = os.listdir(os.path.join(os.getcwd(), "generate"))
    #             output.write_to_report("\nThere are {} candidate structures left to relax\n".format(len(generated_dirs)))
    #         else:
    #             shutil.rmtree(d)
    #             generated_dirs = os.listdir(os.path.join(os.getcwd(), "generate"))
    #             output.write_to_report("\nStructure in folder {} is already in blacklist. Delete.\n".format(d))
    #             output.write_to_report("\nThere are {} candidate structures left to relax\n".format(len(generated_dirs)))
    # # when run out structures 
    # output.write_to_report("All the structures in \"generate\" folder are calculated.")
    # output.write_to_report("Continue to generate and search.\n")
    # blacklist.criteria = "loose"

    while workflow.trials < parameters["trials"]:
        generated_dirs = os.listdir(os.path.join(os.getcwd(), "generate"))
        while workflow.success < parameters["success"]:
            while len(generated_dirs)>0:
                generated_dirs = os.listdir(os.path.join(os.getcwd(), "generate"))
                output.write_to_report("\nThere are {} candidate structures to relax\n".format(len(generated_dirs)))
                d = os.path.join(os.getcwd(), "generate", sorted(generated_dirs)[0])
                output.write_to_report("\nTaking structure from folder {}\n".format(d))
                gen = os.path.join(d, sorted(generated_dirs)[0]+".in")
                configuration = structure.read_configuration(structure, fixed_frame, gen)
                shutil.rmtree(d)
                blacklist.update_blacklist(blacklist.names, os.listdir(blacklist.dir), structure, fixed_frame)
                found = blacklist.find_in_blacklist(structure.torsions_from_conf(configuration), criteria="loose", t=10)
                if not found:
                    dirs.create_directory(parameters)
                    dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
                    calculator.relax(structure, fixed_frame, parameters, dirs.current_dir(parameters), blacklist)                           
                    t = blacklist.find_traj(os.path.join(dirs.current_dir(parameters)))
                    conf = measure_torsion_of_last(Trajectory(os.path.join(dirs.current_dir(parameters), t))[-1], structure.list_of_torsions)
                    ff = blacklist.find_in_blacklist(conf, criteria="loose", t=10)
                    if not ff:
                        dirs.finished(parameters)
                        # blacklist.check_calculated(dirs, parameters)
                        # blacklist.add_to_blacklist_traj(structure, fixed_frame, dirs.current_dir(parameters))
                        workflow.success += 1
                        workflow.trials = 0
                        output.write_successfull_relax(parameters, structure, blacklist, dirs)
                        output.write_to_report("\nGenerated structure in folder {} is deleted\n".format(d))
                        output.write_to_report("\nThere are {} candidate structures left to relax\n".format(len(generated_dirs)))
                        # generated_dirs = os.listdir(os.path.join(os.getcwd(), "generate"))
                    else:
                        output.write_to_report("found in blacklist {}".format(ff))
                        dirs.blacklisted(parameters)
                        # generated_dirs = os.listdir(os.path.join(os.getcwd(), "generate"))                                
                        # blacklist.check_calculated(dirs, parameters)
                        # blacklist.add_to_blacklist_traj(structure, fixed_frame, dirs.current_dir(parameters))

                else:
                    output.write_to_report("\nStructure in folder {} is already in blacklist. Delete.\n".format(d))
                    output.write_to_report("\nThere are {} candidate structures left to relax\n".format(len(generated_dirs)))
                    # generated_dirs = os.listdir(os.path.join(os.getcwd(), "generate"))
            # when run out structures 
            # output.write_to_report("All the structures in \"generate\" folder are calculated.")
            else:
                output.write_to_report("Continue to generate and search.\n")
                blacklist.criteria = "loose"
            # output.write("Start the new Trial {}\n".format(workflow.trials))

            # Generate the vector in internal degrees of freedom
            configuration = structure.create_configuration(parameters)
            structure.apply_configuration(configuration)
            if all_right(structure, fixed_frame):
                blacklist.update_blacklist(blacklist.names, os.listdir(blacklist.dir), structure, fixed_frame)
                print(len(blacklist.blacklist))
                found = blacklist.find_in_blacklist(structure.torsions_from_conf(configuration), criteria="loose", t=10)
                print("\n\n\n", len(blacklist.blacklist), "\n\n\n")
                if not found:
                    dirs.create_directory(parameters)
                    dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
                    calculator.relax(structure, fixed_frame, parameters, dirs.current_dir(parameters), blacklist)
                    t = blacklist.find_traj(os.path.join(dirs.current_dir(parameters)))
                    conf = measure_torsion_of_last(Trajectory(os.path.join(dirs.current_dir(parameters), t))[-1], structure.list_of_torsions)
                    ff = blacklist.find_in_blacklist(conf, criteria="loose", t=10)
                    print("\n\n\n\n\n", ff)
                    if not ff:
                        dirs.finished(parameters)
                        blacklist.send_traj_to_blacklist_folder(dirs, parameters)
                        # blacklist.add_to_blacklist_traj(structure, fixed_frame, dirs.current_dir(parameters))
                        workflow.success += 1
                        workflow.trials = 0
                        output.write_successfull_relax(parameters, structure, blacklist, dirs)
                    else:
                        output.write_to_report("found in blacklist {}".format(ff))
                        dirs.blacklisted(parameters)                                
                        blacklist.send_traj_to_blacklist_folder(dirs, parameters)
                        # blacklist.add_to_blacklist_traj(structure, fixed_frame, dirs.current_dir(parameters))
                        workflow.success += 1
                        workflow.trials = 0
                else:
                    workflow.trials += 1 
                    if workflow.trials == parameters["trials"]:
                        if blacklist.torsional_diff_degree > 10:
                            blacklist.torsional_diff_degree -= 5
                            output.write_to_report("\nDecreasing the criteria for torsional angles to {}\n".format(blacklist.torsional_diff_degree))
                            workflow.trials = 0
                            pass
                        else:
                            print("Swithing to loose criteria:\n")
                            if blacklist.criteria == "strict":
                                blacklist.criteria = "loose"
                                blacklist.torsional_diff_degree = 120
                                output.write_to_report("Start to look with loose criteria\n")
                                workflow.trials = 0
                                pass
                            else:
                               output.write_to_report("Cannot find new structures\n")
                               sys.exit(0)
                    else:
                        pass
            else:
                write("bad_luck.xyz", merge_together(structure, fixed_frame), format="xyz")
                pass
        else:
            output.write_to_report("{} number of structures was successfully relaxed!".format(parameters["success"]))
            output.write_to_report("Terminating algorithm")
            sys.exit(0)

if parameters["calculator"]["optimize"] == "generate":
    dirs = Directories(parameters)
    output = Output("report_generate.out")
    workflow = Workflow()
    structure = Structure(parameters)
    fixed_frame = Fixed_frame(parameters)
    calculator = Calculator(parameters)
    blacklist = Blacklist(structure)
    dirs.find_last_dir(parameters)
    blacklist.check_calculated(dirs, parameters)
    blacklist.analyze_calculated(structure, fixed_frame, parameters)
    dirs.find_last_generated_dir(parameters)
    output.write_parameters(parameters, structure, blacklist, dirs)
    # calculated_dir = os.path.join(os.getcwd(), "search") 
    # snapshots = len(os.listdir(calculated_dir))
    workflow.success = dirs.dir_num
    while workflow.trials < parameters["trials"]:
        while workflow.success < parameters["success"]:
            # Generate the vector in internal degrees of freedom
            configuration = structure.create_configuration(parameters)
            structure.apply_configuration(configuration)
            if all_right(structure, fixed_frame):
                blacklist.update_blacklist(blacklist.names, os.listdir(blacklist.dir), structure, fixed_frame)
                found = blacklist.find_in_blacklist(structure.torsions_from_conf(configuration), criteria=blacklist.criteria, t=blacklist.torsional_diff_degree)
                if not found:
                    dirs.create_directory(parameters)
                    dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
                    blacklist.add_to_blacklist(structure.torsions_from_conf(configuration))
                    workflow.success += 1
                    workflow.trials = 0
                    output.write_successfull_generate(parameters, structure.torsions_from_conf(configuration), dirs)
                else:
                    workflow.trials += 1 
                    print("Trial {}".format(workflow.trials))
                    # if dirs.dir_num > snapshots:
                    #     need_to_visit = range(snapshots+1, len(os.listdir(calculated_dir))+1)
                    #     for d in need_to_visit:
                    #         d_name = os.path.join(os.getcwd(), "search", "{:010d}".format(d))
                    #         if "finished" in os.listdir(d_name):
                    #             output.write_to_report("Adding trajectory from {} to blacklist.\n".format(d_name))
                    #             blacklist.add_to_blacklist_traj(structure, fixed_frame, d_name)
                    #             snapshots += 1 
                    if workflow.trials == parameters["trials"]:
                        if blacklist.torsional_diff_degree > 10:
                            blacklist.torsional_diff_degree -= 5
                            output.write_to_report("\nDecreasing the criteria for torsional angles to {}\n".format(blacklist.torsional_diff_degree))
                            workflow.trials = 0
                            pass
                        else:
                            print("Swithing to loose criteria:\n")
                            if blacklist.criteria == "strict":
                                blacklist.criteria = "loose"
                                blacklist.torsional_diff_degree = 120
                                output.write_to_report("Start to look with loose criteria\n")
                                workflow.trials = 0
                                pass
                            else:
                               output.write_to_report("Cannot find new structures\n")
                               sys.exit(0)
                    else:
                        pass
            else:
                write("bad_luck.xyz", merge_together(structure, fixed_frame), format="xyz")
                pass
        else:
            output.write_to_report("{} number of structures was successfully generated!\n".format(parameters["success"]))
            output.write_to_report("Terminating algorithm\n")
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
# # import random
# # random.seed(3)
# # estimate_mu
# # while workflow.trials < parameters["trials"]:
# #     workflow.trials += 1
# #     print("New Trial", workflow.trials)
# #     configuration = structure.create_configuration(parameters)
# #     structure.apply_configuration(configuration)
# #     if all_right(structure, fixed_frame):
# #         structure.mu = calculator.estimate_mu(structure, fixed_frame, parameters)
# #         structure.A = 1
# #         print(structure.mu)
# #         break

# if parameters["calculator"]["optimize"] == "single":
#     for directory in os.listdir(os.path.join(os.getcwd(), "generate")):
#         current_dir = os.path.join(os.getcwd(), "generate", directory)
#         # dirs.create_directory()
#         structure.mu = np.abs(calculator.estimate_mu(structure, fixed_frame, parameters))
#         for i in ["ID", "Lindh", "Lindh_RMSD"]:
#             parameters["name"] = i
#             if "Lindh" in i:
#                 parameters["calculator"]["preconditioner"]["mol"] = "Lindh"
#                 if i == "Lindh_RMSD":
#                     parameters["calculator"]["preconditioner"]["rmsd_update"] = 0.05
#             else:
#                 parameters["calculator"]["preconditioner"]["mol"] = "ID"
#             calculator.relax(structure, fixed_frame, parameters, current_dir)
#         # os.system("rm /tmp/ipi_*")
#     sys.exit(0)

# if parameters["calculator"]["optimize"] == "generate_and_relax_Diff":
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
#                 for i in ["Lindh", "Lindh_RMSD"]:
#                 # for i in ["ID", "Lindh", "    ", "    _RMSD", "Lindh_ID_    ", "Lindh_ID_    _RMSD", "Lindh_vdW_    ", "Lindh_vdW_    _RMSD",]:
#                     parameters["name"] = i   
#                     if i == "ID":
#                         parameters["calculator"]["preconditioner"]["mol"] = "ID"
#                         parameters["calculator"]["preconditioner"]["fixed_frame"] = "ID"
#                         parameters["calculator"]["preconditioner"]["mol-mol"] = "ID"
#                         parameters["calculator"]["preconditioner"]["mol-fixed_frame"] = "ID"

#                     elif i == "Lindh" or i =="Lindh_RMSD":
#                         parameters["calculator"]["preconditioner"]["mol"] = "Lindh"
#                         parameters["calculator"]["preconditioner"]["fixed_frame"] = "Lindh"
#                         parameters["calculator"]["preconditioner"]["mol-mol"] = "Lindh"
#                         parameters["calculator"]["preconditioner"]["mol-fixed_frame"] = "Lindh"

#                     elif i == "    " or i =="    _RMSD":
#                         parameters["calculator"]["preconditioner"]["mol"] = "    "
#                         parameters["calculator"]["preconditioner"]["fixed_frame"] = "    "
#                         parameters["calculator"]["preconditioner"]["mol-mol"] = "    "
#                         parameters["calculator"]["preconditioner"]["mol-fixed_frame"] = "    "

#                     elif i == "Lindh_ID_    " or i =="Lindh_ID_    _RMSD":
#                         parameters["calculator"]["preconditioner"]["mol"] = "Lindh"
#                         parameters["calculator"]["preconditioner"]["fixed_frame"] = "    "
#                         parameters["calculator"]["preconditioner"]["mol-mol"] = "ID"
#                         parameters["calculator"]["preconditioner"]["mol-fixed_frame"] = "ID"

#                     elif i == "Lindh_vdW_    " or i =="Lindh_vdW_    _RMSD":
#                         parameters["calculator"]["preconditioner"]["mol"] = "Lindh"
#                         parameters["calculator"]["preconditioner"]["fixed_frame"] = "    "
#                         parameters["calculator"]["preconditioner"]["mol-mol"] = "vdW"
#                         parameters["calculator"]["preconditioner"]["mol-fixed_frame"] = "vdW"


#                     if "RMSD" in i:
#                         parameters["calculator"]["preconditioner"]["rmsd_update"] = 0.025
#                     else:
#                         parameters["calculator"]["preconditioner"]["rmsd_update"] = 100.0

#                     calculator.relax(structure, fixed_frame, parameters, dirs.current_dir())
#             else:
#                 ensemble = merge_together(structure, fixed_frame)
#                 write("bad_luck.xyz", ensemble, format="xyz")
#                 pass
#         else:
#             print("{} structures were generated".format(parameters["success"]))
#             sys.exit(0)


# # if parameters["calculator"]["optimize"] == "generate":
# #     os.system("rm /tmp/ipi_*")
# #     while workflow.trials < parameters["trials"]:
# #         while workflow.success < parameters["success"]:
# #             workflow.trials += 1
# #             print("New Trial", workflow.trials)
# #             # output.write("Start the new Trial {}\n".format(workflow.trials))
# #             # Generate the vector in internal degrees of freedom
# #             configuration = structure.create_configuration(parameters)
# #             structure.apply_configuration(configuration)
# #             if all_right(structure, fixed_frame):
# #                 workflow.success +=1
# #                 print("Structuers is ok")
# #                 dirs.create_directory()
# #                 ensemble = merge_together(structure, fixed_frame)
# #                 dirs.save_to_directory(ensemble, parameters)
# #             else:
# #                 ensemble = merge_together(structure, fixed_frame)
# #                 write("bad_luck.xyz", ensemble, format="xyz")
# #                 pass
# #         else:
# #             print("{} structures were generated".format(parameters["success"]))
# #             sys.exit(0)

# def try_relax(structure, fixed_frame, parameters, dir):
#     relaxed = True
#     while True:
#         try:
#             calculator.relax(structure, fixed_frame, parameters, dirs.current_dir(parameters))
#         except:
#             relaxed = False
#     return relaxed