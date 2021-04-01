""" Generate and Search"""

__author__ = "GenSec"
__copyright__ = "Copyright (C) 2020 Dmitrii Maksimov"
__license__ = "Public Domain"
__version__ = "0.1.1"

from optparse import OptionParser

from gensec.known import *
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
    workflow = Workflow()
    structure = Structure(parameters)
    fixed_frame = Fixed_frame(parameters)
    calculator = Calculator(parameters)
    known = Known(structure, parameters)
    if not os.path.exists(parameters["calculator"]["optimize"]):
        os.mkdir(parameters["calculator"]["optimize"])
    os.chdir(parameters["calculator"]["optimize"])
    output = Output(os.path.join(os.getcwd(), "report_{}.out".format(parameters["calculator"]["optimize"])))
    dirs.find_last_dir(parameters)
    known.analyze_calculated(structure, fixed_frame, parameters)
    # output.write_parameters(parameters, structure, known, dirs)
    # for f in glob.glob("/tmp/ipi_*"):
    #     if os.path.exists(os.path.join("/tmp/", f)):
    #         os.remove(os.path.join("/tmp/", f))
    
    structure.mu = np.abs(calculator.estimate_mu(structure, fixed_frame, parameters))
    # Finish unfinished calculation
    calculator.finish_relaxation(structure, fixed_frame, parameters, dirs.current_dir(parameters))
    workflow.success = dirs.dir_num
    generated_dirs = [z for z in os.listdir(dirs.generate_folder) if os.path.isdir(os.path.join(dirs.generate_folder, z))]

    # while len(generated_dirs)>0:
    #     output.write_to_report("\nThere are {} candidate structures to relax\n".format(len(generated_dirs)))
    #     for generated in sorted(generated_dirs):
    #         d = os.path.join(os.getcwd(), "generate", generated)
    #         output.write_to_report("\nTaking structure from folder {}\n".format(d))
    #         gen = os.path.join(d, generated+".in")
    #         configuration = structure.read_configuration(structure, fixed_frame, gen)
    #         found = known.find_in_known(structure.torsions_from_conf(configuration), criteria="all", t=10)
    #         if not found:
    #             dirs.create_directory(parameters)
    #             dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
    #             calculator.relax(structure, fixed_frame, parameters, dirs.current_dir(parameters), known)
    #             dirs.finished(parameters)
    #             known.check_calculated(dirs, parameters)
    #             known.add_to_known_traj(structure, fixed_frame, dirs.current_dir(parameters))
    #             t = known.find_traj(os.path.join(dirs.current_dir(parameters)))
    #             conf = measure_torsion_of_last(Trajectory(os.path.join(dirs.current_dir(parameters), t))[-1], structure.list_of_torsions)
    #             output.write_to_report("found in known {}".format(known.find_in_known(conf, criteria="all", t=10)))
    #             workflow.success += 1
    #             workflow.trials = 0
    #             output.write_successfull_relax(parameters, structure, known, dirs)
    #             shutil.rmtree(d)
    #             output.write_to_report("\nGenerated structure in folder {} is deleted\n".format(d))
    #             generated_dirs = os.listdir(dirs.generate_folder)
    #             output.write_to_report("\nThere are {} candidate structures left to relax\n".format(len(generated_dirs)))
    #         else:
    #             shutil.rmtree(d)
    #             generated_dirs = os.listdir(dirs.generate_folder)
    #             output.write_to_report("\nStructure in folder {} is already in known. Delete.\n".format(d))
    #             output.write_to_report("\nThere are {} candidate structures left to relax\n".format(len(generated_dirs)))
    # # when run out structures 
    # output.write_to_report("All the structures in \"generate\" folder are calculated.")
    # output.write_to_report("Continue to generate and search.\n")
    # known.criteria = "all"

    # Number of trials to generate new structure
    # before decreasing of the criteria 
    while workflow.trials < parameters["trials"]:
        # Requsted number of relaxations to perform
        while workflow.success < parameters["success"]:
            # Check for new structures in generated folder
            print("Check for new structures in generated folder")
            generated_dirs = [z for z in os.listdir(dirs.generate_folder) if os.path.isdir(os.path.join(dirs.generate_folder, z))]
            if len(generated_dirs)>0:
                # generated_dirs = [z for z in os.listdir(dirs.generate_folder) if os.path.isdir(os.path.join(dirs.generate_folder, z))]
                output.write_to_report("\nThere are {} candidate structures to relax\n".format(len(generated_dirs)))
                try:
                    generated_dirs = [z for z in os.listdir(dirs.generate_folder) if os.path.isdir(os.path.join(dirs.generate_folder, z))]
                    d = os.path.join(dirs.generate_folder, sorted(generated_dirs)[0])
                    gen = os.path.join(d, sorted(generated_dirs)[0]+".in")
                    print(gen)
                    configuration = structure.read_configuration(structure, fixed_frame, gen)
                    shutil.rmtree(d)
                except:
                    output.write_to_report("\nSomething went wrong with folder\n")
                    output.write_to_report(d)
                    continue
                known.update_known(known.names, os.listdir(known.dir), structure, fixed_frame, parameters)
                # print("\n\n\n", len(known.torsions), "\n\n\n")
                current_coords = merge_together(structure, fixed_frame)
                found = known.find_in_known(current_coords, parameters, structure, fixed_frame,
                                                criteria=known.criteria, t=known.torsional_diff_degree)
                if not found:
                    output.write_to_report("\nTaking structure from folder {}\n".format(d))
                    dirs.create_directory(parameters)
                    structure.apply_torsions(configuration)
                    dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
                    calculator.relax(structure, fixed_frame, parameters, dirs.current_dir(parameters), known)                           
                    t = known.find_traj(os.path.join(dirs.current_dir(parameters)))
                    conf = measure_torsion_of_last(Trajectory(os.path.join(dirs.current_dir(parameters), t))[-1], structure.list_of_torsions)
                    current_coords = merge_together(structure, fixed_frame)
                    found = known.find_in_known(current_coords, parameters, structure, fixed_frame,
                                                criteria=known.criteria, t=known.torsional_diff_degree)
                    if not found:
                        dirs.finished(parameters)
                        known.send_traj_to_known_folder(dirs, parameters)
                        workflow.success += 1
                        workflow.trials = 0
                        output.write_successfull_relax(parameters, structure, known, dirs)
                        output.write_to_report("\nGenerated structure in folder {} is deleted\n".format(d))
                        # output.write_to_report("\nThere are {} candidate structures left to relax\n".format(len(generated_dirs)))
                    else:
                        output.write_to_report("found in known {}".format(ff))
                        dirs.known(parameters)
                        # known.send_traj_to_known_folder(dirs, parameters)

                else:
                    output.write_to_report("\nStructure in folder {} is already in known. Delete.\n".format(d))
                    output.write_to_report("\nThere are {} candidate structures left to relax\n".format(len(generated_dirs)))


                    # generated_dirs = os.listdir(dirs.generate_folder)
            # when run out structures 
            # output.write_to_report("All the structures in \"generate\" folder are calculated.")
            else:
                output.write_to_report("Continue to generate and search.\n")
                known.criteria = "all"
            # output.write("Start the new Trial {}\n".format(workflow.trials))


            # No structures in generated folder
            print("No new structures in generated folder")
            # Generate the vector in internal degrees of freedom
            configuration = structure.create_configuration(parameters)
            # Apply the configuration to the template molecule
            structure.apply_configuration(configuration)
            # Check if the structure is sencible and don't have clashes
            if all_right(structure, fixed_frame):
                # Check if there are new structures appear in known folder
                known.update_known(known.names, os.listdir(known.dir), structure, fixed_frame, parameters)
                # Two structures are considered similar if all their corresponding torsional angles
                # are different for less than 5 dgrees. 
                current_coords = merge_together(structure, fixed_frame)
                found = known.find_in_known(current_coords, parameters, structure, fixed_frame,
                                                criteria=known.criteria, t=known.torsional_diff_degree)
                if not found:
                    print("Structure is unique")
                    structure.apply_configuration(configuration) # For some reason need to apply again here
                    # Create new directory for outputs
                    dirs.create_directory(parameters)
                    # Save the initial configuration in the folder
                    # This will be used for restart of the calculation in the case of error
                    # if no optimization steps were performed
                    dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
                    # Perform relaxation
                    calculator.relax(structure, fixed_frame, parameters, dirs.current_dir(parameters), known)
                    # Find the final trajectory
                    t = known.find_traj(os.path.join(dirs.current_dir(parameters)))
                    # take final local minima
                    conf = measure_torsion_of_last(Trajectory(os.path.join(dirs.current_dir(parameters), t))[-1], structure.list_of_torsions)
                    # Check if it was among known structures before this relaxation
                    current_coords = merge_together(structure, fixed_frame)
                    was_known = known.find_in_known(current_coords, parameters, structure, fixed_frame,
                                                criteria=known.criteria, t=known.torsional_diff_degree)
                    # Send all the steps of the trajectory to known folder
                    known.send_traj_to_known_folder(dirs, parameters)
                    print("\n\n\n\n\n", was_known)
                    if not was_known:
                        dirs.finished(parameters)
                        workflow.success += 1
                        workflow.trials = 0
                        output.write_successfull_relax(parameters, structure, known, dirs)
                    else:
                        output.write_to_report("found in known {}".format(was_known))
                        dirs.known(parameters)                                
                        workflow.success += 1
                        workflow.trials = 0
                else:
                    workflow.trials += 1 
                    if workflow.trials == parameters["trials"]:
                        if known.torsional_diff_degree > 10:
                            known.torsional_diff_degree -= 5
                            output.write_to_report("\nDecreasing the criteria for torsional angles to {}\n".format(known.torsional_diff_degree))
                            workflow.trials = 0
                            pass
                        else:
                           output.write_to_report("Cannot find new structures\n")
                           sys.exit(0)
                    else:
                        pass
            else:
                # Save not sensible structure for evaluation
                write("bad_luck.xyz", merge_together(structure, fixed_frame), format="xyz")
                pass
        else:
            output.write_to_report("{} number of structures was successfully relaxed!".format(parameters["success"]))
            output.write_to_report("Terminating algorithm")
            sys.exit(0)


if parameters["calculator"]["optimize"] == "generate":
    # Generates unique structures
    dirs = Directories(parameters)
    output = Output("report_generate.out")
    workflow = Workflow()
    structure = Structure(parameters)
    fixed_frame = Fixed_frame(parameters)
    calculator = Calculator(parameters)
    known = Known(structure, parameters)
    os.chdir(parameters["calculator"]["optimize"])
    dirs.find_last_dir(parameters)
    known.check_calculated(dirs, parameters)
    known.analyze_calculated(structure, fixed_frame, parameters)
    dirs.find_last_generated_dir(parameters)
    output.write_parameters(parameters, structure, known, dirs)
    calculated_dir = os.path.join(os.getcwd(), "search") 
    # snapshots = len(os.listdir(calculated_dir))
    print("Initialize")
    workflow.success = dirs.dir_num
    print(workflow.success)

    from ase.constraints import FixAtoms
    while workflow.trials < parameters["trials"]:
        while workflow.success < parameters["success"]:
            # Generate the vector in internal degrees of freedom
            configuration = structure.create_configuration(parameters)
            print(configuration)
            structure.apply_configuration(configuration)
            if all_right(structure, fixed_frame):
                known.update_known(known.names, os.listdir(known.dir), structure, fixed_frame, parameters)
                print("Structure is alright")
                print("Looking if it is known structure")
                current_coords = merge_together(structure, fixed_frame)

                found = known.find_in_known(current_coords,
                                            parameters,
                                            structure, 
                                            fixed_frame,
                                            criteria=known.criteria, 
                                            t=known.torsional_diff_degree)

                if not found:
                    calculator.set_constrains(current_coords, parameters) 
                    dirs.create_directory(parameters)
                    dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
                    t, o, c = known.get_internal_vector(current_coords, structure, fixed_frame, parameters)
                    known.add_to_known(t, o, c)
                    workflow.success += 1
                    workflow.trials = 0
                    output.write_successfull_generate(parameters, structure.torsions_from_conf(configuration), dirs)
                else:
                    workflow.trials += 1 
                    print("Trial {}".format(workflow.trials))
                    if workflow.trials == parameters["trials"]:
                        if known.torsional_diff_degree > 10:
                            known.torsional_diff_degree -= 5
                            output.write_to_report("\nDecreasing the criteria for torsional angles to {}\n".format(known.torsional_diff_degree))
                            workflow.trials = 0
                            pass
                        else:
                           output.write_to_report("Initialization is finished\n")
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
    sys.exit(0)


if "single" in parameters["calculator"]["optimize"]:
    # Relaxes all the structures in the generate folder
    if len(os.listdir(parameters["calculator"]["generate_folder"])) > 0:
        dirs = Directories(parameters)   
        workflow = Workflow()
        gendir = os.path.join(os.getcwd(), parameters["calculator"]["generate_folder"])
        # known = Known(structure, parameters)
        basedir = os.getcwd()
        if not os.path.exists(parameters["calculator"]["optimize"]):
            os.mkdir(parameters["calculator"]["optimize"])
        os.chdir(parameters["calculator"]["optimize"])
        for i in sorted(os.listdir(gendir)):
            parameters["geometry"][0] = os.path.join(gendir, i, i+".in")
            parameters["geometry"][1] = "aims"
            parameters["fixed_frame"]["filename"] = os.path.join(basedir, "slab.in")
            parameters["fixed_frame"]["format"] = "aims"
            parameters["calculator"]["ase_parameters_file"] = os.path.join(basedir, "supporting", "ase_command.py")
            print(parameters["geometry"][0])
            
            structure = Structure(parameters)
            fixed_frame = Fixed_frame(parameters)
            calculator = Calculator(parameters)
            known = Known(structure, parameters)
            # if not os.path.exists(parameters["calculator"]["optimize"]):
                # os.mkdir(parameters["calculator"]["optimize"])
            # output = Output(os.path.join(os.getcwd(), "report_{}.out".format(parameters["calculator"]["optimize"])))
            dirs.find_last_dir(parameters)
            # output.write_parameters(parameters, structure, known, dirs)
            # Estimate parameter mu if needed:
            structure.mu = np.abs(calculator.estimate_mu(structure, fixed_frame, parameters))
            dirs.create_directory(parameters)
            dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
            calculator.relax(structure, fixed_frame, parameters, dirs.current_dir(parameters), known)                           
            dirs.finished(parameters)



    else:
        # Performs relaxation of the structure specified in the parameters file
        dirs = Directories(parameters)   
        workflow = Workflow()
        structure = Structure(parameters)
        fixed_frame = Fixed_frame(parameters)
        calculator = Calculator(parameters)
        known = Known(structure, parameters)
        if not os.path.exists(parameters["calculator"]["optimize"]):
            os.mkdir(parameters["calculator"]["optimize"])
        os.chdir(parameters["calculator"]["optimize"])
        # output = Output(os.path.join(os.getcwd(), "report_{}.out".format(parameters["calculator"]["optimize"])))
        dirs.find_last_dir(parameters)
        # output.write_parameters(parameters, structure, known, dirs)
        # Estimate parameter mu if needed:
        structure.mu = np.abs(calculator.estimate_mu(structure, fixed_frame, parameters))
        dirs.create_directory(parameters)
        dirs.save_to_directory(merge_together(structure, fixed_frame), parameters)
        calculator.relax(structure, fixed_frame, parameters, dirs.current_dir(parameters), known)                           
        dirs.finished(parameters)

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

