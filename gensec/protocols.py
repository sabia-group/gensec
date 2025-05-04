"""Create different search protocols 
"""

import ase.db
import os
import sys
import numpy as np
from ase.io import read, write
from gensec.structure import Structure, Fixed_frame
from gensec.modules import all_right, merge_together, measure_quaternion, run_with_timeout_decorator, return_1000
from gensec.outputs import Directories
from gensec.relaxation import Calculator
from gensec.check_input import Check_input
from gensec.supercell_finder import Supercell_finder
from ase.io.trajectory import Trajectory

from gensec.unit_cell_finder import Unit_cell_finder, gen_base_sheet

# TODO: Add checks 'if '...' in self.parameters' to avoid errors, includes adding default values. Exceptions are for example input files but this also needs a clear error message.

# TODO: Add default values to parameters if not present and safe at the end

class Protocol:

    """Summary

    Attributes:
        success (TYPE): Description
        trials (int): Description
    """

    def init(self):
        """Summary

        Args:
            parameters (TYPE): Description
        """
        pass

    def run(self, parameters):
        """Summary

        Args:
            parameters (TYPE): Description
        """
        parameters = Check_input(parameters)
        
        if parameters["protocol"]["generate"]["activate"] is True:
            # connect to the database and start creating structures there
            print("Start generating of the structures")
            if not os.path.exists("db_generated.db"):
                db_generated = open("db_generated.db", "w")
            if os.path.exists("db_generated.db-journal"):
                os.remove("db_generated.db-journal")
            if os.path.exists("db_generated.lock"):
                os.remove("db_generated.lock")

            db_generated = ase.db.connect("db_generated.db")
            
            if not os.path.exists("db_generated_frames.db") and parameters["fixed_frame"]["is_unit_cell"]:
                db_generated_frames = open("db_generated_frames.db", "w")
            if os.path.exists("db_generated_frames.db-journal"):
                os.remove("db_generated_frames.db-journal")
            if os.path.exists("db_generated_frames.lock"):
                os.remove("db_generated_frames.lock")
            
            db_generated_frames = ase.db.connect("db_generated_frames.db")

            if not os.path.exists("db_relaxed.db"):
                db_relaxed = open("db_relaxed.db", "w")
            if os.path.exists("db_relaxed.db-journal"):
                os.remove("db_relaxed.db-journal")
            if os.path.exists("db_relaxed.lock"):
                os.remove("db_relaxed.lock")

            db_relaxed = ase.db.connect("db_relaxed.db")

            if not os.path.exists("db_trajectories.db"):
                db_trajectories = open("db_trajectories.db", "w")
            if os.path.exists("db_trajectories.db-journal"):
                os.remove("db_trajectories.db-journal")
            if os.path.exists("db_trajectories.db.lock"):
                os.remove("db_trajectories.db.lock")

            db_trajectories = ase.db.connect("db_trajectories.db")

            if not os.path.exists("db_generated_visual.db"):
                db_generated_visual = open("db_generated_visual.db", "w")
            # if os.path.exists("db_generated.db-journal"):
            # os.remove("db_generated.db-journal")
            # if os.path.exists("db_generated.lock"):
            # os.remove("db_generated.lock")

            db_generated_visual = ase.db.connect("db_generated_visual.db")

            self.trials = 0
            self.success = db_generated.count()
            print("Generated structures", db_generated.count())
            
            # Find optimal supercell here
            # Adjust mic, coms and number of atoms accordingly
            # Need to eddit coms to give fixed values on top
            # TODO: Implement a checkpoint for the supercell finder so if there is already a database we can use the same supercell
            # TODO: Does still need to be here? Makes things more complicated.But still interesting for cases where you see the cells on images and dont know the exact orientation.
            if parameters["supercell_finder"]["activate"] and parameters["supercell_finder"]["unit_cell_method"] == "inputfile":
                # TODO: what part of parameters should be overwritten? Should probably do this in check_input already. Thinking of mic and coms.
                supercell_finder = Supercell_finder(parameters)
                supercell_finder.run()
                structure = Structure(parameters, supercell_finder)
                fixed_frame = Fixed_frame(parameters, supercell_finder.S_atoms)
                fixed_frame_sheet = Fixed_frame(parameters, supercell_finder.S_atoms)
            
            elif parameters["supercell_finder"]["activate"]:
                structure = Structure(parameters)
                fixed_frame = Fixed_frame(parameters)
                base_sheet = gen_base_sheet(structure.atoms, fixed_frame.fixed_frame, num_mol = parameters["number_of_replicas"])
                fixed_frame_sheet = Fixed_frame(parameters, base_sheet)
                supercell_finder = Supercell_finder(parameters, set_unit_cells = False)
                
            else:
                structure = Structure(parameters)
                fixed_frame = Fixed_frame(parameters)
                
                # Check if we have a base sheet we want to work with or if we need to create one from a unit cell. Only supposed to be used to chcek for clashes with the configured structure.
                if parameters["fixed_frame"]["activate"]:
                    if parameters["fixed_frame"]["is_unit_cell"]:
                        fixed_frame_sheet = Fixed_frame(parameters)
                    else:                        
                        base_sheet = gen_base_sheet(structure.atoms, fixed_frame.fixed_frame, num_mol = parameters["number_of_replicas"])
                        fixed_frame_sheet = Fixed_frame(parameters, base_sheet)
            dirs = Directories(parameters)
            if parameters["configuration"]["check_forces"]["activate"]:
                calculator = Calculator(parameters)
            while self.success < parameters["success"] and self.trials < parameters["trials"]:
                print(self.trials, self.success)
                # Generate the vector in internal degrees of freedom
                _, conf = structure.create_configuration(parameters)
                # print(conf)
                # Apply the configuration to structure
                structure.apply_conf(conf)
                if parameters["supercell_finder"]["activate"] and parameters["supercell_finder"]["unit_cell_method"] == "find":
                    oriented_mol = structure.atoms_object()
                # Check if that structure is sensible
                is_good = True
                if all_right(structure, fixed_frame_sheet):
                    # Check if it is in database
                    in_db = False
                    if parameters["protocol"]["check_db"]:
                        in_db = True
                        if not structure.find_in_database(conf, db_generated, parameters):
                            if not structure.find_in_database(conf, db_relaxed, parameters ):
                                if not structure.find_in_database(conf, db_trajectories, parameters):
                                    in_db = False

                    if not in_db:
                        if parameters["supercell_finder"]["activate"] and parameters["supercell_finder"]["unit_cell_method"] == "find":
                            oriented_mol_with_cell, _, _ = Unit_cell_finder(oriented_mol, parameters = parameters)
                            supercell_finder.set_unit_cell('find', oriented_mol_with_cell)    # TODO: Input parameters (dont restrict to standard ones in definition of the function) and add all to check input 
                            supercell_finder.run()
                            conf = structure.get_configuration(supercell_finder.F_atoms)
                            fixed_frame_temp = Fixed_frame(parameters, supercell_finder.S_atoms)
                            structure_temp = Structure(parameters, supercell_finder)
                            if all_right(structure_temp, fixed_frame_temp):
                                if parameters["configuration"]["check_forces"]["activate"] == True:
                                    supercell_finder.joined_atoms.calc = calculator.calculator
                                    if not np.max(np.abs(run_with_timeout_decorator(supercell_finder.joined_atoms.get_forces, return_1000, 
                                                                                    timeout = parameters["configuration"]["check_forces"]["max_time"]))) > parameters["configuration"]["check_forces"]["max_force"]:
                                        db_generated.write(supercell_finder.F_atoms, **conf)
                                        db_generated_frames.write(supercell_finder.S_atoms, **conf)
                                        db_generated_visual.write(supercell_finder.joined_atoms, **conf)
                                        write("good_luck.xyz",supercell_finder.joined_atoms,format="extxyz")
                                        
                                    else:
                                        print("Forces too large")
                                        is_good = False
                                        
                                else:
                                    db_generated.write(supercell_finder.F_atoms, **conf)
                                    db_generated_frames.write(supercell_finder.S_atoms, **conf)
                                    db_generated_visual.write(supercell_finder.joined_atoms, **conf)
                                    write("good_luck.xyz",supercell_finder.joined_atoms,format="extxyz")
                                    
                            else:
                                is_good = False
                                
                        else:
                            merged = merge_together(structure, fixed_frame)
                            if parameters["configuration"]["check_forces"]["activate"]:
                                merged.calc = calculator.calculator
                                if not np.max(np.abs(run_with_timeout_decorator(merged.get_forces, return_1000,
                                                                                timeout = parameters["configuration"]["check_forces"]["max_time"]))) > parameters["configuration"]["check_forces"]["max_force"]:
                                    db_generated.write(structure.atoms_object(), **conf)
                                    db_generated_visual.write(merged,**conf)
                                    write("good_luck.xyz",merged,format="extxyz")
                                    
                                else:
                                    print("Forces too large")
                                    is_good = False
                                    
                            else:
                                db_generated.write(structure.atoms_object(), **conf)
                                db_generated_visual.write(merged,**conf)
                                write("good_luck.xyz",merged,format="extxyz")
                                
                    else:
                        is_good = False
                            
                else:
                    is_good = False                        

                if is_good:
                    self.trials = 0
                    self.success = db_generated.count()
                    print("Good", conf)
                    print("Generated structures:", self.success)        
                else:
                    #print("BAD", conf)
                    if parameters["supercell_finder"]["activate"] and hasattr(supercell_finder, "joined_atoms"):
                        write("bad_luck.xyz",supercell_finder.joined_atoms,format="extxyz")
                    else:
                        write("bad_luck.xyz",merge_together(structure, fixed_frame_sheet),format="extxyz")
                    # print("Trials made", self.trials)
                    self.trials += 1
            else:
                sys.exit(0)
                #pass
        

        if parameters["protocol"]["search"]["activate"] is True:
            
            #TODO: Raise error here if db is empty
            
            # connect to the database and start creating structures there
            print("Start relaxing structures")
            # Create database file or connect to existing one,
            # unlock them, if they are locked
            if not os.path.exists("db_generated.db"):
                db_generated = open("db_generated.db", "w")
            if os.path.exists("db_generated.db-journal"):
                os.remove("db_generated.db-journal")
            if os.path.exists("db_generated.lock"):
                os.remove("db_generated.lock")

            db_generated = ase.db.connect("db_generated.db")

            if not os.path.exists("db_relaxed.db"):
                db_relaxed = open("db_relaxed.db", "w")
            if os.path.exists("db_relaxed.db-journal"):
                os.remove("db_relaxed.db-journal")
            if os.path.exists("db_relaxed.lock"):
                os.remove("db_relaxed.lock")

            db_relaxed = ase.db.connect("db_relaxed.db")

            if not os.path.exists("db_trajectories.db"):
                db_trajectories = open("db_trajectories.db", "w")
            if os.path.exists("db_trajectories.db-journal"):
                os.remove("db_trajectories.db-journal")
            if os.path.exists("db_trajectories.db.lock"):
                os.remove("db_trajectories.db.lock")
            if not os.path.exists("db_generated_visual.db"):
                db_generated_visual = open("db_generated_visual.db", "w")
            # if os.path.exists("db_generated.db-journal"):
            # os.remove("db_generated.db-journal")
            # if os.path.exists("db_generated.lock"):
            # os.remove("db_generated.lock")

            db_generated_visual = ase.db.connect("db_generated_visual.db")



            db_trajectories = ase.db.connect("db_trajectories.db")

            name = parameters["name"]

            self.success = db_relaxed.count()
            print("Relaxed structures", db_relaxed.count())
            structure = Structure(parameters)
            fixed_frame = Fixed_frame(parameters)
            dirs = Directories(parameters)
            calculator = Calculator(parameters)
            conf_keys = structure.extract_conf_keys_from_row()
            if not os.path.exists(parameters["protocol"]["search"]["folder"]):
                os.mkdir(parameters["protocol"]["search"]["folder"])
            # Perform optimizations in the folder specified in parameters file
            print("Changing Directory!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            os.chdir(parameters["protocol"]["search"]["folder"])

            # Finish unfinished calculations
            calculator.finish_relaxation(
                structure, fixed_frame, parameters, calculator
            )

            while self.success < parameters["success"]:
                self.success = db_relaxed.count()
                # Take structure from database of generated structures
                # TODO: The generation part is not needed here. Should be deleted. If there is no structure just throw an error.
                if db_generated.count() == 0:
                    raise ValueError("No structures in the database. Please generate some first.")
                    
                    #self.trials = 0
                    #while self.trials < parameters["trials"]:
                    #    configuration, conf = structure.create_configuration(
                    #        parameters
                    #    )
                    #    # Apply the configuration to structure
                    #    structure.apply_conf(conf)
                    #    # Check if that structure is sensible
                    #    if parameters["protocol"]["check_db"]:
                    #        if all_right(structure, fixed_frame):
                    #            # Check if it is in database
                    #            if not structure.find_in_database(
                    #                conf, db_relaxed, parameters
                    #            ):
                    #                if not structure.find_in_database(
                    #                    conf, db_trajectories, parameters
                    #                ):
                    #                    #if hasattr(self, "fixed_frame"):
                    #                    db_generated_visual.write(
                    #                        structure.atoms_object_visual(
                    #                            fixed_frame
                    #                        ),
                    #                        **conf
                    #                    )
                    #                    print("Structure added to generated")
                    #                    break
                    #                else:
                    #                    self.trials += 1
                    #                    print("Found in database")
#
                    #            else:
                    #                self.trials += 1
                    #                print("Found in database")
                    #        else:
                    #            self.trials += 1
                    #            print("Trials made", self.trials)
                    #    else:
                    #        db_generated.write(structure.atoms_object(), **conf)
#
                    #        #if hasattr(self, "fixed_frame"):
                    #        db_generated_visual.write(
                    #            structure.atoms_object_visual(fixed_frame),
                    #            **conf
                    #        )
                    #        self.trials = 0
                    #        self.success = db_generated.count()
                else:
                    for row in db_generated_visual.select():
                        traj_id = row.unique_id
                        # Extract the configuration from the row
                        #conf = {key: row[key] for key in conf_keys}
                        print("added line")
                        print(row.key_value_pairs)
                        print("added line")
                        #structure.apply_conf(conf)
                        dirs.dir_num = row.id
                        del db_generated[row.id]
                        if parameters["protocol"]["check_db"]:
                            if structure.find_in_database(conf, db_relaxed, parameters) or structure.find_in_database(conf, db_trajectories, parameters):
                                print("Found in database")
                                continue
                    
                        print("This is row ID that is taken for calculation",row.id,)
                        row_atoms = row.toatoms().copy()
                        dirs.create_directory(parameters)
                        dirs.save_to_directory(row_atoms,parameters)
                        calculator.simple_relax(row_atoms, parameters, dirs.current_dir(parameters))           #legacy: calculator.relax(structure, fixed_frame,parameters,dirs.current_dir(parameters))
                        
                        calculator.finished(dirs.current_dir(parameters))
                        # Find the final trajectory
                        traj = Trajectory(
                            os.path.join(
                                dirs.current_dir(parameters),
                                "trajectory_{}.traj".format(name)
                            )
                        )
                        print("Structure relaxed")
                        for step in traj:
                            full_conf = structure.get_configuration(step)
                            db_trajectories.write(
                                step, **full_conf, trajectory=traj_id
                            )
                        full_conf = structure.get_configuration(traj[-1])
                        db_relaxed.write(
                            traj[-1], **full_conf, trajectory=traj_id
                        )
                        self.success = db_relaxed.count()
                        #calculator.close()
                        #break
