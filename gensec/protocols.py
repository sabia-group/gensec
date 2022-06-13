"""Create different search protocols 
"""

import ase.db
import os
import sys
from ase.io import read, write
from gensec.structure import Structure, Fixed_frame
from gensec.modules import all_right, merge_together, measure_quaternion
from gensec.outputs import Directories
from gensec.relaxation import Calculator
from ase.io.trajectory import Trajectory


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
            structure = Structure(parameters)
            fixed_frame = Fixed_frame(parameters)
            dirs = Directories(parameters)

            while self.trials < parameters["trials"]:
                while self.success < parameters["success"]:
                    print(self.trials, self.success)
                    # Generate the vector in internal degrees of freedom
                    configuration, conf = structure.create_configuration(
                        parameters
                    )
                    # print(conf)
                    # Apply the configuration to structure
                    structure.apply_conf(conf)
                    # Check if that structure is sensible
                    if all_right(structure, fixed_frame):
                        # Check if it is in database
                        if parameters["configuration"]["torsions"]["activate"]:

                            if not structure.find_in_database(
                                conf, db_generated, parameters
                            ):
                                if not structure.find_in_database(
                                    conf, db_relaxed, parameters
                                ):
                                    if not structure.find_in_database(
                                        conf, db_trajectories, parameters
                                    ):
                                        db_generated.write(
                                            structure.atoms_object(), **conf
                                        )

                                        self.trials = 0
                                        self.success = db_generated.count()

                                        print("Good", conf)
                                        write(
                                            "good_luck.xyz",
                                            merge_together(
                                                structure, fixed_frame
                                            ),
                                            format="xyz",
                                        )
                                        print(
                                            "Generated structures", self.success
                                        )

                        else:
                            db_generated.write(structure.atoms_object(), **conf)
                            db_generated_visual.write(
                                structure.atoms_object_visual(fixed_frame),
                                **conf
                            )
                            self.trials = 0
                            self.success = db_generated.count()

                            print("Good", conf)
                            write(
                                "good_luck.xyz",
                                merge_together(structure, fixed_frame),
                                format="xyz",
                            )
                            print("Generated structures", self.success)
                    else:
                        print("BAD", conf)
                        write(
                            "bad_luck.xyz",
                            merge_together(structure, fixed_frame),
                            format="xyz",
                        )
                        print("Trials made", self.trials)
                        self.trials += 1
                else:
                    sys.exit(0)
            else:
                sys.exit(0)

        if parameters["protocol"]["search"]["activate"] is True:
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
            os.chdir(parameters["protocol"]["search"]["folder"])

            # Finish unfinished calculations
            calculator.finish_relaxation(
                structure, fixed_frame, parameters, calculator
            )

            while self.success < parameters["success"]:
                self.success = db_relaxed.count()
                # Take structure from database of generated structures
                if db_generated.count() == 0:
                    self.trials = 0
                    while self.trials < parameters["trials"]:
                        configuration, conf = structure.create_configuration(
                            parameters
                        )
                        # Apply the configuration to structure
                        structure.apply_conf(conf)
                        # Check if that structure is sensible
                        if parameters["configuration"]["torsions"]["activate"]:
                            if all_right(structure, fixed_frame):
                                # Check if it is in database
                                if not structure.find_in_database(
                                    conf, db_relaxed, parameters
                                ):
                                    if not structure.find_in_database(
                                        conf, db_trajectories, parameters
                                    ):
                                        db_generated_visual.write(
                                            structure.atoms_object_visual(
                                                fixed_frame
                                            ),
                                            **conf
                                        )
                                        print("Structure added to generated")
                                        break
                                    else:
                                        self.trials += 1
                                        print("Found in database")

                                else:
                                    self.trials += 1
                                    print("Found in database")
                            else:
                                self.trials += 1
                                print("Trials made", self.trials)
                        else:
                            db_generated.write(structure.atoms_object(), **conf)
                            db_generated_visual.write(
                                structure.atoms_object_visual(fixed_frame),
                                **conf
                            )
                            self.trials = 0
                            self.success = db_generated.count()
                else:
                    for row in db_generated.select():
                        traj_id = row.unique_id
                        # Extract the configuration from the row
                        conf = {key: row[key] for key in conf_keys}
                        structure.apply_conf(conf)
                        dirs.dir_num = row.id
                        del db_generated[row.id]
                        if parameters["configuration"]["torsions"]["activate"]:
                            if not structure.find_in_database(
                                conf, db_relaxed, parameters
                            ):
                                if not structure.find_in_database(
                                    conf, db_trajectories, parameters
                                ):
                                    print(
                                        "This is row ID that is taken for calculation",
                                        row.id,
                                    )
                                    dirs.create_directory(parameters)
                                    dirs.save_to_directory(
                                        merge_together(structure, fixed_frame),
                                        parameters,
                                    )
                                    calculator.relax(
                                        structure,
                                        fixed_frame,
                                        parameters,
                                        dirs.current_dir(parameters),
                                    )
                                    calculator.finished(
                                        dirs.current_dir(parameters)
                                    )
                                    # Find the final trajectory
                                    traj = Trajectory(
                                        os.path.join(
                                            dirs.current_dir(parameters),
                                            "trajectory_{}.traj".format(name),
                                        )
                                    )
                                    print("Structure relaxed")
                                    for step in traj:
                                        full_conf = (
                                            structure.read_configuration(step)
                                        )
                                        db_trajectories.write(
                                            step,
                                            **full_conf,
                                            trajectory=traj_id
                                        )
                                    full_conf = structure.read_configuration(
                                        traj[-1]
                                    )
                                    db_relaxed.write(
                                        traj[-1],
                                        **full_conf,
                                        trajectory=traj_id
                                    )
                                    self.success = db_relaxed.count()
                                    break
                                else:
                                    print("Found in database")
                                    break
                            else:
                                print("Found in database")
                                break
                        else:
                            print(
                                "This is row ID that is taken for calculation",
                                row.id,
                            )
                            dirs.create_directory(parameters)
                            dirs.save_to_directory(
                                merge_together(structure, fixed_frame),
                                parameters,
                            )
                            calculator.relax(
                                structure,
                                fixed_frame,
                                parameters,
                                dirs.current_dir(parameters),
                            )
                            calculator.finished(dirs.current_dir(parameters))
                            # Find the final trajectory
                            traj = Trajectory(
                                os.path.join(
                                    dirs.current_dir(parameters),
                                    "trajectory_{}.traj".format(name),
                                )
                            )
                            print("Structure relaxed")
                            for step in traj:
                                full_conf = structure.read_configuration(step)
                                db_trajectories.write(
                                    step, **full_conf, trajectory=traj_id
                                )
                            full_conf = structure.read_configuration(traj[-1])
                            db_relaxed.write(
                                traj[-1], **full_conf, trajectory=traj_id
                            )
                            self.success = db_relaxed.count()
                            break
