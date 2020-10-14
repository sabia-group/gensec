import os
import json
from time import gmtime, strftime
from ase.io import read, write
from gensec.modules import measure_torsion_of_last
from ase.io.trajectory import Trajectory
import shutil


class Directories:

    def __init__(self, parameters):

        if not os.path.exists(parameters["calculator"]["generate_folder"]):
            os.mkdir(parameters["calculator"]["generate_folder"])
        else:
            pass

        if not os.path.exists(parameters["calculator"]["optimize"]):
            os.mkdir(parameters["calculator"]["optimize"])
        else:
            pass

        if not os.path.exists(parameters["calculator"]["blacklist_folder"]):
            os.mkdir(parameters["calculator"]["blacklist_folder"])
        else:
            pass

        self.dir_num = 0
        self.generate_folder = parameters["calculator"]["generate_folder"]
        self.blacklist_folder = parameters["calculator"]["blacklist_folder"]

    def create_directory(self, parameters):


        self.dir_num+=1
        dir = os.path.join(os.getcwd(), format(self.dir_num, "010d"))
        if not os.path.exists(dir):
            os.mkdir(dir)
        else:
            pass

    def remove_last_directory(self, parameters):       
        dir = os.path.join(os.getcwd(), format(self.dir_num, "010d"))
        if os.path.exists(dir):
            os.rmdir(dir)
        else:
            pass
        self.dir_num-=1

    def save_to_directory(self, ensemble, parameters):       
        dir = os.path.join(os.getcwd(), format(self.dir_num, "010d"))
        write(os.path.join(dir, "{:010d}.in".format(self.dir_num)), ensemble, format="aims")

    def finished(self, parameters):
        dir = self.current_dir(parameters)
        f = open(os.path.join(dir, "finished"), "w")
        f.write("Calculation was finished")
        f.close()

    def blacklisted(self, parameters):
        dir = self.current_dir(parameters)
        f = open(os.path.join(dir, "blacklisted"), "w")
        f.write("Calculation was terminated and blacklisted")
        f.close()


    def find_last_dir(self, parameters):

        def finished_dir(files, list_dir):
            return any(i in list_dir for i in files) 

        d = os.getcwd()
        dirs = list(filter(os.path.isdir, os.listdir(d)))
        if len(dirs)>0:
            last_dir = [int(i) for i in dirs
                                if finished_dir(["finished", "blacklisted"], os.listdir(os.path.join(d, i)))]
            if len(last_dir) > 0:
                self.dir_num = max(last_dir)
            else:
                self.dir_num = 0
            remove_dirs = [int(i) for i in dirs
                                if finished_dir(["finished", "blacklisted"], os.listdir(os.path.join(d, i)))]
        else:
            self.dir_num = 0
        
    def find_last_generated_dir(self, parameters):
        dirs = list(filter(os.path.isdir, os.listdir(self.generate_folder)))
        if len(dirs)>0:
            self.dir_num = max([int(i) for i in dirs])
        else:
            self.dir_num = 0

    def current_dir(self, parameters):
        dir = os.path.join(os.getcwd(), format(self.dir_num, "010d"))
        return dir






class Output:

    def __init__(self, report_file):

        self.report_file = report_file
        report = open(self.report_file, "w")
        report.write("#    Copyright 2020 Dmitrii Maksimov\n")
        t = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
        report.write("GenSec started {}\n\n\n".format(t))
        report.close()

    def write_to_report(self, text):
        report = open(self.report_file, "a")
        report.write(text)
        report.write("\n")
        report.close()

    def write_configuration(self, configuration):
        report = open("report.out", "a")
        report.write(text)
        report.write("\n")
        report.close()

    def write_parameters(self, parameters, structure, blacklist, dirs):
        report = open(self.report_file, "a")
        report.write("Name of the project is {}\n".format(parameters["name"]))
        report.write("If the unknown structure will not be found {} times in row the criteria for similarity between structures will be decreased.\n".format(parameters["trials"]))
        report.write("After {} structures will be relaxed the algorithm will stop.\n".format(parameters["success"]))
        report.write("{} replicas will be produced.\n".format(parameters["number_of_replicas"]))
        report.write("Reading template geometry from {}.\n".format(parameters["geometry"][0]))
        report.write("Number of atoms in template molecule is {}.\n".format(len(structure.molecules[0])))               
        if parameters["fixed_frame"]["activate"]:
            report.write("Reading fixed frame from {}.\n".format(parameters["fixed_frame"]["filename"]))
        else:
            report.write("Sampling is performed without fixed frame.\n")
        if parameters["mic"]["activate"]:
            report.write("Sampling is performed with periodic boundary conditions.\n")
        else:
            report.write("Sampling is performed without periodic boundary conditions.\n")
        if parameters["configuration"]["torsions"]["list_of_tosrions"] == "auto":
            report.write("Identified rotatable bonds are: {}.\n".format(structure.list_of_torsions))
        if parameters["configuration"]["torsions"]["activate"]:
            report.write("{} values of torsions will be sampled.\n".format(parameters["configuration"]["torsions"]["values"]))
        if parameters["configuration"]["orientations"]["activate"]:
            report.write("{} values of orientations will be sampled.\n".format(parameters["configuration"]["orientations"]["values"]))
        if parameters["configuration"]["coms"]["activate"]:
            report.write("{} values of centres of masses will be sampled.\n".format(parameters["configuration"]["coms"]["values"]))
        report.write("Folder with suporting files is in {}.\n".format(parameters["calculator"]["supporting_files_folder"]))
        report.write("ASE calculator is in {}.\n".format(parameters["calculator"]["ase_parameters_file"]))
        if parameters["calculator"]["optimize"] == "generate":
            report.write("GenSec wil generate structures in \"generate\" folder without relaxation\n")
        elif "search" in parameters["calculator"]["optimize"]:
            report.write("GenSec will generate structures in \"generate\" folder and relax them in {} folder\n".format(parameters["calculator"]["optimize"]))
            report.write("Relaxation will be perfomed until {} remaining forces are reached\n".format(parameters["calculator"]["fmax"]))
            report.write("On molecular part {} preconditioner of the Hessian matrix will be applied\n".format(parameters["calculator"]["preconditioner"]["mol"]))
            report.write("On fixed frame part {} preconditioner of the Hessian matrix will be applied\n".format(parameters["calculator"]["preconditioner"]["mol"]))
            report.write("Between molecular parts {} preconditioner of the Hessian matrix will be applied\n".format(parameters["calculator"]["preconditioner"]["mol"]))
            report.write("Between molecular part and fixed frame part {} preconditioner of the Hessian matrix will be applied\n".format(parameters["calculator"]["preconditioner"]["mol"]))
            report.write("After difference in RMSD during geometry optimization reaches {} - updating of the Hessian will be performed\n".format(parameters["calculator"]["preconditioner"]["rmsd_update"]))

            report.write("Blacklist contains {} snapshots\n".format(len(blacklist.blacklist)))
            report.write("Last calculated directory is {}\n".format(dirs.dir_num))

        report.write("{} structures Already searched.\n".format(dirs.dir_num))
        for struc in range(1, dirs.dir_num+1):
            report.write("Structure {} has torsional angles\n{}\n".format(struc, blacklist.blacklist[struc]))


        report.write("Continue the search.\n")            
        report.close()

    def write_successfull_generate(self, parameters, configuration, dirs):
        report = open(self.report_file, "a")
        dir = os.path.join(os.getcwd(), parameters["calculator"]["optimize"], format(dirs.dir_num, "010d"))
        report.write("Structure {} succsessfully generated and saved in \n{}\n".format(dirs.dir_num, dir))
        report.write("Structure {} has torsional angles configuration \n{}\n".format(dirs.dir_num, configuration))
        report.close()

    def write_successfull_relax(self, parameters, structure, blacklist, dirs):
        report = open(self.report_file, "a")
        dir = os.path.join(os.getcwd(), format(dirs.dir_num, "010d"))
        report.write("Structure {} succsessfully relaxed and saved in \n{}\n".format(dirs.dir_num, dir))
        tors = measure_torsion_of_last(Trajectory(os.path.join(dir ,blacklist.find_traj(dir)))[-1], structure.list_of_torsions)
        report.write("Local minima of structure {} has torsional angles configuration \n{}\n".format(dirs.dir_num, tors))
        report.close()

def load_parameters(parameters_file):
    try:
        with open(os.path.join(os.getcwd(), parameters_file)) as f:
            return json.load(f)
    except:
        print("No parameter file, generate default one?")
    # if not os.path.exists(parameters_file):
        
    #     generate default parameter file
    # else:
    #     with open(parameters_file) as f:
            

class Workflow:

    def __init__(self):
        self.trials = 0
        self.success = 0
