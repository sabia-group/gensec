import os
import json
from ase.io import read, write


class Directories:

    def __init__(self):

        if not os.path.exists("generate"):
            os.mkdir("generate")
        else:
            pass

        if not os.path.exists("search"):
            os.mkdir("search")
        else:
            pass

        if not os.path.exists("current"):
            os.mkdir("current")
        else:
            pass

        self.dir_num = 0

    def create_directory(self):
        self.dir_num+=1
        dir = os.path.join(os.getcwd(), "generate", format(self.dir_num, "010d"))
        if not os.path.exists(dir):
            os.mkdir(dir)
        else:
            pass

    def remove_last_directory(self):       
        dir = os.path.join(os.getcwd(), "generate", format(self.dir_num, "010d"))
        if os.path.exists(dir):
            os.rmdir(dir)
        else:
            pass
        self.dir_num-=1

    def save_to_directory(self, ensemble, parameters):       
        dir = os.path.join(os.getcwd(), "generate", format(self.dir_num, "010d"))
        write(os.path.join(dir, "{:010d}.in".format(self.dir_num)), ensemble, format="aims")

    def finished(self):
        dir = self.current_dir()
        f = open(os.path.join(dir, "finished"), "w")
        f.write("Calculation was finished")
        f.close()

    def find_last_dir(self):
        dirs = os.path.join(os.getcwd(), "generate")
        if len(os.listdir(dirs))>0:
            last_dir = [int(i) for i in os.listdir(dirs)
                                if "finished" in os.listdir(os.path.join(dirs, i))]
            if len(last_dir) > 0:
                self.dir_num = max(last_dir)
            else:
                self.dir_num = 0
        else:
            self.dir_num = 0
        
    # def check_for_restart(self):


    def current_dir(self):
        dir = os.path.join(os.getcwd(), "generate", format(self.dir_num, "010d"))
        return dir

class Output:

    def __init__(self, report_file):

        if not os.path.exists(report_file):
            report = open(report_file, "w")
            report.write("#    Copyright 2020 Dmitrii Maksimov\n")
            report.close()
        else:
            pass

    def write(self, text):
        report = open("report.out", "a")
        report.write(text)
        report.write("\n")
        report.close()



def load_parameters(parameters_file):
    if not os.path.exists(parameters_file):
        print("No parameter file, generate default one?")
        # generate default parameter file
    else:
        with open(parameters_file) as f:
            return json.load(f)

class Workflow:

    def __init__(self):
        self.trials = 0
        self.success = 0
