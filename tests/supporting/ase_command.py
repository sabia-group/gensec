import os
import sys
from ase.calculators.lammpslib import LAMMPSlib

dirname, filename = os.path.split(os.path.abspath(__file__))
airebo_file = os.path.join(dirname, "CH.airebo")
eam_file =  os.path.join(dirname, "Rh.lammps.modified.eam")


lammps_header=["dimension     3",
                "boundary      p p p",
                "atom_style    atomic",
                "units         metal",
                "neighbor      0.5  bin",
                "neigh_modify  delay  1"]

lammps_cmds = ["pair_style hybrid  airebo 3 1 1   eam   lj/cut 15.0",
                "pair_coeff  * *   airebo  {}  H  C NULL".format(airebo_file),
                "pair_coeff  3 3   eam     {}".format(eam_file),
                "pair_coeff  1 3   lj/cut  0.010 2.58",
                "pair_coeff  2 3   lj/cut  0.025 2.78"]

atom_types={'H':1, 'C':2, 'Rh':3}

lammps = LAMMPSlib(lmpcmds=lammps_cmds,
            atom_types=atom_types, 
            #lammps_header=lammps_header,
            log_file='LOG.log', 
            keep_alive=True)    

calculator = lammps       

