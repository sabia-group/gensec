from ase.calculators.emt import EMT
from ase.calculators.loggingcalc import LoggingCalculator
import os, sys
from ase.calculators.dftb import Dftb
from ase.io import read, write
import pathlib
from socket import gethostname
import random
from subprocess import Popen
from ase.calculators.socketio import SocketIOCalculator
import shutil

# working_dir = "/home/damaksimovda/Insync/da.maksimov.da@gmail.com/GoogleDrive/PhD/Preconditioner/DFTB/AminoAcids/DiAla/"
UNIXSOCKET = "dftbplus2"
os.environ[
    "DFTB_PREFIX"
] = "/home/damaksimovda/programs/dftbplus-20.1.x86_64-linux/Slater_Koster/mio-1-1/"
os.environ[
    "ASE_DFTB_COMMAND"
] = "/home/damaksimovda/programs/dftbp/dftbplus/build/prog/dftb+/dftb+ > PREFIX.out"
DFTBP_PATH = "/home/damaksimovda/programs/dftbp/dftbplus/build/prog/dftb+/dftb+"
suppporting = "/home/damaksimovda/Insync/da.maksimov.da@gmail.com/GoogleDrive/PhD/Preconditioner/DFTB/AminoAcids/DiAla/supporting/"

# dirs = os.listdir(os.path.join(working_dir, "generate"))
# name_dir = format(max([int(i) for i in dirs]), "010d")
# at = read(os.path.join(working_dir, "generate", name_dir, name_dir+".in"), format = "aims")
# write(os.path.join(working_dir, 'geo.gen'), at, format='gen')
# shutil.copyfile(os.path.join(suppporting, "dftb_in.hsd"),
# os.path.join(working_dir, "dftb_in.hsd"))

# calc = Dftb(atoms=at,
# Hamiltonian_='DFTB',
# Hamiltonian_SCC='Yes',
# Hamiltonian_SCCTolerance=1e-6,
# Hamiltonian_MaxAngularMomentum_='',
# Hamiltonian_MaxAngularMomentum_H='s',
# Hamiltonian_MaxAngularMomentum_O='p',
# Hamiltonian_MaxAngularMomentum_C='p',
# Hamiltonian_MaxAngularMomentum_N='p',
# Hamiltonian_MaxAngularMomentum_S='d')
# kpts = (1,1,1))

# at.calc=calc
# calc.calculate(at)
# sys.exit(0)
# text = "Driver = Socket {\n\
# File = \"dftbplus\"\n\
# Protocol = i-PI {}\n\
# MaxSteps = 1000\n\
# Verbosity = 0\n\}\n"

# with open(os.path.join(working_dir, "dftb_in.hsd"), "a") as dftb:
# dftb.write(text)
# sys.exit(0)
calculator = SocketIOCalculator(log=sys.stdout, unixsocket=UNIXSOCKET)
Popen(DFTBP_PATH)
