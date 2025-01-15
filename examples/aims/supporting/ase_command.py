import os
import sys
from ase.calculators.aims import Aims
from ase.calculators.socketio import SocketIOCalculator
from socket import gethostname
import random

#hostname = gethostname()
#port = random.randint(11111, 55555)
species_dir = 'path_to_FHIaims/species_defaults/defaults_2020/light/'
command = 'mpirun -n 64 path_to_FHIaims/build/aims.230629.scalapack.mpi.x > {}'.format(os.path.join(os.getcwd(), "aims.out"))
aims = Aims(command=command,
            species_dir=species_dir,
            #use_pimd_wrapper=(hostname, port),
            compute_forces=True,
            xc='pbe',
            charge=0,
            spin='none',
            relativistic='atomic_zora scalar',
            #k_grid=[1,1,1],
            #sc_accuracy_eev='1E-4',
            sc_accuracy_rho='1E-4', 
            #force_correction  = ".true.",
            #sc_accuracy_forces='5E-4', 
            #sc_accuracy_etot='5E-4',
            # trying to reach convergence
            sc_iter_limit='1000',
            # re-initialize Pulay mixer sometimes:
            sc_init_iter='1000',)
            # small mixing needed for metal
            #mixer = 'pulay',
            #n_max_pulay = 10,
            # small mixing needed for metal
            #charge_mix_param='0.02',
            # big blur also needed for metal
            #occupation_type='gaussian 0.15',
            #use_dipole_correction=True,
            #many_body_dispersion_nl='beta=0.81',
            #vdw_correction_hirshfeld=True,
            #vdw_pair_ignore='Cu Cu',
            #output_level='MD_light')
            #output='cube stm -2')
            #
            #elsi_restart='read_and_write 80'
calculator = aims
#calculator = SocketIOCalculator(aims, log=sys.stdout, port=port)










