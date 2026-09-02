import os
from ase.calculators.aims import Aims

species_dir = os.environ.get('AIMS_SPECIES_DIR', '/u/lazerpo/FHIaims/species_defaults/defaults_2020/light')
aims_binary = os.environ.get('AIMS_BINARY', '/u/lazerpo/FHIaims/bin/aims.master.mpi.x')
aims_command = f'srun -n 36 {aims_binary} > {os.path.join(os.getcwd(), "aims.out")}'

calculator = Aims(
    command=aims_command,
    species_dir=species_dir,
    xc='pbe',
    spin='none',
    relativistic='atomic_zora scalar',
    compute_forces=True,
    sc_accuracy_rho=1e-5,
    sc_iter_limit=100,
    )