import os
from ase import Atoms
from ase.calculators.aims import Aims

species_dir = os.environ.get(
    "AIMS_SPECIES_DIR",
    "/u/lazerpo/FHIaims/species_defaults/defaults_2020/light",
)
aims_binary = os.environ.get("AIMS_BINARY", "/u/lazerpo/FHIaims/bin/aims.master.mpi.x")

elements = ["H", "C", "N", "O"]
cell_size = 20.0

for el in elements:
    workdir = os.path.join(os.getcwd(), f"atom_{el}")
    label = "run"
    run_dir = os.path.join(workdir, label)
    os.makedirs(run_dir, exist_ok=True)

    aims_command = f"srun -n 1 {aims_binary} > aims.out"

    calc = Aims(
        command=aims_command,
        species_dir=species_dir,
        xc="pbe",
        spin="collinear",
        default_initial_moment="hund",
        relativistic="atomic_zora scalar",
        compute_forces=False,
        sc_accuracy_rho=1e-5,
        sc_iter_limit=200,
        directory=workdir,
    )

    atom = Atoms(el, cell=[cell_size] * 3, pbc=False)
    atom.calc = calc
    energy = atom.get_potential_energy()
    print(f"{el}: {energy:.10f} eV")