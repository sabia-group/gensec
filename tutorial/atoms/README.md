# Atomic reference energies

This folder contains the isolated-atom setup used to compute MACE `E0s`.

The elements in this example are H, C, N, and O, so the folder contains one
subfolder for each atom:

- `atom_H/`
- `atom_C/`
- `atom_N/`
- `atom_O/`

`atomic_energies.py` collects the isolated-atom energies after the reference
calculations are finished. These values are then placed in
`training.mace_args.E0s` in `parameters.json`.

The atomic calculations should use the same reference setup as the structure
labels: same electronic-structure code, functional, species defaults,
relativistic treatment, and similar numerical settings.
