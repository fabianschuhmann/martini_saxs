from pathlib import Path

import MDAnalysis as mda

from saxs_tools.saxs_martini_beads import get_base_electrons, get_bead_radii


def get_valid_bead_names():
    """
    Build the set of valid Martini bead names known to the SAXS code.
    """
    valid = set(get_base_electrons().keys()) | set(get_bead_radii().keys())
    return valid


def pdb_has_valid_bead_names(pdb_path):
    """
    Return True if every atom name in the PDB is already a valid Martini bead name.
    """
    pdb_path = Path(pdb_path)
    valid_beads = get_valid_bead_names()

    u = mda.Universe(str(pdb_path))
    atom_names = set(u.atoms.names)

    invalid = sorted(name for name in atom_names if name not in valid_beads)
    return len(invalid) == 0, invalid


def load_itp(path, molecules):
    """
    Parse one Martini-style .itp and add entries to `molecules`.

    Result format:
        molecules[mol_name] = [bead_type_1, bead_type_2, ...]
    """
    path = Path(path)
    with path.open("r", encoding="utf8") as f:
        lines = f.readlines()

    clean_lines = [
        line.rstrip("\n").replace("\t", " ")
        for line in lines
        if line.strip() and not line.strip().startswith((";", "#"))
    ]

    current_name = None
    for i, line in enumerate(clean_lines):
        compact = line.replace(" ", "")

        if compact == "[moleculetype]":
            if i + 1 < len(clean_lines):
                current_name = clean_lines[i + 1].split()[0]

        elif compact == "[atoms]":
            if current_name is None:
                raise ValueError(f"Found [atoms] before [moleculetype] in {path}")

            bead_types = []
            k = i + 1
            while k < len(clean_lines):
                row = clean_lines[k].strip()
                if not row:
                    k += 1
                    continue
                if row.startswith("["):
                    break

                fields = row.split()
                if len(fields) < 2:
                    raise ValueError(f"Malformed [atoms] line in {path}: {row}")

                bead_types.append(fields[1])
                k += 1

            molecules[current_name] = bead_types

    return molecules


def load_all_itps(itp_dir=".", pattern="*.itp"):
    """
    Load all .itp files from a directory/pattern.

    Returns
    -------
    dict
        Mapping residue/molecule name -> list of bead types
    """
    molecules = {}
    itp_dir = Path(itp_dir)

    for fp in sorted(itp_dir.glob(pattern)):
        load_itp(fp, molecules)

    return molecules


def update_universe_atom_names(universe, mapping):
    """
    In-place update of atom names based on residue name -> bead-type list mapping.
    """
    for res in universe.residues:
        resname = res.resname
        if resname not in mapping:
            raise KeyError(f"No bead mapping provided for residue '{resname}'")

        bead_types = mapping[resname]
        atoms = res.atoms

        if len(atoms) != len(bead_types):
            raise ValueError(
                f"Residue '{resname}' atom count mismatch: "
                f"PDB has {len(atoms)}, mapping has {len(bead_types)}"
            )

        for atom, bead_type in zip(atoms, bead_types):
            atom.name = bead_type

    return universe


def convert_pdb_atom_names(pdb_path, output_path, itp_dir=".", itp_pattern="*.itp"):
    """
    Convert atom names in a PDB using residue definitions from .itp files.
    """
    pdb_path = Path(pdb_path)
    output_path = Path(output_path)

    mapping = load_all_itps(itp_dir=itp_dir, pattern=itp_pattern)
    if not mapping:
        raise FileNotFoundError(
            f"No .itp files found in '{itp_dir}' matching pattern '{itp_pattern}'"
        )

    u = mda.Universe(str(pdb_path))
    update_universe_atom_names(u, mapping)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    u.atoms.write(str(output_path))
    return output_path


def ensure_pdb_has_bead_names(
    pdb_path,
    itp_dir=".",
    itp_pattern="*.itp",
    converted_suffix="_updated",
    verbose=True,
):
    """
    Ensure that the PDB atom names are valid Martini bead names.

    Detection is done against saxs_martini_beads.py.
    Conversion, if needed, is done using .itp files.

    Returns
    -------
    Path
        Path to a PDB usable by the SAXS code.
    """
    pdb_path = Path(pdb_path)

    already_ok, invalid = pdb_has_valid_bead_names(pdb_path)
    if already_ok:
        if verbose:
            print(f"[converter] PDB already has valid bead names: {pdb_path}")
        return pdb_path

    if verbose:
        print(f"[converter] PDB has non-bead atom names: {invalid}")
        print(f"[converter] Converting using ITP files from: {itp_dir}")

    out_path = pdb_path.with_name(pdb_path.stem + converted_suffix + pdb_path.suffix)
    convert_pdb_atom_names(
        pdb_path=pdb_path,
        output_path=out_path,
        itp_dir=itp_dir,
        itp_pattern=itp_pattern,
    )

    if verbose:
        print(f"[converter] Wrote converted PDB: {out_path}")

    return out_path