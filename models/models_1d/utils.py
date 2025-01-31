import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from e3fp.pipeline import fprints_from_mol
from concurrent.futures import ProcessPoolExecutor


def construct_fingerprint(smiles):
    mol = [Chem.MolFromSmiles(s) for s in smiles]

    # Morgan fingerprint
    mfp = np.asarray(
        [AllChem.GetMorganFingerprintAsBitVect(x, radius=3, nBits=3 * 1024) for x in mol], dtype=np.int8)

    # RDKit topological fingerprint
    rdkbi = {}
    rdfp = np.asarray([Chem.RDKFingerprint(x, maxPath=5, bitInfo=rdkbi) for x in mol], dtype=np.int8)

    # MACCS keys
    maccs = np.asarray([MACCSkeys.GenMACCSKeys(x) for x in mol], dtype=np.int8)

    fingerprint = np.concatenate([mfp, rdfp, maccs], axis=1)

    return drop_redundant(fingerprint)


def drop_redundant(fingerprint):
    # dropout redundant features
    correlation = np.corrcoef(np.transpose(fingerprint))
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    correlation[mask] = 0
    drop = np.where(abs(correlation) > 0.9)[1]
    return np.delete(fingerprint, drop, axis=1)


def construct_molecules(dataset, field_name='smiles'):
    molecule_idx = dataset.data.molecule_idx[dataset._indices]
    molecules, counts = molecule_idx.unique(return_counts=True)
    cursor = 0
    mapping = []
    for count in counts:
        mapping.append(list(range(cursor, cursor + count)))
        cursor += count
    conformer_index = [mapping[i][0] for i in range(len(molecules))]
    dataset = dataset[conformer_index]
    return dataset.__getattr__(field_name)


def concatenate_molecules(dataset, variable_name, field_name='smiles'):
    molecule_idx = dataset.data.molecule_idx[dataset._indices]
    variables = dataset.data[variable_name][dataset._indices]
    unique_variables = sorted(variables.unique().tolist())

    molecules, counts = molecule_idx.unique(return_counts=True)
    cursor = 0
    conformer_mapping = []
    variable_mapping = []
    for count in counts:
        conformer_mapping.append(list(range(cursor, cursor + count)))
        variable_mapping.append(variables[cursor:cursor + count])
        cursor += count

    fields = []
    for i in range(len(molecules)):
        conformer_index = conformer_mapping[i]
        variable_by_conformer = variable_mapping[i]

        fields_per_molecule = []
        for var in unique_variables:
            for idx, id_var in zip(conformer_index, variable_by_conformer):
                if id_var == var:
                    fields_per_molecule.append(dataset.data[field_name][dataset._indices[idx]])
                    break
        if field_name == 'smiles':
            fields.append('.'.join(fields_per_molecule))
        else:
            fields.append(fields_per_molecule)
    return fields


def compute_3d_fingerprint(mol_list, fprint_params):
    if type(mol_list) is list:
        fps = []
        for mol in mol_list:
            mol.SetProp('_Name', 'conformer')
            fp = fprints_from_mol(mol, fprint_params=fprint_params)[0].to_vector(sparse=False)
            fps.append(fp)
        fps = np.asarray(fps, dtype=np.int8)
        fps = np.concatenate(fps)
    else:
        mol_list.SetProp('_Name', 'conformer')
        fps = fprints_from_mol(mol_list, fprint_params=fprint_params)[0].to_vector(sparse=False)
        fps = np.asarray(fps, dtype=np.int8)
    return fps


def construct_3d_fingerprint(mols, fprint_params, max_workers=None):
    all_fps = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for fps in executor.map(
                compute_3d_fingerprint, mols, [fprint_params] * len(mols), chunksize=len(mols) // (10 * max_workers)):
            all_fps.append(fps)
    return drop_redundant(all_fps)

# def construct_3d_fingerprint(mols, fprint_params):
#     all_fps = []
#     for mol_list in mols:
#         if type(mol_list) is list:
#             fps = []
#             for mol in mol_list:
#                 mol.SetProp('_Name', 'conformer')
#                 fp = fprints_from_mol(mol, fprint_params=fprint_params)[0].to_vector(sparse=False)
#                 fps.append(fp)
#             fps = np.asarray(fps, dtype=np.int8)
#             fps = np.concatenate(fps)
#         else:
#             mol_list.SetProp('_Name', 'conformer')
#             fps = fprints_from_mol(mol_list, fprint_params=fprint_params)[0].to_vector(sparse=False)
#             fps = np.asarray(fps, dtype=np.int8)
#         all_fps.append(fps)
#
#     return drop_redundant(all_fps)
