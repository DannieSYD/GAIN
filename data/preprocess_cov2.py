import argparse
import multiprocessing as mp
from functools import partial

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina
from tqdm import tqdm

from nff.data.features import graph


def add_properties_to_mol(mol, mol_group, idx):
    # props = ["sars_cov_two_active", "uniqueconfs", "energy", "weights", "geom_id"]
    props = [
        "sars_cov_two_cl_protease_active",
        "uniqueconfs",
        "energy",
        "weights",
        "geom_id",
    ]
    for prop in props:
        if mol_group[prop].dim() == 0:
            mol.SetProp(prop, str(mol_group[prop].item()))
        else:
            mol.SetProp(prop, str(mol_group[prop][idx].item()))
    mol.SetProp("conf_id", str(idx))


def process_single_molecule(mol_group, threshold=0.8, output_file=""):
    mol_list = mol_group["rd_mols"]
    energies = mol_group["energy"]
    geom_id = mol_group["geom_id"].item()

    # Handle single conformer case
    if len(mol_list) < 2:
        mol = mol_list[0]
        add_properties_to_mol(mol, mol_group, idx=0)
        with Chem.SDWriter(f"{output_file}_{geom_id}.sdf") as w:
            w.write(mol)
        return [mol]

    # Add properties to all molecules
    for i, mol in enumerate(mol_list):
        add_properties_to_mol(mol, mol_group, idx=i)

    # Calculate RMSD distance matrix
    dists = []
    n_mols = len(mol_list)
    for i in range(n_mols):
        for j in range(n_mols):
            if i <= j:  # Skip diagonal and upper triangle
                continue
            rmsd = rdMolAlign.GetBestRMS(mol_list[i], mol_list[j])
            dists.append(rmsd)

    # Cluster using Butina algorithm
    clusters = Butina.ClusterData(
        dists, n_mols, threshold, isDistData=True, reordering=True
    )

    # Process each cluster
    kept_conformers = []  # List to store indices of conformers to keep
    for cluster in clusters:
        if len(cluster) == 0:
            continue

        # Find conformer with lowest energy in the cluster
        cluster_energies = [energies[i].item() for i in cluster]
        min_energy_idx = cluster[np.argmin(cluster_energies)]
        kept_conformers.append(min_energy_idx)

    print(
        f"Selected {len(kept_conformers)} conformers out of {len(mol_list)} conformers for molecule {geom_id}"
    )

    # kept_conformers = list(range(len(mol_list)))  # Keep all conformers for now

    with Chem.SDWriter(f"{output_file}_{geom_id}.sdf") as w:
        for i in kept_conformers:
            w.write(mol_list[i])

    # Create new molecule with selected conformers
    # if kept_conformers:
    #     base_mol = mol_list[kept_conformers[0]]
    #
    #     # Add remaining conformers
    #     for i, conf_idx in enumerate(kept_conformers[1:], 1):
    #         conf = mol_list[conf_idx].GetConformer()
    #         positions = conf.GetPositions()
    #
    #         base_mol.AddConformer(Chem.Conformer(base_mol.GetNumAtoms()), assignId=True)
    #         new_conf = base_mol.GetConformer(i)
    #
    #         # Copy atomic positions
    #         for j, pos in enumerate(positions):
    #             new_conf.SetAtomPosition(j, pos)
    #     return base_mol
    return [mol_list[i] for i in kept_conformers]


def cluster_conformers_parallel(
    dataset, threshold=0.8, n_processes=128, output_file=""
):
    process_func = partial(
        process_single_molecule, threshold=threshold, output_file=output_file
    )
    with mp.Pool(processes=n_processes) as pool, tqdm(
        total=len(dataset), desc="Processing molecules"
    ) as pbar:
        for _ in pool.imap_unordered(process_func, dataset):
            pbar.update()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--threshold", type=float, default=0.8)
    args.add_argument("--n_processes", type=int, default=128)
    args.add_argument("--files", type=str, nargs="+", help="Path to dataset files")
    args.add_argument("--output_file", type=str, help="Path to output file")
    args = args.parse_args()

    threshold = args.threshold
    n_processes = args.n_processes
    output_file = args.output_file

    for path in args.files:
        print(f"Processing {path}")
        dataset = torch.load(path)
        dataset = graph.make_rd_mols(dataset)

        cluster_conformers_parallel(dataset, threshold, n_processes, output_file)

    # pkl.dump(unified_mols, open(f"cov2_{output_file}.pkl", "wb"))
