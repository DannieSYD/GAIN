from collections import defaultdict
from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.data import extract_zip
from tqdm import tqdm

from loaders.ensemble import EnsembleMultiPartDatasetV2
from loaders.utils import mol_to_data_obj


class CommonCOV2V2(EnsembleMultiPartDatasetV2):
    def __init__(
        self,
        root,
        max_num_conformers=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.max_num_conformers = max_num_conformers
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def num_parts(self):
        return 1

    def _process_molecules(self, return_molecule_lists=False):
        raw_file = self.raw_paths[0]
        extract_zip(raw_file, self.raw_dir)

        molecule_dict = defaultdict(list)
        splits = dict()
        for split in ["train", "test", "val"]:
            for file in tqdm(Path(self.raw_dir).glob(f"{split}*.sdf")):
                # print(f"Processing {file}")
                with Chem.SDMolSupplier(str(file), removeHs=False) as suppl:
                    for mol in suppl:
                        data = mol_to_data_obj(mol)
                        data.smiles = Chem.MolToSmiles(mol)
                        data.mol = mol
                        data.name = int(mol.GetProp("geom_id"))
                        data.ori_conf_id = int(mol.GetProp("conf_id"))
                        data.y = [
                            int(mol.GetProp(descriptor))
                            for descriptor in self.descriptors
                        ]
                        data.energy = float(mol.GetProp("energy"))
                        data.weights = float(mol.GetProp("weights"))
                        data.split = split
                        splits[data.name] = split

                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)

                        molecule_dict[data.name].append(data)

        data_list, y, cursor, split_ids = [], [], 0, defaultdict(list)
        for i, (name, conformers) in enumerate(tqdm(molecule_dict.items())):
            if self.max_num_conformers is not None:
                conformers = sorted(conformers, key=lambda x: x.energy)
                conformers = conformers[: self.max_num_conformers]

            for conformer in conformers:
                conformer.molecule_idx = cursor
            cursor += 1

            split_ids[conformers[0].split].append(i)

            if return_molecule_lists:
                data_list.append(conformers)
            else:
                data_list.extend(conformers)

            y.append(
                torch.Tensor([conformers[0].y[i] for i in range(len(self.descriptors))])
            )

        if return_molecule_lists:
            data_list = [data_list]
        return data_list, torch.stack(y, dim=0), split_ids

    def process(self):
        molecule_lists, y, split_ids = self._process_molecules(
            return_molecule_lists=True
        )
        torch.save((molecule_lists, y), self.processed_paths[0])
        torch.save(split_ids, self.processed_paths[1])

    def get_split_idx(self):
        split_ids = torch.load(self.processed_paths[1])
        return split_ids


class COV23CLV2(CommonCOV2V2):
    @property
    def descriptors(self):
        return ["sars_cov_two_cl_protease_active"]

    @property
    def raw_file_names(self):
        return "COV2_3CL.zip"

    @property
    def processed_file_names(self):
        data_file_name = (
            "COV2_3CL_V2_processed.pt"
            if self.max_num_conformers is None
            else f"COV2_3CL_V2_{self.max_num_conformers}_processed.pt"
        )
        split_file_name = "COV2_3CL_V2_split_ids.pt"
        return [data_file_name, split_file_name]


class COV2V2(CommonCOV2V2):
    @property
    def descriptors(self):
        return ["sars_cov_two_active"]

    @property
    def raw_file_names(self):
        return "cov2.zip"

    @property
    def processed_file_names(self):
        data_file_name = (
            "COV2_V2_processed.pt"
            if self.max_num_conformers is None
            else f"COV2_V2_{self.max_num_conformers}_processed.pt"
        )
        split_file_name = "COV2_V2_split_ids.pt"
        return [data_file_name, split_file_name]