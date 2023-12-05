from rdkit import Chem
import torch
from torch_geometric.data import Data

class process_drug():

    def __init__(self):

        self.hybridizations = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other']

        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]

        self.symbols = []

        self.electronegativities = []

        self.radii = []
        # Find the atomic elements in these drugs and then list electronegativity, radii, and symbol

    def to_graph(self, smile):

        mol = Chem.MolFromSmiles(smile)
        new_molecule = []
        edge_indices = []
        edge_attrs = []
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = [0.] * len(self.symbols)
            symbol[self.symbols.index(atom.GetSymbol())] = 1.
            degree = [0.] * 5
            degree[atom.GetDegree()] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(self.hybridizations)
            hybridization[self.hybridizations.index(atom.GetHybridization())] = 1.
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            electronegativity = self.electronegativities[atom.GetSymbol()]
            radius = self.radii[atom.GetSymbol()]
            x = torch.tensor(symbol + degree + [formal_charge] +
                             [radical_electrons] + hybridization + [aromaticity] + hydrogens + [chirality] +
                             chirality_type + [electronegativity] + [radius])

            new_molecule.append(x)
            features = torch.stack(new_molecule, dim=0)

            for bond in mol.GetBonds():
                edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
                edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

                bond_type = bond.GetBondType()
                single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
                double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
                triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
                aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
                conjugation = 1. if bond.GetIsConjugated() else 0.
                ring = 1. if bond.IsInRing() else 0.
                stereo = [0.] * 4
                edge_attr = torch.tensor(
                    [single, double, triple, aromatic, conjugation, ring] + stereo)

                edge_attrs += [edge_attr, edge_attr]

            if len(edge_attrs) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 10), dtype=torch.float)
            else:
                edge_index = torch.tensor(edge_indices).t().contiguous()
                edge_attr = torch.stack(edge_attrs, dim=0)

            graph = Data(x=torch.Tensor(features),
                         edge_index=edge_index,
                         edge_attr=edge_attr,
                         smile=smile)

            return graph
