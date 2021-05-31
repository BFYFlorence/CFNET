import numpy as np


class PDB_single:
    def __init__(self):
        self.atoms = []
        self.aas = []
        self.resSeq = []
        self.cors = []


class PDB_record:
    def __init__(self):
        self.name = ""
        self.resName = ""


class PDBedit:
    def generate_pdb(self, path, pdb_single, save=False):
        # atoms: list of atoms       (n,)
        # aas  : list of amino acids (n,)
        # cors : list of coordinates (n,3)
        if len(pdb_single.atoms) != len(pdb_single.aas) or len(pdb_single.atoms) != len(pdb_single.cors):
            raise Exception("something wrong about contents in input!")

        with open(path, 'a') as file:
            for line in range(len(pdb_single.atoms)):
                file.write("{:<4s}  {:>5d}  {:<4s} {:<3s} {:<1s}{:>4s}    {:>8.3f} {:>8.3f} {:>8.3f} \n".format(
                    "ATOM",  # ATOM
                    line + 1,  # serial
                    pdb_single.atoms[line],  # name
                    pdb_single.aas[line],  # resName
                    "A",  # chainID
                    pdb_single.resSeq[line],  # resSeq
                    pdb_single.cors[line][0],  # X
                    pdb_single.cors[line][1],  # Y
                    pdb_single.cors[line][2],  # Z
                ))
