import numpy as np
from collections import defaultdict
import os

class Preprocess:

    # bond_len: N-CA [1.4,1.55]  'laplace': (1.4589551055464296, 0.008568664703522886)
    #           CA-C [1.45,1.6]  'laplace': (1.5256277396534188, 0.008097397922186955)
    #           C-N  [1.28,1.38] 'cauchy':  (1.329622236704048, 0.003281794770536163)
    # bond_ang: CA-C-N [1.9,2.2] 'logistic': (2.0338830957259755, 0.011090343628058264)
    #           N-CA-C [1.75,2.25] 'laplace': (1.9380537313835406, 0.03440290262409793)
    #           C-N-CA [1.95,2.3]  'laplace': (2.122176463637426, 0.02021244939054669)

    def __init__(self):
        self.aa = ["GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "TYR", "ASP", "ASN",
                   "GLU", "LYS", "GLN", "MET", "SER", "THR", "CYS", "PRO", "HIS", "ARG",
                   "HID", "ASN", "ASH", "HIE", "HIP"]

        self.DIM = 3  # 坐标维数
        self.Heavy = ["C", "CA", "N"]
        self.ATOM = "ATOM"
        self.dataset_name = ['0','1','2','3','4','5','6','7','8','9',
                            'a','b','c','d','e','f','g','h','i','j',
                            'k','l','m','n','o','p','q','r','s','t',
                            'u','v','w','x','y','z']
        self.altLoc = ["", "A"]
        self.torsion_atom = ["N", "C"]

    def bond_len(self, atom1, atom2):  # N-CA CA-C C-N
        sum = 0
        for i in range(self.DIM):
            sum += np.power(atom1[i]-atom2[i], 2)
        return np.sqrt(sum)

    def bond_ang(self,  # N-CA-C
                        # CA-C-N
                        # C-N-CA
                 atom1:np.ndarray,
                 atom2:np.ndarray,  # 以atom2为中心原子计算键角
                 atom3:np.ndarray):
        vec1 = atom1 - atom2
        vec2 = atom3 - atom2
        return np.arccos(vec1.dot(vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))

    def dihedral(self,  # 原子坐标
                 p1:np.ndarray,
                 p2:np.ndarray,
                 p3:np.ndarray,
                 p4:np.ndarray):
        p12 = p2-p1
        p13 = p3-p1
        p42 = p2-p4
        p43 = p3-p4

        vec1 = np.cross(p12,p13)
        vec2 = np.cross(p42,p43)

        signature = [1 if vec1.dot(p42)<0 else -1]

        result = np.arccos(np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))) * signature[0]
        return result

    def processIon(self, aa):  # 处理质子化条件
        if aa in ['ASH', 'ASN']:
            return 'ASP'
        if aa in ['HIE', 'HID', 'HIP']:
            return 'HIS'
        return aa

    def check_dih(self, name:list):
        if name[0] == "C" and name[1] == "N" and name[2] == "CA" and name[3] == "C":
            return "phi"
        if name[0] == "N" and name[1] == "CA" and name[2] == "C" and name[3] == "N":
            return "psi"

        print(name)
        return "error"

    def defaultdict_list_1(self):  # nesting_1
        return defaultdict(list)

    def defaultdict_list_2(self):  # nesting_2
        return defaultdict(self.defaultdict_list_1)

    def readHeavyAtom(self, path, bond_len, bond_ang, dihedral_phi, dihedral_psi):
        print(path)
        cors = []  # cors
        name = []  # name
        ang_resName = []  # ang res
        torsion_resName = []  # dihedral res
        current_chain = ""
        with open(path, 'r') as file:
            for line in file.readlines():
                record = []
                record.append(line[:4].strip())  # ATOM
                if record[0] == self.ATOM:
                    record.append(line[6:11].strip())  # serial
                    record.append(line[12:16].strip())  # name
                    record.append(line[16].strip())  # altLoc
                    if record[2] in self.Heavy and record[3] in self.altLoc:  # select heavy atoms and A altLoc
                        resName = self.processIon(line[17:20].strip())
                        record.append(resName)  # resName
                        ang_resName.append(resName)

                        if record[2] in self.torsion_atom:  # update dihedral res
                            torsion_resName.append(resName)
                        chain = line[21].strip()
                        if current_chain == "":
                            current_chain = chain
                        elif current_chain != chain:  # capture the change of chain
                            if len(cors) == 3:  # calculating if more than 2
                                bond_len[name[0] + '-' + name[1]].append(self.bond_len(cors[0], cors[1]))
                                bond_len[name[1] + '-' + name[2]].append(self.bond_len(cors[1], cors[2]))
                                bond_ang[name[0] + '-' + name[1] + '-' + name[2]][ang_resName[0]+'-'+ang_resName[2]].append(self.bond_ang(np.array(cors[0]),
                                                                                                                                          np.array(cors[1]),
                                                                                                                                          np.array(cors[2])))
                            cors = []  # cors
                            name = []  # name
                            current_chain = chain

                        record.append(chain)  # chainID
                        record.append(line[22:26].strip())  # resSeq
                        record.append(line[30:38].strip())  # x
                        record.append(line[38:46].strip())  # y
                        record.append(line[46:54].strip())  # z
                        # print(record)
                        name.append(record[2])
                        cors.append([float(record[7]),float(record[8]),float(record[9])])

                        if len(cors) == 4:
                            # dih
                            if name[0] == "C":  # phi
                                state = self.check_dih(name)
                                if state == "error":
                                    return
                                dihedral_phi[torsion_resName[0]+'-'+torsion_resName[1]].append(self.dihedral(np.array(cors[0]),
                                                                                                             np.array(cors[1]),
                                                                                                             np.array(cors[2]),
                                                                                                             np.array(cors[3])))
                                torsion_resName.pop(0)
                            if name[0] == "N":  # psi
                                state = self.check_dih(name)
                                if state == "error":
                                    return
                                dihedral_psi[torsion_resName[0]+'-'+torsion_resName[1]].append(self.dihedral(np.array(cors[0]),
                                                                                                             np.array(cors[1]),
                                                                                                             np.array(cors[2]),
                                                                                                             np.array(cors[3])))
                                torsion_resName.pop(0)
                            # bond_len
                            bond_len[name[0]+'-'+name[1]].append(self.bond_len(cors[0], cors[1]))

                            # bond_ang
                            bond_ang[name[0]+'-'+name[1]+'-'+name[2]][ang_resName[0]+'-'+ang_resName[2]].append(self.bond_ang(np.array(cors[0]),
                                                                                                                              np.array(cors[1]),
                                                                                                                              np.array(cors[2])))
                            cors.pop(0)
                            name.pop(0)
                            ang_resName.pop(0)

        file.close()
        # process the remaining three record
        if len(cors) == 3:  # because of this(3), some irregular items may be generated. It could be modified later
            bond_len[name[0]+'-'+name[1]].append(self.bond_len(cors[0], cors[1]))
            bond_len[name[1]+'-'+name[2]].append(self.bond_len(cors[1], cors[2]))
            bond_ang[name[0]+'-'+name[1]+'-'+name[2]][ang_resName[0]+'-'+ang_resName[2]].append(self.bond_ang(np.array(cors[0]),
                                                                                                              np.array(cors[1]),
                                                                                                              np.array(cors[2])))

        print("done")

    def statistics(self):
        bond_len = defaultdict(list)
        bond_ang = defaultdict(self.defaultdict_list_1)
        dihedral_phi = defaultdict(list)
        dihedral_psi = defaultdict(list)

        for i in self.dataset_name:
            for j in self.dataset_name:
                path = "/home/caofan/learning/dataset/pdbstyle-2.07/{0}/".format(i+j)
                if os.path.exists(path):
                    file_list = os.listdir(path)
                    for k in file_list:
                            self.readHeavyAtom(path+k, bond_len, bond_ang, dihedral_phi, dihedral_psi)

        np.save("./bond_len.npy", bond_len)
        np.save("./bond_ang.npy", bond_ang)
        np.save("./dihedral_phi.npy", dihedral_phi)
        np.save("./dihedral_psi.npy", dihedral_psi)

    def bondlenclean(self):  # reduce the scale of data with no bias for the convenience of modeling
        bond_len = np.load("./bond_len.npy", allow_pickle=True).item()
        for types in bond_len.keys():
            temp_list = []
            for i in bond_len[types]:
                seed = np.random.uniform(low=0, high=1, size=1)[0]
                if seed <= 0.05:  # reduce by 20 times
                    temp_list.append(i)
            np.save("./{0}_reduced.npy".format(types), temp_list)

    def bondangclean(self):  # reduce the scale of data with no bias for the convenience of modeling
        bond_ang = np.load("./bond_ang.npy", allow_pickle=True).item()
        for types in bond_ang.keys():
            temp_list = []
            for i in bond_ang[types].keys():
                for j in bond_ang[types][i]:
                    seed = np.random.uniform(low=0, high=1, size=1)[0]
                    if seed <= 0.05:  # reduce by 20 times
                        temp_list.append(j)
            np.save("./{0}_reduced.npy".format(types), temp_list)

    def load_mol_E(self, path):
        print("Reading: ", path)
        E = []
        with open(path, 'r') as file:
            for i in file.readlines():
                record = i.strip()
                if record[0] != '#':
                    energy = float(record.split()[-1])
                    E.append(energy)
        # print(len(E))
        return E
        # np.save("./c7o2h10_equilibrium.npy", E)

    def load_mol_cor(self, path):
        print("Reading: ", path)
        cors_over_trajectory = []
        atoms_over_trajectory = []
        single_atoms = []
        single_cor = []
        natoms = 0
        atoms_i = 0
        with open(path, 'r') as file:
            for i in file.readlines():
                record = i.strip()
                if len(record.split()) != 1:
                    if record.split(":")[0] == 'Iteration':
                        single_cor = []
                        single_atoms = []  # be careful when using list.clear()
                    else:
                        record = record.split()
                        X = float(record[-3])
                        Y = float(record[-2])
                        Z = float(record[-1])
                        single_atoms.append(record[0])
                        single_cor.append([X,Y,Z])
                        atoms_i += 1
                        if atoms_i == natoms:
                            cors_over_trajectory.append(single_cor)
                            atoms_over_trajectory.append(single_atoms)
                            atoms_i = 0
                else:
                    natoms = int(record.split()[0])
        return cors_over_trajectory, atoms_over_trajectory

    def load_mol_E_batch(self):
        start = 1000
        data = defaultdict(self.defaultdict_list_1)
        for i in range(2010):
            serial = i + start
            path_E = "/Users/erik/Downloads/c7o2h10_md/c7o2h10_md/{0}.energy.dat".format(serial)
            path_XYZ = "/Users/erik/Downloads/c7o2h10_md/c7o2h10_md/{0}.xyz".format(serial)
            if os.path.exists(path_E):
                E_over_trajectory = self.load_mol_E(path=path_E)
                cors_atoms = self.load_mol_cor(path=path_XYZ)

                data[str(serial)]["cors"] = cors_atoms[0]
                data[str(serial)]["E"] = E_over_trajectory
                data[str(serial)]["atoms"] = cors_atoms[1]

        np.save("./c7o2h10_md.npy", data)

# pp=Preprocess()
# pp.load_mol_E_batch()
# print(pp.load_mol_E(path="/Users/erik/Downloads/c7o2h10_md/c7o2h10_equilibrium.dat"))
# c7o2h10_md = np.load("./c7o2h10_md.npy", allow_pickle=True).item()
# Keys = c7o2h10_md.keys()
# print(Keys)
# print(c7o2h10_md["1000"].keys())

