import os, glob
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from slugtest import inputs


class Base(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_full_path_name(self):
        pass


class Child(Base):

    def __init__(self, folder, name):
        super().__init__(name)
        self.folder = folder

    def get_full_path_name(self):
        return os.path.join(self.folder.get_full_path_name(), self.name)


class Folder(Base):

    def __init__(self, name):
        super().__init__(name)
        self.subfolder = []

    def get_full_path_name(self):
        return self.name

    def connect(self):
        subfolders = os.listdir(self.name)
        for name in subfolders:
            if os.path.isdir(os.path.join(self.name, name)):
                subfolder = Subfolder(self, name)
                self.subfolder.append(subfolder)
                subfolder.set_cut_files()
                subfolder.set_raw_files()

    def get_subfolder_names(self):
        return [subfolder.name for subfolder in self.subfolder]

    def get_subfolder(self, subfolder_name):
        i = self.get_subfolder_names().index(subfolder_name)
        return self.subfolder[i]

    def get_cut_file(self, subfolder_name, test_num):
        subfolder = self.get_subfolder(subfolder_name)
        return subfolder.get_cut_file(test_num)

    def get_raw_file(self, subfolder_name, test_num):
        subfolder = self.get_subfolder(subfolder_name)
        return subfolder.get_raw_file(test_num)

    def get_test(self, subfolder_name, test_num, nt=None):
        subfolder = self.get_subfolder(subfolder_name)
        test, well, aquifer = subfolder.get_test(test_num, nt)
        return test, well, aquifer


class Subfolder(Child):

    def __init__(self, folder, name):
        super().__init__(folder, name)
        self.raw_file = []
        self.cut_file = []

    def set_cut_files(self):
        files = glob.glob(os.path.join(self.get_full_path_name(), '*_cut.csv'))
        files = [os.path.basename(file) for file in files]
        files.sort()
        for file in files:
            self.cut_file.append(CutFile(self, file))

    def set_raw_files(self):
        files = glob.glob(os.path.join(self.get_full_path_name(), '*_ruw.csv'))
        files = [os.path.basename(file) for file in files]
        files.sort()
        for file in files:
            self.raw_file.append(RawFile(self, file))

    def get_test_nums(self):
        return [cut_file.test_num for cut_file in self.cut_file]

    def get_cut_file(self, test_num):
        i = self.get_test_nums().index(test_num)
        return self.cut_file[i]

    def get_raw_file(self, test_num):
        i = self.get_test_nums().index(test_num)
        return self.raw_file[i]

    def get_test(self, test_num, nt=None):
        cut_file = self.get_cut_file(test_num)
        test, well, aquifer = cut_file.read(nt)
        return test, well, aquifer


class File(Child, ABC):

    def __init__(self, folder, name):
        super().__init__(folder, name)
        self.test_num = int([ch for ch in name if ch.isdigit()][0])

    @abstractmethod
    def read(self, *args):
        pass


class CutFile(File):

    def __init__(self, folder, name):
        super().__init__(folder, name)
        self.aquifer = None
        self.well = None
        self.test = None
        self.t = None
        self.sobs = None

    def read(self, nt=None):

        m = np.genfromtxt(self.get_full_path_name())

        aquifer = inputs.Aquifer(confined=m[0, 0], D=m[0, 1], Kh=m[1, 0],
                                 Kv=m[1, 1], Ss=m[2, 0])
        well = aquifer.add_well(depth=m[3, 0], L=m[3, 1], rs=m[4, 0], rc=m[4, 1])

        t = m[6:, 0]
        sobs = m[6:, 1]
        b = t > 0.0
        t = t[b]
        sobs = sobs[b]
        if sobs[0] < 0:
            sobs = -sobs

        self.t = t.copy()
        self.sobs = sobs.copy()

        if nt is not None:
            t, sobs = self.select(t, sobs, nt)

        test = well.add_test(m[2, 1], t / 3600 / 24, sobs)
        test.dt = m[5, 0] / 3600 / 24
        test.ds = m[5, 1]

        self.aquifer = aquifer
        self.well = well
        self.test = test

        return test, well, aquifer

    def write(self):
        m = np.zeros((6, 2))
        m[0, 0] = float(self.aquifer.confined)
        m[0, 1] = self.aquifer.D
        m[1, 0] = self.aquifer.Kh
        m[1, 1] = self.aquifer.Kv
        m[2, 0] = self.aquifer.Ss
        m[2, 1] = self.test.s0
        m[3, 0] = self.well.depth
        m[3, 1] = self.well.L
        m[4, 0] = self.well.rs
        m[4, 1] = self.well.rc
        m[5, 0] = self.test.dt
        m[5, 1] = self.test.ds
        ts = np.concatenate((self.t[:, np.newaxis], self.sobs[:, np.newaxis]),
                            axis=1)
        m = np.concatenate((m, ts), axis=0)
        np.savetxt(self.get_full_path_name(), m)

    @staticmethod
    def select(t, sobs, nt):

        # determine frequency
        freq = np.max(np.round(1 / (t[1:] - t[:-1])))

        # transform times to integers
        tfreq = np.round(t * freq)
        tfreq.astype(np.int_)

        # create the new logspaced series
        tlog = np.logspace(np.log10(tfreq[0]), np.log10(tfreq[-1]), nt)
        tlog = np.unique(np.round(tlog))
        tlog.astype(np.int_)

        # the new reduced time series
        tred = []
        sred = []

        # select times
        for k in range(len(tlog)):
            i = tlog[k] == tfreq
            if np.any(i):
                tred.append(t[i])
                sred.append(sobs[i])

        return np.array(tred).flatten(), np.array(sred).flatten()


class RawFile(File):

    def __init__(self, folder, name):
        super().__init__(folder, name)
        self.t = None
        self.h = None
        self.selected = None

    def read(self):
        m = np.genfromtxt(self.get_full_path_name())
        self.t = m[:, 0]
        self.h = m[:, 1]
        self.selected = m[:, 2].astype(np.bool)

    def plot(self):

        # initialize output dict
        h = dict()

        # figure and axes
        h['fig'] = plt.figure()
        h['ax'] = plt.axes()

        # h and selected
        h['h'] = plt.plot(self.t, self.h, 'k-')
        h['selected'] = plt.plot(self.t[self.selected], self.h[self.selected], 'ro')

        # labels
        h['ax'].set_xlabel('t (sec)')
        h['ax'].set_ylabel('h (m)')

        # output
        return h
