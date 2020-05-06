import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from slugtest import functions

matplotlib.rcParams.update({"font.size": 16})


class Model(ABC):

    def __init__(self, test):
        self.test = test
        self.aquifer = test.aquifer
        self.well = test.well
        self.s = None

    def plot(self, hax=None, plot_function='semilogx', show_legend=True):

        # initialize output dict
        h = dict()

        # hax given?
        if hax is None:
            h['fig'] = plt.figure()
            h['ax'] = plt.axes()
        else:
            h['fig'] = hax.figure
            h['ax'] = hax

        # get plot_function
        plot_function = getattr(h['ax'], plot_function)

        # plot
        h['s'], = plot_function(self.test.t, self.s, 'k-',
                                linewidth=2, label="calculated")

        if self.test.sobs is not None:
            h['sobs'] = plot_function(self.test.t, self.test.sobs, 'kx',
                                      markersize=8, label="observed")

        if hax is None:
            h['ax'].set_xlabel('t')
            h['ax'].set_ylabel('s')
            h['ax'].grid(which='both', axis="both")

        if show_legend:
            h["ax"].legend(loc="upper right")

        return plt, h

    @abstractmethod
    def run(self):
        pass


class BouwerRice(Model):

    def __init__(self, test):
        super().__init__(test)
        self.add_Kv = True
        self.R = None

    def run(self):
        self.s, self.R = functions.bouwer_rice(
            self.aquifer.D, self.well.depth + self.well.L, self.well.L,
            self.well.rs, self.well.rc, self.test.s0, self.aquifer.Kh,
            self.test.t, self.get_anisotropy()
        )

    def plot(self, hax=None, plot_function="semilogy", show_legend=True):
        plt, h = super().plot(hax, plot_function, show_legend)
        return plt, h

    def get_anisotropy(self):
        if self.add_Kv:
            return np.sqrt(self.aquifer.Kv / self.aquifer.Kh)
        else:
            return 1


class Hvorslev(BouwerRice):

    def __init__(self, test):
        super().__init__(test)
        self.R = 200 * self.well.rs  # see Butler (1998)

    def run(self):
        self.s = functions.hvorslev(
            self.well.L, self.well.rs, self.well.rc, self.R,
            self.test.s0, self.aquifer.Kh, self.test.t, self.get_anisotropy()
        )


class Cooper(Model):

    def __init__(self, test):
        super().__init__(test)
        self.ns = 16
        self.full_aquifer = True

    def run(self):

        # take aquifer thickness or screen length?
        if self.full_aquifer:
            D = self.aquifer.D
        else:
            D = self.well.L

        # call cooper function
        self.s = functions.cooper(
            self.test.s0, D,
            self.aquifer.Kh, self.aquifer.Ss,
            self.well.rs, self.well.rc,
            self.test.t, self.ns
        )


class KGS(Model):

    def __init__(self, test):
        super().__init__(test)
        self.miniter = 10
        self.maxiter = 500
        self.maxerr = 1e-6
        self.htol = 1e-5
        self.ns = 16
        self.niter = None
        self.err = None

    def run(self):
        self.s, self.niter, self.err = functions.kgs_no_skin(
            self.test.t, self.test.s0,
            self.well.rs, self.well.rc,
            self.aquifer.D, self.well.L, self.well.depth, self.aquifer.confined,
            self.aquifer.Kh, self.aquifer.Kv, self.aquifer.Ss,
            self.ns, self.maxerr, self.miniter, self.maxiter, self.htol
        )

