import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from slugtest import parameters, functions, models
from scipy.optimize import fmin, least_squares

matplotlib.rcParams.update({"font.size": 16})


class Optimizer(ABC):

    def __init__(self, model):
        self.model = model
        self.parameter = []
        self.composed = []
        self.dependent = []
        self.inv_dependent = []

    def add_parameter(self, obj, name, ini_value=None):
        if type(obj) == str:
            obj = getattr(self.model, obj)
        parameter = parameters.Parameter(obj, name, ini_value)
        self.parameter.append(parameter)
        return parameter

    def add_composed(self, independent, dependent, func, inv_func):
        # independent and dependent can be objects
        # or lists with input arguments [object, name, ini_value]

        def create_parameter(parameter):
            if type(parameter) == list:
                if type(parameter[0]) == str:
                    parameter[0] = getattr(self.model, parameter[0])
                parameter = parameters.Parameter(*parameter)
            return parameter

        independent = create_parameter(independent)
        dependent = create_parameter(dependent)
        composed = parameters.Composed(independent, dependent, func, inv_func)
        self.composed.append(composed)

        return composed, independent, dependent

    def add_dependent(self, parameter, obj, name, ini_value=None):
        if type(obj) == str:
            obj = getattr(self.model, obj)
        dependent = parameters.Dependent(parameter, obj, name, ini_value)
        self.dependent.append(dependent)
        return dependent

    def add_inv_dependent(self, parameter, obj, name, ini_value=None):
        if type(obj) == str:
            obj = getattr(self.model, obj)
        inv_dependent = parameters.InverseDependent(parameter, obj, name, ini_value)
        self.inv_dependent.append(inv_dependent)
        return inv_dependent

    def plot(self, plot_function_s='semilogx', plot_function_eta='semilogx'):

        # initialize output dict
        h = dict()

        # create figure
        h['fig'] = plt.figure()

        # create axes
        h['ax'] = dict()
        h['ax']['s'] = h['fig'].add_subplot(121)
        h['ax']['eta'] = h['fig'].add_subplot(122)

        # left plot: drawdowns
        _, hs = self.model.plot(hax=h['ax']['s'], plot_function=plot_function_s)
        h['s'] = hs['s']
        h['ax']['s'].set_xlabel('t')
        h['ax']['s'].set_ylabel('s')
        h['ax']['s'].grid(which='both', axis="both")

        # get plot_function for eta
        plot_function_eta = getattr(h['ax']['eta'], plot_function_eta)

        # right plot: residuals
        h['eta'], = plot_function_eta(self.model.test.t, self.get_eta(),
                                      'k.', markersize=12)
        h['ax']['eta'].set_xlabel('t')
        h['ax']['eta'].yaxis.tick_right()
        h['ax']['eta'].yaxis.set_label_position("right")
        h['ax']['eta'].set_ylabel('s - s' + '$\mathregular{_{obs}}$')
        h['ax']['eta'].grid(which='both', axis="both")

        # output
        return plt, h

    def objective_function(self, x):
        self.set_values(x)
        self.model.run()
        self.reset_values()
        return self.get_ssr()

    def get_eta(self):
        return self.model.s - self.model.test.sobs

    def get_ssr(self):
        eta = self.get_eta()
        return np.inner(eta, eta)

    def get_rmse(self):
        return np.sqrt(self.get_ssr() / len(self.model.s))

    def get_ini_values(self):

        def get_x0(parameter):
            if parameter.log:
                x0 = np.log10(parameter.ini_value)
            else:
                x0 = parameter.ini_value
            return x0

        x0 = [get_x0(par) for par in self.parameter] + \
             [get_x0(comp) for comp in self.composed]

        return np.array(x0)

    def get_values(self):

        def get_x(parameter):
            if parameter.log:
                x = np.log10(parameter.get_value())
            else:
                x = parameter.get_value()
            return x

        x = [get_x(par) for par in self.parameter] + \
            [get_x(comp) for comp in self.composed]

        return np.array(x)

    def set_values(self, x):
        i = 0
        for par in self.parameter:
            if par.log:
                par.set_value(10**x[i])
            else:
                par.set_value(x[i])
            i += 1
        for comp in self.composed:
            if comp.log:
                comp.set_value(10**x[i])
            else:
                comp.set_value(x[i])
            i += 1
        for dep in self.dependent:
            dep.set_value()
        for inv in self.inv_dependent:
            inv.set_value()

    def reset_values(self):
        for par in self.parameter:
            par.reset_value()
        for comp in self.composed:
            comp.reset_value()
        for dep in self.dependent:
            dep.reset_value()
        for inv in self.inv_dependent:
            inv.reset_value()

    @abstractmethod
    def run(self):
        pass


class NelderMead(Optimizer):

    def __init__(self, model):
        super().__init__(model)
        self.Kh = None
        self.Ss = None
        self.s0 = None
        self.Kv = None
        self.ani = None

    def add_Kh(self, ini_value=None, log=True):
        self.Kh = self.add_parameter(self.model.aquifer, 'Kh', ini_value)
        self.Kh.log = log

    def add_Ss(self, ini_value=None, log=True):
        self.Ss = self.add_parameter(self.model.aquifer, 'Ss', ini_value)
        self.Ss.log = log

    def add_s0(self, ini_value=None, log=True):
        self.s0 = self.add_parameter(self.model.test, 's0', ini_value)
        self.s0.log = log

    def add_Kv(self, ini_value=None, log=True):
        self.Kv = self.add_parameter(self.model.aquifer, 'Kv', ini_value)
        self.Kv.log = log

    def keep_ani(self, ini_value=None):
        # ani = Kv / Kh
        # Kv_ini_value = Kh_ini_value * ini_value
        if ini_value is not None:
            ini_value *= self.Kh.ini_value
        self.Kv = self.add_dependent(self.Kh, self.model.aquifer, 'Kv', ini_value)

    def add_ani(self, ini_value=None, log=True):
        # ani = Kv / Kh
        # Kv_ini_value = Kh_ini_value * ini_value
        if ini_value is not None:
            ini_value *= self.Kh.ini_value
        self.ani, _, self.Kv = self.add_composed(self.Kh,
                                                 [self.model.aquifer, 'Kv', ini_value],
                                                 lambda Kh, Kv: Kv / Kh,
                                                 lambda Kh, ani: Kh * ani)
        self.ani.log = log

    def run(self):
        x0 = self.get_ini_values()
        xopt = fmin(lambda x: self.objective_function(x), x0)
        self.set_values(xopt)


class LevenbergMarquardt(NelderMead):

    def __init__(self, model):
        super().__init__(model)
        self.diff_step = 0.01
        self.status = None
        self.message = None

    def run(self):
        x0 = self.get_ini_values()
        result = least_squares(lambda x: self.objective_function(x), x0,
                               method='lm', diff_step=self.diff_step)
        self.status = result.status
        self.message = result.message
        if self.status > 0:
            self.set_values(result.x)

    def objective_function(self, x):
        self.set_values(x)
        self.model.run()
        self.reset_values()
        return self.get_eta()


class LinearRegression(Optimizer):

    def __init__(self, model):
        super().__init__(model)
        self.Kh = self.add_parameter(model.aquifer, 'Kh')
        self.s0 = None
        self.smin = 1e-2

    def add_s0(self):
        self.s0 = self.add_parameter(self.model.test, 's0')

    def run(self):

        # select sobs < smin
        t = np.array(self.model.test.t)
        sobs = np.array(self.model.test.sobs)
        keep = sobs >= self.smin
        t = t[keep]
        sobs = sobs[keep]

        # derive s0?
        if self.s0 is None:
            s0 = self.model.test.s0
        else:
            s0 = None

        # Hvorslev model
        if type(self.model) == models.Hvorslev:
            values = functions.hvorslev(
                self.model.well.L, self.model.well.rs, self.model.well.rc,
                self.model.R, s0, sobs, t, self.model.get_anisotropy()
            )

        elif type(self.model) == models.BouwerRice:
            values, _ = functions.bouwer_rice(
                self.model.aquifer.D, self.model.well.depth + self.model.well.L,
                self.model.well.L, self.model.well.rs, self.model.well.rc,
                s0, sobs, t, self.model.get_anisotropy()
            )

        else:
            raise Exception('Linear regression is applicable only to Hvorslev and Bouwer & Rice models!')

        # set derived parameter values
        if np.isscalar(values):
            values = [values]
        self.set_values(np.array(values))

        # run model with the optimal values
        self.model.run()

    def plot(self, plot_function_s='semilogy', plot_function_eta='plot'):
        plt, h = super().plot(plot_function_s, plot_function_eta)
        return plt, h
