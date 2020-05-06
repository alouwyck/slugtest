from abc import ABC, abstractmethod


class Base(ABC):

    def __init__(self):
        self.ini_value = None

    @abstractmethod
    def get_value(self):
        # must return value
        pass

    @abstractmethod
    def set_value(self, *args):
        # no return
        pass

    @abstractmethod
    def reset_value(self):
        # no return
        pass


class LogTransformable:

    def __init__(self):
        self.log = False


class Single(Base):

    def __init__(self, obj, name, ini_value=None):
        super().__init__()
        self.object = obj
        self.name = name
        if ini_value is None:
            self.ini_value = self.get_value()
        else:
            self.ini_value = ini_value
            self.reset_value()

    def get_value(self):
        return getattr(self.object, self.name)

    def reset_value(self):
        setattr(self.object, self.name, self.ini_value)


class Parameter(Single, LogTransformable):

    def __init__(self, obj, name, ini_value=None):
        Single.__init__(self, obj, name, ini_value)
        LogTransformable.__init__(self)

    def set_value(self, value):
        setattr(self.object, self.name, value)


class Composed(Base, LogTransformable):

    def __init__(self, independent, dependent, func, inv_func):
        Base.__init__(self)
        LogTransformable.__init__(self)
        self.independent = independent
        self.dependent = dependent
        self.func = func
        self.inv_func = inv_func
        self.ini_value = func(independent.ini_value, dependent.ini_value)

    def get_value(self):
        return self.func(self.independent.get_value(), self.dependent.get_value())

    def set_value(self, value):
        self.dependent.set_value(
            self.inv_func(self.independent.get_value(), value)
        )

    def reset_value(self):
        self.independent.reset_value()
        self.dependent.reset_value()
        self.ini_value = self.func(self.independent.ini_value, self.dependent.ini_value)


class Dependent(Single):

    def __init__(self, parameter, obj, name, ini_value=None):
        # factor = dependent / parameter
        super().__init__(obj, name, ini_value)
        self.parameter = parameter
        self.factor = self.ini_value / parameter.ini_value

    def set_value(self):
        # dependent = factor * parameter
        setattr(self.object, self.name, self.factor * self.parameter.get_value())


class InverseDependent(Dependent):

    def __init__(self, parameter, obj, name, ini_value=None):
        # factor = dependent * parameter
        super().__init__(parameter, obj, name, ini_value)
        self.factor = self.ini_value * self.parameter.ini_value

    def set_value(self):
        # dependent = factor / parameter
        setattr(self.object, self.name, self.factor / self.parameter.get_value())

