

class ManueleInterpretatie:

    def __init__(self, opt):
        self.opt = opt
        self.model = opt.model
        self.aquifer = opt.model.aquifer
        self.well = opt.model.well
        self.test = opt.model.test

    def set_model_input(self, confined, D, depth, L, rs, rc):
        self.aquifer.confined = confined
        self.aquifer.D = D
        self.well.depth = depth
        self.well.L = L
        self.well.rs = rs
        self.well.rc = rc

    def run(self, Kh=None, Ss=None, s0=None, Ani=None):

        if Kh is not None:
            self.aquifer.Kv *= Kh / self.aquifer.Kh
            self.aquifer.Kh = Kh
        if Ss is not None:
            self.aquifer.Ss = Ss
        if s0 is not None:
            self.test.s0 = s0
        if Ani is not None:
            self.aquifer.Kv = self.aquifer.Kh * Ani

        self.model.run()

        print("RMSE = " + str(self.opt.get_rmse()))

        plt, h = self.opt.plot()
        plt.setp(h['s'], color="red", linewidth=3, zorder=5)
        h['fig'].show()
        h['fig'].canvas.draw()

        return {'Kh': self.aquifer.Kh,
                'Ss': self.aquifer.Ss,
                's0': self.test.s0,
                'Ani': self.aquifer.Kv / self.aquifer.Kh}