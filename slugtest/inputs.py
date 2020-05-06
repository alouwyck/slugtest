class Aquifer:

    def __init__(self, confined, D, Kh, Kv, Ss):
        self.confined = confined
        self.D = D
        self.Kh = Kh
        self.Kv = Kv
        self.Ss = Ss
        self.well = []

    def add_well(self, depth, L, rs, rc):
        well = Well(self, depth, L, rs, rc)
        self.well.append(well)
        return well


class Well:

    def __init__(self, aquifer, depth, L, rs, rc):
        self.aquifer = aquifer
        self.depth = depth
        self.L = L
        self.rs = rs
        self.rc = rc
        self.skin = None
        self.test = []

    def add_skin(self, r, Kh, Kv, Ss):
        self.skin = Skin(self, r, Kh, Kv, Ss)

    def add_test(self, s0, t, sobs=None):
        test = SlugTest(self, s0, t, sobs)
        self.test.append(test)
        return test


class Skin:

    def __init__(self, well, r, Kh, Kv, Ss):
        self.well = well
        self.r = r
        self.Kh = Kh
        self.Kv = Kv
        self.Ss = Ss


class SlugTest:

    def __init__(self, well, s0, t, sobs=None):
        self.aquifer = well.aquifer
        self.well = well
        self.s0 = s0
        self.t = t
        self.sobs = sobs