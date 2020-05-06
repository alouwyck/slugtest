import numpy as np
from scipy.special import factorial, k0, k1


def hvorslev(L, rw, rc, Re, y0, y_or_K, t, alpha=None):

    # y or K?
    forward = np.isscalar(y_or_K)
    if forward:
        K = y_or_K
    else:
        y = y_or_K

    # correct rw using alpha = sqrt(Kv/Kh)
    if alpha is not None:
        rw = rw * alpha  # see Butler (1998)

    # forward simulation: return y
    if forward:
        y_or_K = y0 * np.exp(-2 * K * L / rc**2 / np.log(Re / rw) * t)  # y_or_K = y

    # linear regression to derive K: return K
    elif y0 is not None:

        # K = rc^2 * log(Re/rw)/2/L *1./t .* log(y0./y)
        # => log(y/y0) = -2*K*L/rc^2/log(Re/rw) * t = a*t
        lnyy0 = np.log(y / y0)
        a, _, _, _ = np.linalg.lstsq(t[:, np.newaxis], lnyy0, rcond=None)
        y_or_K = -a.item() / 2 / L * rc**2 * np.log(Re / rw)  # y_or_K = K

    # linear regression to derive K and y0: return K and y0
    else:  # y0 is None

        # K = rc^2 * log(Re/rw)/2/L *1./t .* log(y0./y)
        # => log(y) - log(y0) = -2*K*L/rc^2/log(Re/rw) * t
        # => log(y) = log(y0) -2*K*L/rc^2/log(Re/rw) * t = b + a*t
        a_b = np.polyfit(t, np.log(y), 1)
        K = -a_b[0] / 2 / L * rc**2 * np.log(Re / rw)
        y0 = np.exp(a_b[1])
        y_or_K = [K, y0]

    # output
    return y_or_K


def bouwer_rice(D, H, L, rw, rc, y0, y_or_K, t, alpha=None):

    # y or K?
    forward = np.isscalar(y_or_K)
    if forward:
        K = y_or_K
    else:
        y = y_or_K

    # correct rw using alpha = sqrt(Kv/Kh)
    if alpha is not None:
        rw = rw * alpha  # see Butler (1998)

    # empirical graphs (Lrw = L/rw) according to Van Rooy(1988) - see Butler(1998)
    A = lambda Lrw: 1.4720 + 3.537e-2 * Lrw - 8.148e-5 * Lrw**2 + 1.028e-7 * Lrw**3 \
                    - 6.484e-11 * Lrw**4 + 1.573e-14 * Lrw**5
    B = lambda Lrw: 0.2372 + 5.151e-3 * Lrw - 2.682e-6 * Lrw**2 - 3.491e-10 * Lrw**3 \
                    + 4.738e-13 * Lrw**4
    C = lambda Lrw: 0.7920 + 3.993e-2 * Lrw - 5.743e-5 * Lrw**2 + 3.858e-8 * Lrw**3 \
                    - 9.659e-12 * Lrw**4

    # calculate ln(R/rw) = lnRerw
    Lrw = L / rw
    if np.abs(H - D) < 1e-5:  # fully penetrating
        lnRerw = 1 / (1.1 / np.log(H / rw) + C(Lrw) / Lrw)
    else:  # partially penetrating
        lnRerw = 1 / (1.1 / np.log(H / rw) +
                      (A(Lrw) + B(Lrw) * np.log((D - H) / rw)) / Lrw)

    # effective radius R
    Re = np.exp(lnRerw) * rw

    # forward simulation: return y
    if forward:

        # log(y/y0) = -2*K*L/rc^2/lnRerw*t
        # => y = y0*exp(-2*K*L/rc^2/lnRerw*t)
        y_or_K = y0 * np.exp(-2 * K * L / rc**2 / lnRerw * t)  # y_or_K = y

    # linear regression to derive K: return K
    elif y0 is not None:

        # K = rc^2 * lnRerw/2/L *1./t .* log(y0./y)
        # => log(y/y0) = -2*K*L/rc^2/lnRerw * t = a*t
        lnyy0 = np.log(y / y0)
        a, _, _, _ = np.linalg.lstsq(t[:, np.newaxis], lnyy0, rcond=None)
        y_or_K = -a.item() / 2 / L * rc**2 * lnRerw  # y_or_K = K

    # linear regression to derive K and y0: return K and y0
    else:  # y0 is None

        # K = rc^2 * lnRerw/2/L *1./t .* log(y0./y)
        # => log(y) - log(y0) = -2*K*L/rc^2/lnRerw * t
        # => log(y) = log(y0) -2*K*L/rc^2/lnRerw * t = b + a*t
        a_b = np.polyfit(t, np.log(y), 1)
        K = -a_b[0] / 2 / L * rc**2 * lnRerw
        y0 = np.exp(a_b[1])
        y_or_K = [K, y0]

    # output
    return y_or_K, Re


def cooper(H0, B, Kr, Ss, rw, rc, t, ns=16):
    rsw = lambda p: rw * np.sqrt(p * Ss / Kr)
    Hp = lambda p: H0 * k0(rsw(p)) / (2 * rsw(p) / rc/rc * Kr * B * k1(rsw(p))
                                      + p * k0(rsw(p)))
    return stehfest(Hp, t, ns)


def stehfest(F, t, ns):

    def W(j):
        ns2 = ns // 2
        m = min(j, ns2)
        k0 = np.floor((j + 1) / 2)
        w = 0
        for k in np.arange(k0, m+1):
            w += k**ns2 * factorial(2*k, exact=True) / factorial(ns2-k, exact=True) \
                 / factorial(k, exact=True) / factorial(k-1, exact=True) \
                 / factorial(j-k, exact=True) / factorial(2*k-j, exact=True)
        w *= (-1)**(ns2 + j)
        return w

    ln2t = np.log(2) / t
    f = 0
    for i in range(1, ns+1):
        f += W(i) * F(i * ln2t)
    f *= ln2t
    return f


def kgs_no_skin(t, H0,
                rw, rc,
                B, b, d, confined,
                Kr, Kz, Ss,
                ns=16, maxerr=1e-6, miniter=10, maxiter=500, htol=1e-5):

    # dimensionless parameters
    beta = B / b
    zeta = d / b
    rw2 = rw * rw
    rc2 = rc * rc
    A = Kz / Kr
    a = b * b / rw2
    psi = A / a
    R = rw2 * Ss * b / rc2
    tau = t * b * Kr / rc2
    nt = len(t)
    phreatic = not confined
    pi2 = np.pi * np.pi
    o2 = pi2 / beta / beta

    # stehfest parameters
    ns2 = ns // 2
    ln2t = np.log(2) / tau  # len(ln2t) = nt

    # Laplace parameter p and stehfest weights w
    # p.shape = w.shape = (nt, ns)
    p = np.empty((nt, ns))
    w = np.empty(ns)
    for j in range(1, ns+1):
        p[:, j-1] = j * ln2t
        k = np.arange(np.floor((j + 1) / 2), min(j, ns2) + 1)
        w[j-1] = np.sum(k**ns2 * factorial(2*k, exact=True) /
                        factorial(ns2-k, exact=True) / factorial(k, exact=True) /
                        factorial(k-1, exact=True) / factorial(j-k, exact=True) /
                        factorial(2*k-j, exact=True)) * \
                 (-1)**(ns2 + j)
    w = np.tile(w, (nt, 1))

    # f1 and phi
    u = 4 * beta / pi2
    c1 = np.pi / 2 / beta
    if phreatic:
        f0 = 0
        u = 4 * u
        c1 = c1 / 2
        o2 = o2 / 4
        sin_cos = np.sin
    else:
        nu0 = np.sqrt(R * p)
        f0 = k0(nu0) / k1(nu0) / nu0 / beta / 2  # f0.shape = (nt, ns)
        sin_cos = np.cos
    c2 = c1 * (1 + 2 * zeta)

    def f1(n):
        # n is scalar
        # f.shape = (nt, ns)
        n2 = n * n
        nu = np.sqrt(psi * o2 * n2 + R * p)
        f = k0(nu) / n2 / k1(nu) / nu * \
            np.sin(n * c1)**2 * sin_cos(n * c2)**2
        return f

    def phi(f):
        # f.shape = (nt, ns)
        omega = f0 + f * u
        f = omega / (1 + omega * p)
        return f

    # back transform to obtain head change
    f = f1(1)
    h = np.sum(w * phi(f), axis=1) * ln2t  # len(h) = nt
    err = np.Inf
    i = 1
    n = 2 + phreatic
    while (i < miniter or err > maxerr) and i < maxiter:
        df = f1(n)
        if np.nanmax(np.abs(df)) > 1e-16:  # f1 can be periodically equal to zero
            f = f + df  # size(f) = [nt, ns]
            hnew = np.sum(w * phi(f), axis=1) * ln2t  # len(hnew) = nt
            ok = ~np.isnan(hnew)
            err = max(np.abs(h[ok] - hnew[ok]))
            h[ok] = hnew[ok]
        i += 1
        n += 1 + phreatic

    # set normalized head smaller than given tolerance to zero
    # and denormalize
    h[h < htol] = 0
    h = h * H0

    return h, i, err

