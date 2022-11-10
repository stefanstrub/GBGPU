import time
import warnings
from abc import ABC

import numpy as np

YEAR = 31457280.0
Larm = 2.5e9
Clight = 299792458.
kappa = 0.000000
lambda0 = 0.000000
fm = 3.168753578738106e-08
AU = 1.49597870660e11
ec = 0.004824185218078991

# import for GPU if available
try:
    import cupy as xp
except (ModuleNotFoundError, ImportError):
    import numpy as xp

class GBGPU(object):
    """Generate Galactic Binary Waveforms

    This class generates galactic binary waveforms in the frequency domain,
    in the form of LISA TDI channels X, A, and E. It generates waveforms in batches.
    It can also provide injection signals and calculate likelihoods in batches.
    These batches are run on GPUs or CPUs. When CPUs are used, all available threads
    are leveraged with OpenMP. To adjust the available threads, use ``OMP_NUM_THREADS``
    environmental variable or :func:`gbgpu.utils.set_omp_num_threads`.

    This class can generate waveforms for two different types of GB sources:

        * Circular Galactic binaries
        * Circular Galactic binaries with an eccentric third body

    The class determines which waveform is desired based on the number of argmuments
    input by the user (see the ``*args`` description below).

    Args:
        use_gpu (bool, optional): If True, run on GPUs. Default is ``False``.

    Attributes:
        xp (obj): NumPy if on CPU. CuPy if on GPU.
        use_gpu (bool): Use GPU if True.
        get_basis_tensors (obj): Cython function.
        GenWave (obj): Cython function.
        GenWaveThird (obj): Cython function.
        unpack_data_1 (obj): Cython function.
        XYZ (obj): Cython function.
        get_ll_func (obj): Cython function.
        num_bin (int): Number of binaries in the current calculation.
        N_max (int): Maximum points in a waveform based on maximum harmonic mode considered.
        start_inds (list of 1D int xp.ndarray): Start indices into data stream array. q - N/2.
        df (double): Fourier bin spacing.
        X_out, A_out, E_out (1D complex xp.ndarrays): X, A, or E channel TDI templates.
            Each array is a 2D complex array
            of shape (number of points, number of binaries) that is flattened. These can be
            accessed in python with the properties ``X``, ``A``, ``E``.
        N (int): Last N value used.
        d_d (double): <d|d> term in the likelihood.

    """

    def __init__(self, use_gpu=False):

        self.use_gpu = use_gpu

        # setup Cython/C++/CUDA calls based on if using GPU
        if self.use_gpu:
            self.xp = xp
        else:
            self.xp = np
        self.d_d = None

    @property
    def citation(self):
        """Get citations for this class"""
        return zenodo + cornish_fastb + robson_triple

    def run_wave(
        self,
        amp,
        f0,
        fdot,
        fddot,
        phi0,
        iota,
        psi,
        lam,
        beta,
        *args,
        N=None,
        T=4 * YEAR,
        dt=10.0,
        oversample=1,
        tdi2=False,
    ):
        """Create waveforms in batches.

        This call creates the TDI templates in batches.

        The parameters and code below are based on an implementation of Fast GB
        in the LISA Data Challenges' ``ldc`` package.

        This class can be inherited to build fast waveforms for systems
        with additional astrophysical effects.

        # TODO: add citation property

        Args:
            amp (double or 1D double np.ndarray): Amplitude parameter.
            f0 (double or 1D double np.ndarray): Initial frequency of gravitational
                wave in Hz.
            fdot (double or 1D double np.ndarray): Initial time derivative of the
                frequency given as Hz/s.
            fddot (double or 1D double np.ndarray): Initial second derivative with
                respect to time of the frequency given in Hz/s^2.
            phi0 (double or 1D double np.ndarray): Initial phase angle of gravitational
                wave given in radians.
            iota (double or 1D double np.ndarray): Inclination of the Galactic binary
                orbit given in radians.
            psi (double or 1D double np.ndarray): Polarization angle of the Galactic
                binary orbit in radians.
            lam (double or 1D double np.ndarray): Ecliptic longitutude of the source
                given in radians.
            beta (double or 1D double np.ndarray): Ecliptic Latitude of the source
                given in radians. This is converted to the spherical polar angle.
            *args (tuple, optional): Flexible parameter to allow for a flexible
                number of argmuments when inherited by other classes.
                If running a circular Galactic binarys, ``args = ()``.
                If ``len(args) != 0``, then the inheriting class must have a
                ``prepare_additional_args`` method.
            N (int, optional): Number of points in waveform.
                This should be determined by the initial frequency, ``f0``. Default is ``None``.
                If ``None``, will use :func:`gbgpu.utils.utility.get_N` function to determine proper ``N``.
            T (double, optional): Observation time in seconds. Default is ``4 * YEAR``.
            dt (double, optional): Observation cadence in seconds. Default is ``10.0`` seconds.
            oversample(int, optional): Oversampling factor compared to the determined ``N``
                value. Final N will be ``oversample * N``. This is only used if N is
                not provided. Default is ``1``.
            tdi2 (bool, optional): If ``True``, produce the TDI channels for TDI 2nd-generation.
                If ``False``, produce TDI 1st-generation. Technically, the current TDI computation
                is not valid for generic LISA orbits, which are dealth with with 2nd-generation TDI,
                only those with an "equal-arm length" condition. Default is ``False``.

            Raises:
                ValueError: Length of ``*args`` is not 0 or 5.

        """

        # get number of observation points and adjust T accordingly
        N_obs = int(T / dt)
        T = N_obs * dt

        # if given scalar parameters, make sure at least 1D
        amp = np.atleast_1d(amp)
        f0 = np.atleast_1d(f0)
        fdot = np.atleast_1d(fdot)
        fddot = np.atleast_1d(fddot)
        phi0 = np.atleast_1d(phi0)
        iota = np.atleast_1d(iota)
        psi = np.atleast_1d(psi)
        lam = np.atleast_1d(lam)
        beta = np.atleast_1d(beta)

        # if circular base
        if len(args) == 0:
            add_args = ()

        else:
            if not hasattr(self, "prepare_additional_args"):
                raise ValueError(
                    "If providing more args than the base args, must be a class derived from GBGPU that has a 'prepare_additional_args' method."
                )

            add_args = self.prepare_additional_args(*args)

        # set N if it is not given based on timescales in the waveform
        if N is None:
            if hasattr(self, "special_get_N"):
                # take the original extra arguments
                N_temp = self.special_get_N(amp, f0, T, *args, oversample=oversample)
            else:
                N_temp = get_N(amp, f0, T, oversample=oversample)
            N = N_temp.max()

        # number of binaries is determined from length of amp array
        self.num_bin = num_bin = len(amp)

        # polar angle from ecliptic latitude
        theta = np.pi / 2 - beta

        # bin spacing
        self.df = df = 1 / T

        # instantiate GPU/CPU arrays

        # copy to GPU if needed
        amp = self.xp.asarray(amp.copy())
        f0 = self.xp.asarray(f0.copy())  # in mHz
        fdot = self.xp.asarray(fdot.copy())
        fddot = self.xp.asarray(fddot.copy())
        phi0 = self.xp.asarray(phi0.copy())
        iota = self.xp.asarray(iota.copy())
        psi = self.xp.asarray(psi.copy())
        lam = self.xp.asarray(lam.copy())
        theta = self.xp.asarray(theta.copy())

        cosiota = self.xp.cos(iota.copy())

        self.N = N

        # figure out start inds
        q_check = (f0 * T).astype(np.int32)
        #self.start_inds = (q_check - N / 2).astype(xp.int32)

        cosiota = self.xp.cos(iota)

        # transfer frequency
        fstar = Clight / (Larm * 2 * np.pi)

        cosps, sinps = self.xp.cos(2.0 * psi), self.xp.sin(2.0 * psi)

        Aplus = amp * (1.0 + cosiota * cosiota)
        Across = -2.0 * amp * cosiota

        DP = Aplus * cosps - 1.0j * Across * sinps
        DC = -Aplus * sinps - 1.0j * Across * cosps

        # sky location basis vectors
        sinth, costh = self.xp.sin(theta), self.xp.cos(theta)
        sinph, cosph = self.xp.sin(lam), self.xp.cos(lam)
        u = self.xp.array([costh * cosph, costh * sinph, -sinth]).T[:, None, :]
        v = self.xp.array([sinph, -cosph, self.xp.zeros_like(cosph)]).T[:, None, :]
        k = self.xp.array([-sinth * cosph, -sinth * sinph, -costh]).T[:, None, :]

        # polarization tensors
        eplus = self.xp.matmul(v.transpose(0, 2, 1), v) - self.xp.matmul(
            u.transpose(0, 2, 1), u
        )
        ecross = self.xp.matmul(u.transpose(0, 2, 1), v) + self.xp.matmul(
            v.transpose(0, 2, 1), u
        )

        # time points evaluated
        tm = self.xp.linspace(0, T, num=N, endpoint=False)

        # get the spacecraft positions from orbits
        Ps = self._spacecraft(tm)

        # time domain information
        Gs, q = self._construct_slow_part(
            T,
            Larm,
            Ps,
            tm,
            f0,
            fdot,
            fddot,
            fstar,
            phi0,
            k,
            DP,
            DC,
            eplus,
            ecross,
            *add_args,
        )

        # transform to TDI observables
        XYZf, f_min = self._computeXYZ(T, Gs, f0, fdot, fddot, fstar, amp, q, tm)

        self.start_inds = self.kmin = self.xp.round(f_min/df).astype(int)
        fctr = 0.5 * T / N

        # adjust for TDI2 if needed
        if tdi2:
            omegaL = 2 * np.pi * f0_out * (Larm / Clight)
            tdi2_factor = 2.0j * self.xp.sin(2 * omegaL) * self.xp.exp(-2j * omegaL)
            fctr *= tdi2_factor

        XYZf *= fctr

        # we do not care about T right now
        Af, Ef, Tf = AET(XYZf[:, 0], XYZf[:, 1], XYZf[:, 2])

        # setup waveforms for efficient GPU likelihood or global template building
        self.A_out = Af.T.flatten()
        self.E_out = Ef.T.flatten()

        self.X_out = XYZf[:, 0].T.flatten()

    def _computeXYZ(self, T, Gs, f0, fdot, fddot, fstar, ampl, q, tm):
        """Compute TDI X, Y, Z from y_sr"""

        # get true frequency as a function of time
        f = (
            f0[:, None]
            + fdot[:, None] * tm[None, :]
            + 1 / 2 * fddot[:, None] * tm[None, :] ** 2
        )

        # compute transfer function
        omL = f / fstar
        SomL = self.xp.sin(omL)
        fctr = self.xp.exp(-1.0j * omL)
        fctr2 = 4.0 * omL * SomL * fctr / ampl[:, None]

        # Notes from LDC below

        ### I have factored out 1 - exp(1j*omL) and transformed to
        ### fractional frequency: those are in fctr2
        ### I have rremoved Ampl to reduce dynamical range, will restore it later

        Xsl = Gs["21"] - Gs["31"] + (Gs["12"] - Gs["13"]) * fctr
        Ysl = Gs["32"] - Gs["12"] + (Gs["23"] - Gs["21"]) * fctr
        Zsl = Gs["13"] - Gs["23"] + (Gs["31"] - Gs["32"]) * fctr

        # time domain slow part
        XYZsl = fctr2[:, None, :] * self.xp.array([Xsl, Ysl, Zsl]).transpose(1, 0, 2)

        # frequency domain slow part
        XYZf_slow = ampl[:, None, None] * self.xp.fft.fft(XYZsl, axis=-1)

        # for testing
        # Xtry = 4.0*(self.G21 - self.G31 + (self.G12 - self.G13)*fctr)/self.ampl

        M = XYZf_slow.shape[2]  # len(XYZf_slow)
        XYZf = self.xp.fft.fftshift(XYZf_slow, axes=-1)

        # closest bin frequency
        f0 = (q - M / 2) / T  # freq = (q + self.xp.arange(M) - M/2)/T
        return XYZf, f0

    def _spacecraft(self, t):
        """Compute space craft positions as a function of time"""
        # kappa and lambda are constants determined in the Constants.h file

        # angular quantities defining orbit
        alpha = 2.0 * np.pi * fm * t + kappa

        beta1 = 0.0 + lambda0
        beta2 = 2.0 * np.pi / 3.0 + lambda0
        beta3 = 4.0 * np.pi / 3.0 + lambda0

        sa = self.xp.sin(alpha)
        ca = self.xp.cos(alpha)

        # output arrays
        P1 = self.xp.zeros((len(t), 3))
        P2 = self.xp.zeros((len(t), 3))
        P3 = self.xp.zeros((len(t), 3))

        # spacecraft 1
        sb = self.xp.sin(beta1)
        cb = self.xp.cos(beta1)

        P1[:, 0] = AU * ca + AU * ec * (sa * ca * sb - (1.0 + sa * sa) * cb)
        P1[:, 1] = AU * sa + AU * ec * (sa * ca * cb - (1.0 + ca * ca) * sb)
        P1[:, 2] = -SQ3 * AU * ec * (ca * cb + sa * sb)

        # spacecraft 2
        sb = self.xp.sin(beta2)
        cb = self.xp.cos(beta2)
        P2[:, 0] = AU * ca + AU * ec * (sa * ca * sb - (1.0 + sa * sa) * cb)
        P2[:, 1] = AU * sa + AU * ec * (sa * ca * cb - (1.0 + ca * ca) * sb)
        P2[:, 2] = -SQ3 * AU * ec * (ca * cb + sa * sb)

        # spacecraft 3
        sb = self.xp.sin(beta3)
        cb = self.xp.cos(beta3)
        P3[:, 0] = AU * ca + AU * ec * (sa * ca * sb - (1.0 + sa * sa) * cb)
        P3[:, 1] = AU * sa + AU * ec * (sa * ca * cb - (1.0 + ca * ca) * sb)
        P3[:, 2] = -SQ3 * AU * ec * (ca * cb + sa * sb)

        return [P1, P2, P3]

    def _construct_slow_part(
        self,
        T,
        arm_length,
        Ps,
        tm,
        f0,
        fdot,
        fddot,
        fstar,
        phi0,
        k,
        DP,
        DC,
        eplus,
        ecross,
        *add_args,
    ):
        """Construct the time-domain function for the slow part of the waveform."""

        # these are the orbits (equal-arm lengths assumed)
        P1, P2, P3 = Ps
        r = dict()

        # unit vectors of constellation arms
        r["12"] = (P2 - P1) / arm_length  ## [3xNt]
        r["13"] = (P3 - P1) / arm_length
        r["23"] = (P3 - P2) / arm_length
        r["31"] = -r["13"]

        # wave propagation axis dotted with constellation unit vectors
        kdotr = dict()
        for ij in ["12", "13", "23"]:
            kdotr[ij] = self.xp.dot(k.squeeze(), r[ij].T)  ### should be size Nt
            kdotr[ij[-1] + ij[0]] = -kdotr[ij]

        # wave propagation axis dotted with spacecraft positions
        kdotP = self.xp.array(
            [self.xp.dot(k, P1.T), self.xp.dot(k, P2.T), self.xp.dot(k, P3.T)]
        )[:, :, 0].transpose(1, 0, 2)

        kdotP /= Clight

        Nt = len(tm)

        # delayed time at the spacecraft
        xi = tm - kdotP

        # instantaneous frequency of wave at the spacecraft at xi
        fi = (
            f0[:, None, None]
            + fdot[:, None, None] * xi
            + 1 / 2.0 * fddot[:, None, None] * xi**2
        )

        if hasattr(self, "shift_frequency"):
            # shift is performed in place to save memory
            fi[:] = self.shift_frequency(fi, xi, *add_args)

        # transfer frequency ratio
        fonfs = fi / fstar  # Ratio of true frequency to transfer frequency

        # LDC notes with '###'
        ### compute transfer f-n
        q = np.rint(f0 * T)  # index of nearest Fourier bin
        df = 2.0 * np.pi * (q / T)
        om = 2.0 * np.pi * f0

        ### The expressions below are arg2_i with om*kR_i factored out
        A = dict()
        for ij in ["12", "23", "31"]:
            aij = (
                self.xp.dot(eplus, r[ij].T) * r[ij].T * DP[:, None, None]
                + self.xp.dot(ecross, r[ij].T) * r[ij].T * DC[:, None, None]
            )
            A[ij] = aij.sum(axis=1)

        # below is information from the LDC about matching the original LDC.
        # The current code matches the time-domain-generated tempaltes in the LDC.

        # These are wfm->TR + 1j*TI in c-code

        # arg2_1 = 2.0*np.pi*f0*xi[0] + phi0 - df*tm + np.pi*fdot*(xi[0]**2)
        # arg2_2 = 2.0*np.pi*f0*xi[1] + phi0 - df*tm + np.pi*fdot*(xi[1]**2)
        # arg2_3 = 2.0*np.pi*f0*xi[2] + phi0 - df*tm + np.pi*fdot*(xi[2]**2)

        ### These (y_sr) reproduce exactly the FastGB results
        # self.y12 = 0.25*np.sin(arg12)/arg12 * np.exp(1.j*(arg12 + arg2_1)) * ( Dp12*self.DP + Dc12*self.DC )
        # self.y23 = 0.25*np.sin(arg23)/arg23 * np.exp(1.j*(arg23 + arg2_2)) * ( Dp23*self.DP + Dc23*self.DC )
        # self.y31 = 0.25*np.sin(arg31)/arg31 * np.exp(1.j*(arg31 + arg2_3)) * ( Dp31*self.DP + Dc31*self.DC )
        # self.y21 = 0.25*np.sin(arg21)/arg21 * np.exp(1.j*(arg21 + arg2_2)) * ( Dp12*self.DP + Dc12*self.DC )
        # self.y32 = 0.25*np.sin(arg32)/arg32 * np.exp(1.j*(arg32 + arg2_3)) * ( Dp23*self.DP + Dc23*self.DC )
        # self.y13 = 0.25*np.sin(arg13)/arg13 * np.exp(1.j*(arg13 + arg2_1)) * ( Dp31*self.DP + Dc31*self.DC )

        ### Those are corrected values which match the time domain results.
        ## om*kdotP_i singed out for comparison with another code.

        argS = (
            phi0[:, None, None]
            + (om[:, None, None] - df[:, None, None]) * tm[None, None, :]
            + np.pi * fdot[:, None, None] * (xi**2)
            + 1 / 3 * np.pi * fddot[:, None, None] * (xi**3)
        )

        if hasattr(self, "add_to_argS"):
            # performed in place to save memory
            argS[:] = self.add_to_argS(argS, f0, fdot, fddot, xi, *add_args)

        kdotP = om[:, None, None] * kdotP - argS

        # get Gs transfer functions
        Gs = dict()
        for ij, ij_sym, s in [
            ("12", "12", 0),
            ("23", "23", 1),
            ("31", "31", 2),
            ("21", "12", 1),
            ("32", "23", 2),
            ("13", "31", 0),
        ]:

            arg_ij = 0.5 * fonfs[:, s, :] * (1 + kdotr[ij])
            Gs[ij] = (
                0.25
                * self.xp.sin(arg_ij)
                / arg_ij
                * self.xp.exp(-1.0j * (arg_ij + kdotP[:, s]))
                * A[ij_sym]
            )
        ### Lines blow are extractions from another python code and from C-code in LDC
        # y = -0.5j*self.omL*A*sinc(args)*np.exp(-1.0j*(args + self.om*kq))
        # args = 0.5*self.omL*(1.0 - kn)
        # arg12 = 0.5*fonfs[0,:] * (1 + kdotr12)
        # arg2_1 = 2.0*np.pi*f0*xi[0] + phi0 - df*tm + np.pi*self.fdot*(xi[0]**2)  -> om*k.Ri
        # arg1 = 0.5*wfm->fonfs[i]*(1. + wfm->kdotr[i][j])
        # arg2 =  PI*2*f0*wfm->xi[i] + phi0 - df*t
        # sinc = 0.25*sin(arg1)/arg1
        # tran1r = aevol*(wfm->dplus[i][j]*wfm->DPr + wfm->dcross[i][j]*wfm->DCr)
        # tran1i = aevol*(wfm->dplus[i][j]*wfm->DPi + wfm->dcross[i][j]*wfm->DCi)
        # tran2r = cos(arg1 + arg2)
        # tran2i = sin(arg1 + arg2)
        # wfm->TR[i][j] = sinc*(tran1r*tran2r - tran1i*tran2i)
        # wfm->TI[i][j] = sinc*(tran1r*tran2i + tran1i*tran2r)
        return Gs, q

    @property
    def X(self):
        """return X channel reshaped based on number of binaries"""
        return self.X_out.reshape(self.N, self.num_bin).T

    @property
    def A(self):
        """return A channel reshaped based on number of binaries"""
        return self.A_out.reshape(self.N, self.num_bin).T

    @property
    def E(self):
        """return E channel reshaped based on number of binaries"""
        return self.E_out.reshape(self.N, self.num_bin).T

    @property
    def freqs(self):
        """Return frequencies associated with each signal"""
        freqs_out = (
            self.xp.arange(self.N)[None, :] + self.start_inds[:, None]
        ) * self.df
        return freqs_out


