from numpy import pi, abs, ceil
from numpy import array, zeros, arange, dot
from numpy import newaxis as na
from numpy.linalg import norm, inv
from time import perf_counter
import logging

# requires numba 0.49.1 or later to iterate over arrays
from numba import jit
from multiprocessing import Pool

# physical constants and unit conversions
from constants import *

r"""Excitation
                                       |  \                 /             
                                       |   \_             _/    
        p4 \                / p3 ------|-> k'\__       __/                     
            \              /          e|        \_____/                        
             \            /           n|                    
              \  q --->  /            e|                    
              /\/\/\/\/\/\            r|        _____                          
             /            \           g|     __/     \__                       
            /              \          y|   _/           \_                     
        p1 /                \ p2 <-----|-k/               \                    
                                       |_/_________________\_____   
                                               momentum

python requirements:
* python 3.8 or later to avoid multiprocess struct.error
* python 3.9 or later to avoid numba error

DFT Requirements:
* Coefficients C^n_{G+k} are normalized so that \sum_G |C^n_{G+k}|^2 = 1
* Does not work if spin breaks degeneracy.
* All valence bands must be filled in ground state.

Notes:
p(E) = ((m**2 + p**2)**.5 - m**2)**.5
E(p) = (m**2 + m**2)**.5 - m
P(E=200 eV) = 14298 eV
p(E=600 eV) = 24770 eV (24763 eV nonrelativistially)
p(E=80 keV) = 296917 eV (285937 eV nonrelativistially)

1 hBN RLV = 5727 eV
1 MoS2 RLV = 4498 eV
E(1 hBN RLV) = 32.087 eV
E(1 MoS2 RLV) = 19.792 eV

Subscripts x, y, and z are used in place of reciprocal lattice indices 1, 2,
and 3 for legibility (though they do not generally denote the x, y, or z
directions)

to do:
* use logging
* work for arbitrary k-point mesh (need to fix readExport first)
"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# formatter = logging.Formatter('')

file_handler = logging.FileHandler('moller.log')
# file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

#-------------------------- CALCULATE PROBABILITY -----------------------------

def getProbA6(
        C_a6,
        p_a6,  # eV
        E_a3,  # eV
        k_a3,  # eV
        b_a2,  # eV
        area,  # eV^{-2}
        ne,
        q0max = 10,  # eV
        Eb = 6e4,  # eV
        nprocs = 8,
        ):
    """Returns array of excitation probabilitiies.

    * considers valence and conduction band pair for each k-point pair
    * prob_a6[k2x, k2y, k3x, k3y, v, c] --> prob of |vk2> --> |ck3>
    * must divide by area corresponding to smallest q
    * assumes uniform k-point mesh. No reduction by symmetry!

    C_a6[kx, ky, n, Kx, Ky, Kz] --> C^n_{K+k}
    p_a6[kx, ky, Kx, Ky, Kz, i] --> p^i (eV)
    E_a3[kx, ky, n] --> E_nk (eV)
    k_a3[kx, ky] --> k_ar (eV)
    b_a2: reciprocal lattice vectors in eV (3x3 array dtype=float)
    area: perpendicular unit cell area in eV^{-2} (float)
    ne: number of electrons (pos int)
    q0max: maximum virtual photon energy in eV considered (pos float)
        * Truly a misnomer: This is not the actual virtual photon energy, but
        the kinetic energy that would be associated with a an electron with the
        same momentum, much like cutoff energy in DFT calculations
        * ignores difference in eigenvalues
        * only care about momentum since change in energy is miniscule
        * doesn't remove "diagonal" terms, so all 8 adjacent BZs are included
          in summation, but still helps
    Eb: beam energy in eV (pos float)
    nprocs: number of cpus for Pool (pos int)
    """
    startTime = perf_counter()
    Cc_a6 = C_a6.conjugate()

    # RLV magnitudes
    bx, by, bz = norm(b_a2, axis=0)  # eV
    invb_a2 = inv(b_a2)  # eV^{-1}
    qmax = ((m + q0max)**2 - m**2)**.5  # eV

    # integer values
    nkx, nky, nb, nKx, nKy, nKz = C_a6.shape
    nk = nkx*nky
    nv = int(round(ne/2))
    nc = nb - nv

    # beam momentum
    E1 = Eb + m  # eV
    gamma = E1/m
    p1z = (E1**2 - m**2)**.5  # eV

    # overlaps for chi
    Qxmax = int(ceil(qmax/bx))
    Qymax = int(ceil(qmax/by))

    # time _getProbA2.
    getProbA2_time = _time_getProbA2(
            nv, nc,
            p_a6[1,1], p_a6[1,1], 
            C_a6[1,1], Cc_a6[1,1],
            E_a3[1,1], E_a3[1,1],
            arange(nKx)[:,na], arange(nKy)[:,na], nKz,
            p1z, E1, gamma,
            )

    # use function that optimizes parallelization
    if getProbA2_time > 0.1:
        getListFunct = _getProbA2List
        print('using _getProbA2')
    else:
        getListFunct = _getProbA4List
        print('using _getProbA4')

    prob_list = getListFunct(
            nkx, nky, nc, nv, nKx, nKy, nKz,
            Qxmax, Qymax, qmax,
            p_a6, C_a6, Cc_a6, k_a3, E_a3,
            p1z, E1, gamma,
            bx, by, invb_a2, 
            nprocs,
            )

    # reshape array so that indices correspond to transitions
    prob_a6 = array(prob_list).reshape((nkx, nky, nkx, nky, nv, nc))

    print('prob_a6 calculation time = {:.5g} seconds'
            .format(perf_counter() - startTime))

    return prob_a6 / nk**2 / area**2


def _getProbA2List(
        nkx, nky, nc, nv, nKx, nKy, nKz,
        Qxmax, Qymax, qmax,
        p_a6, C_a6, Cc_a6, k_a3, E_a3,
        p1z, E1, gamma,
        bx, by, invb_a2, 
        nprocs,
        ):
    """Returns list of rank 2 arrays."""
    # holds arguments of _getProbA2 for multiprocess Pool
    args_list = []

    # loop over incoming and outgoing k-points
    for k2x in range(nkx):
        for k2y in range(nky):
            p2_a4 = p_a6[k2x, k2y]
            k2_ar = k_a3[k2x, k2y]
            C2_a4 = C_a6[k2x, k2y]
            eig2_ar = E_a3[k2x, k2y]

            for k3x in range(nkx):
                for k3y in range(nky):
                    p3_a4 = p_a6[k3x, k3y]
                    k3_ar = k_a3[k3x, k3y]
                    Cc3_a4 = Cc_a6[k3x, k3y]
                    eig3_ar = E_a3[k3x, k3y]

                    # only consider virtual photons with momenta less than qmax
                    dkx, dky, dkz = dot(k3_ar - k2_ar, invb_a2)  # RLV
                    Qxmin = int(qmax/bx + dkx)
                    Qxmax = int(qmax/bx - dkx)
                    Qymin = int(qmax/by + dky)
                    Qymax = int(qmax/by - dky)

                    K3x_a2 = (arange(-Qxmin, Qxmax+1) + arange(nKx)[:,na])%nKx
                    K3y_a2 = (arange(-Qymin, Qymax+1) + arange(nKy)[:,na])%nKy

                    # get probability of transition
                    args_list.append((
                            nv, nc,
                            p2_a4, p3_a4,
                            C2_a4, Cc3_a4,
                            eig2_ar, eig3_ar,
                            K3x_a2, K3y_a2, nKz,
                            p1z, E1, gamma,
                            ))

    # calculate probabilities in parallel
    with Pool(nprocs) as p:
        probA2_list = p.starmap(_getProbA2, args_list)

    return probA2_list


@jit(nopython=True)
def _getProbA2(
        nv, nc,
        p2_a4, p3_a4,
        C2_a4, Cc3_a4,
        eig2_ar, eig3_ar,
        K3x_a2, K3y_a2, nKz,
        p1z, E1, gamma,
        ):
    """Returns excitation probability.

    Result must be divided by nk**2 and area**2
    """
    prob_a2 = zeros((nv, nc))

    for v in range(nv):
        for c_ in range(nc):
            c = c_ + nv

            # determine d\sigma and K3z for (k2, k3)
            amp = 0.+0.j

            for K2x, K3x_ar in enumerate(K3x_a2):
                for K2y, K3y_ar in enumerate(K3y_a2):
                    for K2z in range(nKz):

                        # skip if no coefficient
                        Cv = C2_a4[v, K2x, K2y, K2z]
                        if Cv==0.+0.j:
                            continue

                        p2x, p2y, p2z = p2_a4[K2x, K2y, K2z]
                        eig2 = eig2_ar[v]
                        E2 = eig2 + m
                        
                        for K3x in K3x_ar:
                            for K3y in K3y_ar:

                                # skip if no coefficient
                                Cc = Cc3_a4[c, K3x, K3y, K2z]
                                if Cc==0.+0.j:
                                    continue

                                p3x, p3y = p3_a4[K3x, K3y, 0, :2]
                                eig3 = eig3_ar[c]
                                eigdiff = eig2 - eig3

                                # p3z(p1, p2, p3perp) section S3
                                p3z = p1z + p2z - (p1z**2
                                        -
                                        (p2x - p3x)**2 - (p2y - p3y)**2
                                        +
                                        2*gamma*m*eigdiff)**.5

                                # remaining constrained Es and ps
                                E3 = eig3 + m
                                E4 = E1 + eigdiff
                                p4x = p2x - p3x
                                p4y = p2y - p3y
                                p4z = p1z + p2z - p3z

                                q0 = E1 - E4  # eels energy loss = init - fin
                                u = q0**2 - p4x**2 - p4y**2 - (p1z - p4z)**2

                                # common factor in M terms
                                pre = ((E1+m)*(E2+m)*(E3+m)*(E4+m))**-.5

                                # u-channel from ornl/notes/noSpinSums.nb
                                M = -1 / u * (\
                                (E1 + m)*((E3 + m)*p2x + (E2 + m)*p3x)*p4x\
                                +(m*p2y + E3*(p2y + 1j*p2z) + 1j*m*p2z\
                                    +E2*p3y + m*p3y - 1j*(E2 + m)*p3z)\
                                *(1j*E4*p1z + 1j*m*p1z + E1*p4y\
                                    +m*p4y - 1j*(E1 + m)*p4z)\
                                -((E2 + m)*(E3 + m) + p2x*p3x\
                                    +(p2y + 1j*p2z)*(p3y - 1j*p3z))\
                                *((E1 + m)*(E4 + m) + p1z*(1j*p4y + p4z))\
                                +(E3*((-1j)*p2y + p2z) + E2*(1j*p3y + p3z)\
                                    +m*((-1j)*p2y + p2z + 1j*p3y + p3z))\
                                *(E4*p1z + E1*(1j*p4y + p4z)\
                                +m*(p1z + 1j*p4y + p4z)))

                                # factor resulting from wavepacket overlap
                                overlap = (E4*E3/E2/E1)**.5 \
                                        /abs(E3*p4z - E4*p3z)

                                # d\sigma divided by factorable constants
                                dsig = overlap * pre * M

                                # sum over pw coeff to determine prob
                                damp = Cv * Cc * dsig
                                amp += damp

            prob_a2[v, c_] = (amp * amp.conjugate()).real

    # (4\pi\alpha)^2, x2^2 t-u symmetry, x4^2 noSpinSums.nb
    # /2^2 beam and material ground states equal parts spin up and down
    # /4^2 overlap prefactor
    # spin degeneracy: S = sum of all probabilities x4
    return 16. * (pi * alpha)**2 * prob_a2


def _getProbA4List(
        nkx, nky, nc, nv, nKx, nKy, nKz,
        Qxmax, Qymax, qmax,
        p_a6, C_a6, Cc_a6, k_a3, E_a3,
        p1z, E1, gamma,
        bx, by, invb_a2, 
        nprocs,
        ):
    """Returns list of rank 4 arrays."""
    # holds arguments of _getProbA2 for multiprocess Pool
    args_list = []

    # loop over incoming and outgoing k-points
    for k2x in range(nkx):
        for k2y in range(nky):
            # get probability of transition
            args_list.append((
                    k2x, k2y,
                    nkx, nky, nc, nv, nKx, nKy, nKz,
                    Qxmax, Qymax, qmax,
                    p_a6, C_a6, Cc_a6, k_a3, E_a3,
                    p1z, E1, gamma,
                    bx, by, invb_a2, 
                    ))

    # calculate probabilities in parallel
    with Pool(nprocs) as p:
        probA4_list = p.starmap(_getProbA4, args_list)

    return probA4_list


@jit(nopython=True)
def _getProbA4(
        k2x, k2y,
        nkx, nky, nc, nv, nKx, nKy, nKz,
        Qxmax, Qymax, qmax,
        p_a6, C_a6, Cc_a6, k_a3, E_a3,
        p1z, E1, gamma,
        bx, by, invb_a2, 
        ):
    """Returns excitation probabilities.

    Result must be divided by nk**2 and area**2
    """
    prob_a4 = zeros((nkx, nky, nv, nc))

    p2_a4 = p_a6[k2x, k2y]
    k2_ar = k_a3[k2x, k2y]
    C2_a4 = C_a6[k2x, k2y]
    eig2_ar = E_a3[k2x, k2y]

    for k3x in range(nkx):
        for k3y in range(nky):
            p3_a4 = p_a6[k3x, k3y]
            k3_ar = k_a3[k3x, k3y]
            Cc3_a4 = Cc_a6[k3x, k3y]
            eig3_ar = E_a3[k3x, k3y]

            # only consider virtual photons with momenta less than qmax
            dkx, dky, dkz = dot(k3_ar - k2_ar, invb_a2)  # RLV
            Qxmin = int(qmax/bx + dkx)
            Qxmax = int(qmax/bx - dkx)
            Qymin = int(qmax/by + dky)
            Qymax = int(qmax/by - dky)

            # jit can't handle numpy.newaxis (na)
            K3x_a2 = (arange(-Qxmin, Qxmax+1) + arange(nKx).reshape(-1,1))%nKx
            K3y_a2 = (arange(-Qymin, Qymax+1) + arange(nKy).reshape(-1,1))%nKy

            for v in range(nv):
                for c_ in range(nc):
                    c = c_ + nv

                    # determine d\sigma and K3z for (k2, k3)
                    amp = 0.+0.j

                    for K2x, K3x_ar in enumerate(K3x_a2):
                        for K2y, K3y_ar in enumerate(K3y_a2):
                            for K2z in range(nKz):

                                # skip if no coefficient
                                Cv = C2_a4[v, K2x, K2y, K2z]
                                if Cv==0.+0.j:
                                    continue

                                p2x, p2y, p2z = p2_a4[K2x, K2y, K2z]
                                eig2 = eig2_ar[v]
                                E2 = eig2 + m
                                
                                for K3x in K3x_ar:
                                    for K3y in K3y_ar:

                                        # skip if no coefficient
                                        Cc = Cc3_a4[c, K3x, K3y, K2z]
                                        if Cc==0.+0.j:
                                            continue

                                        p3x, p3y = p3_a4[K3x, K3y, 0, :2]
                                        eig3 = eig3_ar[c]
                                        eigdiff = eig2 - eig3

                                        # p3z(p1, p2, p3perp) section S3
                                        p3z = p1z + p2z - (p1z**2
                                                -
                                                (p2x - p3x)**2 - (p2y - p3y)**2
                                                +
                                                2*gamma*m*eigdiff)**.5

                                        # remaining constrained Es and ps
                                        E3 = eig3 + m
                                        E4 = E1 + eigdiff
                                        p4x = p2x - p3x
                                        p4y = p2y - p3y
                                        p4z = p1z + p2z - p3z

                                        q0 = E1 - E4  # eels energy loss
                                        u = q0**2 - p4x**2 - p4y**2 \
                                                - (p1z - p4z)**2

                                        # common factor in M terms
                                        pre = ((E1+m)*(E2+m) \
                                                *(E3+m)*(E4+m))**-.5

                                        # u-channel (ornl/notes/noSpinSums.nb)
                                        M = -1 / u * (\
                                        (E1 + m)*((E3 + m)*p2x\
                                            + (E2 + m)*p3x)*p4x\
                                        +(m*p2y + E3*(p2y + 1j*p2z)\
                                            + 1j*m*p2z + E2*p3y + m*p3y\
                                            - 1j*(E2 + m)*p3z)\
                                        *(1j*E4*p1z + 1j*m*p1z + E1*p4y\
                                            +m*p4y - 1j*(E1 + m)*p4z)\
                                        -((E2 + m)*(E3 + m) + p2x*p3x\
                                            +(p2y + 1j*p2z)*(p3y - 1j*p3z))\
                                        *((E1 + m)*(E4 + m)\
                                            +p1z*(1j*p4y + p4z))\
                                        +(E3*((-1j)*p2y + p2z)\
                                            +E2*(1j*p3y + p3z)\
                                            +m*((-1j)*p2y + p2z\
                                            + 1j*p3y + p3z))\
                                        *(E4*p1z + E1*(1j*p4y + p4z)\
                                        +m*(p1z + 1j*p4y + p4z)))

                                        # factor from wavepacket overlap
                                        overlap = (E4*E3/E2/E1)**.5 \
                                                /abs(E3*p4z - E4*p3z)

                                        # d\sigma div by factorable constants
                                        dsig = overlap * pre * M

                                        # sum over pw coeff to determine prob
                                        damp = Cv * Cc * dsig
                                        amp += damp

                        prob_a4[k3x, k3y, v, c_] = (amp * amp.conjugate()).real

    # (4\pi\alpha)^2, x2^2 t-u symmetry, x4^2 noSpinSums.nb
    # /2^2 beam and material ground states equal parts spin up and down
    # /4^2 overlap prefactor
    # spin degeneracy: S = sum of all probabilities x4
    return 16. * (pi * alpha)**2 * prob_a4


def _time_getProbA2(
        nv, nc,
        p2_a4, p3_a4,
        C2_a4, Cc3_a4,
        eig2_ar, eig3_ar,
        K3x_a2, K3y_a2, nKz,
        p1z, E1, gamma,
        ):
    """Time single call of _getProbA2."""
    #run twice to allow jit to compile
    _getProbA2(
        nv, nc,
        p2_a4, p3_a4,
        C2_a4, Cc3_a4,
        eig2_ar, eig3_ar,
        K3x_a2, K3y_a2, nKz,
        p1z, E1, gamma,
    )

    t0 = perf_counter()
    _getProbA2(
        nv, nc,
        p2_a4, p3_a4,
        C2_a4, Cc3_a4,
        eig2_ar, eig3_ar,
        K3x_a2, K3y_a2, nKz,
        p1z, E1, gamma,
    )

    _getProbA2_time = perf_counter() - t0
    print(f'_getProbA2_time = {_getProbA2_time:.5g} seconds')
    return _getProbA2_time

#---------------------------------- SCRATCH -----------------------------------

#     print(f'nprocs just before Pool = {nprocs}')
#             for Kx in range(nKx):
#                 K3x_a2 = (arange(-Qxmin, Qxmax + 1) + Kx)%nKx
#                 K3x_l2.append(K3x_ar)
#             K3x_a2 = array(K3x_l2)
#             K3x_a2 = array([(arange(-Qxmin, Qxmax + 1) + Kx)%nKx
#                     for Kx in range(nKx)])
#             K3y_a2 = array([(arange(-Qymin, Qymax + 1) + Ky)%nKy
#                     for Ky in range(nKy)])

