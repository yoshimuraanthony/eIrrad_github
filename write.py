from numpy import zeros, array, dot, floor, log10
from numpy.linalg import inv
from time import perf_counter
import sys

# include date on outfile file names
from datetime import date
from shutil import copyfile

# read quantum espresso output files
from quantumEspresso.read import readExport
from eIrrad.moller import getProbA6

# physical constants and unit conversions
from constants import *

"""Write probabilities from moller.py."""

def writeProb(
        prob_a6 = [],
        infile = 'input',
        exroot = 'tmp/hBN.export',
        headOutfile = 'newhead.txt',
        probOutfile = 'newprob.txt',
        q0max = 10,  # eV
        Eb = 6e4,  # eV
        nprocs = 8,
        ):
    """Writes prob_a6 to probOutfile.

    prob_a6[k2x, k2y, k3x, k3y, v, c] --> prob of |vk2> --> |ck3>
        * if prob_a6 not included, read infile for getProbA6 parameters
    infile: input file containing parameters for getProbA6 (str)
    headOutfile: file to which summary is written (str)
    probOutfile: file to which all transition probabilities are written (str)
    """
    startTime = perf_counter()

    # calculate prob_a6 if not provided
    if len(prob_a6)==0:

        # read input file for getProbA6 parameters
        try:
            with open(infile) as f:
                print(f'reading from {infile}')

                for line in f:
                    if 'Eb' in line:
                        Eb = float(line.split()[-1])
                        print(f'    Eb = {Eb:.5g} eV')

                    if 'q0max' in line:
                        q0max = float(line.split()[-1])
                        print(f'    q0max = {q0max:.5g} eV')

                    if 'exroot' in line:
                        exroot = line.split()[-1]
                        print(f'    exroot = {exroot}')
    
        except FileNotFoundError:
            pass

        # calculate all transition probabilities
        C_a6, p_a6, E_a3, k_a3, b_a2, ne, volume, area, encut = \
                readExport(exroot)
        prob_a6 = getProbA6(C_a6, p_a6, E_a3, k_a3, b_a2, area, ne, q0max, Eb,
                nprocs)

    # total probability is the likelihood of "missing" all transitions    
    noscatter = 1
    nkx, nky, _2, _3, nv, nc = prob_a6.shape
    nk = nkx*nky
    for k2x in range(nkx):
        for k2y in range(nky):
            for k3x in range(nkx):
                for k3y in range(nky):
                    for v in range(nv):
                        for c_ in range(nc):
                            prob = prob_a6[k2x, k2y, k3x, k3y, v, c_]
                            noscatter *= (1 - prob) 

    # spin degeneracy: square noscatter twice
    totProb = 1 - noscatter**4  # probability of exactly one excitation
    totSum = prob_a6.sum()*4  # S
    totTime = perf_counter() - startTime

    #  write relavent info that can affect probabilities
    nkx, nky, _2, _3, nv, nc = prob_a6.shape
    height = volume / area * invÅtoEV   # Å
    print(f'Sum of probabilities = {totSum:.5g}')
    print(f'number of cpus       = {nprocs}')
    print(f'total time           = {totTime:.5g} seconds')

    # write summary file                        
    print('writing summary to {}'.format(headOutfile))
    summary = \
        f'beam energy = {Eb:.10g} eV\n' \
        f'bands       = {nv+nc}\n' \
        f'val bands   = {nv}\n' \
        f'k-points    = {nk}\n' \
        f'area        = {area:.10g} eV^{-2}\n' \
        f'elec cutoff = {encut:.10g} eV\n' \
        f'phot cutoff = {q0max:.10g} eV\n' \
        f'cell-height = {height:.10g} Å\n' \
        f'total prob  = {totProb:.10g}\n' \
        f'total sum   = {totSum:.10g}\n' \
        f'total time  = {totTime:.10g} seconds\n' \
        f'cpus        = {nprocs}\n\n'

    with open(headOutfile, 'w') as f:
        f.write(summary)

    # make copy with date in name
    today = date.today()
    year = today.year - 2000
    month = today.month
    day = today.day

    if '.' in headOutfile:
        name, ext = headOutfile.split('.')
        headOutfileCopy = f'{name}{year:02d}{month:02d}{day:02d}.{ext}'
    else:
        headOutfileCopy = f'{headOutfile}{year:02d}{month:02d}{day:02d}'

    copyfile(headOutfile, headOutfileCopy)

    # write all transition probabilities
    print('writing all transition probabilities to {}'.format(probOutfile))
    with open(probOutfile, 'w') as f:
        f.write(summary)
        f.write("energy (eV)   probability    k   k'  n   n'\n")
        for k2x in range(nkx):
            for k2y in range(nky):
                k2ind = nky*k2x + k2y + 1
                for k3x in range(nkx):
                    for k3y in range(nky):
                        k3ind = nky*k3x + k3y + 1
                        for v in range(nv):
                            vind = v + 1
                            for c_ in range(nc):
                                cind = c_ + nv + 1
                                prob = prob_a6[k2x, k2y, k3x, k3y, v, c_]
                                energy = E_a3[k3x, k3y, c_ + nv] \
                                        - E_a3[k2x, k2y, v]

                                f.write(
                                    f'{energy:>11.6f}{prob:>15.6E}'
                                    f'   {k2ind:<4}{k3ind:<4}'
                                    f'{vind:<4}{cind}\n'
                                    )


def writeKpts(
        infile = 'input',
        exroot = 'tmp/hBN.export',
        outfile = 'kpts.txt',
        top = '.',
        ):
    """Writes k-points from QuantumEspresso."""
    try:
        with open(f'{top}/{infile}') as f:
            print(f'reading from {infile}:')

            for line in f:
                if 'exroot' in line:
                    exroot = line.split()[-1]
                    print(f'    exroot = {exroot}')

    except FileNotFoundError:
        pass

    with open(f'{top}/{exroot}/index.xml') as f:
        for line in f:

            if '<Kpoints' in line:
                nk = int(line.split('"')[1])

            if '<Cell' in line:
                alat = float(f.readline().split('"')[1])  # bohr

            if '<a3' in line:
                b_a2 = zeros((3, 3))
                for n in range(3):
                    b_a2[n] = array(f.readline().split('"')[1].split())\
                            .astype(float)  # bohr^{-1}
                invb_a2 = inv(b_a2)  # bohr

            if '<k' in line:
                blat = 2*pi/alat  # bohr^{-1}
                k_a2 = zeros((nk, 3))
                for k in range(nk):
                    k_ar = array([float(val) for val in
                            f.readline().split()])*blat  # bohr^{-1}
                    k_a2[k] = dot(k_ar, invb_a2)

    k_a2 = roundCoords(k_a2)

    with open(f'{top}/{outfile}', 'w') as f:
        f.write('k-point momenta in reciprocal lattice coordinates\n\n')
        for k_ar in k_a2:
#             f.write('{:>22.14e}{:>22.14e}{:>22.14e}\n'.format(*k_ar))
            f.write('{:>20.14f}{:>20.14f}{:>20.14f}\n'.format(*k_ar))


def roundCoords(comp_a2, degree=5):
    """Rounds specified coordinates to the specified degree

    a2: rank 2 array of floats
    degree: degree of rounding (pos int)
        e.g., rounds all coords of 5 or more repeating decimals
    """
    new_comp_a2 = zeros(comp_a2.shape)
    for i, comp_ar in enumerate(comp_a2):
        for j, comp in enumerate(comp_ar):
            comp_str = f'{comp} '  # to iterate from 0 to n + 1
            if 'e' in comp_str:
                new_comp = 0.0
            else:
                consecutive = 0
                n = 0
                while consecutive<degree and n<len(comp_str)-1:
                    if comp_str[n] == comp_str[n + 1]:
                        consecutive += 1
                    else:
                        consecutive = 0
                    n += 1               

                if consecutive==degree:
                    n -= consecutive
                    rep = comp_str[n]

                    # round .999 to 1
                    if int(rep)==9:
                        new_comp = round(comp, n-int(floor(log10(abs(comp)))))
                        if new_comp == 1.0:
                            new_comp = 0.0

                    # round up
                    elif 4 < int(rep) < 9:
                        new_comp_str = comp_str[:n] + rep * (17 - n) \
                                + str(int(rep) + 1)
                        new_comp = float(new_comp_str)

                    # round down
                    else:
                        new_comp_str = comp_str[:n] + comp_str[n] \
                                * (18 - n)
                        new_comp = float(new_comp_str)

                    new_comp_a2[i][j] = new_comp
                else:
                    new_comp_a2[i][j] = float(comp_str)
    return new_comp_a2

                    
#---------------------------- CALL FROM TERMINAL ------------------------------

if __name__ == '__main__':
    nprocs = int(sys.argv[1])
    writeProb(nprocs=nprocs)
    
#---------------------------------- SCRATCH -----------------------------------

#                 alat = float(f.readline().split('"')[1])*bohrtoInvEV  # eV^{-1}
# 
#             if '<b1' in line:
#                 b_a2 = zeros((3, 3))
#                 for n in range(3):
#                     b_a2[n] = array(f.readline().split('"')[1].split())\
#                             .astype(float)  # bohr^{-1}
#                 b_a2 *= invBohrtoEV  # eV
# 
#             if '<k' in line:
#                 blat = 2*pi/alat
#                 k_a2 = zeros((nk, 3))
#                 for k in range(nk):
#                     k_a2[k] = array([float(val) for val in
#                             f.readline().split()])*blat  # eV
#             f.write('{:>22.14e}{:>22.14e}{:>22.14e}\n'.format(*k_ar))
