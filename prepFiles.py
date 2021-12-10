from numpy import array, zeros, arange, linspace, dot
from numpy import sin, cos, log10, pi
from numpy.linalg import inv
import os

import vasp.POSCAR as p

# catch warnings (e.g. RuntimeWarning) as an exception
import warnings
warnings.filterwarnings('error')

"""
version 201118:
    general cell height convergence
    read from .xml instead of .out
"""

sputAtom_dict = {'MoS2': 68, 'hBN': 15}

def prepKCon(
        infile = 'qscf.in',
        kmin = 6,
        kmax = 30,
        layer = True,
        ):
    """Prepares scf.in files with various k-point meshes."""
    line_list = []
    with open(infile) as f:
        for line in f:
            line_list.append(line)
            if 'K_POINTS' in line:
                break

    pos = p.getFromQuanteumEspresso(infile=infile)
    k_list = pos.getK_list(kmin=kmin, kmax=kmax, layer=layer)

    for mesh in k_list:
        outdir = '{:0>2}x{:0>2}x{:0>2}'.format(*mesh)
        os.makedirs(outdir, exist_ok=True)
        with open(f'{outdir}/{infile}', 'w') as f:
            for line in line_list:
                f.write(line)
            f.write('{} {} {} 0 0 0\n'.format(*mesh))

        
def prepCrossing(
        D0 = 0,
        D1 = 5,
        N = 51,
        dim = 6,
        infile = 'POSCAR',
        ):
    """
    prepares POSCAR files with sputtered atom at various distances
    infile: pristine POSCAR file (str)
    N: number of images (pos int)
    D: distance spanned by images in Å (pos float)
    """
    pris = p.POSCAR(infile)
    pris.makeSuperCell([dim, dim, 1])

    if 'B' in pris.specs:
        sputAtom = 15
    elif 'S' in pris.specs:
        sputAtom = 68
    else:
        sputAtom = 1
    
    dist_ar = linspace(D0, D1, N)
    for dist in dist_ar:
        sput = pris.copy()
        sput.translate([0, 0, dist], cartesian=True, atom_list=[sputAtom])

        outfile = '{:.3f}/POSCAR'.format(dist)
        sput.write(outfile)
        sput.write('{}_scr'.format(outfile))


def prepZCon(
        infile = 'scf.in',
        newh_list = 'auto',  # Å
        ):
    """
    prepares scf.in files for cell height convergence with Quantum Espresso
        * run in directory containing infile
        * assumes infile contains tag: CELL_PARAMETERS (angstrom)
    infile: template scf.in file (str)
    newh_list: list of cell heights in Å (list of floats)
    """
    # read template file
    with open(infile) as f:
        line_list = f.readlines()

    for n, line in enumerate(line_list):
        if 'nat' in line:
            nat = int(line.split()[-1])

        if 'CELL_PARAMETERS' in line:
            hlineind = n + 3
            oldh = float(line_list[hlineind].split()[-1])

        if 'ATOMIC_POSITIONS' in line:
            coordlineind = n + 1
            oldcoordline_list = line_list[coordlineind: coordlineind + nat]

    # list cell heights if not defined
    if newh_list == 'auto':
        newh_list = arange(10, 40)

    # generate scf.in files for each cell height
    for newh in newh_list:

        # change cell height
        hline = '{:>23.16f}{:>22.16f}{:>22.16f}\n'.format(0, 0, newh)
        line_list[hlineind] = hline

        # scale atomic coordinates
        newcoordline_list = []
        for oldcoordline in oldcoordline_list:
            str_list = oldcoordline.split()
            oldz = float(str_list[3])
            newz = (oldz - 0.5)*oldh/newh + 0.5
            str_list[3] = '{:.9f}'.format(newz)
            newcoordline = '{:<2}{:>18}{:>14}{:>14}{:>5}{:>4}{:>4}\n' \
                    .format(*str_list)
            newcoordline_list.append(newcoordline)
            
        line_list[coordlineind: coordlineind + nat] = newcoordline_list

        # make outfile directory if it doesn't already exist
        outfile = '{}/{}'.format(newh, infile)
        os.makedirs(os.path.dirname(outfile), exist_ok = True)

        # write new scf.in
        with open(outfile, 'w') as f:
            for line in line_list:
                f.write(line)


def prepShiftedKPOINTSCircle(
        b_a2,
        infile = 'scf.in',
        outfile = 'scf_shift.in',
        DIM = [6, 6, 1],
        dim = [3, 3, 1],
        ):
    """
    prepares q-shifted QE input files that unformly fill the space between
        * run with gamma-centered input file in current directory
        * all q-shifts lie on a uniform k-point mesh
        * label q-shifted by k-point indices
    infile: tamplate QE input file (str)
        * must list k-points at the end
    DIM: major k-point grid (list of 3 post ints)
    dim: grid of q-shifts (list of 3 pos ints)
    """
    DIM = array(DIM)
    dim = array(dim)

    line_list = []
    with open(infile) as f:
        for line in f:
            if 'K_POINTS' in line:
                break
            else:
                line_list.append(line)
                
    ROW = (arange(n)/n + .5)%1 - .5
    MESH = zeros((DIM.prod(), 3))
    for I in ROW_ar:
        for J in ROW_ar:
            MESH.append
    angle_ar = arange(N)*2*pi/N
    for n, angle in enumerate(angle_ar):
        q_l2.append([cos(angle), sin(angle), 0])
    q_a2 = array(q_l2) * 0.1

    for q_ar, angle in zip(q_a2, angle_ar):
        qx, qy, qz = q_ar
#        directory = '{:3g}_{:3g}'.format(qx, qy)
        directory = '{:.3g}'.format(angle * 360/2/pi)
        outfile = '%s/KPOINTS' %directory
        print('writing to {}'.format(outfile))
        infile = '0.00/KPOINTS'
        prepShiftedKPOINTS(q_ar, infile, outfile)


def prepShiftedKPOINTS(
        q_list = [0.0001],
        log = True,
        in0 = '0.0/scf.in',
        out0 = '0.0/tmp/hBN.export/index.xml',
        outfile = 'scf_shift.in',
        ):
    """
    writes .in file with k-points shifted from Gamma-centered mesh
        * assumes k-points are written and the end of the QE input file
        * run in directory where 0.0/scf.in contains Gamma-centered run
    q_list: list of q-shifts in reciprocal space (list of floats)
    in0: Gamma-centered QE input file (str)
    out0: Gamma-centered QE output file (str)
        * use .out instead of .xml to access crystal coordinates
    outfile: q-shifted QE input file (str)
    """
    line_list = []
    with open(in0) as f:
        for line in f:
            if 'K_POINTS' in line:
                break
            else:
                line_list.append(line)

    with open(out0) as f:
        for line in f:

            if '<Kpoints' in line and 'nk' not in locals():
                nk = int(line.split('"')[1])

            elif '<Cell' in line and 'b_a2' not in locals():
                for n in range(4):
                    f.readline()

                b_a2 = zeros((3, 3))
                for n in range(3):
                    b_a2[n] = array(f.readline().split('"')[1].split())\
                            .astype(float)  # bohr^{-1}

            elif '<k' in line and 'k_a2' not in locals():
                k_a2 = zeros((nk, 3))
                for k in range(nk):
                    k_a2[k] = array([float(val) for val in
                            f.readline().split()])  # cartesian RLV

                k_a2 = dot(k_a2, inv(b_a2)) * b_a2[0, 0]  # RLV
                break

    for q in q_list:
        k_a2 += array([0., q, 0.])

        if log:
            outdir = abs(round(log10(q), 1))
        else:
            outdir = '{:>4.3f}'.format(q)

        qoutfile = '{}/{}'.format(outdir, outfile)
        os.makedirs(os.path.dirname(qoutfile), exist_ok = True)

        with open(qoutfile, 'w') as f:
            for line in line_list:
                f.write(line)

            f.write('K_POINTS (crystal)\n{}\n'.format(nk))
            for k_ar in k_a2:
                f.write('{:>14.9f}{:>14.9f}{:>14.9f}\t1\n'.format(*k_ar))


def prepUniformKPOINTS(infile = 'IBZKPT', outfile = 'KPOINTS_uni'):
    """
    writes KPOINTS file with uniform mesh (no symmetry assumptions)
    infile: IBZKPT from ISYM = 0 run (str)
        * weights should only be 1 or 2
        * seems to only work reliably for orthorhobmic symmetry
    outfile: KPOINTS file to write to (str)
    """
    kpt_l2 = []
    with open(infile) as f:
        f.readline()
        nkpts = int(f.readline())
        f.readline()
        for line in f:
            kpt_l2.append([float(val) for val in line.split()])
            
    newKpt_l2 = []
    for kpt_list in kpt_l2:
        if kpt_list[-1] > 1.5:
            newKpt_l2.append([-val for val in kpt_list])

    kpt_l2 += newKpt_l2

    with open(outfile, 'w') as f:
        f.write('uniform k-point mesh generated from prepUniformKPOINTS\n')
        f.write('\t{}\n'.format(len(kpt_l2)))
        f.write('Reciprocal lattice\n')
        for kpt_list in kpt_l2:
            f.write('{:>20.14f}{:>20.14f}{:>20.14f}\t1\n'
                    .format(*kpt_list[:3]))

#---------------------------- PHYSICAL CONSTANTS ------------------------------

m = 5.109989461e5  # mass of electron (eV)
M = 9.3827231e8  # mass of proton (eV)
c  = 299792458  # speed of light (m/s)
a0 = 5.29177e-11  # Bohr radius (m)
ev = 1.60217662e-19  # electronVolt (J)
e0 = 8.85418782e-12  # permittivity of free space (SI)
kb = 8.6173303e-5  # Boltzmann constant (eV/K)
bm = 9.37e-11  # minimum impact parameter in MoS2 (m)
alpha = 1/137.035999084  # fine structure constant (unitless)
hbar = 6.582119569e-16  # planck constant (eV s)

#----------------------------- UNIT CONVERSIONS -------------------------------

scale1 = 1e28 * c**2                  # s^2 to 100 fm^2 (barn)
const = pi*hbar**2*alpha**2 * scale1  # prefactor to cross sections
invÅtoEV = 1e10*c*hbar
invÅtomeV = 1e10*c*hbar*1000
invEVSqtoÅSq = hbar**2*c**2*1e20

#---------------------------- CALL FROM TERMINAL ------------------------------

if __name__ == '__main__':
#    prepUniformKPOINTS()
#    prepShiftedKPOINTS()
    prepZCon()


#------------------------------------------------------------------------------

#        q_list = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002,
#            0.0001],
#            if 'number of k points' in line and 'k_a2' not in locals():
#                nk = int(line.split()[4])
#
#                # jump to crystal representation
#                inCart = True 
#                while inCart:
#                    if 'cryst. coord.' in f.readline():
#                        inCart = False
#
#                k_l2 = []
#                while len(k_l2) < nk:
#
#                    line = f.readline().strip()
#                    if len(line) > 0 and line[:2] == 'k(':
#                        k_l2.append([float(val.strip('),')) for val in
#                            line.split()[4:7]])
#                    else:
#                        continue
#
#                k_a2 = array(k_l2)
#        out0 = '0.0/scf.out',
