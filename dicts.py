from numpy import pi, array, arange

# periodic table for atomic masses
from periodic import p_dict

#-------------------------- MATERIAL DICTIONARIES -----------------------------

# species being sputtered from each material
spec_dict = {
        'hBN': 'B', 'MoS2': 'S', 'Barm': 'B', 'Narm': 'N',
        'hBN2': 'B', 'MoS22': 'S', 'Barm2': 'B', 'Narm2': 'N',
        }

# cross sectional area of unit cells in eV^{-2} (obtained from LDA)
Omega_dict = {
        'hBN': 1.3900733428437345e-6,
        'MoS2': 2.253578894519087e-6,
        'Barm': 1.3900733428437345e-6,
        'Narm': 1.3900733428437345e-6,
        'MoS22': 2.253578894519087e-6,
        'Barm2': 1.3900733428437345e-6,
        'Narm2': 1.3900733428437345e-6,
        }  # eV^{-2}

# vibrational frequencies of A_1' mode in eV (DOI: 10.1038/srep04215)
omega_dict = {
        # TO mode in hBN DOI: 10.1103/PhysRevLett.98.095503
        'hBN': 1378.4 * invCmtoEV,
        'MoS2': 406.1 * invCmtoEV,
        'Barm': 1378.4 * invCmtoEV,
        'Narm': 1378.4 * invCmtoEV,
        'MoS22': 406.1 * invCmtoEV,
        'Barm2': 1378.4 * invCmtoEV,
        'Narm2': 1378.4 * invCmtoEV,
        }

# displacement thresholds
Ed_dict = {
        'hBN': array([
            21.17928845, 15.22928845, 9.27928845, 3.32928845, -2.62071155,
            ]),  # optB88 - (band gap * nex)
        'MoS2': 6.9155224 - 1.88*arange(13),  # optB88 - (band gap * nex)
        'Barm': 12.8537 - 4.07*arange(8),  # optB88 - (band gap * nex)
        'Narm': 12.70692655 - 4.07*arange(8),  # optB88 - (band gap * nex)
        'MoS22': 6.9155224 - 1.88*arange(4),  # optB88 - (band gap * nex)
        'Barm2': 12.8537 - 4.07*arange(4),  # optB88 - (band gap * nex)
        'Narm2': 12.70692655 - 4.07*arange(4),  # optB88 - (band gap * nex)
        }

# exp(inverse) fit for sum of excitation probabilities, S(Eb)
fit_dict = {
        'hBN': [7.654762575, -0.7695393786, 0.06525836682],  # R = 0.999957155
        'MoS2': [49.0586673, -3.867010425, 0.3546924901],  # R = 0.9999947621
        'Barm': [7.654762575, -0.7695393786, 0.06525836682],
        'Narm': [7.654762575, -0.7695393786, 0.06525836682],
        'Barm2': [7.654762575, -0.7695393786, 0.06525836682],  # for edgePeaks
        'Narm2': [7.654762575, -0.7695393786, 0.06525836682],
        }

# experimental cross section data
expt_dict = {
        'MoS2': (  # SI of Kretchmer et al. 10.1021/acs.nanolett.0c00670
            array([20, 30, 40, 60, 80]),  # Eb (keV)
            array([6.7, 11.6, 9., 4.2, 5.3]),  # sigma (barn)
            array([0.5, 0.5, 1.0, 0.4, 0.4])   # sigma err (barn)
            ),
        'hBN': (  # Cretu et al. doi:10.1016/j.micron.2015.02.002
            array([
                30, 30, 30, 30, 60, 60, 60, 60, 60, 60, 60, 60
                ]),  # Eb (keV)
            array([
                650, 1000, 1200, 1200, 500, 650, 650, 800, 1000, 1000, 1200,
                1200
                ]) + 273.15,  # temp (K)
            array([
                27.64, 21.46, 97.19, 64.08, 29.5, 13, 13.56, 15.96, 33.66,
                20.44, 80.64, 122.6
                ]),  # sigma (barn)
            ),
            }

# Cretu et al. doi:10.1016/j.micron.2015.02.002
abszero = -273.15
hBNTemp_dict = {  # barn
    500 - abszero: ((60), (29.5)),
    650 - abszero: ((30, 60), (27.63, 13)),
    800 - abszero: ((60), (15.96)),
    1000 - abszero: ((30, 60, 60), (21.46, 33.66, 20.44)),
    1200 - abszero: ((30, 30, 60, 60), (97.19, 64.08, 80.64, 122.6)),
    }



#---------------------------------- SCRATCH ----------------------------------

#         'MoS2': array([
#             6.9155224, 5.0355224, 3.1555224, 1.2755224, -0.6044776,
#             -1, -2, -3, -4,
#             ]),  # optB88 - (band gap * nex)

#         'MoS2': array([
#             6.9155224, 5.0355224, 3.1555224, 2.36258838, 1.66982051,
#             1.15995519, 0.74414126, 0.55279596,
#             ]),  # optB88 - (band gap * nex)
#         'Barm': array([
#             12.85371804, 8.78731804, 4.72091804, 2.11702136, 0.37020869,
#             -3.69619131, -1, -2
#             ]),  # optB88 - (band gap * nex)
#         'Narm': array([
#             12.70692655, 8.64052655, 4.57412655, 0.50772655, -3.55867345, -1,
#             -2,
#             ]),  # optB88 - (band gap * nex)

#         'hBN': [5.778794485, -1.073949207, 0.05713493359],
#         'Barm': [5.462949624, -0.8497685369, 0.08390400208],
#         'Narm': [5.462949624, -0.8497685369, 0.08390400208],
#         'Barm2': [5.462949624, -0.8497685369, 0.08390400208],
#         'Narm2': [5.462949624, -0.8497685369, 0.08390400208],
#         'MoS22': [49.0586673, -3.867010425, 0.3546924901],  # R = 0.9999947621

# Cretu unmodified
#     500 - abszero: ((60), (29.5)),
#     650 - abszero: ((30, 60), (27.63, 13)),
#     800 - abszero: ((60), (15.96)),
#     1000 - abszero: ((30, 60, 60), (21.46, 33.66, 20.44)),
#     1200 - abszero: ((30, 30, 60, 60), (97.19, 64.08, 80.64, 122.6)),

# Cretu modified
#     500 - abszero: ((60), (472.4)),
#     650 - abszero: ((30, 60), (389.4, 96.00, 164.7)),
#     800 - abszero: ((60), (209.2)),
#     1000 - abszero: ((30, 60, 60), (109.6, 289.3, 270.9)),
#     1200 - abszero: ((30, 30, 60, 60), (1189, 1175, 1915, 1613)),
