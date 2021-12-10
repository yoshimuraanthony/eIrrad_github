def writeEELS(
        el_ar = [],
        eels_ar = [],
        infile = 'input',
        exroot = 'tmp/hBN.export',
        outfile = 'eels.txt',

        q0max = 50,  # eV
        Eb = 6e4,  # eV
        N = 500,

        printInfo = True,
        ):
    """Writes EELS.

    outfile: file to which probabilities of all transitions are written (str)
    """
    # calculate eels_ar if not provided
    if len(eels_ar) == 0:
        
        # read input file for getProbA6 parameters
        try:
            with open(infile) as f:
                print('reading from {}:'.format(infile))

                for line in f:
                    if 'Eb' in line:
                        Eb = float64(line.split()[-1])
                        print('    Eb = {:.5g} eV'.format(Eb))

                    if 'q0max' in line:
                        q0max = float64(line.split()[-1])
                        print('    q0max = {:.5g} eV'.format(q0max))

                    if 'exroot' in line:
                        exroot = line.split()[-1]
                        print('    exroot = {} eV'.format(exroot))
    
        except FileNotFoundError:
            pass

        # calculate all transition probabilities
        startTime = perf_counter()
        C_a6, p_a6, E_a3, k_a3, b_a2, ne, volume, area, encut = \
                _readOutput(exroot)
        prob_a6, eels_ar, el_ar = getProbA6(C_a6, p_a6, E_a3, k_a3, b_a2,
                area, ne, q0max, Eb, N, printInfo)

        totTime = perf_counter() - startTime
        print('prob_a6 calculation time = {:.5g} seconds'.format(totTime))
    
    print('writing to {}'.format(outfile))
    with open(outfile, 'w') as f:
        f.write('beam energy = {:.5g} eV\n'.format(Eb))
        f.write("energy loss (eV)   probability\n")
        for el, eels in zip(el_ar, eels_ar):
            f.write('{:>11.4f}{:>15.6E}\n'.format(el, eels))


def plotEELS(
        el_ar = [],
        eels_ar = [],
        infile = 'input',
        eelsInfile = 'eels.txt',
        exroot = 'tmp/hBN.export',

        q0max = 50,  # eV
        Eb = 6e4,  # eV
        N = 500,

        plot = True,
        smear = True,
        figsize = (6,5),
        title = None,
        sigma = 0.2,
        save = True,
        outfile = 'eels.png',

        getData = False,
        ):
    """
    returns total probability for all valence to conduction band excitations
    scat.in: optional input file containing Eb and q0max (str)
    Eb: energy of beam electron in eV (pos float)
    q0max: maximum virtual photon energy in eV considered (pos float)
        * only care about momentum since change in energy is miniscule
        * doesn't remove "diagonal" terms, but still helps
    """
    if len(eels_ar) == 0 and eelsInfile == '':
        startTime = perf_counter()
        p_a6, C_a6, E_a3, k_a3, b_a2, ne, volume, area, encut = \
                _readOutput(exroot)
        prob_a6, eels_ar, el_ar = getProbA6(C_a6, p_a6, E_a3, k_a3, b_a2,
                area, ne, q0max, Eb, N)

    if len(eels_ar) == 0:
        print('reading from {}'.format(eelsInfile))
        el_ar = zeros(N)
        eels_ar = zeros(N)
        with open(eelsInfile) as f:
            for n in range(2):
                f.readline()
    
            for n in range(N):
                el, eels = [float(val) for val in f.readline().split()]
                el_ar[n] = el
                eels_ar[n] = eels
                
    smear_ar = zeros(len(eels_ar))
    for el, eels in zip(el_ar, eels_ar):
        smear_ar += eels * exp(-((el_ar - el)/sigma)**2/2)

    if plot:
        fig, ax = plt.subplots(figsize = figsize)
        if smear:
            ax.plot(el_ar, smear_ar)
        else:
            ax.plot(el_ar, eels_ar)
        ax.set_xlabel('energy loss (eV)', fontsize = 16)
        ax.set_ylabel('probability', fontsize = 16)
        ax.set_title(title, fontsize = 16)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
        ax.grid()
        ax.set_xlim(0, q0max)
    
        plt.tight_layout()
        if save:
            plt.savefig(outfile, dpi = 300)
        plt.show()

    # make copy with date in name
    today = date.today()
    year = today.year - 2000
    month = today.month
    day = today.day

    if '.' in outfile:
        name, ext = outfile.split('.')
        outfileCopy = '{}{}{}{}.{}'.format(name, year, month, day, ext)
    else:
        outfileCopy = '{}{}{}{}'.format(outfile, year, month, day)

    copyfile(outfile, outfileCopy)

    if getData:
        return eels_ar, el_ar

