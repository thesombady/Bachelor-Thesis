import sys

gamma_h = 1

def parser():
    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    argv = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    assert len(argv) <= 5, 'To many arguments'

    def name(argument):
        if 'Above' in argument:
            return 150 * gamma_h, 'Above'
        elif 'Below' in argument:
            return 34 * gamma_h, 'Below'
        elif 'Lasing' in argument:
            return 37.5 * gamma_h, 'Lasing'

    if "-h" in opts:
        print("""Arg1 = Number of photon modes, Arg2 = Method to use, Euler & Runge,
        Arg3 = Delta t, Arg4 = Mode of operation""")
        sys.exit()

    try:
        N = int(argv[0])
        KEY = str(argv[1])
        deltas = float(argv[2])
        w_2 = name(argv[3])[0]
        NAME = name(argv[3])[1]
        NAME = NAME + KEY + '1000_' + str(N) + '_' + str(argv[4]) + '.npy'
        return N, KEY, deltas, NAME
    except Exception as e:
        raise e


print(parser())
