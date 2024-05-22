from structures.divisive_normalization_heeger import DivisiveNormHeeger
from structures.divisive_normalization_plain import DivNorm2d
from structures.divisive_normalization_miller import DivisiveNormKenMiller

def get_divnorm(out_channels, args_divnorm=None):

    if args_divnorm['type'] == 'ours':
        if args_divnorm:
            p = args_divnorm['p']
        else:
            p = 0.5

        if p == 1:
            return DivNorm2d(args_divnorm)
        else:
            return DivisiveNormHeeger(out_channels, args_divnorm)
    elif args_divnorm['type'] == 'ken':
        return DivisiveNormKenMiller(args_divnorm)