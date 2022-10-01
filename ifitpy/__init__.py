try:
    # make iminuit importable even if it is not installed yet for setup.cfg
    from .Fitter import Fitter
    from .Fitter import Functions
except ImportError:  # pragma: no cover
    pass  # pragma: no cover