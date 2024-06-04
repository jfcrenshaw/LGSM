"""This module defines the physics layers.

e.g. redshifting, IGM extinction, bandpass zero-point offsets...
"""

import torch


def redshift_sed(sed: torch.Tensor, redshift: float) -> torch.Tensor:
    """Redshift the SED.

    Parameters
    ----------
    sed: torch.Tensor
        The rest-frame SED to redshift
    redshift: float
        Target redshift

    Returns
    -------
    torch.Tensor
        Redshifted SED
    """
    raise NotImplementedError


def extinct_sed(sed: torch.Tensor, redshift: float) -> torch.Tensor:
    """Apply IGM extinction to observed-frame SED.

    Parameters
    ----------
    sed: torch.Tensor
        Observed-frame SED
    redshift: float
        Redshift of galaxy

    Returns
    -------
    torch.Tensor
        SED with IGM extinction
    """
    raise NotImplementedError


def observe_sed(sed: torch.Tensor, redshift: float) -> torch.Tensor:
    """Convert intrinsic SED to observed SED.

    Parameters
    ----------
    sed: torch.Tensor
        Rest-frame SED
    redshift: float
        Redshift of galaxy

    Returns
    -------
    torch.Tensor
        Observed SED
    """
    return extinct_sed(redshift_sed(sed, redshift), redshift)


def correct_zero_points(photometry: torch.Tensor) -> torch.Tensor:
    """Correct the zero-point offsets of the photometry."""
    raise NotImplementedError
