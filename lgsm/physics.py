"""This module defines the physics layers.

e.g. redshifting, IGM extinction, bandpass zero-point offsets...
"""

from collections import OrderedDict

import numpy as np
import torch
from torchinterp1d import interp1d

from .paths import paths

# Constants for the luminosity distance below
_lum_dist_powers = torch.Tensor([8, 7, 6, 5, 4, 3, 2, 1, 0])
_lum_dist_coeff = torch.Tensor(
    [
        3.986328649940791e-05,
        -0.00116780796758629,
        0.014749896043820264,
        -0.1058478438176513,
        0.4814424486992418,
        -1.4877543059552116,
        3.46341245063575,
        4.426282964783882,
        -0.00021275062264485727,
    ]
)

# Load the bandpasses
bandpasses = {}
_bp_dir = paths.data / "bandpasses"
for file in _bp_dir.glob("*"):
    # Load the bandpasses
    wavelengths, R = np.genfromtxt(file, unpack=True)

    # Convert wavelengths from nm -> Angstrom
    wavelengths *= 10

    # Make sure the bandpass is normalized
    R /= np.trapz(R, wavelengths)

    # Calculate the effective wavelength
    eff_wavelength = np.trapz(wavelengths * R, wavelengths)

    bandpasses[file.stem] = {
        "wavelengths": torch.from_numpy(wavelengths),
        "R": torch.from_numpy(R),
        "eff_wavelength": eff_wavelength,
    }

# Sort the bandpasses by effective wavelength
bandpasses = OrderedDict(
    sorted(bandpasses.items(), key=lambda item: item[1]["eff_wavelength"])
)


def luminosity_distance(redshift: torch.Tensor) -> torch.Tensor:
    """Return luminosity distance in Gpc (Planck 2018 cosmology).

    This uses an 8th-order polynomial approximation that is accurate
    to < 0.5% between redshifts 0.01 and 6.

    Parameters
    ----------
    redshift: torch.Tensor
        Redshifts of galaxies

    Returns
    -------
    torch.Tensor
        Luminosity distances in Gpc
    """
    return torch.sum(
        _lum_dist_coeff[None, :]
        * torch.float_power(redshift, _lum_dist_powers[None, :]),
        axis=1,
    )


def redshift_sed(
    wavelengths: torch.Tensor,
    sed: torch.Tensor,
    redshift: torch.Tensor,
) -> torch.Tensor:
    """Redshift the SED.

    Parameters
    ----------
    wavelengths: torch.Tensor
        Wavelength grid for the SED
    sed: torch.Tensor
        The rest-frame SED to redshift (units: solar luminosities / angstrom)
    redshift: torch.Tensor
        Target redshift

    Returns
    -------
    torch.Tensor
        Redshifted SED (units: erg / s / cm^2 / angstrom)
    """
    # Redshift the wavelength grid
    redshifted_sed = torch.where(
        wavelengths >= (1 + redshift) * wavelengths[0],
        interp1d((1 + redshift) * wavelengths, sed, wavelengths),
        0,  # TODO: change this back to torch.nan?
    )
    redshifted_sed /= 1 + redshift

    # Multiply normalization
    solar_luminosity_cgs = 3.828e33  # L* -> erg / s / cm^2
    centimeters_per_Gpc = 3.086e27
    pi = 3.14159
    dL_cent = luminosity_distance(redshift) * centimeters_per_Gpc
    norm = solar_luminosity_cgs / (4 * pi * dL_cent**2)

    redshifted_sed *= norm[:, None]

    return redshifted_sed


def extinct_sed(
    wavelengths: torch.Tensor,
    sed: torch.Tensor,
    redshift: torch.Tensor,
) -> torch.Tensor:
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
    return sed


def observe_sed(
    wavelengths: torch.Tensor,
    sed: torch.Tensor,
    redshift: torch.Tensor,
) -> torch.Tensor:
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
    return extinct_sed(
        wavelengths,
        redshift_sed(wavelengths, sed, redshift),
        redshift,
    )


def calc_photometry(
    wavelengths: torch.Tensor,
    sed: torch.Tensor,
) -> torch.Tensor:
    """Integrate SED against bandpasses to get photometry.

    Parameters
    ----------
    """
    # Create a list for fluxes
    fluxes = []

    # Loop over bands
    for band in bandpasses.values():
        # Interpolate the SED onto the bandpass grid
        sed_interp = interp1d(
            torch.tile(wavelengths, (len(sed), 1)),
            sed,
            band["wavelengths"],
        )

        # Integrate the SED against the bandpass
        fluxes.append(torch.trapz(sed_interp * band["R"], band["wavelengths"]))

    # Stack fluxes into a tensor
    fluxes = torch.vstack(fluxes).T

    return fluxes
