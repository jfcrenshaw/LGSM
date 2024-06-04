"""This module combines the normalizing flow, spectral decoder, and physics
layers into a single LGS Model.
"""

import torch
from torch import nn

from .physics import bandpasses, calc_photometry, observe_sed
from .speculator import Speculator


class LGSModel(nn.Module):
    """Latent Galaxy SED Model.

    Currently this is just schematic code.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sed_model = Speculator()
        self.bandpasses = bandpasses

    def encode(
        self,
        photometry: torch.Tensor,
        redshift: float,
        seed: int,
    ) -> tuple[float, torch.Tensor]:
        """Encode the photometry in the latent space.

        Parameters
        ----------
        photometry: torch.Tensor
            The array of galaxy photometry to encode
        redshift: float
            The redshift of the galaxy. If NaN, the redshift flow is used
            to sample a redshift
        seed: int, default=123
            The random seed to use for probabilistic encoding.

        Returns
        -------
        float
            The predicted redshift of the galaxy
        torch.Tensor
            The latent SED parameters
        """
        # Sample redshifts for any NaNs
        # Then pass everything through the latent flow
        raise NotImplementedError

    def decode_sed(self, latents: torch.Tensor, log_mass: torch.Tensor) -> torch.Tensor:
        """Decode latent parameters into a rest-frame SED.

        Parameters
        ----------
        latents: torch.Tensor
            Array of latent SED parameters
        log_mass: torch.Tensor

        Returns
        -------
        torch.Tensor
            Decoded rest-frame galaxy SED
        """
        return self.sed_model(latents, log_mass)

    def calc_photometry(self, sed: torch.Tensor, redshift: float) -> torch.Tensor:
        """Calculate photometry for the SED.

        Parameters
        ----------
        sed: torch.Tensor
            The galaxy SED
        redshift: float
            The galaxy redshift

        Returns
        -------
        torch.Tensor
            The galaxy photometry
        """
        # Get observed SED
        observed_sed = observe_sed(self.sed_model.wavelengths, sed, redshift)

        # Calculate photometry
        photometry = calc_photometry(self.sed_model.wavelengths, observed_sed)

        return photometry

    def autoencode(
        self,
        photometry: torch.Tensor,
        redshift: float,
        seed: int,
    ) -> torch.Tensor:
        """Pass photometry through autoencoder, returning new photometry.

        Parameters
        ----------
        photometry: torch.Tensor
            The photometry to autoencode.
        redshift: float
            The redshift of the galaxy. If NaN, the redshift flow is used
            to sample a redshift
        seed: int, default=123
            The random seed to use for probabilistic encoding.

        Returns
        -------
        torch.Tensor
            The autoencoded photometry
        """
        latents = self.encode(photometry, redshift, seed)
        sed = self.decode_sed(latents)
        new_photometry = self.calc_photometry(sed, redshift)
        return new_photometry
