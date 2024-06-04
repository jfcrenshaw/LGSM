"""This module combines the normalizing flow, spectral decoder, and physics
layers into a single LGS Model.
"""

import numpy as np

from .physics import adjust_zero_points, calc_photometry, observe_sed


class LGSModel:
    """Latent Galaxy SED Model.

    Currently this is just schematic code.
    """

    def encode(
        self,
        photometry: np.array,
        redshift: float,
        seed: int,
    ) -> tuple[float, np.ndarray]:
        """Encode the photometry in the latent space.

        Parameters
        ----------
        photometry: np.array
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
        np.ndarray
            The latent SED parameters
        """
        # Sample redshifts for any NaNs
        # Then pass everything through the latent flow
        raise NotImplementedError

    def decode_sed(self, sed_params: np.array) -> np.array:
        """Decode latent parameters into a rest-frame SED.

        Parameters
        ----------
        sed_params: np.array
            Array of latent SED parameters

        Returns
        -------
        np.array
            Decoded rest-frame galaxy SED
        """
        raise NotImplementedError

    def calc_photometry(self, sed: np.array, redshift: float) -> np.array:
        """Calculate photometry for the SED.

        Parameters
        ----------
        sed: np.array
            The galaxy SED
        redshift: float
            The galaxy redshift

        Returns
        -------
        np.array
            The galaxy photometry
        """
        # Get observed SED
        observed_sed = observe_sed(sed, redshift)

        # Calculate photometry
        photometry = calc_photometry(observed_sed)

        # Adjust zero points
        photometry = adjust_zero_points(photometry)

        raise photometry

    def autoencode(
        self,
        photometry: np.ndarray,
        redshift: float,
        seed: int,
    ) -> np.ndarray:
        """Pass photometry through autoencoder, returning new photometry.

        Parameters
        ----------
        photometry: np.ndarray
            The photometry to autoencode.
        redshift: float
            The redshift of the galaxy. If NaN, the redshift flow is used
            to sample a redshift
        seed: int, default=123
            The random seed to use for probabilistic encoding.

        Returns
        -------
        np.ndarray
            The autoencoded photometry
        """
        latents = self.encode(photometry, redshift, seed)
        sed = self.decode_sed(latents)
        new_photometry = self.calc_photometry(sed, redshift)
        return new_photometry

    def sample_prior(self, N: int, seed: int) -> np.ndarray:
        """Return a sample from the latent prior.

        Parameters
        ----------
        N: int
            The number of samples to return.
        seed: int
            The random seed.

        Returns
        -------
        np.ndarray
            The latent samples
        """
        raise NotImplementedError
