"""Cross-site leave-one-out split utilities."""

from __future__ import annotations

import logging

from nee_classification.config.schema import ExperimentConfig

logger = logging.getLogger(__name__)


def get_cross_site_splits(
    all_sites: list[str],
) -> dict[str, list[str]]:
    """Generate leave-one-out cross-site validation splits.

    For each site in *all_sites*, one split is produced where that site is
    the holdout and all remaining sites are used for training.

    Parameters
    ----------
    all_sites:
        Complete list of site names.

    Returns
    -------
    dict[str, list[str]]
        Mapping from holdout site name → list of training site names.

    Examples
    --------
    >>> splits = get_cross_site_splits(["a", "b", "c"])
    >>> splits["a"]
    ['b', 'c']
    """
    splits: dict[str, list[str]] = {}
    for holdout in all_sites:
        train = [s for s in all_sites if s != holdout]
        splits[holdout] = train

    logger.info(
        "Generated %d leave-one-out cross-site splits from %d sites",
        len(splits),
        len(all_sites),
    )
    return splits


def get_single_split(
    config: ExperimentConfig,
) -> tuple[list[str], str]:
    """Extract the (train_sites, holdout_site) pair from a configuration.

    If ``config.data.train_sites`` and ``config.data.holdout_site`` are
    explicitly set they are returned directly.  Otherwise this function
    raises a :class:`ValueError` instructing the caller to specify them or
    use :func:`get_cross_site_splits` for exhaustive leave-one-out.

    Parameters
    ----------
    config:
        Validated experiment configuration.

    Returns
    -------
    train_sites:
        List of site names for training.
    holdout_site:
        Name of the single holdout site for testing.

    Raises
    ------
    ValueError
        If either ``train_sites`` or ``holdout_site`` is *None*.
    """
    if config.data.train_sites is None or config.data.holdout_site is None:
        raise ValueError(
            "Both 'data.train_sites' and 'data.holdout_site' must be set in "
            "the configuration.  Use get_cross_site_splits() for automatic "
            "leave-one-out enumeration."
        )

    logger.info(
        "Single split: train=%s, holdout='%s'",
        config.data.train_sites,
        config.data.holdout_site,
    )
    return list(config.data.train_sites), config.data.holdout_site
