"""TabTransformer trainer — inherits from FTTransformerTrainer."""

from __future__ import annotations

import logging
from typing import Any

import optuna

from nee_classification.models.ft_transformer import FTTransformerTrainer

logger = logging.getLogger(__name__)


class TabTransformerTrainer(FTTransformerTrainer):
    """Trainer for TabTransformer models.

    Inherits all pipeline logic from :class:`FTTransformerTrainer`;
    only overrides the model configuration to use ``TabTransformerConfig``.
    """

    MODEL_CONFIG_CLASS_NAME = "TabTransformerConfig"

    def _create_model_config(
        self, trial: optuna.Trial, n_features: int
    ) -> Any:
        """Create a TabTransformer config from Optuna suggestions."""
        from pytorch_tabular.models import TabTransformerConfig

        num_attn_blocks = trial.suggest_int("num_attn_blocks", 2, 6)
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.3)
        ff_dropout = trial.suggest_float("ff_dropout", 0.0, 0.5)
        embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        max_epochs = self.config.model_config_extra.max_epochs_optuna
        patience = self.config.model_config_extra.early_stopping_patience

        return TabTransformerConfig(
            task="classification",
            input_embed_dim=embed_dim,
            num_attn_blocks=num_attn_blocks,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            learning_rate=lr,
            metrics=["accuracy"],
            metrics_prob_input=[False],
        ), batch_size, max_epochs, patience
