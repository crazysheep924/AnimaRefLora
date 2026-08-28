from __future__ import annotations

from .config import TrainConfig


class UnsupportedFeatureError(RuntimeError):
    pass


def validate_supported_training_features(config: TrainConfig) -> None:
    """Fail before training when exposed CLI switches do not have real behavior yet."""
    problems: list[str] = []

    invalid_train_args = [value for value in config.train_args if "=" not in value]
    if invalid_train_args:
        problems.append(f"--train-arg only supports KEY=VALUE network kwargs here: {invalid_train_args}")
    if config.train_args and config.backend != "external":
        problems.append("--train-arg only applies to the external sd-scripts backend")

    if problems:
        joined = "\n  - ".join(problems)
        raise UnsupportedFeatureError(f"Unsupported or incomplete training features requested:\n  - {joined}")
