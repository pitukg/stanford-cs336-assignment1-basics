import argparse
from pathlib import Path

import yaml


def parse_config(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser.add_argument("--config", type=Path)

    # Read only the config path first; the full parser runs afterward.
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    preliminary, _ = bootstrap.parse_known_args(argv)

    config = {}
    if preliminary.config is not None:
        try:
            with preliminary.config.open() as f:
                config = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as exc:
            parser.error(f"Cannot read configuration: {exc}")

        if config is None:
            config = {}
        if not isinstance(config, dict):
            parser.error("The YAML configuration must be a mapping.")

    actions = {
        action.dest: action
        for action in parser._actions
        if action.option_strings and action.dest not in {"help", "config"}
    }

    defaults = {}
    for key, value in config.items():
        if not isinstance(key, str):
            parser.error("Configuration keys must be strings.")

        dest = key.replace("-", "_")
        if dest not in actions:
            parser.error(f"Unknown configuration key: {key}")
        if dest in defaults:
            parser.error(f"Duplicate configuration key: {key}")
        if value is None:
            parser.error(f"{key} cannot be null.")

        action = actions[dest]

        try:
            if isinstance(
                action,
                (argparse._StoreTrueAction, argparse._StoreFalseAction),
            ):
                if not isinstance(value, bool):
                    raise ValueError("expected a YAML boolean")
            elif action.type is not None:
                # Match CLI conversion, including rejecting fractional integers.
                value = action.type(str(value))
            elif not isinstance(value, str):
                raise ValueError("expected a string")

            if action.choices is not None and value not in action.choices:
                raise ValueError(f"expected one of {action.choices}")
        except (ValueError, TypeError, argparse.ArgumentTypeError) as exc:
            parser.error(f"Invalid value for {key}: {exc}")

        defaults[dest] = value

    parser.set_defaults(**defaults)

    # A required option may be supplied through YAML instead of the CLI.
    for dest in defaults:
        actions[dest].required = False

    return parser.parse_args(argv)


def save_resolved_config(
    args: argparse.Namespace,
    destination: Path,
) -> None:
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"config", "overwrite"}
    }
    with destination.open("w") as f:
        yaml.safe_dump(values, f, sort_keys=True)
