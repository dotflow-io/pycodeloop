"""Providers __init__ module."""

import importlib
import os
from pathlib import Path

from pycodeloop.constants import ENV_API_KEY
from pycodeloop.providers.generic import GenericProvider

PROVIDERS = {"generic": GenericProvider}

TEMPLATES_DIR = Path(__file__).parent / "templates"
BUNDLED_TEMPLATES = {p.stem for p in TEMPLATES_DIR.glob("*.json")}

__all__ = ["GenericProvider", "get_provider", "PROVIDERS", "BUNDLED_TEMPLATES"]


def get_provider(name: str, **kwargs):
    """Build a `Provider` — always a `GenericProvider` under the hood.

    `name` is one of:
      - a path to a JSON config file (see `pycodeloop.providers.generic`),
        the standard way to point at any HTTP LLM API;
      - the bare name of a template shipped in `providers/templates/`
        (e.g. `"anthropic"`, `"ollama"`) — resolved from the installed
        package, so it works from any directory without a path;
      - `"generic"`, paired with `url=`/`model=` kwargs for an ad-hoc
        config with no file;
      - `'module.path:ClassName'` for a custom `Provider` subclass.
    """
    if name.endswith(".json"):
        return _from_json_path(name, kwargs)

    if name in BUNDLED_TEMPLATES:
        return _from_json_path(str(TEMPLATES_DIR / f"{name}.json"), kwargs)

    if ":" in name:
        module_path, class_name = name.split(":", 1)
        module = importlib.import_module(module_path)
        provider_cls = getattr(module, class_name)

        return provider_cls(**_with_env_api_key(kwargs))

    try:
        provider_cls = PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {list(PROVIDERS)}, "
            f"a bundled template ({', '.join(sorted(BUNDLED_TEMPLATES))}), "
            "a path to a JSON config file, or 'module.path:ClassName' "
            "for a custom Provider."
        ) from None

    return provider_cls(**_with_env_api_key(kwargs))


def _from_json_path(path: str, kwargs: dict) -> GenericProvider:
    provider = GenericProvider.from_json(path)

    if kwargs.get("model"):
        provider.model = kwargs["model"]

    if kwargs.get("api_key"):
        provider.api_key = kwargs["api_key"]

    if kwargs.get("inference_params"):
        provider.inference_params = {
            **provider.inference_params,
            **kwargs["inference_params"],
        }

    return provider


def _with_env_api_key(kwargs: dict) -> dict:
    if kwargs.get("api_key"):
        return kwargs
    env_key = os.environ.get(ENV_API_KEY)
    if not env_key:
        return kwargs
    return {**kwargs, "api_key": env_key}
