"""Base class for information environments."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Environment(ABC):
    """
    An information environment that the agent navigates.

    The environment executes actions and returns results.
    Different environments represent different information spaces.
    """

    @abstractmethod
    def execute(self, action) -> object:
        """Execute an action and return an ActionResult."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the environment for a new navigation episode."""

    @property
    @abstractmethod
    def substrate_names(self) -> list[str]:
        """Names of available substrates."""
