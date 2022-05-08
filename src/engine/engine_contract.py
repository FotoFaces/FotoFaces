from logging import Logger
from typing import Optional, List

from model import Meta
import appCore


class IPluginRegistry(type):
    plugin_registries: List[type] = list()

    def __init__(cls, name, bases, attrs):
        super().__init__(cls)
        if name != 'PluginCore':
            print(cls)
            IPluginRegistry.plugin_registries.append(cls)
        else:
            print("ERROR")


class PluginCore(object, metaclass=IPluginRegistry):
    """
    Plugin core class
    """

    meta: Optional[Meta]

    def __init__(self, logger: Logger, coreApplication : appCore.ApplicationCore ) -> None:
        """
        Entry init block for plugins
        :param logger: logger that plugins can make use of
        """
        self._logger = logger
        self._coreApplication = logger

    def invoke(self, **args):
        """
        Starts main plugin flow
        :param args: possible arguments for the plugin
        :return: a device for the plugin
        """
        pass
