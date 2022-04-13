from logging import Logger

from engine import PluginCore
from model import Meta


class SamplePlugin(PluginCore):

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self.meta = Meta(
            name='Sample Plugin',
            description='Sample plugin template',
            version='0.0.1'
        )

    def invoke(self, args ):
        #self._logger.debug(f'Command: {args} -> {self.meta}')
        return ""
