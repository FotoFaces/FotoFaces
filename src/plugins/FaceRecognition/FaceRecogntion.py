from logging import Logger

from engine import PluginCore
from model import Meta


class SamplePlugin(PluginCore):

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger)
        self.meta = Meta(
            name='Face Recognition plugin',
            description='Plugin that returns a distance between two faces within a roi, the bigger the distance the lower the people look a like.Meaning they are not the same person.',
            version='0.0.1'
        )

    def invoke(self, **args):
        """
            Logic of the plugin
            :args is a dictionaire
            :returns a a value related to the metric analysed
        """
        self._logger.debug(f'Command: {command} -> {self.meta}')
        return device
