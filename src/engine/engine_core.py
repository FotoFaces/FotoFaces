from logging import Logger

from usecase import PluginUseCase
from util import LogUtil


class PluginEngine:
    _logger: Logger

    def __init__(self, **args) -> None:
        self._logger = LogUtil.create(args['options']['log_level'])
        self.use_case = PluginUseCase(args['options'])
        self._coreApplication = args['options']['coreApplication']
        self.reload = True


    def start(self, **args):
        """
            start all plugins with some data
        """
        self.__reload_plugins()
        return self.__invoke_on_plugins(**args)


    def __reload_plugins(self) -> None:
        """Reset the list of all plugins and initiate the walk over the main
        provided plugin package to load all available plugins
        """
        self.use_case.discover_plugins(self.reload)
        self.reload=False

    def __invoke_on_plugins(self, **args):
        """ Invoke evey plugin and gather
            metics analysed by every plugin
        """
        dict_feedback = {} # gather all plugins

        print(self.use_case.modules)
        for module in self.use_case.modules:
            plugin = self.use_case.register_plugin(module, self._logger, self._coreApplication)
            delegate = self.use_case.hook_plugin(plugin) # callable invoke of the curren plugin

            # plugin_resp should be a tuple or a list composed of (key, value)
            plugin_resp = delegate(args) # invoke(args)
            if isinstance(plugin_resp, list):
                for tpl in plugin_resp:
                    dict_feedback[tpl[0]] = tpl[1]
            elif isinstance(plugin_resp, tuple):
                dict_feedback[plugin_resp[0]] = plugin_resp[1]
            else:
                self._logger.error(f'Return datatype invalid')

            self._logger.info(f'Plugin Returned: {plugin_resp}')
        self._logger.info(f'Plugin Returned: {dict_feedback}')
        return dict_feedback
