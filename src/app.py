import argparse

from engine import pluginengine
from util import filesystem
# Change this to be a flask app



def __description() -> str:
    return "This is an application with a plugin architecture\n \
            the main objective of this application is to provide mechanisms\n \
            of face recogtion, identification, and provide quality parameters related to a photo.\n \
            Lastly this application returns a list of metrics that are to be used in the decision of updating a photo"


def __usage() -> str:
    return "python3 app.py"


def __init_cli() -> argparse:
    parser = argparse.ArgumentParser(description=__description(), usage=__usage())
    parser.add_argument(
        '-l', '--log', default='DEBUG', help="""
        Specify log level which should use. Default will always be DEBUG, choose between the following options
        CRITICAL, ERROR, WARNING, INFO, DEBUG
        """
    )
    parser.add_argument(
        '-d', '--directory', default=f'{FileSystem.get_plugins_directory()}', help="""
        (Optional) Supply a directory where plugins should be loaded from. The default is ./plugins
        """
    )
    return parser


def __print_program_end() -> None:
    print("-----------------------------------")
    print("End of execution")
    print("-----------------------------------")


def __init_app(parameters: dict, **args) -> None:
    PluginEngine(options=parameters).start(args)


if __name__ == '__main__':
    __cli_args = __init_cli().parse_args()
    __init_app({
        'log_level': __cli_args.log,
        'directory': __cli_args.directory
    })
