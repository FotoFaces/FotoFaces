#core.py
import importlib
import sys

class MyApplication:
    # We are going to receive a list of plugins as parameter
    def __init__(self, plugins:list=[]):
        # Checking if plugin were sent
        if plugins != []:
            # create a list of plugins
            self._plugins = [
                # Import the module and initialise it at the same time
                importlib.import_module(plugin,".").Plugin() for plugin in plugins
            ]
        else:
            # If no plugin were set we use our default
            self._plugins = [importlib.import_module('default',".").Plugin()]


    # no pedido
    def run(self):
        print("Starting my application")
        print("-" * 10)
        print("This is my core system")

        # We is were magic happens, and all the plugins are going to be printed
        for plugin in self._plugins:
            plugin.process(1,2)

        print("-" * 10)
        print("Ending my application")
        print()


if __name__ == "__main__":
    # Initialize our app with the parameters received from CLI with the sys.argv
    # Starts from the possion one due to position 0 will be main.py
    app = MyApplication(sys.argv[1:])
    #app = MyApplication()
    # Run our application
    app.run()
