
import inspect
import logging

from typing import Callable
from threading import Thread
import multiprocessing as mp


def subproc(cls):
    """ Subprocess decorator """
    _reserved_subproc_keywords = (
        '_process', '_methods', '_p_client', '_p_server')

    class SubprocWrapper:
        def __init__(self, *args, **kwargs):
            # Start bidirectional pipe.
            p_client, p_server = mp.Pipe()

            def run(*args, **kwargs):
                # Spawn class instance.
                instance = cls(*args, **kwargs)
                logging.debug('spawned instance')

                # Get some information about routines
                methods = inspect.getmembers(instance, inspect.isroutine)
                p_server.send([m[0] for m in methods])

                # Listen to INCOMING data
                logging.debug('defining callback ...')

                def on_data(data):
                    # unpack data ...
                    name, args, kwargs = data
                    attr = getattr(instance, name)

                    # Deal with functions.
                    # syntax = (name, args, kwargs)
                    if inspect.isroutine(attr):
                        method = attr
                        p_server.send(method(*args, **kwargs))
                        return

                    # Deal with properties.
                    if args:
                        # setattr syntax = (name, [value], {})
                        setattr(instance, name, args[0])
                        return

                    # getattr syntax = (name)
                    p_server.send(attr)

                # listener.start()
                logging.debug('starting listening ...')
                while True:
                    try:
                        data = p_server.recv()
                    except Exception as e:
                        logging.error(e)
                        # continue
                        break
                    on_data(data)

            self._process = mp.Process(target=run, args=args, kwargs=kwargs)
            self._process.start()

            # Retrieve list of methods and close temporary pipe.
            methods = p_client.recv()
            self._methods = methods

            # Save pipes.
            self._p_server = p_server
            self._p_client = p_client

        def __getattr__(self, name):
            # Overrides for reserved keywords.
            if name in _reserved_subproc_keywords:
                return super().__getattribute__(name)

            # Deal with methods.
            if name in self._methods:
                def caller(*args, **kwargs):
                    self._p_client.send((name, args, kwargs))
                    return self._p_client.recv()
                return caller
            else:
                # Deal with attributes.
                self._p_client.send((name, [], {}))
                return self._p_client.recv()

        def __setattr__(self, name, value):
            # Overrides for reserved keywords.
            if name in _reserved_subproc_keywords:
                return super().__setattr__(name, value)
            self._p_client.send((name, [value], {}))

        def __del__(self):
            self._process.terminate()

    return SubprocWrapper


def main():
    class SimpleRunner(object):
        """ Sample class to be wrapped. """

        def __init__(self, x=5, y=4):
            self.x = x
            self.y = y

        def get_y(self):
            return self.y

        def set_y(self, value):
            self.y = value

    @subproc
    class SubprocRunner(SimpleRunner):
        pass
    runner = SubprocRunner(x=6)
    print(runner.x)
    print(runner.get_y())
    print(runner.set_y(15))
    print(runner.get_y())  # 15
    runner.x = 6
    print(runner.x)  # 6


if __name__ == '__main__':
    main()
