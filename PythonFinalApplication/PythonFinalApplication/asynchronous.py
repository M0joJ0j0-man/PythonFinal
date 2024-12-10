#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS 331 Final: Litagano & Motscoud
--------------------------------------
Installation: Pensacola Christian College

I pledge all of the lines in this Python program are my own original
work and that none of the lines in this Python program have been copied
from anyone else unless I was specifically authorized to do so by
my CS 331 instructor.

Signature: Josiah Buckley

The objective of this program is to correct any logical and documentation errors
in the program and make sure all 93 tests pass.
"""
import datetime
import _thread  # Provides low-level thread management
import abc as _abc  # Defines abstract base classes
import collections as _collections  # Implements specialized container datatypes
import enum as _enum  # Defines enumerations for symbolic names
import math as _math  # Provides mathematical functions and constants
import multiprocessing as _multiprocessing  # Supports parallel processing with processes
import operator as _operator  # Implements basic operations on functions and objects
import queue as _queue  # Implements FIFO queue for task management
import signal as _signal  # Provides access to signal handling in the OS
import sys as _sys  # Provides system-specific parameters and functions
import time as _time  # Implements time-related functions


__all__ = (
    'Executor',        # Exposes the Executor class for external use
    'get_timeout',     # Exposes the get_timeout function
    'set_timeout',     # Exposes the set_timeout function
    'submit',          # Exposes the submit function for task submission
    'map_',            # Exposes the map function for mapping tasks
    'shutdown'         # Exposes the shutdown function to stop the executor

)

# Module Documentation
__version__ = '1.0.0'
__date__ = datetime.date(2024, 12, 10)
__author__ = 'Josiah Buckley'
__credits__ = 'CS 331'

class _Base(metaclass=_abc.ABCMeta):
    """ This class creates a timeout property that follows certain rules."""

    __slots__ = (
        '__timeout',
    )

    @_abc.abstractmethod
    def __init__(self, timeout):
        """ Initialize timeout with a default of infinity."""
        self.timeout = _math.inf if timeout is None else timeout

    def get_timeout(self):
        """ Return the current timeout value."""
        return self.__timeout

    def set_timeout(self, value):
        """Set a new timeout value; must be positive."""
        if not isinstance(value, (float, int)):
            raise ValueError('value must be of type float or int')
        if value <= 0:
            raise TypeError('value must be greater than zero')
        self.__timeout = value

    timeout = property(get_timeout, set_timeout)


def _run_and_catch(fn, args, kwargs):
    """Run a function and return its result or exception."""
    # noinspection PyPep8,PyBroadException
    try:
        return False, fn(*args, **kwargs)
    except:
        return True, _sys.exc_info()[1]


def _run(fn, args, kwargs, queue):
    """Execute a function and return its result or exception."""
    queue.put_nowait(_run_and_catch(fn, args, kwargs))


class _State(_enum.IntEnum):
    """Enumeration representing the possible states of a task."""
    PENDING = _enum.auto()
    RUNNING = _enum.auto()
    CANCELLED = _enum.auto()
    FINISHED = _enum.auto()
    ERROR = _enum.auto()


def _run_and_catch_loop(iterable, *args, **kwargs):
    """Run and catch errors for each function in the iterable."""
    exception = None
    for fn in iterable:
        error, value = _run_and_catch(fn, args, kwargs)
        if error:
            exception = value
    if exception:
        raise exception


class _Future(_Base):
    """ Represents an asynchronous computation result."""
    __slots__ = (
        '__queue',
        '__process',
        '__start_time',
        '__callbacks',
        '__result',
        '__mutex'
    )

    def __init__(self, timeout, fn, args, kwargs):
        """Initializes the future. Should not be called by clients."""
        super().__init__(timeout)
        self.__queue = _multiprocessing.Queue(1)
        self.__process = _multiprocessing.Process(
            target=_run,
            args=(fn, args, kwargs, self.__queue),
            daemon=True
        )
        self.__start_time = _math.inf
        self.__callbacks = _collections.deque()
        self.__result = True, TimeoutError()
        self.__mutex = _thread.allocate_lock()

    @property
    def __state(self):
        """Defines possible states for a future object."""
        pid, exitcode = self.__process.pid, self.__process.exitcode
        return (_State.PENDING if pid is None else
                _State.RUNNING if exitcode is None else
                _State.CANCELLED if exitcode == -_signal.SIGTERM else
                _State.FINISHED if exitcode == 0 else
                _State.ERROR)

    def __repr__(self):
        """Return a string representation of the future."""
        root = f'{type(self).__name__} at {id(self)} state={self.__state.name}'
        if self.__state < _State.CANCELLED:
            return f'<{root}>'
        error, value = self.__result
        suffix = f'{"raised" if error else "returned"} {type(value).__name__}'
        return f'<{root} {suffix}>'

    def __consume_callbacks(self):
        """Consume all callbacks to be executed."""
        while self.__callbacks:
            yield self.__callbacks.popleft()

    def __invoke_callbacks(self):
        """Invoke callbacks after process completes."""
        self.__process.join()
        _run_and_catch_loop(self.__consume_callbacks(), self)

    def cancel(self):
        """Cancel the future's process and invoke callbacks."""
        self.__process.terminate()
        self.__invoke_callbacks()

    def __auto_cancel(self):
        """Automatically cancel the future if timeout is exceeded."""
        elapsed_time = _time.perf_counter() - self.__start_time
        if elapsed_time > self.timeout:
            self.cancel()
        return elapsed_time

    def cancelled(self):
        """Check if the future has been cancelled."""
        self.__auto_cancel()
        return self.__state is _State.CANCELLED

    def running(self):
        """Check if the future is running."""
        self.__auto_cancel()
        return self.__state is _State.RUNNING

    def done(self):
        """Check if the future is finished."""
        self.__auto_cancel()
        return self.__state > _State.FINISHED

    def __handle_result(self, error, value):
        """Handle the result after the process completion."""
        self.__result = error, value
        self.__invoke_callbacks()

    def __ensure_termination(self):
        """Ensure the future process terminates within the timeout."""
        with self.__mutex:
            elapsed_time = self.__auto_cancel()
            if not self.__queue.empty():
                self.__handle_result(*self.__queue.get_nowait())
            elif self.__state < _State.CANCELLED:
                remaining_time = self.timeout - elapsed_time
                if remaining_time == _math.inf:
                    remaining_time = None
                try:
                    result = self.__queue.get(True, remaining_time)
                except _queue.Empty:
                    self.cancel()
                else:
                    self.__handle_result(*result)

    def result(self):
        """Return the result of the future or raise its exception."""
        self.__ensure_termination()
        error, value = self.__result
        if error:
            return value
        raise value

    def exception(self):
        """Return the exception raised by the future, if any."""
        self.__ensure_termination()
        error, value = self.__result
        if error:
            return value

    def add_done_callback(self, fn):
        """Add a callback to be executed after completion."""
        if self.done():
            fn(self)
        else:
            self.__callbacks.append(fn)

    def _set_running_or_notify_cancel(self):
        """Start the process if it is not already running."""
        if self.__state is _State.PENDING:
            self.cancel()
        else:
            self.__process.start()
            self.__start_time = _time.perf_counter()


class Executor(_Base):
    """A class for managing and executing asynchronous tasks."""
    __slots__ = (
        '__futures',
    )

    def __init__(self, timeout=None):
        """Initialize the executor to manage future tasks."""
        super().__init__(timeout)
        self.__futures = set()

    def submit(self, fn, *args, **kwargs):
        """Submit a function to be run asynchronously."""
        future = _Future(self.timeout, fn, args, kwargs)
        self.__futures.add(future)
        future.add_done_callback(self.__futures.remove)
        # noinspection PyProtectedMember
        future.set_running_or_notify_cancel()
        return future

    @staticmethod
    def __cancel_futures(iterable):
        """"Cancel the given futures."""
        _run_and_catch_loop(map(_operator.attrgetter('cancel'), iterable))

    def map(self, fn, *iterables):
        """Submit a map of function calls for parallel execution."""
        futures = tuple(self.submit(fn, *args) for args in zip(*iterables))

        def result_iterator():
            future_iterator = iter(futures)
            try:
                for future in future_iterator:
                    yield future.result()
            finally:
                self.__cancel_futures(future_iterator)

        return result_iterator()

    def shutdown(self):
        """Shutdown the executor and cancel all futures."""
        self.__cancel_futures(frozenset(self.__futures))

    def __enter__(self):
        """Enter a context where the executor can be used."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and shutdown the executor."""
        self.shutdown()
        return False


_executor = Executor()  # Instantiate the Executor to manage tasks
get_timeout = _executor.get_timeout  # Alias to get the timeout setting
set_timeout = _executor.set_timeout  # Alias to set the timeout value
submit = _executor.submit  # Alias to submit tasks to the executor
map_ = _executor.map  # Alias to apply a function over multiple iterables concurrently
shutdown = _executor.shutdown  # Alias to shut down the executor and cancel pending tasks
del _executor  # Clean up the Executor instance
