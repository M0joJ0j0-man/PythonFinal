#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the asynchronous module.

Installation: Pensacola Christian College

I pledge all the lines in this Python program are my own original work and that
none of the lines in this Python program have been copied from anyone else
unless I was specifically authorized to do so by my CS 331 instructor.

    Signed: _______________________Stephen_Paul_Chappell_______________________
                                        (signature)

This program will validate the operation of the asynchronous module for the
final exam of the class. The first week provided the opportunity to examine
the assigned code, and the second week allows for fixing its many problems."""

import _thread
import atexit
import contextlib
import datetime
import functools
import inspect
import io
import itertools
import math
import operator
import os
import queue
import sys
import time
import unittest

import asynchronous

# Public Names
__all__ = (
    'TestExecutorAPI',
    'TestModuleAPI',
    'TestFutureAPI'
)

# Module Documentation
__version__ = 1, 1, 0
__date__ = datetime.date(2024, 12, 4)
__author__ = 'Stephen Paul Chappell'
__credits__ = 'CS 331'


# noinspection PyUnresolvedReferences
class TestConstructor:
    """Class that assists with creating instances and testing the timeout."""

    # noinspection SpellCheckingInspection
    KILOBIT = 1 << 7
    MEGABYTE = 1 << 20

    def instantiate(self, *args):
        """Create a new instance based on the CLASS attribute."""
        parameters = len(inspect.signature(self.CLASS).parameters)
        return self.CLASS(*args[:parameters])

    def test_valid_timeout_with_none(self):
        """Validate that a 'none' timeout value is handled properly."""
        instance = self.instantiate(None, print, (), {})
        self.assertEqual(
            math.inf,
            instance.get_timeout(),
            'Timeout should be infinity if initialized with None.'
        )

    def test_valid_timeout_with_one(self):
        """Validate that a 'one' timeout value is handled properly."""
        instance = self.instantiate(1, print, (), {})
        self.assertEqual(
            1,
            instance.get_timeout(),
            'Timeout should be one if initialized with one.'
        )

    def test_valid_timeout_with_small(self):
        """Validate that a 'small' timeout value is handled properly."""
        float_timeout = (math.e ** (1j * math.pi) + 1).imag
        self.assertIsInstance(
            float_timeout,
            float,
            'float_timeout is expected to be a float.'
        )
        instance = self.instantiate(float_timeout, print, (), {})
        self.assertEqual(
            float_timeout,
            instance.get_timeout(),
            'Timeout should retain any value above zero even if small.'
        )

    def test_error_timeout_with_str(self):
        """Validate that a 'str' timeout causes type error exception."""
        self.assertRaises(TypeError, self.instantiate, '60', print, (), {})

    def test_error_timeout_with_zero(self):
        """Validate that a 'zero' timeout causes value error exception."""
        self.assertRaises(ValueError, self.instantiate, 0, print, (), {})

    def test_error_timeout_with_negative(self):
        """Validate that a 'negative' timeout causes value error exception."""
        self.assertRaises(ValueError, self.instantiate, -1, print, (), {})


# noinspection PyUnresolvedReferences
class TestTimeout(TestConstructor):
    """Class that tests the timeout property of an instance."""

    def test_valid_int_property(self):
        """Validate that the timeout property can be set as an int."""
        instance = self.instantiate(None, None, None, None)
        instance.timeout = 1
        self.assertIsInstance(
            instance.timeout,
            int,
            'Timeout should be int if initialized with int.'
        )

    def test_valid_float_property(self):
        """Validate that the timeout property can be set as a float."""
        instance = self.instantiate(None, None, None, None)
        instance.timeout = 0.5
        self.assertIsInstance(
            instance.timeout,
            float,
            'Timeout should be float if initialized with float.'
        )

    def test_valid_set_and_get_property(self):
        """Validate that the timeout property can be mutated and accessed."""
        instance = self.instantiate(None, None, None, None)
        kilo_bit = int.from_bytes(os.urandom(self.KILOBIT), 'big')
        instance.timeout = kilo_bit
        self.assertEqual(
            kilo_bit,
            instance.timeout,
            'Timeout should be whatever value it was set to.'
        )

    def test_error_property_with_str(self):
        """Validate that str timeout causes type error exception."""
        instance = self.instantiate(None, None, None, None)
        with self.assertRaises(TypeError, msg='Timeout cannot be str.'):
            instance.timeout = 'inf'
        self.assertEqual(
            math.inf,
            instance.timeout,
            'Timeout should retain original value on error.'
        )

    def test_error_property_with_complex(self):
        """Validate that complex timeout causes type error exception."""
        instance = self.instantiate(None, None, None, None)
        with self.assertRaises(TypeError, msg='Timeout cannot be complex.'):
            instance.timeout = complex(123456789, 0)
        self.assertEqual(
            math.inf,
            instance.timeout,
            'Timeout should retain original value on error.'
        )

    def test_error_property_with_zero_int(self):
        """Validate that zero as int causes value error exception."""
        instance = self.instantiate(None, None, None, None)
        with self.assertRaises(ValueError, msg='Timeout cannot be 0 int.'):
            instance.timeout = 0
        self.assertEqual(
            math.inf,
            instance.timeout,
            'Timeout should retain original value on error.'
        )

    def test_error_property_with_zero_float(self):
        """Validate that zero as float causes value error exception."""
        instance = self.instantiate(None, None, None, None)
        with self.assertRaises(ValueError, msg='Timeout cannot be 0 float.'):
            instance.timeout = 0.0
        self.assertEqual(
            math.inf,
            instance.timeout,
            'Timeout should retain original value on error.'
        )

    def test_error_property_with_negative_one(self):
        """Validate that negative one causes value error exception."""
        instance = self.instantiate(None, None, None, None)
        with self.assertRaises(ValueError, msg='Timeout cannot be < 0 int.'):
            instance.timeout = -1
        self.assertEqual(
            math.inf,
            instance.timeout,
            'Timeout should retain original value on error.'
        )

    def test_error_property_with_negative_pi(self):
        """Validate that negative pi causes value error exception."""
        instance = self.instantiate(None, None, None, None)
        with self.assertRaises(ValueError, msg='Timeout cannot be < 0 float.'):
            instance.timeout = -math.pi
        self.assertEqual(
            math.inf,
            instance.timeout,
            'Timeout should retain original value on error.'
        )


class Timer:
    """Class that provides a timing interface per thread."""

    DEFAULT_ERROR_RANGE = 1 / 2 # one half second
    __timers = {}

    @classmethod
    def start_timer(cls):
        """Begin a timer for the current thread if not yet running."""
        ident, now = _thread.get_ident(), time.perf_counter()
        if now is not cls.__timers.setdefault(ident, now):
            raise KeyError(ident)

    @classmethod
    def stop_timer(cls, expected_time, error=None):
        """End this thread's timer and check if timing is as expected."""
        if error is None:
            error = cls.DEFAULT_ERROR_RANGE
        used = time.perf_counter() - cls.__timers.pop(_thread.get_ident())
        absolute_difference = abs(used - expected_time)
        return absolute_difference <= error


# noinspection PyUnresolvedReferences
class TestTimer(Timer):
    """Class that extends Timer with a TestCase API for easy use."""

    def stop_timer(self, expected_time, error=None):
        """Validate that the expected timing is within margin of error."""
        self.assertTrue(
            super().stop_timer(expected_time, error),
            'Timer should end within margin of error.'
        )


def delay_run(delay, fn, *args, sync=True, **kwargs):
    """Run a function with a desired delay (possibly in a separate thread)."""

    def wrapper():
        """Create a closure in case the function is run asynchronously."""
        time.sleep(delay)
        return fn(*args, **kwargs)

    if sync:
        return wrapper()
    _thread.start_new_thread(wrapper, ())


# noinspection PyUnresolvedReferences
class TestModuleOrInstance(TestTimer):
    """Class that can assist with testing either a module or class instance."""

    @property
    def moi(self):
        """Property that provides shortcut to accessing MODULE_OR_INSTANCE."""
        return self.MODULE_OR_INSTANCE

    def test_valid_timeout_with_inf(self):
        """Validate that the timeout can be set to infinity."""
        self.moi.set_timeout(math.inf)
        self.assertEqual(
            math.inf,
            self.moi.get_timeout(),
            'Timeout should be infinity if set to infinity.'
        )

    def test_valid_timeout_with_one_minute(self):
        """Validate that the timeout can be set to 1 minute."""
        self.moi.set_timeout(60)
        self.assertEqual(
            60,
            self.moi.get_timeout(),
            'Timeout should be 1 minutes if set to 1 minute.'
        )

    def test_valid_timeout_with_one_twentieth_second(self):
        """Validate that the timeout can be set to 1 / 20 second."""
        self.moi.set_timeout(0.05)
        self.assertEqual(
            0.05,
            self.moi.get_timeout(),
            'Timeout should be 1 / 20 second if set to 1 / 20 second.'
        )

    def test_error_timeout_with_none(self):
        """Validate type error exception raised with None timeout."""
        self.moi.set_timeout(math.inf)
        self.assertRaises(TypeError, self.moi.set_timeout, None)
        self.assertEqual(
            math.inf,
            self.moi.get_timeout(),
            'Timeout should retain original value on error.'
        )

    def test_error_timeout_with_zero(self):
        """Validate value error exception raised with 0 timeout."""
        self.moi.set_timeout(math.inf)
        self.assertRaises(ValueError, self.moi.set_timeout, 0)
        self.assertEqual(
            math.inf,
            self.moi.get_timeout(),
            'Timeout should retain original value on error.'
        )

    def test_error_timeout_with_negative(self):
        """Validate value error exception raised with -1 timeout."""
        self.moi.set_timeout(math.inf)
        self.assertRaises(ValueError, self.moi.set_timeout, -1)
        self.assertEqual(
            math.inf,
            self.moi.get_timeout(),
            'Timeout should retain original value on error.'
        )

    def run_submit_check(self):
        """Validate that submit returns a properly operating _Future."""
        self.start_timer()
        future = self.moi.submit(delay_run, 0.5, operator.add, 1, 2)
        self.assertRegex(
            repr(future),
            r'^<_Future at \d+ state=RUNNING>$',
            'Representation should indicate running.'
        )
        self.assertEqual(
            3,
            future.result(),
            'Adding 1 and 2 should yield 3.'
        )
        self.stop_timer(0.5)
        self.assertRegex(
            repr(future),
            r'^<_Future at \d+ state=FINISHED returned int>$',
            'Representation should indicate finished.'
        )

    def test_submit_one_second_timeout(self):
        """Validate submit works properly with a 1-second timeout."""
        self.moi.set_timeout(1)
        self.run_submit_check()

    def test_submit_no_timeout(self):
        """Validate submit works properly with an infinite timeout."""
        self.moi.set_timeout(math.inf)
        self.run_submit_check()

    def test_submit_short_timeout(self):
        """Validate submit works properly with a half-second timeout."""
        self.moi.set_timeout(0.5)
        self.start_timer()
        future = self.moi.submit(delay_run, 1, operator.add, 1, 2)
        self.assertRegex(
            repr(future),
            r'^<_Future at \d+ state=RUNNING>$',
            'Representation should indicate running.'
        )
        self.assertIsInstance(
            future.exception(),
            TimeoutError,
            'Exception should indicate a timeout error.'
        )
        self.stop_timer(0.5)
        self.assertRegex(
            repr(future),
            r'^<_Future at \d+ state=CANCELLED raised TimeoutError>$',
            'Representation should indicate cancelled.'
        )

    def run_map(self, *args):
        """Assist with running the map method of a module or instance."""
        return getattr(self.moi, self.NAME_OF_MAP)(delay_run, *args)

    def test_valid_map(self):
        """Validate the map method when it is being using properly."""
        self.moi.set_timeout(1.5)
        for result in self.run_map(
                [1, 1, 1, 1],
                [operator.add] * 4,
                [0, 1, 2, 3],
                [3, 2, 1, 0]
        ):
            self.assertEqual(
                3,
                result,
                'Adding pairs across last two lists should yield 3.'
            )

    def test_error_map(self):
        """Validate the map method when it causes an expected error."""
        self.moi.set_timeout(1.5)
        success = 0
        with self.assertRaises(TimeoutError, msg='The 3rd add should fail.'):
            for result in self.run_map(
                    [1, 1, 2, 1],
                    [operator.add] * 4,
                    [0, 1, 2, 3],
                    [3, 2, 1, 0]
            ):
                self.assertEqual(
                    3,
                    result,
                    'Adding pairs across last two lists should yield 3.'
                )
                success += 1
        self.assertEqual(
            2,
            success,
            'Only first two add operations should be successful.'
        )

    def run_shutdown_check(self, running, future):
        """Assist with testing operations when shutting down an Executor."""
        self.assertRaises(TimeoutError, future.result)
        running.remove(future)

    def run_submit_loop(self, executor):
        """Create several _Future instances to test during shutdown."""
        running = set()
        done_callback = functools.partial(self.run_shutdown_check, running)
        for _ in range(10):
            future = executor.submit(delay_run, 2, operator.add, 10, 20)
            running.add(future)
            future.add_done_callback(done_callback)
        time.sleep(0.5)
        return running

    def test_valid_shutdown(self):
        """Validate the shutdown method can complete resource cleanup."""
        self.moi.set_timeout(1.5)
        running = self.run_submit_loop(self.moi)
        self.moi.shutdown()
        self.assertFalse(
            running,
            'Nothing should be running after shutdown is complete.'
        )

    def test_error_shutdown(self):
        """Validate shutdown method responds properly with errors."""
        self.moi.set_timeout(1.5)
        running = self.run_submit_loop(self.moi)
        running.pop()
        self.assertRaises(KeyError, self.moi.shutdown)
        self.assertFalse(
            running,
            'Nothing should be running after shutdown is complete.'
        )


class TestExecutorAPI(TestTimeout, TestModuleOrInstance, unittest.TestCase):
    """Class that combines many tests into one for the Executor interface."""

    CLASS = asynchronous.Executor
    MODULE_OR_INSTANCE = CLASS()
    NAME_OF_MAP = 'map'

    def test_valid_context_manager(self):
        """Validate instance usage when using Executors in a context."""
        with self.instantiate(1.5) as executor:
            running = self.run_submit_loop(executor)
        self.assertFalse(
            running,
            'Nothing should be running after with block completes.'
        )

    def test_error_context_manager_with_exception(self):
        """Validate with context block response to raise statement."""
        error = Exception()
        with self.assertRaises(Exception, msg='Error should be caught.') as cm:
            with self.instantiate(1.5) as executor:
                running = self.run_submit_loop(executor)
                raise error
        self.assertIs(
            cm.exception,
            error,
            'Exception should be same as that raised.'
        )
        self.assertFalse(
            running,
            'Nothing should be running after with block completes.'
        )

    def test_error_context_manager_with_key_error(self):
        """Validate with context block response to key error exception."""
        with self.assertRaises(KeyError, msg='Problem should be detected.'):
            with self.instantiate(1.5) as executor:
                running = self.run_submit_loop(executor)
                running.pop()
        self.assertFalse(
            running,
            'Nothing should be running after with block completes.'
        )


class TestModuleAPI(TestModuleOrInstance, unittest.TestCase):
    """Class that specifically tests the module's functional interface."""

    MODULE_OR_INSTANCE = asynchronous
    NAME_OF_MAP = 'map_'


def verify_error():
    """Look for a very specific problem and check that it occurs."""
    sys.stderr.seek(0, io.SEEK_SET)
    for line in sys.stderr:
        if line == 'queue.Full\n':
            break
    else:
        sys.stderr.seek(0, io.SEEK_SET)
        sys.__stderr__.write(sys.stderr.read())
        sys.__stderr__.flush()


def cause_error(obj):
    """Using unconventional instructions, break the asynchronous module."""
    sys.stderr = io.StringIO()
    atexit.register(verify_error)
    inspect.currentframe().f_back.f_back.f_locals['queue'].put_nowait(obj)


def return_(obj):
    """Since return is a keyword, give it a functional interface."""
    return obj


# noinspection PyUnusedLocal
def throw(exception, *args):
    """Since raise is a keyword, give it a functional interface."""
    raise exception


class Silencer:
    """Class that can modify an object on a per-thread basis."""

    def __init__(self, silenced):
        """Initialize the Silencer instance."""
        self.__silenced = silenced
        self.__ident = _thread.get_ident()

    @property
    def silenced(self):
        """Property allowing access to the silenced object."""
        return self.__silenced

    def __getattr__(self, name):
        """Return the desired attribute only on the creating thread."""
        return (getattr(self.__silenced, name)
                if _thread.get_ident() == self.__ident else
                self)

    def __call__(self, *args, **kwargs):
        """Capture object invocations and return generic results."""
        return self


@contextlib.contextmanager
def silence_other_threads():
    """Generate a with context to silence standard output and error."""
    sys.stdout, sys.stderr = Silencer(sys.stdout), Silencer(sys.stderr)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = sys.stdout.silenced, sys.stderr.silenced


class TestFutureAPI(TestTimer, TestTimeout, unittest.TestCase):
    """Class that does extensive testing on _Future instances."""

    CLASS = asynchronous._Future

    def test_valid_representation_when_cancelled(self):
        """Validate _Future representations when they are cancelled."""
        future = self.instantiate(None, time.sleep, (0.1,), {})
        self.assertRegex(
            repr(future),
            r'^<_Future at \d+ state=PENDING>$',
            'Representation should indicate pending.'
        )
        future._set_running_or_notify_cancel()
        self.assertRegex(
            repr(future),
            r'^<_Future at \d+ state=RUNNING>$',
            'Representation should indicate running.'
        )
        future._set_running_or_notify_cancel()
        self.assertRegex(
            repr(future),
            r'^<_Future at \d+ state=CANCELLED raised TimeoutError>$',
            'Representation should indicate cancelled.'
        )

    def test_valid_representation_when_finished(self):
        """Validate _Future representations when they are finished."""
        future = self.instantiate(None, time.sleep, (0.1,), {})
        future._set_running_or_notify_cancel()
        time.sleep(0.5)
        self.assertRegex(
            repr(future),
            r'^<_Future at \d+ state=FINISHED raised TimeoutError>$',
            'Representation should indicate finished.'
        )
        self.assertIsNone(
            future.exception(),
            'There should be no exception on success.'
        )
        self.assertRegex(
            repr(future),
            r'^<_Future at \d+ state=FINISHED returned NoneType>$',
            'Representation should indicate finished.'
        )

    def test_error_representation_with_exception(self):
        """Validate that a _Future can detect an error with an exception."""
        future = self.instantiate(0.5, cause_error, (None,), {})
        future._set_running_or_notify_cancel()
        self.assertRaises(TypeError, future.result)
        self.assertIsInstance(
            future.exception(),
            TimeoutError,
            'Exception should indicate a timeout error.'
        )
        self.assertRegex(
            repr(future),
            r'^<_Future at \d+ state=ERROR raised TimeoutError>$',
            'Representation should indicate error.'
        )

    def test_error_representation_with_result(self):
        """Validate that a _Future can detect an error with a result."""
        future = self.instantiate(0.5, cause_error, ((False, 'okay'),), {})
        future._set_running_or_notify_cancel()
        self.assertEqual(
            'okay',
            future.result(),
            'Bad code can cause errors but still return values.'
        )
        self.assertRegex(
            repr(future),
            r'^<_Future at \d+ state=ERROR returned str>$',
            'Representation should indicate error.'
        )

    def test_cancel_without_callback(self):
        """Validate that the cancel method works without a done callback."""
        future = self.instantiate(None, time.sleep, (0.1,), {})
        self.assertRaises(AttributeError, future.cancel)
        future._set_running_or_notify_cancel()
        future.cancel()
        self.assertTrue(
            future.cancelled(),
            'Cancelled should be true after cancel is called.'
        )

    def test_cancel_with_callback(self):
        """Validate that the cancel method works with a done callback."""
        future = self.instantiate(None, time.sleep, (0.1,), {})
        checker = set()
        future.add_done_callback(checker.add)
        future._set_running_or_notify_cancel()
        future.cancel()
        # XXX Should cancel be called again in this test?
        future.cancel()
        self.assertIs(
            checker.pop(),
            future,
            'Callback should receive future as first argument.'
        )
        self.assertFalse(
            checker,
            'Set should be empty at this point in the test.'
        )

    def test_cancelled_with_result(self):
        """Validate that cancelled returns proper values during normal run."""
        future = self.instantiate(None, time.sleep, (0.1,), {})
        self.assertFalse(
            future.cancelled(),
            'Pending future should not be cancelled.'
        )
        future._set_running_or_notify_cancel()
        self.assertFalse(
            future.cancelled(),
            'Running future should not be cancelled.'
        )
        self.assertIsNone(
            future.result(),
            'time.sleep does not return anything.'
        )
        self.assertFalse(
            future.cancelled(),
            'Finished future should not be cancelled.'
        )

    def test_cancelled_with_cancel(self):
        """Validate that cancelled returns proper value when cancelled."""
        future = self.instantiate(None, time.sleep, (0.1,), {})
        future._set_running_or_notify_cancel()
        future.cancel()
        self.assertTrue(
            future.cancelled(),
            'Future should be cancelled after calling cancel.'
        )

    def test_cancelled_with_timeout(self):
        """Validate that cancelled returns proper value on timeout."""
        future = self.instantiate(0.1, time.sleep, (1,), {})
        future._set_running_or_notify_cancel()
        time.sleep(0.5)
        self.assertTrue(
            future.cancelled(),
            'Future should be cancelled after timeout expires.'
        )

    def test_running_with_result(self):
        """Validate that running returns proper values during normal run."""
        future = self.instantiate(None, time.sleep, (0.1,), {})
        self.assertFalse(
            future.running(),
            'Pending future should not be running.'
        )
        future._set_running_or_notify_cancel()
        self.assertTrue(
            future.running(),
            'Running future should be running.'
        )
        self.assertIsNone(
            future.result(),
            'time.sleep does not return anything.'
        )
        self.assertFalse(
            future.running(),
            'Finished future should not be running.'
        )

    def test_running_with_cancel(self):
        """Validate that running returns proper value when cancelled."""
        future = self.instantiate(None, time.sleep, (0.1,), {})
        future._set_running_or_notify_cancel()
        future.cancel()
        self.assertFalse(
            future.running(),
            'Future should not be running after calling cancel.'
        )

    def test_running_with_timeout(self):
        """Validate that running returns proper value on timeout."""
        future = self.instantiate(0.1, time.sleep, (1,), {})
        future._set_running_or_notify_cancel()
        time.sleep(0.5)
        self.assertFalse(
            future.running(),
            'Future should not be running after timeout expires.'
        )

    def test_done_with_result(self):
        """Validate that done returns proper values during normal run."""
        future = self.instantiate(None, time.sleep, (0.1,), {})
        self.assertFalse(
            future.done(),
            'Pending future should not be done.'
        )
        future._set_running_or_notify_cancel()
        self.assertFalse(
            future.done(),
            'Running future should not be done.'
        )
        self.assertIsNone(
            future.result(),
            'time.sleep does not return anything.'
        )
        self.assertTrue(
            future.done(),
            'Finished future should be done.'
        )

    def test_done_with_exception(self):
        """Validate that done returns proper value when there is an error."""
        future = self.instantiate(None, time.sleep, (None,), {})
        future._set_running_or_notify_cancel()
        self.assertIsInstance(
            future.exception(),
            TypeError,
            'Exception should indicate a type error.'
        )
        self.assertTrue(
            future.done(),
            'Finished future should be done.'
        )

    def test_result_immediate_with_result(self):
        """Validate that results can be received without delay."""
        data = os.urandom(self.MEGABYTE)
        future = self.instantiate(None, return_, (data,), {})
        future._set_running_or_notify_cancel()
        self.assertEqual(
            data,
            future.result(),
            'Data should pass through future unchanged.'
        )

    def test_result_immediate_with_exception(self):
        """Validate that exceptions can be received without delay."""
        test_exception = Exception('test')
        future = self.instantiate(None, throw, (test_exception,), {})
        future._set_running_or_notify_cancel()
        with self.assertRaises(Exception, msg='Error is expected.') as cm:
            future.result()
        self.assertIsInstance(
            cm.exception,
            type(test_exception),
            'Exception type should match that of test value.'
        )
        self.assertEqual(
            test_exception.args,
            cm.exception.args,
            'Exception arguments should match that of test value.'
        )

    def test_result_delay_without_timeout_without_sleep(self):
        """Validate proper result without timeout and without sleeping."""
        future = self.instantiate(None, delay_run, (0, operator.add, 1, 2), {})
        self.start_timer()
        future._set_running_or_notify_cancel()
        self.assertEqual(
            3,
            future.result(),
            'Adding 1 and 2 should yield 3.'
        )
        self.stop_timer(0.1)

    def test_result_delay_without_timeout_with_sleep(self):
        """Validate proper result without timeout and with sleeping."""
        future = self.instantiate(None, delay_run, (1, operator.add, 2, 3), {})
        self.start_timer()
        future._set_running_or_notify_cancel()
        self.assertEqual(
            5,
            future.result(),
            'Adding 2 and 3 should yield 5.'
        )
        self.stop_timer(1)

    def test_result_delay_with_timeout_without_sleep(self):
        """Validate proper result with timeout and without sleeping."""
        future = self.instantiate(0.5, delay_run, (0, operator.add, 1, 2), {})
        self.start_timer()
        future._set_running_or_notify_cancel()
        self.assertEqual(
            3,
            future.result(),
            'Adding 1 and 2 should yield 3.'
        )
        self.stop_timer(0.1)

    def test_result_delay_with_timeout_with_sleep(self):
        """Validate proper result with timeout and with sleeping."""
        future = self.instantiate(0.5, delay_run, (1, operator.add, 2, 3), {})
        self.start_timer()
        future._set_running_or_notify_cancel()
        self.assertRaises(TimeoutError, future.result)
        self.stop_timer(0.5)

    def test_result_before_running(self):
        """Validate the result method before _Future begins running."""
        future = self.instantiate(0.1, delay_run, (0, operator.add, 1, 2), {})
        delay_run(0.5, future._set_running_or_notify_cancel, sync=False)
        self.start_timer()
        self.assertEqual(
            3,
            future.result(),
            'Adding 1 and 2 should yield 3.'
        )
        self.stop_timer(0.5)

    def run_time_check(self, test):
        """Assist with check the expected speed of the test."""
        self.start_timer()
        test()
        self.stop_timer(0.5)

    def run_waiter_check(self, threads, *tests):
        """Assist with running several tests in a threaded environment."""
        future = self.instantiate(1, delay_run, (0.5, operator.add, 1, 2), {})
        future._set_running_or_notify_cancel()
        result = queue.SimpleQueue()
        with silence_other_threads():
            for test in itertools.islice(itertools.cycle(tests), threads):
                args = self.run_time_check, (lambda: test(future),), {}, result
                _thread.start_new_thread(asynchronous._run, args)
            for _ in range(threads):
                error, value = result.get(True, 1.5)
                self.assertFalse(
                    error,
                    'None of the threads should experience an exception.'
                )

    def test_result_with_waiters(self):
        """Validate the result method works properly when used with threads."""
        self.run_waiter_check(
            10,
            lambda future: self.assertEqual(
                3,
                future.result(),
                'Adding 1 and 2 should yield 3.'
            )
        )

    def test_exception_immediate_without_error(self):
        """Validate absence of exception when there is no error."""
        data = os.urandom(self.MEGABYTE)
        future = self.instantiate(None, return_, (data,), {})
        future._set_running_or_notify_cancel()
        self.assertIsNone(
            future.exception(),
            'There should be no exception on success.'
        )

    def test_exception_immediate_with_error(self):
        """Validate presence of exception when there is an error."""
        test_exception = Exception('test')
        future = self.instantiate(None, throw, (test_exception,), {})
        future._set_running_or_notify_cancel()
        self.assertIsInstance(
            future.exception(),
            type(test_exception),
            'Exception type should match that of test value.'
        )
        self.assertEqual(
            test_exception.args,
            future.exception().args,
            'Exception arguments should match that of test value.'
        )

    def test_exception_delay_without_timeout_without_sleep(self):
        """Validate exception value without timeout and without sleeping."""
        future = self.instantiate(None, delay_run, (0, operator.add, 1, 2), {})
        self.start_timer()
        future._set_running_or_notify_cancel()
        self.assertIsNone(
            future.exception(),
            'There should be no exception on success.'
        )
        self.stop_timer(0.1)

    def test_exception_delay_without_timeout_with_sleep(self):
        """Validate exception value without timeout and with sleeping."""
        future = self.instantiate(None, delay_run, (1, operator.add, 2, 3), {})
        self.start_timer()
        future._set_running_or_notify_cancel()
        self.assertIsNone(
            future.exception(),
            'There should be no exception on success.'
        )
        self.stop_timer(1)

    def test_exception_delay_with_timeout_without_sleep(self):
        """Validate exception value with timeout and without sleeping."""
        future = self.instantiate(0.5, delay_run, (0, operator.add, 1, 2), {})
        self.start_timer()
        future._set_running_or_notify_cancel()
        self.assertIsNone(
            future.exception(),
            'There should be no exception on success.'
        )
        self.stop_timer(0.1)

    def test_exception_delay_with_timeout_with_sleep(self):
        """Validate exception value with timeout and with sleeping."""
        future = self.instantiate(0.5, delay_run, (1, operator.add, 2, 3), {})
        self.start_timer()
        future._set_running_or_notify_cancel()
        self.assertIsInstance(
            future.exception(),
            TimeoutError,
            'Exception should indicate a timeout error.'
        )
        self.assertFalse(
            future.exception().args,
            'Exception arguments should be empty.'
        )
        self.stop_timer(0.5)

    def test_exception_before_running(self):
        """Validate the exception method before _Future begins running."""
        future = self.instantiate(0.1, delay_run, (0, operator.add, 1, 2), {})
        delay_run(0.5, future._set_running_or_notify_cancel, sync=False)
        self.start_timer()
        self.assertIsNone(
            future.exception(),
            'There should be no exception on success.'
        )
        self.stop_timer(0.5)

    def test_exception_with_waiters(self):
        """Validate the exception method is a threaded environment."""
        self.run_waiter_check(
            10,
            lambda future: self.assertIsNone(
                future.exception(),
                'There should be no exception on success.'
            )
        )

    def test_result_followed_by_exception_waiters(self):
        """Validate usage of result method followed by exception method."""
        self.run_waiter_check(
            10,
            lambda future: self.assertEqual(
                3,
                future.result(),
                'Adding 1 and 2 should yield 3.'
            ),
            lambda future: self.assertIsNone(
                future.exception(),
                'There should be no exception on success.'
            )
        )

    def test_exception_followed_by_result_waiters(self):
        """Validate usage of exception method followed by result method."""
        self.run_waiter_check(
            10,
            lambda future: self.assertIsNone(
                future.exception(),
                'There should be no exception on success.'
            ),
            lambda future: self.assertEqual(
                3,
                future.result(),
                'Adding 1 and 2 should yield 3.'
            )
        )

    def test_valid_add_done_callback(self):
        """Validate that callbacks can be added to _Future instances."""
        future = self.instantiate(None, time.sleep, (0,), {})
        requires_callback = {future}
        future.add_done_callback(requires_callback.remove)
        self.assertIn(
            future,
            requires_callback,
            'The callback should still be pending execution.'
        )
        future._set_running_or_notify_cancel()
        self.assertIsNone(
            future.exception(),
            'There should be no exception on success.'
        )
        self.assertFalse(
            requires_callback,
            'The callback should have removed the future.'
        )
        requires_callback.add(future)
        future.add_done_callback(requires_callback.remove)
        self.assertFalse(
            requires_callback,
            'Callback should be run immediately with finished future.'
        )

    def test_error_add_done_callback(self):
        """Validate the operation of erroneous callbacks."""
        future = self.instantiate(None, time.sleep, (0,), {})
        requires_callback = [{future} for _ in range(10)]
        callbacks = [s.remove for s in requires_callback]
        error = Exception()
        callbacks.insert(5, functools.partial(throw, error))
        for fn in callbacks:
            future.add_done_callback(fn)
        future._set_running_or_notify_cancel()
        with self.assertRaises(Exception, msg='Error is expected.') as cm:
            future.exception()
        self.assertIs(
            cm.exception,
            error,
            'Exception should be same as that raised.'
        )
        self.assertFalse(
            any(requires_callback),
            'All callbacks should be run to completion.'
        )

    def test_set_running_or_notify_cancel(self):
        """Validate the _Future's semi-private method used internally."""
        future = self.instantiate(None, time.sleep, (0.1,), {})
        self.assertFalse(
            future.running() or future.done(),
            'Future should not be running or done when pending.'
        )
        future._set_running_or_notify_cancel()
        self.assertTrue(
            future.running(),
            'First state transition should be from pending to running.'
        )
        future._set_running_or_notify_cancel()
        self.assertTrue(
            future.cancelled(),
            'Second state transition should be from running to cancelled.'
        )

    def test_not_empty_queue(self):
        """Validate the usage of queues for communication purposes."""
        data = os.urandom(self.MEGABYTE)
        future = self.instantiate(None, return_, (data,), {})
        future._set_running_or_notify_cancel()
        result = queue.SimpleQueue()
        with silence_other_threads():
            for _ in range(2):
                delay_run(
                    0.1,
                    asynchronous._run,
                    lambda: self.assertEqual(
                        data,
                        future.result(),
                        'Data should pass through future unchanged.'
                    ),
                    (),
                    {},
                    result,
                    sync=False
                )
            for _ in range(2):
                error, value = result.get(True, 0.2)
                self.assertFalse(
                    error,
                    'None of the threads should experience an exception.'
                )


if __name__ == '__main__':
    unittest.main()
