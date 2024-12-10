import asynchronous


# noinspection PyPep8Naming
class add_timeout:
    def __init__(self, function, limit=60):
        self._executor = asynchronous.Executor(limit)
        self._function = function
        self._future = None

    def __call__(self, *args, **kwargs):
        self._future = self._executor.submit(self._function, *args, **kwargs)

    def cancel(self):
        self._future.cancel()

    @property
    def ready(self):
        return (False if not self._future.done() else
                True if not self._future.cancelled() else
                None)

    @property
    def value(self):
        return self._future.result()
