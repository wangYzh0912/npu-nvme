"""Archived pre-cleanup reproducer, not runnable against ABI3; see README.md."""
from pathlib import Path
import json
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'tests/d1')]

from conftest import coordinator, publish, spec
from test_runtime import runtime
from test_strict import Target


def main():
    schema = spec.__wrapped__()
    catalog = coordinator.__wrapped__()
    _, _, _, disk, _ = publish(catalog, schema, 1)
    instance, transport, _ = runtime(catalog, schema)
    transport.disk = disk
    restore_entered, restore_release = threading.Event(), threading.Event()
    freeze_entered, freeze_release = threading.Event(), threading.Event()
    close_entered, close_done = threading.Event(), threading.Event()
    errors, saved = [], []

    def read(offset, size):
        restore_entered.set()
        assert restore_release.wait(5)
        return transport.read(offset, size)

    def freeze(params, generation):
        freeze_entered.set()
        assert freeze_release.wait(5)
        return params

    instance.restore_session.reader = read
    instance.freeze = freeze
    restored = instance.begin_restore(lambda unused: Target(), schema, 1)
    assert restore_entered.wait(2)

    def save():
        try:
            saved.append(instance.save({}, {}, 2, schema))
        except BaseException as error:
            errors.append(repr(error))

    saver = threading.Thread(target=save)
    saver.start()
    original_drain = instance.drain

    def drain(timeout):
        close_entered.set()
        return original_drain(timeout)

    instance.drain = drain

    def close():
        try:
            instance.close(2)
        except BaseException as error:
            errors.append(repr(error))
        finally:
            close_done.set()

    closer = threading.Thread(target=close)
    try:
        assert freeze_entered.wait(2)
        closer.start()
        assert close_entered.wait(2)
        restore_release.set()
        assert close_done.wait(3)
        target, receipt = restored.result()
        result = dict(real_restore_ready=target.ready,
                      restored_generation=receipt.generation,
                      transport_closed=transport.closed,
                      accepted_save_still_busy=instance._busy,
                      capture_still_blocked=not freeze_release.is_set(),
                      errors=errors)
        print(json.dumps(result, indent=2))
        assert all(result[name] for name in (
            'real_restore_ready', 'transport_closed',
            'accepted_save_still_busy', 'capture_still_blocked'))
        assert not errors
    finally:
        freeze_release.set()
        restore_release.set()
        saver.join(5)
        if closer.ident is not None:
            closer.join(5)
        for handle in saved:
            handle.wait(5)


if __name__ == '__main__':
    main()
