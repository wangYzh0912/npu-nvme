"""Retired legacy heap exporter. Archive: archive-legacy-full-20260914."""
def export_to_heap(*args, **kwargs):
    raise RuntimeError('legacy metadata writer retired; use strict FULL save_state and restore_full_state')

if __name__ == '__main__':
    export_to_heap()
