"""Same-clock task interval overlap; never equates tasks with core occupancy."""
from collections import defaultdict


def overlap(intervals, *, step_start, step_end, ready_time=None):
    if step_end <= step_start:
        raise ValueError('empty step interval')
    events = defaultdict(lambda: [0, 0])
    events[step_start]; events[step_end]
    for row in intervals:
        if row['engine'] not in ('cube', 'vector'):
            raise ValueError('unknown engine classification')
        begin, end = max(step_start, row['start']), min(step_end, row['end'])
        if end <= begin:
            continue
        axis = 0 if row['engine'] == 'cube' else 1
        events[begin][axis] += 1
        events[end][axis] -= 1
    times = sorted(events)
    active = [0, 0]
    durations = {key: 0 for key in ('cube_only', 'vector_only', 'both', 'neither')}
    windows = []
    segments = []
    for index, start in enumerate(times[:-1]):
        active = [a + b for a, b in zip(active, events[start])]
        end = times[index + 1]
        label = ('both' if active[1] else 'cube_only') if active[0] else ('vector_only' if active[1] else 'neither')
        durations[label] += end - start
        segments.append(dict(start=start, end=end, state=label))
        if label == 'cube_only':
            if windows and windows[-1]['end'] == start:
                windows[-1]['end'] = end
            else:
                windows.append(dict(start=start, end=end))
    for row in windows:
        row['duration'] = row['end'] - row['start']
        row['ready_duration'] = None if ready_time is None else max(0, row['end'] - max(row['start'], ready_time))
    return dict(metric='task_activity_overlap', duration=step_end-step_start,
                fractions={key: value/(step_end-step_start) for key,value in durations.items()},
                durations=durations, segments=segments, candidate_windows=windows,
                candidate_count=len(windows), candidate_total=sum(row['duration'] for row in windows),
                candidate_longest=max((row['duration'] for row in windows), default=0),
                core_occupancy=None, instruction_activity=None)
