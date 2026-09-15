"""Validate measured request events without synthesizing missing data."""
def validate_sample(sample):
    events = sample.get("events", [])
    timestamps = [event.get("monotonic_ns") for event in events
                  if event.get("monotonic_ns") is not None]
    errors = []
    if not events or len(timestamps) != len(events):
        errors.append("missing event timestamps")
    if timestamps != sorted(timestamps):
        errors.append("event timestamps are not monotonic")
    if len(timestamps) != len(set(timestamps)):
        errors.append("event timestamps contain duplicates")
    for key, value in sample.get("timeline_us", {}).items():
        if isinstance(value, (int, float)) and value < 0:
            errors.append(f"negative duration: {key}")
    return errors

def request_timing(request):
    """Derive request latency from the request's monotonic state transitions."""
    event_times = {
        event["state"]: int(event["monotonic_ns"])
        for event in request.get("events", [])
    }
    persisted_ns = event_times.get("PERSISTED")
    created_ns = event_times.get("CREATED")
    api_enter_ns = request.get("api_enter_ns")
    if persisted_ns is None or created_ns is None or api_enter_ns is None:
        raise ValueError("request is missing API, CREATED, or PERSISTED timestamp")
    if persisted_ns < created_ns or persisted_ns < int(api_enter_ns):
        raise ValueError("request timestamps are not monotonic")
    return {
        "persist_seconds": (persisted_ns - int(api_enter_ns)) / 1e9,
        "state_machine_seconds": (persisted_ns - created_ns) / 1e9,
    }
