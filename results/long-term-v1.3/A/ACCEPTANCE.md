# A software exit

`gate-002`: **completed/pass, 70 measured pytest cases**, including 17 production-C
host cases under ASan/UBSan. Production shared-library rebuild also passed.
The evidence manifest includes the actual C test executable and object, raw output,
JUnit, profile and source digests. This is the §7 **software** A exit only.

The earlier gate-001 incorrectly required persistent request lookup and H02;
those requirements belong to later batches. Its original blocked evidence remains
intact. A.json now matches §7: G02, selected G03, software G04, FULL input bounds,
and the G14 schema/runner subset. No hardware DMA-stop or power-loss claim is made.

Since gate-001: retained native slots expose request ID/bytes/offset/reason;
Python quarantined snapshots expose owner and release prerequisites. Observation
timeout closes native admission while allowing late completion polling. Queue
counts no longer dereference rings concurrently with reactor shutdown. Tests cover
metadata timeout references, bounded close, lost completion followed by a late
callback, descriptor/restore budgets, and invalid performance evidence.

Frozen limits: `config/safety_budgets.json`. These describe implementation limits;
they are not arbitrary runtime overrides. Unsafe resource retention consumes the
admitted budget and does not authorize release.

Compatibility boundary: old synchronous raw-pointer APIs still drain after an
observation timeout to protect existing callers' buffer lifetimes. New request
waits and close are finite. B2/C2 must migrate callers before retiring the old APIs.
Live framework acceptance, complete restore transactions, durable commit ownership,
and all hardware gates remain outside this A result.
