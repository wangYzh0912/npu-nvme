"""Legacy restore ordering with explicit catalog, transport and target interfaces.

This preserves the old in-place behavior. It is not D1 RestoreSession and does
not issue ready tokens or guarantee rollback for late integrity failures.
"""


class LegacyRestore:
    def __init__(self, *, select_record, transport, rank_id, chunk_size, schema_version):
        self.select_record = select_record
        self.transport = transport
        self.rank_id = rank_id
        self.chunk_size = chunk_size
        self.schema_version = schema_version

    def load_state(self, target, step: int = None,
                   verify_checksums: bool = True):
        """Restore namespaced parameters and return decoded control state."""
        selected_step, record = self.select_record(step)
        if record.get("type") == "MULTI_TRAINING_STATE_FULL":
            rank_record = record.get("ranks", {}).get(str(self.rank_id))
            if rank_record is None:
                raise ValueError(
                    f"checkpoint has no shard for rank {self.rank_id}")
            # Coordinator metadata carries the global commit fields at the
            # outer level and a normal TRAINING_STATE_FULL-shaped manifest in
            # each rank record.  Keep all subsequent validation and DMA code
            # shared with the single-rank path.
            record = {
                **rank_record,
                "type": "TRAINING_STATE_FULL",
                "schema_version": record.get("schema_version"),
                "state_step": record.get("state_step"),
                "chunk_size": record.get("chunk_size", self.chunk_size),
            }
        if record.get("type") != "TRAINING_STATE_FULL":
            raise ValueError(
                f"step {selected_step} is not a complete training-state checkpoint")
        if record.get("schema_version") != self.schema_version:
            raise ValueError("unsupported training-state schema version")
        if record.get("state_step") != selected_step:
            raise ValueError("training-state step does not match metadata key")
        if record.get("components") != target.component_names():
            raise ValueError("training-state component manifest mismatch")

        saved = record.get("params", {})
        saved_control_names = {
            name.split("/", 1)[1] for name in saved
            if name.startswith("control/")
        }
        if saved_control_names != set(record.get("control_names", [])):
            raise ValueError("training-state control manifest mismatch")
        target_items = target.prepare()
        targets = {item["name"]: item for item in target_items}
        saved_parameter_names = {
            name for name in saved
            if not name.startswith("control/")
        }
        if set(targets) != saved_parameter_names:
            missing = sorted(saved_parameter_names - set(targets))
            extra = sorted(set(targets) - saved_parameter_names)
            raise ValueError(
                f"training-state parameter set mismatch: missing={missing[:3]} "
                f"extra={extra[:3]}")

        dev_buffers, host_buffers, control_buffers = target.allocate(saved, targets)
        chunk_size = min(int(record.get("chunk_size", self.chunk_size)), self.chunk_size)
        self.transport.read_state(dev_buffers, host_buffers, chunk_size)
        target.apply(host_buffers)
        target.verify(targets, saved, record, verify_checksums)
        return target.decode(control_buffers)
