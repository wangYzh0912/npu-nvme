/* Archived pre-cleanup reproducer; not an ABI3 test. See README.md. */
#define main original_host_safety_main
#include "../../tests/c/host_safety_test.c"
#undef main

int main(int argc, char **argv) {
    assert(argc == 2);
    NPUNVMEContext ctx = context();
    pthread_mutex_init(&ctx.state_lock, NULL);

    if (!strcmp(argv[1], "zero-tasks")) {
        emulate_copy = 1;
        copy_result = ACL_SUCCESS;
        int device_step = 4, host_step = 0;
        ctx.listener.dev_step_ptr = &device_step;
        ctx.listener.step_poll_buf = &host_step;
        ctx.listener.ckpt_interval = 4;
        ctx.last_step_seen = -1;
        assert(step_poller_fn(&ctx) == 0);
        assert(ctx.write_fsm.state == WRITE_FSM_RUNNING);
        assert(ctx.write_fsm.faf_req.num_tasks == 0);
        assert(write_fsm_poller_fn(&ctx) == 0);
        assert(atomic_load(&ctx.write_fsm.faf_req.done));
        assert(ctx.write_fsm.faf_req.result == 0);
        puts("REPRODUCED: an unregistered zero-task FaF trigger completed successfully");
        pthread_mutex_destroy(&ctx.state_lock);
        return 0;
    }

    assert(!strcmp(argv[1], "reregister"));
    char data[4096] = {0};
    void *pointers[] = {data};
    uint64_t offsets[] = {0};
    size_t sizes[] = {4};
    assert(npu_nvme_register_tasks(&ctx, pointers, offsets, sizes, 1) == 0);
    ctx.write_fsm.faf_req.tasks = ctx.listener.registered_tasks;
    ctx.write_fsm.faf_req.num_tasks = 1;
    initiate_write_fsm(&ctx, &ctx.write_fsm.faf_req);
    assert(npu_nvme_register_tasks(&ctx, pointers, offsets, sizes, 1) == 0);
    assert(npu_nvme_register_tasks(&ctx, pointers, offsets, sizes, 1) == 0);
    /* The unfinished FSM still borrows the first registration. */
    volatile size_t active_size = ctx.write_fsm.req->tasks[0].size;
    (void)active_size;
    return 0;
}
