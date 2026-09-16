/* runtime: behavior-preserving extraction from npu_nvme.c. */
#include "internal/implementation.h"
#include <sys/file.h>
#include <fcntl.h>

static pthread_mutex_t native_owner_guard = PTHREAD_MUTEX_INITIALIZER;
static NPUNVMEContext *native_owner;

static int acquire_native_owner(NPUNVMEContext *ctx, const char *address) {
    unsigned domain, bus, device, function;
    char tail, path[128];
    if (sscanf(address, "%x:%x:%x.%x%c", &domain, &bus, &device, &function, &tail) != 4 ||
        domain > 65535 || bus > 255 || device > 31 || function > 7) return -EINVAL;
    snprintf(path, sizeof(path), "/tmp/npu-nvme-native-owner-%04x:%02x:%02x.%x-ns1",
             domain, bus, device, function);
    pthread_mutex_lock(&native_owner_guard);
    if (native_owner) { pthread_mutex_unlock(&native_owner_guard); return -EBUSY; }
    int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0666);
    if (fd < 0) { int rc = -errno; pthread_mutex_unlock(&native_owner_guard); return rc; }
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        int rc = errno == EWOULDBLOCK ? -EBUSY : -errno;
        close(fd); pthread_mutex_unlock(&native_owner_guard); return rc;
    }
    ctx->native_owner_fd = fd;
    ctx->native_owner_pid = getpid();
    ctx->native_owner_held = true;
    native_owner = ctx;
    pthread_mutex_unlock(&native_owner_guard);
    return 0;
}

static void release_native_owner(NPUNVMEContext *ctx) {
    pthread_mutex_lock(&native_owner_guard);
    if (ctx->native_owner_held && ctx->native_owner_pid == getpid()) {
        flock(ctx->native_owner_fd, LOCK_UN);
        close(ctx->native_owner_fd);
        ctx->native_owner_held = false;
        if (native_owner == ctx) native_owner = NULL;
    }
    pthread_mutex_unlock(&native_owner_guard);
}

int read_int_from_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int val;
    if (fscanf(f, "%d", &val) != 1) { fclose(f); return -1; }
    fclose(f);
    return val;
}

void ensure_hugepages(void) {
    int nr = read_int_from_file(NR_HUGEPAGES_PATH);
    int free_2mb = read_int_from_file(HUGEPAGE_2MB_PATH);
    if (nr < 0) return;

    int target = nr;
    if (free_2mb < HUGEPAGE_PADDING) {
        target = nr + (HUGEPAGE_PADDING - free_2mb);
        char buf[64];
        snprintf(buf, sizeof(buf), "%d", target);
        FILE *fp = fopen(NR_HUGEPAGES_PATH, "w");
        if (fp) { fprintf(fp, "%s", buf); fclose(fp); }
        fprintf(stderr, "[NPU-NVMe] Expanded hugepages %d -> %d\n", nr, target);
    }
}

bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                    struct spdk_nvme_ctrlr_opts *opts) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)cb_ctx;
    if (strncmp(trid->traddr, ctx->pci_addr, sizeof(ctx->pci_addr)) == 0) {
        fprintf(stderr, "[NPU-NVMe] Found matching controller at %s\n", trid->traddr);
        return 1;
    }
    return 0;
}

void attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
                      struct spdk_nvme_ctrlr *ctrlr,
                      const struct spdk_nvme_ctrlr_opts *opts) {
    NPUNVMEContext *ctx = (NPUNVMEContext *)cb_ctx;
    ctx->ctrlr = ctrlr;
    ctx->ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1);
    if (ctx->ns) {
        ctx->block_size = spdk_nvme_ns_get_sector_size(ctx->ns);
        ctx->total_blocks = spdk_nvme_ns_get_num_sectors(ctx->ns);
        printf("[NPU-NVMe] Attached NS: block_size=%u total_blocks=%lu\n",
               ctx->block_size, ctx->total_blocks);
    }
}

void context_put(NPUNVMEContext *ctx) {
    if (atomic_fetch_sub_explicit(&ctx->refs, 1, memory_order_acq_rel) == 1) {
        release_native_owner(ctx);
        if (ctx->state_lock_initialized) pthread_mutex_destroy(&ctx->state_lock);
        free(ctx);
    }
}

int npu_nvme_get_max_transfer(NPUNVMEContext *ctx) {
    return ctx ? (int)ctx->dma.chunk_size : 0;
}

uint64_t npu_nvme_get_total_blocks(NPUNVMEContext *ctx) {
    return ctx ? ctx->total_blocks * ctx->block_size : 0;
}

int npu_nvme_set_io_timeout_ms(NPUNVMEContext *ctx, uint32_t timeout_ms) {
    if (!ctx || timeout_ms == 0) return -1;
    ctx->io_timeout_ms = timeout_ms;
    return 0;
}

uint32_t npu_nvme_get_io_timeout_ms(NPUNVMEContext *ctx) {
    return ctx ? ctx->io_timeout_ms : 0;
}

int npu_nvme_wait_quiescent(NPUNVMEContext *ctx, uint32_t timeout_ms) {
    return wait_reactor_quiescent(ctx, timeout_ms);
}

int ensure_acl_context(NPUNVMEContext *ctx) {
    aclError ret = aclrtSetDevice(ctx->acl.npu_id);
    if (ret != 0) return -1;
    return aclrtSetCurrentContext(ctx->acl.acl_ctx);
}

static int scheduler_value(const char *name, uint32_t fallback, uint32_t maximum,
                           uint32_t *out) {
    const char *value=getenv(name);
    *out=fallback;
    if (!value) return 0;
    if (!value[0]) return -EINVAL;
    for (const char *p=value; *p; ++p) if (*p<'0' || *p>'9') return -EINVAL;
    errno=0; char *end=NULL; unsigned long long parsed=strtoull(value,&end,10);
    if (errno || *end || !parsed || parsed>maximum) return -EINVAL;
    *out=(uint32_t)parsed; return 0;
}
int npu_nvme_get_capabilities(NPUNVMEContext *ctx, NPUNVMECapabilities *out, uint32_t size) {
    if (!ctx || !out || size!=sizeof(*out)) return -EINVAL;
    uint64_t capacity;
    if (namespace_capacity(ctx,&capacity)) return -EINVAL;
    *out=(NPUNVMECapabilities){.struct_size=sizeof(*out),.version=1,
        .operation_mask=(1U<<(NPU_NVME_COPY_H2D+1))-1,
        .block_size=ctx->block_size,.chunk_size=ctx->dma.chunk_size,
        .pipe_depth=ctx->dma.max_pipe_depth,.max_request_items=MAX_BATCH_ITEMS,
        .max_pending_requests=EFFECTIVE(ctx->max_pending_requests,64),
        .copy_bytes_per_tick=EFFECTIVE(ctx->copy_bytes_per_tick,65536),
        .checksum_bytes_per_tick=EFFECTIVE(ctx->checksum_bytes_per_tick,65536),
        .submit_items_per_tick=EFFECTIVE(ctx->submit_items_per_tick,4),
        .quantum_items=EFFECTIVE(ctx->quantum_items,DEFAULT_QUANTUM(ctx)),
        .max_request_bytes=MAX_BATCH_BYTES,.namespace_bytes=capacity,
        .dma_pool_bytes=(uint64_t)ctx->dma.chunk_size*ctx->dma.max_pipe_depth};
    return 0;
}

int npu_nvme_init(NPUNVMEContext **out_ctx, const char *pci_addr, int npu_id,
                  int pipe_depth, uint32_t chunk_size, bool enable_profiling,
                  const char *prof_dir) {
    if (!out_ctx || !pci_addr || pci_addr[0] == '\0' || chunk_size == 0 ||
        chunk_size % 4096 != 0 || chunk_size > INT_MAX ||
        pipe_depth < MIN_PIPE_DEPTH || pipe_depth > MAX_PIPE_DEPTH) {
        fprintf(stderr, "[Fatal] Invalid npu_nvme_init arguments.\n");
        return -1;
    }
    *out_ctx = NULL;

    NPUNVMEContext *ctx = calloc(1, sizeof(NPUNVMEContext));
    if (!ctx) return -1;
    if (scheduler_value("NPU_NVME_COPY_BYTES",65536,16*1024*1024,&ctx->copy_bytes_per_tick) ||
        scheduler_value("NPU_NVME_CHECKSUM_BYTES",65536,16*1024*1024,&ctx->checksum_bytes_per_tick) ||
        scheduler_value("NPU_NVME_SUBMIT_ITEMS",4,64,&ctx->submit_items_per_tick) ||
        scheduler_value("NPU_NVME_QUANTUM_ITEMS",pipe_depth>4 ? pipe_depth : 4,4096,&ctx->quantum_items) ||
        scheduler_value("NPU_NVME_MAX_PENDING",64,4096,&ctx->max_pending_requests)) {
        free(ctx); return -EINVAL;
    }
    int owner_rc = acquire_native_owner(ctx, pci_addr);
    if (owner_rc != 0) { free(ctx); return owner_rc; }

    atomic_init(&ctx->refs, 1);
    atomic_init(&ctx->pending_requests, 0);
    atomic_init(&ctx->quarantined, 0);
    atomic_init(&ctx->reactor_exited, 0);
    atomic_init(&ctx->nvme_submit_count, 0);
    atomic_init(&ctx->nvme_complete_count, 0);
    atomic_init(&ctx->nvme_outstanding, 0);
    atomic_init(&ctx->nvme_outstanding_peak, 0);
    atomic_init(&ctx->dma_inflight, 0);
    atomic_init(&ctx->dma_inflight_peak, 0);
    atomic_init(&ctx->request_ring_peak, 0);
    atomic_init(&ctx->async_dma_submit_count, 0);
    atomic_init(&ctx->async_event_query_count, 0);
    atomic_init(&ctx->async_event_query_error_count, 0);
    atomic_init(&ctx->stream_sync_fallback_count, 0);
    atomic_init(&ctx->spdk_retry_count, 0);
    atomic_init(&ctx->completion_error_count, 0);
    atomic_init(&ctx->reactor_cpu_us, 0);

    ctx->io_timeout_ms = 60000;
    const char *timeout_env = getenv("NPU_NVME_IO_TIMEOUT_MS");
    if (timeout_env && timeout_env[0] != '\0') {
        char *end = NULL;
        unsigned long parsed = strtoul(timeout_env, &end, 10);
        if (end != timeout_env && *end == '\0' && parsed <= UINT32_MAX)
            ctx->io_timeout_ms = (uint32_t)parsed;
    }

    strncpy(ctx->pci_addr, pci_addr, sizeof(ctx->pci_addr) - 1);
    ctx->acl.npu_id = npu_id;
    ctx->dma.chunk_size = chunk_size;
    ctx->dma.max_pipe_depth = pipe_depth;
    ctx->enable_profiling = enable_profiling;
    if (prof_dir) {
        strncpy(ctx->profiling_dir, prof_dir, sizeof(ctx->profiling_dir) - 1);
    } else {
        strcpy(ctx->profiling_dir, ".");
    }
    /* Listener-state lock — protects registered_tasks, dev_step_ptr,
     * probe_flag_* from concurrent Python <-> reactor access.  I/O is
     * async via rings and does NOT use this lock.  Non-recursive. */
    if (pthread_mutex_init(&ctx->state_lock, NULL) != 0) {
        fprintf(stderr, "[Fatal] Failed to init state_lock.\n");
        release_native_owner(ctx);
        free(ctx);
        return -1;
    }
    ctx->state_lock_initialized = true;

    /* Initialise SPDK environment (once per process via SPDK_SHM_ID).
     * MUST be called BEFORE spdk_thread_lib_init — the thread library
     * internally creates spdk_ring (rte_ring/rte_mempool) which requires
     * DPDK EAL to be fully initialised. */
    {
        static int spdk_inited = 0;
        if (!spdk_inited) {
            struct spdk_env_opts env_opts;
            spdk_env_opts_init(&env_opts);
            env_opts.name = "npu_nvme_app";

            /* uio_pci_generic cannot advertise a reliable IOVA mode to DPDK.
             * Use PA explicitly for the authorized raw-device deployment;
             * callers may override this with NPU_NVME_DPDK_ARGS. */
            const char *dpdk_args = getenv("NPU_NVME_DPDK_ARGS");
            env_opts.env_context = (dpdk_args && dpdk_args[0])
                ? (void *)dpdk_args : (void *)"--iova-mode=pa";

            const char *shm = getenv("SPDK_SHM_ID");
            if (shm) { env_opts.shm_id = atoi(shm); }

            ensure_hugepages();

            if (spdk_env_init(&env_opts) < 0) {
                fprintf(stderr, "[Fatal] Unable to initialize SPDK env.\n"
                        "[Fatal] Check: (1) run as root, (2) free hugepages > 0 "
                        "per NUMA node.\n"
                        "[Fatal] Quick fix: echo %d > %s\n",
                        read_int_from_file(NR_HUGEPAGES_PATH) + HUGEPAGE_PADDING,
                        NR_HUGEPAGES_PATH);
                npu_nvme_cleanup(ctx);
                return -1;
            }
            spdk_inited = 1;
        }
    }

    /* --- diagnostic: verify EAL state after spdk_env_init --- */
    fprintf(stderr, "[Diag] after spdk_env_init: rte_lcore_count=%u "
            "rte_socket_count=%u spdk_core_count=%u\n",
            rte_lcore_count(), rte_socket_count(),
            spdk_env_get_core_count());

    /* Re-register DPDK ring mempool ops after EAL init (once per process).
     *
     * The RTE_INIT constructors in librte_mempool_ring.a register ops at
     * dlopen time via init_array, but EAL was not yet available.  We
     * explicitly re-register "ring_mp_mc" (the default) here, now that
     * spdk_env_init has completed and EAL is fully initialised.
     */
    {
        static int ops_registered = 0;
        if (!ops_registered) {
        extern int common_ring_alloc(struct rte_mempool *mp);
        extern void common_ring_free(struct rte_mempool *mp);
        extern int common_ring_mp_enqueue(struct rte_mempool *mp,
                void * const *obj_table, unsigned n);
        extern int common_ring_mc_dequeue(struct rte_mempool *mp,
                void **obj_table, unsigned n);
        extern unsigned common_ring_get_count(const struct rte_mempool *mp);

        struct rte_mempool_ops ops = {
            .name = "ring_mp_mc", .alloc = common_ring_alloc,
            .free = common_ring_free, .enqueue = common_ring_mp_enqueue,
            .dequeue = common_ring_mc_dequeue,
            .get_count = common_ring_get_count,
        };
        int rc = rte_mempool_register_ops(&ops);
        if (rc < 0) {
            fprintf(stderr, "[NPU-NVMe] WARNING: rte_mempool_register_ops "
                    "(ring_mp_mc) failed (rc=%d)\n", rc);
        }
            ops_registered = 1;
        }
    }

    /* Start reactor pthread via SPDK thread library.
     * spdk_thread_lib_init creates an spdk_ring (rte_ring → rte_mempool)
     * — EAL must be initialised first (done above).
     * Like spdk_env_init, spdk_thread_lib_init is once-per-process. */
    ctx->app_should_stop = 0;
    ctx->reactor_init_result = 0;
    if (pthread_barrier_init(&ctx->init_barrier, NULL, 2) != 0) {
        fprintf(stderr, "[Fatal] Failed to initialize Reactor barrier.\n");
        npu_nvme_cleanup(ctx);
        return -1;
    }

    {
        static int thread_lib_inited = 0;
        if (!thread_lib_inited) {
            if (spdk_thread_lib_init((spdk_new_thread_fn)reactor_new_thread_fn, 0) != 0) {
                fprintf(stderr, "[Fatal] spdk_thread_lib_init failed.\n");
                pthread_barrier_destroy(&ctx->init_barrier);
                npu_nvme_cleanup(ctx);
                return -1;
            }
            thread_lib_inited = 1;
        }
    }

    /* spdk_thread_create: the second argument is a cpumask pointer.
     * Passing ctx as cpumask causes a SEGV on ARM64 because SPDK
     * internally reads cpumask as a potentially large cpu_set bitmask.
     * V0 passes NULL; we do the same and communicate ctx via g_reactor_ctx. */
    g_reactor_ctx = ctx;
    struct spdk_thread *th = spdk_thread_create("npu_nvme", NULL);
    if (!th) {
        fprintf(stderr, "[Fatal] spdk_thread_create failed.\n");
        pthread_barrier_destroy(&ctx->init_barrier);
        npu_nvme_cleanup(ctx);
        return -1;
    }
    if (!ctx->reactor_pthread_started) {
        fprintf(stderr, "[Fatal] Reactor pthread did not start.\n");
        pthread_barrier_destroy(&ctx->init_barrier);
        reactor_finish_thread(ctx->reactor_thread);
        npu_nvme_cleanup(ctx);
        return -1;
    }

    /* Wait for reactor pthread to reach its main loop. */
    pthread_barrier_wait(&ctx->init_barrier);
    pthread_barrier_destroy(&ctx->init_barrier);
    if (ctx->reactor_init_result != 0) {
        npu_nvme_cleanup(ctx);
        return -1;
    }

    /* Probe and attach NVMe device */
    struct spdk_nvme_transport_id trid = {};
    spdk_nvme_trid_populate_transport(&trid, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(trid.traddr, sizeof(trid.traddr), "%s", pci_addr);

    if (spdk_nvme_probe(&trid, ctx, probe_cb, attach_cb, NULL) != 0) {
        fprintf(stderr, "[Fatal] spdk_nvme_probe failed.\n");
        goto init_fail;
    }
    if (!ctx->ctrlr) {
        fprintf(stderr, "[Fatal] Controller not found at %s.\n", pci_addr);
        goto init_fail;
    }

    /* Allocate SPDK I/O queue pair */
    struct spdk_nvme_io_qpair_opts qopts;
    spdk_nvme_ctrlr_get_default_io_qpair_opts(ctx->ctrlr, &qopts, sizeof(qopts));
    qopts.io_queue_size = 512;
    ctx->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctx->ctrlr, &qopts, sizeof(qopts));
    if (!ctx->qpair) {
        fprintf(stderr, "[Fatal] Cannot allocate NVMe I/O qpair.\n");
        goto init_fail;
    }

    /* Allocate dedicated metadata qpair (small, no contention with data path). */
    {
        struct spdk_nvme_io_qpair_opts meta_opts;
        spdk_nvme_ctrlr_get_default_io_qpair_opts(ctx->ctrlr, &meta_opts,
                                                   sizeof(meta_opts));
        meta_opts.io_queue_size = 64;
        ctx->meta_qpair = spdk_nvme_ctrlr_alloc_io_qpair(ctx->ctrlr, &meta_opts,
                                                          sizeof(meta_opts));
        if (!ctx->meta_qpair) {
            fprintf(stderr, "[Fatal] Cannot allocate metadata qpair.\n");
            goto init_fail;
        }
    }

    /* Initialise NPU environment */
    aclError ret = aclrtSetDevice(ctx->acl.npu_id);
    if (ret != ACL_SUCCESS) goto init_fail;
    if (aclrtGetCurrentContext(&ctx->acl.acl_ctx) != ACL_SUCCESS) {
        fprintf(stderr, "[Fatal] Failed to get ACL context.\n");
        goto init_fail;
    }
    ret = aclrtCreateStream(&ctx->acl.copy_stream);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[Fatal] Failed to create NPU Stream.\n");
        goto init_fail;
    }

    /* Allocate DMA buffer pool + NPU events */
    ctx->dma.pool = calloc(ctx->dma.max_pipe_depth, sizeof(dma_buf_t));
    ctx->acl.events = calloc(ctx->dma.max_pipe_depth, sizeof(aclrtEvent));
    if (!ctx->dma.pool || !ctx->acl.events) {
        fprintf(stderr, "[Fatal] Failed to allocate DMA/event descriptor arrays.\n");
        goto init_fail;
    }
    /* ring capacity = pipe_depth + 1 (one overflow slot for full-vs-empty). */
    if (ring_init(&ctx->dma.free_ring, ctx->dma.max_pipe_depth + 1) != 0) {
        fprintf(stderr, "[Fatal] Failed to allocate DMA free ring.\n");
        goto init_fail;
    }

    for (int i = 0; i < ctx->dma.max_pipe_depth; i++) {
        ctx->dma.pool[i].buf = spdk_zmalloc(ctx->dma.chunk_size,
                                             2 * 1024 * 1024, NULL,
                                             SPDK_ENV_SOCKET_ID_ANY,
                                             SPDK_MALLOC_DMA);
        if (!ctx->dma.pool[i].buf) {
            fprintf(stderr, "[Fatal] spdk_zmalloc failed at slot %d.\n", i);
            goto init_fail;
        }
        ctx->dma.pool[i].phys_addr = spdk_vtophys(ctx->dma.pool[i].buf, NULL);

        ret = aclrtCreateEvent(&ctx->acl.events[i]);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[Fatal] Failed to create NPU Event at slot %d.\n", i);
            goto init_fail;
        }
        ring_push(&ctx->dma.free_ring, i);
    }

    printf("[Init] NPUNVME Fully Initialized! Stream/Events ready. "
           "Max Pipe Depth: %d\n", ctx->dma.max_pipe_depth);
    /* Allocate dedicated buffer for metadata I/O */
    ctx->meta_dma_buf = spdk_zmalloc(META_DMA_BUF_SIZE, 2 * 1024 * 1024, NULL,
                                      SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
    if (!ctx->meta_dma_buf) {
        fprintf(stderr, "[NPU-NVMe] Failed to allocate meta DMA buffer.\n");
        goto init_fail;
    }

    /* Listener poller runs on the reactor thread (registered in reactor_loop).
     * The NPU_NVME_NO_LISTENER env var still controls whether the step poller
     * is active — it does nothing until dev_step_ptr is set. */
    ctx->listener.probe_flag_dev_ptr = NULL;
    ctx->listener.probe_flag_host = NULL;
    ctx->listener.dev_step_ptr = NULL;
    ctx->listener.step_poll_buf = NULL;

    *out_ctx = ctx;
    printf("[Init] Initialisation complete.\n");
    return 0;

init_fail:
    /* spdk_thread_create can succeed while the pthread callback fails. */
    if (ctx->reactor_thread && !ctx->reactor_pthread_started)
        reactor_finish_thread(ctx->reactor_thread);
    npu_nvme_cleanup(ctx);
    return -1;
}

static int close_owned_context(NPUNVMEContext *ctx, uint32_t timeout_ms) {
    if (!ctx) return 0;
    uint64_t start = get_time_us();
    timeout_ms = finite_timeout(timeout_ms);
    if (ctx->state_lock_initialized) pthread_mutex_lock(&ctx->state_lock);
    atomic_store(&ctx->app_should_stop, 1);
    if (ctx->state_lock_initialized) pthread_mutex_unlock(&ctx->state_lock);
    if (atomic_load(&ctx->quarantined)) return -EIO;
    if (ctx->reactor_pthread_started) {
        while (!atomic_load_explicit(&ctx->reactor_exited, memory_order_acquire)) {
            if (get_time_us() - start >= (uint64_t)timeout_ms * 1000ULL) return -ETIMEDOUT;
            if (atomic_load(&ctx->quarantined)) return -EIO;
            usleep(1000);
        }
        pthread_join(ctx->reactor_pthread, NULL);
        ctx->reactor_pthread_started = false;
    }
    if (ctx->reactor_thread) {
        if (!spdk_thread_is_exited(ctx->reactor_thread)) return -EBUSY;
        spdk_thread_destroy(ctx->reactor_thread);
        ctx->reactor_thread = NULL;
    }
    if (g_reactor_ctx == ctx) g_reactor_ctx = NULL;
    return 0;
}

int npu_nvme_close(NPUNVMEContext *ctx, uint32_t timeout_ms) {
    if (!ctx) return 0;
    uint64_t begin = get_time_us();
    timeout_ms = finite_timeout(timeout_ms);
    for (;;) {
        bool expected = false;
        if (atomic_compare_exchange_strong(&ctx->close_in_progress, &expected, true)) break;
        if (get_time_us() - begin >= (uint64_t)timeout_ms * 1000) return -ETIMEDOUT;
        usleep(100);
    }
    uint64_t elapsed_ms = (get_time_us() - begin) / 1000;
    int rc = elapsed_ms >= timeout_ms ? -ETIMEDOUT :
             close_owned_context(ctx, timeout_ms - (uint32_t)elapsed_ms);
    atomic_store(&ctx->close_in_progress, false);
    return rc;
}

void npu_nvme_cleanup(NPUNVMEContext *ctx) {
    if (!ctx) return;
    if (npu_nvme_close(ctx, finite_timeout(ctx->io_timeout_ms)) != 0) {
        fprintf(stderr, "[NPU-NVMe] cleanup retained context: DMA stop/drain not proven\n");
        return;
    }

    /* Bind ACL context — needed for aclrtDestroyEvent/FreeHost below. */
    if (ctx->acl.acl_ctx) {
        aclrtSetDevice(ctx->acl.npu_id);
        aclrtSetCurrentContext(ctx->acl.acl_ctx);
    }

    /* Release ACL resources */
    if (ctx->acl.events) {
        for (int i = 0; i < ctx->dma.max_pipe_depth; i++) {
            if (ctx->acl.events[i]) aclrtDestroyEvent(ctx->acl.events[i]);
        }
        free(ctx->acl.events);
    }
    if (ctx->acl.copy_stream) aclrtDestroyStream(ctx->acl.copy_stream);

    /* Release listener host buffers */
    if (ctx->listener.probe_flag_host)
        aclrtFreeHost(ctx->listener.probe_flag_host);
    if (ctx->listener.step_poll_buf)
        aclrtFreeHost(ctx->listener.step_poll_buf);
    if (ctx->listener.owns_probe_flag && ctx->listener.probe_flag_dev_ptr)
        aclrtFree(ctx->listener.probe_flag_dev_ptr);

    /* Release DMA pool */
    if (ctx->dma.pool) {
        for (int i = 0; i < ctx->dma.max_pipe_depth; i++) {
            if (ctx->dma.pool[i].buf) spdk_free(ctx->dma.pool[i].buf);
        }
        free(ctx->dma.pool);
    }
    ring_free(&ctx->dma.free_ring);

    /* Release metadata DMA buffer */
    if (ctx->meta_dma_buf) spdk_free(ctx->meta_dma_buf);

    /* Detach NVMe */
    if (ctx->qpair) spdk_nvme_ctrlr_free_io_qpair(ctx->qpair);
    if (ctx->meta_qpair) spdk_nvme_ctrlr_free_io_qpair(ctx->meta_qpair);
    if (ctx->ctrlr) spdk_nvme_detach(ctx->ctrlr);

    /* Release registered tasks (both current and deferred-free). */
    if (ctx->listener.registered_tasks) {
        free(ctx->listener.registered_tasks);
        ctx->listener.registered_tasks = NULL;
        ctx->listener.num_registered_tasks = 0;
    }
    if (ctx->listener.old_tasks) {
        free(ctx->listener.old_tasks);
        ctx->listener.old_tasks = NULL;
    }

    context_put(ctx);
}
