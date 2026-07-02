/* =======================================================================
 * spdk_reactor_poller_test.c — SPDK thread/poller framework verification
 *
 * Verifies that spdk_env_init / spdk_thread_lib_init / spdk_thread_create /
 * spdk_poller_register / spdk_thread_poll work on this Ascend 910B + ARM64
 * + DPDK 25.07 platform.
 *
 * The SPDK thread library requires a new_thread_fn callback that creates an
 * OS thread and binds the SPDK thread to it.  This test creates a dedicated
 * pthread whose body polls the SPDK thread, driving the registered poller.
 * ======================================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>

#include <spdk/env.h>
#include <spdk/thread.h>

#define POLLER_PERIOD_US    1000000   /* 1 second */
#define POLLER_MAX_COUNT    5         /* stop after 5 callbacks */
#define THREAD_NAME         "spdk_reactor_test"

/* ---- shared state ---- */

static atomic_int g_poller_ticks = 0;
static atomic_int g_done = 0;
static struct spdk_thread *g_thread = NULL;

/* ---- poller callback (runs on reactor pthread) ---- */

/**
 * @brief Poller callback invoked by spdk_thread_poll.
 *
 * Increments a counter on each invocation.  After reaching
 * POLLER_MAX_COUNT, signals the main thread to exit.
 *
 * @param arg  unused (NULL)
 * @return     0 to continue polling, -1 to unregister after max count
 */
static int
reactor_poller_fn(void *arg)
{
    int n = atomic_fetch_add(&g_poller_ticks, 1) + 1;
    printf("[spdk-reactor-poller] poller tick %d/%d\n", n, POLLER_MAX_COUNT);
    fflush(stdout);

    if (n >= POLLER_MAX_COUNT) {
        printf("[spdk-reactor-poller] max count reached, signalling done\n");
        atomic_store(&g_done, 1);
        return -1;   /* unregister poller */
    }
    return 0;
}

/* ---- reactor pthread body ---- */

/**
 * @brief Reactor pthread body that polls the SPDK thread.
 *
 * Binds the SPDK thread to this OS thread, then loops calling
 * spdk_thread_poll until the main thread signals completion.
 *
 * @param arg  unused (NULL)
 * @return     NULL
 */
static void *
reactor_thread(void *arg)
{
    struct spdk_poller *poller;

    spdk_set_thread(g_thread);

    /* register the poller on this thread */
    poller = spdk_poller_register(reactor_poller_fn, NULL, POLLER_PERIOD_US);
    printf("[spdk-reactor-poller] spdk_poller_register OK (poller=%p, period=%d us)\n",
           (void *)poller, POLLER_PERIOD_US);

    while (!atomic_load(&g_done)) {
        spdk_thread_poll(g_thread, 0, 0);
        usleep(1000);
    }

    spdk_poller_unregister(&poller);
    spdk_thread_exit(g_thread);
    return NULL;
}

/* ---- new_thread_fn callback (called by spdk_thread_create) ---- */

/**
 * @brief SPDK new-thread callback.
 *
 * SPDK invokes this when spdk_thread_create is called.  We store the
 * thread pointer and spawn the reactor pthread that will poll it.
 *
 * @param thread  the newly created SPDK thread
 * @return        0 on successful pthread creation, non-zero on error
 */
static int
reactor_new_thread_fn(struct spdk_thread *thread)
{
    pthread_t tid;
    pthread_attr_t attr;
    int rc;

    printf("[spdk-reactor-poller] new_thread_fn callback (thread=%p)\n", (void *)thread);
    g_thread = thread;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    rc = pthread_create(&tid, &attr, reactor_thread, NULL);
    pthread_attr_destroy(&attr);
    return rc;
}

/* ---- main ---- */

/**
 * @brief Entry point for the SPDK thread/poller verification test.
 *
 * Initialises the SPDK environment and thread library, creates a thread
 * with a registered poller, then waits for the poller to signal
 * completion.
 *
 * @param argc  argument count
 * @param argv  argument vector
 * @return      0 on success, non-zero on error
 */
int
main(void)
{
    struct spdk_env_opts env_opts;
    struct spdk_thread *thread;
    int rc;

    printf("[spdk-reactor-poller] === SPDK Thread/Poller Framework Verification ===\n");

    /* init SPDK environment */
    spdk_env_opts_init(&env_opts);
    env_opts.name = "spdk_reactor_poller_test";
    env_opts.shm_id = -1;

    rc = spdk_env_init(&env_opts);
    if (rc < 0) {
        fprintf(stderr, "[spdk-reactor-poller] spdk_env_init failed (rc=%d)\n", rc);
        return EXIT_FAILURE;
    }
    printf("[spdk-reactor-poller] spdk_env_init OK (lcores=%u)\n", spdk_env_get_core_count());

    /* init SPDK thread library (creates msg mempool internally) */
    rc = spdk_thread_lib_init(reactor_new_thread_fn, 0);
    if (rc != 0) {
        fprintf(stderr, "[spdk-reactor-poller] spdk_thread_lib_init failed (rc=%d)\n", rc);
        spdk_env_fini();
        return EXIT_FAILURE;
    }
    printf("[spdk-reactor-poller] spdk_thread_lib_init OK\n");

    /* create a thread (triggers reactor_new_thread_fn -> reactor pthread) */
    thread = spdk_thread_create(THREAD_NAME, NULL);
    if (thread == NULL) {
        fprintf(stderr, "[spdk-reactor-poller] spdk_thread_create failed\n");
        spdk_env_fini();
        return EXIT_FAILURE;
    }
    printf("[spdk-reactor-poller] spdk_thread_create OK (thread=%p)\n", (void *)thread);

    /* wait for poller to complete (runs inside reactor pthread) */
    printf("[spdk-reactor-poller] waiting for %d poller ticks...\n", POLLER_MAX_COUNT);
    while (!atomic_load(&g_done)) {
        usleep(100000);   /* 100 ms */
    }

    printf("[spdk-reactor-poller] poll loop exited after %d ticks\n",
           atomic_load(&g_poller_ticks));

    spdk_env_fini();
    printf("[spdk-reactor-poller] === PASS ===\n");
    return EXIT_SUCCESS;
}
