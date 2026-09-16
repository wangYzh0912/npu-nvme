/* =======================================================================
 * reactor_v0_spdk_thread_test.c — V0 diagnostic for DPDK mempool init
 *
 * VERIFICATION-ONLY: This file was used to isolate the rte_mempool_create
 * failure after spdk_env_init (root cause: missing --whole-archive for
 * DPDK driver libs, which stripped the ring_mp_mc mempool ops constructor).
 *
 * Kept for regression testing of the SPDK env + thread lib init path.
 * Not linked into libnpu_nvme.so.
 * ======================================================================= */

#include <stdio.h>
#include <stdlib.h>
#include <spdk/env.h>
#include <spdk/thread.h>
#include <rte_mempool.h>
#include <rte_malloc.h>
#include <rte_errno.h>
#include <rte_memzone.h>

int
main(void)
{
    struct spdk_env_opts env_opts;
    int rc;

    spdk_env_opts_init(&env_opts);
    env_opts.name = "v0_diag";
    env_opts.shm_id = -1;

    rc = spdk_env_init(&env_opts);
    printf("spdk_env_init: %d (lcores=%u)\n", rc, spdk_env_get_core_count());
    if (rc < 0) return EXIT_FAILURE;

    /* Verify mempool creation works (V0 regression test) */
    struct rte_mempool *mp = rte_mempool_create(
        "v0_diag_mp", 256, 64, 0, 0,
        NULL, NULL, NULL, NULL, SOCKET_ID_ANY, 0);
    printf("rte_mempool_create: %s (err=%d)\n",
           mp ? "OK" : "FAIL", rte_errno);
    if (mp) rte_mempool_free(mp);

    /* Verify thread lib init works */
    rc = spdk_thread_lib_init(NULL, 0);
    printf("spdk_thread_lib_init: %d\n", rc);

    spdk_env_fini();
    return (mp && rc == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
