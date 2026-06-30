/* Internal: SPSC ring buffer for DMA slot management.
 *
 * Used by the write/read pipelines to track free buffer slots.
 * Single-producer, single-consumer with acquire/release barriers
 * for cross-thread visibility on weakly-ordered architectures (ARM64).
 */
#ifndef NPU_NVME_RING_BUFFER_H
#define NPU_NVME_RING_BUFFER_H

#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    int *slots;
    int capacity;
    int head;   /* consumer index (Python thread or reactor, not both) */
    int tail;   /* producer index (reactor thread) */
} ring_t;

static inline int ring_init(ring_t *r, int capacity) {
    r->slots = calloc(capacity, sizeof(int));
    if (!r->slots) return -1;
    r->capacity = capacity;
    r->head = 0;
    r->tail = 0;
    return 0;
}

static inline void ring_free(ring_t *r) {
    free(r->slots);
    r->slots = NULL;
}

static inline bool ring_is_full(const ring_t *r) {
    /* head is read by producer — need acquire to see consumer's updates. */
    int h = __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
    return (r->tail + 1) % r->capacity == h;
}

static inline bool ring_is_empty(const ring_t *r) {
    /* tail is read by consumer — need acquire to see producer's updates. */
    int t = __atomic_load_n(&r->tail, __ATOMIC_ACQUIRE);
    return r->head == t;
}

/* Push a slot index.  Called from the reactor thread (producer). */
static inline int ring_push(ring_t *r, int val) {
    if (ring_is_full(r)) return -1;
    r->slots[r->tail] = val;
    /* Ensure slot write is visible before updating tail. */
    __atomic_store_n(&r->tail, (r->tail + 1) % r->capacity, __ATOMIC_RELEASE);
    return 0;
}

/* Pop a slot index.  Called from the Python thread (consumer). */
static inline int ring_pop(ring_t *r, int *out) {
    if (ring_is_empty(r)) return -1;
    *out = r->slots[r->head];
    /* Ensure slot read happens after reading tail. */
    __atomic_store_n(&r->head, (r->head + 1) % r->capacity, __ATOMIC_RELEASE);
    return 0;
}

#endif
