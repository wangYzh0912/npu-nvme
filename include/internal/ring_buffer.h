/* Internal: SPSC ring buffer for DMA slot management.
 *
 * Used exclusively by the write/read pipelines to track free buffer slots.
 * Single-producer, single-consumer — no locks needed.
 */
#ifndef NPU_NVME_RING_BUFFER_H
#define NPU_NVME_RING_BUFFER_H

#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    int *slots;
    int capacity;
    int head;
    int tail;
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
    return (r->tail + 1) % r->capacity == r->head;
}

static inline bool ring_is_empty(const ring_t *r) {
    return r->head == r->tail;
}

static inline int ring_push(ring_t *r, int val) {
    if (ring_is_full(r)) return -1;
    r->slots[r->tail] = val;
    r->tail = (r->tail + 1) % r->capacity;
    return 0;
}

static inline int ring_pop(ring_t *r, int *out) {
    if (ring_is_empty(r)) return -1;
    *out = r->slots[r->head];
    r->head = (r->head + 1) % r->capacity;
    return 0;
}

#endif
