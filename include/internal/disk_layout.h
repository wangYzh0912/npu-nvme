/* Shared raw disk layout constants.
 *
 * These values mirror python/disk_layout.py and are used by the C layer
 * to keep raw NPU-SSD transfers outside the checkpoint metadata region.
 */
#ifndef NPU_NVME_DISK_LAYOUT_H
#define NPU_NVME_DISK_LAYOUT_H

#include <stdint.h>

#define NPU_NVME_SUPERBLOCK_OFFSET        0ULL
#define NPU_NVME_SUPERBLOCK_HEADER_BYTES  28U

#define NPU_NVME_META_SLOT_A_OFFSET       4096ULL
#define NPU_NVME_META_SLOT_BYTES          (400ULL * 1024ULL)
#define NPU_NVME_META_SLOT_B_OFFSET       (NPU_NVME_META_SLOT_A_OFFSET + \
                                           NPU_NVME_META_SLOT_BYTES)

/* Default start of the user raw-IO region: after the two metadata slots. */
#define NPU_NVME_RAW_IO_START_OFFSET      (NPU_NVME_META_SLOT_B_OFFSET + \
                                           NPU_NVME_META_SLOT_BYTES)

#define NPU_NVME_MAGIC_NUMBER_BYTES       8U

#endif
