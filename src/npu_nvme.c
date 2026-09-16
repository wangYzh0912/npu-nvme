/* Temporary C2a compatibility wrappers: one submit/wait implementation.
 * Raw-pointer callers cannot observe a timeout and then free DMA borrowers.
 * These wrappers retain that old lifetime contract until the ABI2 retirement. */
#include "internal/implementation.h"

static int legacy_transfer(NPUNVMEContext *ctx, void **ptrs, uint64_t *offsets,
                           size_t *sizes, int count, bool host, bool read,
                           uint32_t *crc) {
    if (validate_io_batch(ctx,ptrs,offsets,sizes,count)) return -EINVAL;
    NPUNVMETransferItem *items=calloc(count,sizeof(*items));
    if (!items) return -ENOMEM;
    for (int i=0;i<count;++i) {
        items[i].address=ptrs[i];items[i].offset=offsets[i];items[i].length=sizes[i];
    }
    NPUNVMETransferSpec spec={.struct_size=sizeof(spec),.version=1,
        .operation=read ? NPU_NVME_TRANSFER_READ : NPU_NVME_TRANSFER_WRITE,
        .memory_kind=host ? NPU_NVME_MEMORY_HOST : NPU_NVME_MEMORY_HBM,
        .checksum_flags=crc ? NPU_NVME_CHECK_CRC32 : 0,.item_count=count,.items=items};
    NPUNVMERequest *req=NULL;
    int rc=npu_nvme_submit_transfer(ctx,&spec,&req);free(items);
    if (rc) return rc;
    rc=npu_nvme_wait_request(req,0);
    if (rc==-ETIMEDOUT) {
        while (!atomic_load_explicit(&req->done,memory_order_acquire)) usleep(1000);
    }
    if (!rc && crc) for (int i=0;i<count;++i) crc[i]=req->tasks[i].crc32;
    npu_nvme_release_request(req);return rc;
}
int npu_nvme_write_batch(NPUNVMEContext *ctx, void **p,uint64_t *o,size_t *s,int n) {
    return legacy_transfer(ctx,p,o,s,n,false,false,NULL);
}
int npu_nvme_write_batch_host(NPUNVMEContext *ctx, void **p,uint64_t *o,size_t *s,int n) {
    return legacy_transfer(ctx,p,o,s,n,true,false,NULL);
}
int npu_nvme_write_batch_crc(NPUNVMEContext *ctx,void **p,uint64_t *o,size_t *s,uint32_t *crc,int n) {
    if (!crc) return -EINVAL;
    return legacy_transfer(ctx,p,o,s,n,false,false,crc);
}
int npu_nvme_read_batch(NPUNVMEContext *ctx,void **p,uint64_t *o,size_t *s,int n) {
    return legacy_transfer(ctx,p,o,s,n,false,true,NULL);
}
int npu_nvme_read_batch_host(NPUNVMEContext *ctx,void **p,uint64_t *o,size_t *s,int n) {
    return legacy_transfer(ctx,p,o,s,n,true,true,NULL);
}
