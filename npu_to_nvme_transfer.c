/*
 * NPU to NVMe Transfer Test using ACL and SPDK
 * 
 * This program demonstrates direct data transfer from NPU memory to NVMe storage
 * using Huawei ACL (Ascend Computing Language) and SPDK (Storage Performance Development Kit)
 */

#include "spdk/stdinc.h"
#include "spdk/nvme.h"
#include "spdk_internal/nvme_util.h"
#include "spdk/vmd.h"
#include "spdk/nvme_zns.h"
#include "spdk/env.h"
#include "spdk/string.h"
#include "spdk/log.h"
#include <acl/acl.h>
#include <string.h>


// 测试配置
#define TEST_DATA_SIZE (4096)  // 4KB 数据块大小
#define NUM_BLOCKS 1           // 要传输的块数量
#define STARTING_LBA 0         // NVMe起始LBA

// ACL 相关结构和函数
aclError checkAclError(aclError ret, const char* funcName) {
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "ACL Error in %s: %d\n", funcName, ret);
    }
    return ret;
}

void* allocateNPUMemory(size_t size) {
    void* d_ptr = NULL;
    
    aclError ret = aclrtMalloc(&d_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    checkAclError(ret, "aclrtMalloc");
    
    if (ret == ACL_SUCCESS && d_ptr != NULL) {
        printf("[NPU] 成功分配 %zu 字节内存，地址: %p\n", size, d_ptr);
    } else {
        fprintf(stderr, "[NPU] 内存分配失败！\n");
        d_ptr = NULL;
    }
    
    return d_ptr;
}

void* allocateHostPinnedMemory(size_t size) {
    void* h_ptr = NULL;
    
    aclError ret = aclrtMallocHost(&h_ptr, size);
    checkAclError(ret, "aclrtMallocHost");
    
    if (ret == ACL_SUCCESS && h_ptr != NULL) {
        printf("[Host] 成功分配 %zu 字节固定内存，地址: %p\n", size, h_ptr);
    } else {
        fprintf(stderr, "[Host] 固定内存分配失败！\n");
        h_ptr = NULL;
    }
    
    return h_ptr;
}

void transferNPUToHostPinned(void* d_src, void* h_dst, size_t size) {
    aclError ret = aclrtMemcpy(h_dst, size, d_src, size, ACL_MEMCPY_DEVICE_TO_HOST);
    checkAclError(ret, "aclrtMemcpy (NPU to Host)");
    
    if (ret == ACL_SUCCESS) {
        printf("[Transfer] NPU -> Host: %zu 字节 (%p -> %p)\n", size, d_src, h_dst);
    } else {
        fprintf(stderr, "[Transfer] NPU到Host传输失败！\n");
    }
}

void transferHostPinnedToNPU(void* h_src, void* d_dst, size_t size) {
    aclError ret = aclrtMemcpy(d_dst, size, h_src, size, ACL_MEMCPY_HOST_TO_DEVICE);
    checkAclError(ret, "aclrtMemcpy (Host to NPU)");
    
    if (ret == ACL_SUCCESS) {
        printf("[Transfer] Host -> NPU: %zu 字节 (%p -> %p)\n", size, h_src, d_dst);
    } else {
        fprintf(stderr, "[Transfer] Host到NPU传输失败！\n");
    }
}

void freeNPUMemory(void* d_ptr) {
    if (d_ptr != NULL) {
        aclError ret = aclrtFree(d_ptr);
        checkAclError(ret, "aclrtFree");
        if (ret == ACL_SUCCESS) {
            printf("[NPU] 已释放NPU内存: %p\n", d_ptr);
        }
    }
}

void freeHostPinnedMemory(void* h_ptr) {
    if (h_ptr != NULL) {
        aclError ret = aclrtFreeHost(h_ptr);
        checkAclError(ret, "aclrtFreeHost");
        if (ret == ACL_SUCCESS) {
            printf("[Host] 已释放Host固定内存: %p\n", h_ptr);
        }
    }
}

// SPDK 相关结构
struct ctrlr_entry {
    struct spdk_nvme_ctrlr      *ctrlr;
    TAILQ_ENTRY(ctrlr_entry)    link;
    char                        name[1024];
};

struct ns_entry {
    struct spdk_nvme_ctrlr  *ctrlr;
    struct spdk_nvme_ns     *ns;
    TAILQ_ENTRY(ns_entry)   link;
    struct spdk_nvme_qpair  *qpair;
};

static TAILQ_HEAD(, ctrlr_entry) g_controllers = TAILQ_HEAD_INITIALIZER(g_controllers);
static TAILQ_HEAD(, ns_entry) g_namespaces = TAILQ_HEAD_INITIALIZER(g_namespaces);
static struct spdk_nvme_transport_id g_trid = {};
static bool g_vmd = false;

// NPU到NVMe传输序列结构
struct npu_nvme_sequence {
    struct ns_entry *ns_entry;
    void            *npu_buffer;     // NPU内存指针
    void            *host_buffer;    // Host pinned内存指针
    size_t          buffer_size;
    int             is_write_completed;
    int             is_read_completed;
    int             error_occurred;
};

// SPDK 回调函数
static void
read_complete_cb(void *arg, const struct spdk_nvme_cpl *completion)
{
    struct npu_nvme_sequence *sequence = (struct npu_nvme_sequence *)arg;

    if (spdk_nvme_cpl_is_error(completion)) {
        spdk_nvme_qpair_print_completion(sequence->ns_entry->qpair, 
                                         (struct spdk_nvme_cpl *)completion);
        fprintf(stderr, "[NVMe] Read I/O error: %s\n", 
                spdk_nvme_cpl_get_status_string(&completion->status));
        sequence->error_occurred = 1;
        sequence->is_read_completed = 1;
        return;
    }

    printf("[NVMe] 读取完成，数据已在Host buffer中\n");
    sequence->is_read_completed = 1;
}

static void
write_complete_cb(void *arg, const struct spdk_nvme_cpl *completion)
{
    struct npu_nvme_sequence *sequence = (struct npu_nvme_sequence *)arg;

    if (spdk_nvme_cpl_is_error(completion)) {
        spdk_nvme_qpair_print_completion(sequence->ns_entry->qpair, 
                                         (struct spdk_nvme_cpl *)completion);
        fprintf(stderr, "[NVMe] Write I/O error: %s\n", 
                spdk_nvme_cpl_get_status_string(&completion->status));
        sequence->error_occurred = 1;
        sequence->is_write_completed = 1;
        return;
    }

    printf("[NVMe] 写入完成\n");
    sequence->is_write_completed = 1;
}

static void
reset_zone_complete_cb(void *arg, const struct spdk_nvme_cpl *completion)
{
    struct npu_nvme_sequence *sequence = (struct npu_nvme_sequence *)arg;

    if (spdk_nvme_cpl_is_error(completion)) {
        spdk_nvme_qpair_print_completion(sequence->ns_entry->qpair, 
                                         (struct spdk_nvme_cpl *)completion);
        fprintf(stderr, "[NVMe] Reset zone error: %s\n", 
                spdk_nvme_cpl_get_status_string(&completion->status));
        sequence->error_occurred = 1;
    }
    
    sequence->is_write_completed = 1;
}

// 注册NVMe namespace
static void
register_ns(struct spdk_nvme_ctrlr *ctrlr, struct spdk_nvme_ns *ns)
{
    struct ns_entry *entry;

    if (!spdk_nvme_ns_is_active(ns)) {
        return;
    }

    entry = (struct ns_entry *)malloc(sizeof(struct ns_entry));
    if (entry == NULL) {
        perror("ns_entry malloc");
        exit(1);
    }

    entry->ctrlr = ctrlr;
    entry->ns = ns;
    entry->qpair = NULL;
    TAILQ_INSERT_TAIL(&g_namespaces, entry, link);

    printf("[NVMe] Namespace ID: %d, Size: %juGB\n", 
           spdk_nvme_ns_get_id(ns),
           spdk_nvme_ns_get_size(ns) / 1000000000);
}

// NPU到NVMe传输主函数
static int
npu_to_nvme_transfer(void)
{
    struct ns_entry *ns_entry;
    struct npu_nvme_sequence sequence;
    int rc;
    size_t data_size = TEST_DATA_SIZE * NUM_BLOCKS;

    // 检查是否找到NVMe设备
    if (TAILQ_EMPTY(&g_namespaces)) {
        fprintf(stderr, "[Error] 没有找到可用的NVMe namespace\n");
        return -1;
    }

    // 使用第一个找到的namespace
    ns_entry = TAILQ_FIRST(&g_namespaces);
    
    printf("\n========================================\n");
    printf("开始 NPU -> NVMe 传输测试\n");
    printf("数据大小: %zu 字节\n", data_size);
    printf("========================================\n\n");

    // 分配IO队列对
    ns_entry->qpair = spdk_nvme_ctrlr_alloc_io_qpair(ns_entry->ctrlr, NULL, 0);
    if (ns_entry->qpair == NULL) {
        fprintf(stderr, "[Error] 分配IO qpair失败\n");
        return -1;
    }
    printf("[NVMe] 成功分配IO qpair\n");

    // 初始化sequence结构
    memset(&sequence, 0, sizeof(sequence));
    sequence.ns_entry = ns_entry;
    sequence.buffer_size = data_size;

    // 1. 分配NPU内存
    printf("\n[步骤 1] 在NPU上分配内存...\n");
    sequence.npu_buffer = allocateNPUMemory(data_size);
    if (sequence.npu_buffer == NULL) {
        fprintf(stderr, "[Error] NPU内存分配失败\n");
        rc = -1;
        goto cleanup_qpair;
    }

    // 2. 分配Host固定内存（用于初始化和验证）
    printf("\n[步骤 2] 分配Host固定内存...\n");
    sequence.host_buffer = allocateHostPinnedMemory(data_size);
    if (sequence.host_buffer == NULL) {
        fprintf(stderr, "[Error] Host内存分配失败\n");
        rc = -1;
        goto cleanup_npu;
    }

    // 3. 在Host上准备测试数据
    printf("\n[步骤 3] 在Host上初始化测试数据...\n");
    uint32_t *host_data = (uint32_t *)sequence.host_buffer;
    for (size_t i = 0; i < data_size / sizeof(uint32_t); i++) {
        host_data[i] = 0x12345678 + i;
    }
    printf("[Host] 数据初始化完成 (首个值: 0x%08x)\n", host_data[0]);

    // 4. 传输数据从Host到NPU
    printf("\n[步骤 4] 将数据从Host传输到NPU...\n");
    transferHostPinnedToNPU(sequence.host_buffer, sequence.npu_buffer, data_size);

    // 5. 清空host buffer以便后续验证
    memset(sequence.host_buffer, 0, data_size);

    // 6. 从NPU传输数据到Host（准备写入NVMe）
    printf("\n[步骤 5] 将数据从NPU传输回Host（准备写入NVMe）...\n");
    transferNPUToHostPinned(sequence.npu_buffer, sequence.host_buffer, data_size);

    // 验证数据
    printf("[验证] 检查NPU往返数据...\n");
    for (size_t i = 0; i < 10 && i < data_size / sizeof(uint32_t); i++) {
        printf("  host_data[%zu] = 0x%08x\n", i, host_data[i]);
    }

    // 7. 如果是ZNS设备，重置zone
    if (spdk_nvme_ns_get_csi(ns_entry->ns) == SPDK_NVME_CSI_ZNS) {
        printf("\n[步骤 6] 检测到ZNS namespace，重置zone...\n");
        sequence.is_write_completed = 0;
        rc = spdk_nvme_zns_reset_zone(ns_entry->ns, ns_entry->qpair,
                                      STARTING_LBA, false,
                                      reset_zone_complete_cb, &sequence);
        if (rc != 0) {
            fprintf(stderr, "[Error] Reset zone失败\n");
            goto cleanup_host;
        }
        
        while (!sequence.is_write_completed) {
            spdk_nvme_qpair_process_completions(ns_entry->qpair, 0);
        }
        
        if (sequence.error_occurred) {
            rc = -1;
            goto cleanup_host;
        }
    }

    // 8. 将Host buffer中的数据写入NVMe
    printf("\n[步骤 7] 将数据写入NVMe设备 (LBA %d)...\n", STARTING_LBA);
    sequence.is_write_completed = 0;
    sequence.error_occurred = 0;
    
    rc = spdk_nvme_ns_cmd_write(ns_entry->ns, ns_entry->qpair, 
                                sequence.host_buffer,
                                STARTING_LBA, NUM_BLOCKS,
                                write_complete_cb, &sequence, 0);
    if (rc != 0) {
        fprintf(stderr, "[Error] 提交写入命令失败\n");
        goto cleanup_host;
    }

    // 等待写入完成
    while (!sequence.is_write_completed) {
        spdk_nvme_qpair_process_completions(ns_entry->qpair, 0);
    }

    if (sequence.error_occurred) {
        fprintf(stderr, "[Error] 写入过程中发生错误\n");
        rc = -1;
        goto cleanup_host;
    }

    // 9. 清空host buffer
    printf("\n[步骤 8] 清空Host buffer准备读取验证...\n");
    memset(sequence.host_buffer, 0, data_size);

    // 10. 从NVMe读取数据回Host
    printf("\n[步骤 9] 从NVMe读取数据回Host...\n");
    sequence.is_read_completed = 0;
    sequence.error_occurred = 0;
    
    rc = spdk_nvme_ns_cmd_read(ns_entry->ns, ns_entry->qpair,
                               sequence.host_buffer,
                               STARTING_LBA, NUM_BLOCKS,
                               read_complete_cb, &sequence, 0);
    if (rc != 0) {
        fprintf(stderr, "[Error] 提交读取命令失败\n");
        goto cleanup_host;
    }

    // 等待读取完成
    while (!sequence.is_read_completed) {
        spdk_nvme_qpair_process_completions(ns_entry->qpair, 0);
    }

    if (sequence.error_occurred) {
        fprintf(stderr, "[Error] 读取过程中发生错误\n");
        rc = -1;
        goto cleanup_host;
    }

    // 11. 验证读取的数据
    printf("\n[步骤 10] 验证从NVMe读取的数据...\n");
    bool data_correct = true;
    for (size_t i = 0; i < data_size / sizeof(uint32_t); i++) {
        uint32_t expected = 0x12345678 + i;
        if (host_data[i] != expected) {
            fprintf(stderr, "[验证失败] 索引 %zu: 期望 0x%08x, 实际 0x%08x\n", 
                    i, expected, host_data[i]);
            data_correct = false;
            if (i >= 10) break;  // 只显示前几个错误
        }
    }

    if (data_correct) {
        printf("[验证成功] ✓ 所有数据匹配！\n");
        printf("前10个值:\n");
        for (size_t i = 0; i < 10 && i < data_size / sizeof(uint32_t); i++) {
            printf("  [%zu] = 0x%08x\n", i, host_data[i]);
        }
        rc = 0;
    } else {
        fprintf(stderr, "[验证失败] ✗ 数据不匹配\n");
        rc = -1;
    }

    printf("\n========================================\n");
    printf("NPU -> NVMe 传输测试完成\n");
    printf("========================================\n\n");

cleanup_host:
    freeHostPinnedMemory(sequence.host_buffer);
cleanup_npu:
    freeNPUMemory(sequence.npu_buffer);
cleanup_qpair:
    if (ns_entry->qpair) {
        spdk_nvme_ctrlr_free_io_qpair(ns_entry->qpair);
        ns_entry->qpair = NULL;
    }

    return rc;
}

// SPDK probe回调
static bool
probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
         struct spdk_nvme_ctrlr_opts *opts)
{
    printf("[NVMe] 正在连接到: %s\n", trid->traddr);
    return true;
}

// SPDK attach回调
static void
attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
          struct spdk_nvme_ctrlr *ctrlr, const struct spdk_nvme_ctrlr_opts *opts)
{
    int nsid;
    struct ctrlr_entry *entry;
    struct spdk_nvme_ns *ns;
    const struct spdk_nvme_ctrlr_data *cdata;

    entry = (struct ctrlr_entry *)malloc(sizeof(struct ctrlr_entry));
    if (entry == NULL) {
        perror("ctrlr_entry malloc");
        exit(1);
    }

    printf("[NVMe] 已连接到: %s\n", trid->traddr);

    cdata = spdk_nvme_ctrlr_get_data(ctrlr);
    snprintf(entry->name, sizeof(entry->name), "%-20.20s (%-20.20s)", 
             cdata->mn, cdata->sn);

    entry->ctrlr = ctrlr;
    TAILQ_INSERT_TAIL(&g_controllers, entry, link);

    // 注册所有活动的namespace
    for (nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); nsid != 0;
         nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid)) {
        ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
        if (ns == NULL) {
            continue;
        }
        register_ns(ctrlr, ns);
    }
}

// 清理资源
static void
cleanup(void)
{
    struct ns_entry *ns_entry, *tmp_ns_entry;
    struct ctrlr_entry *ctrlr_entry, *tmp_ctrlr_entry;
    struct spdk_nvme_detach_ctx *detach_ctx = NULL;

    TAILQ_FOREACH_SAFE(ns_entry, &g_namespaces, link, tmp_ns_entry) {
        TAILQ_REMOVE(&g_namespaces, ns_entry, link);
        free(ns_entry);
    }

    TAILQ_FOREACH_SAFE(ctrlr_entry, &g_controllers, link, tmp_ctrlr_entry) {
        TAILQ_REMOVE(&g_controllers, ctrlr_entry, link);
        spdk_nvme_detach_async(ctrlr_entry->ctrlr, &detach_ctx);
        free(ctrlr_entry);
    }

    if (detach_ctx) {
        spdk_nvme_detach_poll(detach_ctx);
    }
}

// 使用说明
static void
usage(const char *program_name)
{
    printf("%s [options]\n", program_name);
    printf("\n");
    printf("选项:\n");
    printf("  -d <MB>     DPDK huge memory大小 (MB)\n");
    printf("  -g          为DPDK内存段使用单一文件描述符\n");
    printf("  -i <ID>     共享内存组ID\n");
    printf("  -r <traddr> NVMe传输地址 (例如: traddr:0000:00:04.0)\n");
    printf("  -V          枚举VMD设备\n");
    printf("  -L <flag>   启用调试日志\n");
    printf("  -h          显示此帮助信息\n");
}

// 解析命令行参数
static int
parse_args(int argc, char **argv, struct spdk_env_opts *env_opts)
{
    int op, rc;

    spdk_nvme_trid_populate_transport(&g_trid, SPDK_NVME_TRANSPORT_PCIE);
    snprintf(g_trid.subnqn, sizeof(g_trid.subnqn), "%s", SPDK_NVMF_DISCOVERY_NQN);

    while ((op = getopt(argc, argv, "d:ghi:r:L:V")) != -1) {
        switch (op) {
        case 'V':
            g_vmd = true;
            break;
        case 'i':
            env_opts->shm_id = spdk_strtol(optarg, 10);
            if (env_opts->shm_id < 0) {
                fprintf(stderr, "无效的共享内存ID\n");
                return env_opts->shm_id;
            }
            break;
        case 'g':
            env_opts->hugepage_single_segments = true;
            break;
        case 'r':
            if (spdk_nvme_transport_id_parse(&g_trid, optarg) != 0) {
                fprintf(stderr, "解析传输地址错误\n");
                return 1;
            }
            break;
        case 'd':
            env_opts->mem_size = spdk_strtol(optarg, 10);
            if (env_opts->mem_size < 0) {
                fprintf(stderr, "无效的DPDK内存大小\n");
                return env_opts->mem_size;
            }
            break;
        case 'L':
            rc = spdk_log_set_flag(optarg);
            if (rc < 0) {
                fprintf(stderr, "未知的日志标志\n");
                usage(argv[0]);
                exit(EXIT_FAILURE);
            }
#ifdef DEBUG
            spdk_log_set_print_level(SPDK_LOG_DEBUG);
#endif
            break;
        case 'h':
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        default:
            usage(argv[0]);
            return 1;
        }
    }

    return 0;
}

int
main(int argc, char **argv)
{
    int rc;
    struct spdk_env_opts opts;
    aclError acl_ret;

    printf("======================================\n");
    printf("NPU to NVMe 直接传输测试程序\n");
    printf("======================================\n\n");

    // 1. 初始化ACL
    printf("[初始化] ACL运行时...\n");
    acl_ret = aclInit(NULL);
    checkAclError(acl_ret, "aclInit");
    if (acl_ret != ACL_SUCCESS) {
        fprintf(stderr, "[Error] ACL初始化失败\n");
        return 1;
    }

    // 2. 设置NPU设备
    acl_ret = aclrtSetDevice(0);
    checkAclError(acl_ret, "aclrtSetDevice");
    if (acl_ret != ACL_SUCCESS) {
        fprintf(stderr, "[Error] 设置NPU设备失败\n");
        aclFinalize();
        return 1;
    }
    printf("[ACL] 初始化成功，使用设备 0\n\n");

    // 3. 初始化SPDK环境
    printf("[初始化] SPDK环境...\n");
    opts.opts_size = sizeof(opts);
    spdk_env_opts_init(&opts);
    rc = parse_args(argc, argv, &opts);
    if (rc != 0) {
        aclrtResetDevice(0);
        aclFinalize();
        return rc;
    }

    opts.name = "npu_nvme_transfer";
    if (spdk_env_init(&opts) < 0) {
        fprintf(stderr, "[Error] 无法初始化SPDK环境\n");
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }
    printf("[SPDK] 环境初始化成功\n\n");

    // 4. 初始化VMD（如果需要）
    if (g_vmd && spdk_vmd_init()) {
        fprintf(stderr, "[Warning] VMD初始化失败，某些NVMe设备可能不可用\n");
    }

    // 5. 探测并连接NVMe控制器
    printf("[初始化] 探测NVMe控制器...\n");
    rc = spdk_nvme_probe(&g_trid, NULL, probe_cb, attach_cb, NULL);
    if (rc != 0) {
        fprintf(stderr, "[Error] spdk_nvme_probe()失败\n");
        rc = 1;
        goto cleanup;
    }

    if (TAILQ_EMPTY(&g_controllers)) {
        fprintf(stderr, "[Error] 未找到NVMe控制器\n");
        rc = 1;
        goto cleanup;
    }
    printf("[SPDK] NVMe控制器初始化完成\n\n");

    // 6. 执行NPU到NVMe的传输测试
    rc = npu_to_nvme_transfer();

cleanup:
    // 7. 清理SPDK资源
    printf("\n[清理] SPDK资源...\n");
    fflush(stdout);
    cleanup();
    if (g_vmd) {
        spdk_vmd_fini();
    }
    spdk_env_fini();

    // 8. 清理ACL资源
    printf("[清理] ACL资源...\n");
    aclrtResetDevice(0);
    aclFinalize();

    printf("\n程序执行完毕！返回码: %d\n", rc);
    return rc;
}