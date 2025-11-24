#include <acl/acl.h>
#include <stdio.h>

aclError checkAclError(aclError ret, const char* funcName) {
    if (ret != ACL_SUCCESS) {
        printf("Error in %s: %d\n", funcName, ret);
    }
    return ret;
}

void* allocateNPUMemory(size_t size) {
    void* d_ptr = nullptr;
    
    // 在NPU上分配内存
    aclError ret = aclrtMalloc(&d_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    checkAclError(ret, "aclrtMalloc");
    
    if (ret == ACL_SUCCESS && d_ptr != nullptr) {
        printf("成功在NPU上分配 %zu 字节内存，地址: %p\n", size, d_ptr);
    } else {
        printf("NPU内存分配失败！\n");
        d_ptr = nullptr;
    }
    
    return d_ptr;
}

void transferNPUToHostPinned(void* d_src, void* h_dst, size_t size) {
    // 从NPU复制数据到Host的固定内存
    aclError ret = aclrtMemcpy(h_dst, size, d_src, size, ACL_MEMCPY_DEVICE_TO_HOST);
    checkAclError(ret, "aclrtMemcpy (Device to Host)");
    
    if (ret == ACL_SUCCESS) {
        printf("成功从NPU地址 %p 传输 %zu 字节数据到Host固定内存 %p\n", 
               d_src, size, h_dst);
    } else {
        printf("NPU到Host传输失败！\n");
    }
}

void transferHostPinnedToNPU(void* h_src, void* d_dst, size_t size) {
    // 从Host固定内存复制数据到NPU
    aclError ret = aclrtMemcpy(d_dst, size, h_src, size, ACL_MEMCPY_HOST_TO_DEVICE);
    checkAclError(ret, "aclrtMemcpy (Host to Device)");
    
    if (ret == ACL_SUCCESS) {
        printf("成功从Host固定内存 %p 传输 %zu 字节数据到NPU地址 %p\n", 
               h_src, size, d_dst);
    } else {
        printf("Host到NPU传输失败！\n");
    }
}

void* allocateHostPinnedMemory(size_t size) {
    void* h_ptr = nullptr;
    
    // 分配固定内存（页锁定内存）
    aclError ret = aclrtMallocHost(&h_ptr, size);
    checkAclError(ret, "aclrtMallocHost");
    
    if (ret == ACL_SUCCESS && h_ptr != nullptr) {
        printf("成功在Host上分配 %zu 字节固定内存，地址: %p\n", size, h_ptr);
    } else {
        printf("Host固定内存分配失败！\n");
        h_ptr = nullptr;
    }
    
    return h_ptr;
}

/**
 * @brief 释放NPU内存
 */
void freeNPUMemory(void* d_ptr) {
    if (d_ptr != nullptr) {
        aclError ret = aclrtFree(d_ptr);
        checkAclError(ret, "aclrtFree");
        if (ret == ACL_SUCCESS) {
            printf("已释放NPU内存: %p\n", d_ptr);
        } else {
            printf("释放NPU内存失败！\n");
        }
    }
}

/**
 * @brief 释放Host端固定内存
 */
void freeHostPinnedMemory(void* h_ptr) {
    if (h_ptr != nullptr) {
        aclError ret = aclrtFreeHost(h_ptr);
        checkAclError(ret, "aclrtFreeHost");
        if (ret == ACL_SUCCESS) {
            printf("已释放Host固定内存: %p\n", h_ptr);
        } else {
            printf("释放Host固定内存失败！\n");
        }
    }
}

// 示例主函数
int main() {
    aclError ret = aclInit(NULL);  // 初始化 ACL 运行时
    checkAclError(ret, "aclInit");
    if (ret != ACL_SUCCESS) {
        printf("ACL 初始化失败，无法继续！\n");
        return 1;
    }
    
    // 设置默认设备（假设设备ID为0，如果有多个NPU，需要指定）
    ret = aclrtSetDevice(0);
    checkAclError(ret, "aclrtSetDevice");
    if (ret != ACL_SUCCESS) {
        printf("设置NPU设备失败，无法继续！\n");
        aclFinalize();
        return 1;
    }
    
    const int N = 1024;  // 元素数量
    const size_t bytes = N * sizeof(float);
        
    printf("=== NPU 内存分配和传输示例 ===\n\n");

    float* d_data = (float*)allocateNPUMemory(bytes);
    if (d_data == nullptr) {
        printf("NPU内存分配失败，程序退出！\n");
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }

    float* h_src = (float*)allocateHostPinnedMemory(bytes);
    if (h_src == nullptr) {
        printf("Host源内存分配失败，程序退出！\n");
        freeNPUMemory(d_data);
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }

    float* h_dst = (float*)allocateHostPinnedMemory(bytes);
    if (h_dst == nullptr) {
        printf("Host目标内存分配失败，程序退出！\n");
        freeNPUMemory(d_data);
        freeHostPinnedMemory(h_src);
        aclrtResetDevice(0);
        aclFinalize();
        return 1;
    }

    // 在 Host 上初始化源数据
    printf("\n在Host上初始化源数据...\n");
    for (int i = 0; i < N; i++) {
        h_src[i] = 100.0f + i;
    }
    printf("Host源数据初始化完成\n");

    // 将 Host 源数据传输到 NPU
    printf("\n开始 Host 到 NPU 数据传输...\n");
    transferHostPinnedToNPU(h_src, d_data, bytes);
    
    // 将 NPU 数据传输到 Host 目标内存
    printf("\n开始 NPU 到 Host 数据传输...\n");
    transferNPUToHostPinned(d_data, h_dst, bytes);
    
    // 验证数据（打印前10个元素）
    printf("\n验证传输数据（前10个元素）:\n");
    for (int i = 0; i < 10; i++) {
        printf("h_dst[%d] = %.2f\n", i, h_dst[i]);
    }
    
    // 清理内存
    printf("\n清理内存...\n");
    freeNPUMemory(d_data);
    freeHostPinnedMemory(h_src);
    freeHostPinnedMemory(h_dst);
    
    // 清理设备和 ACL
    ret = aclrtResetDevice(0);
    checkAclError(ret, "aclrtResetDevice");
    aclFinalize();  // 清理 ACL 运行时
    
    printf("\n程序执行完毕！\n");
    
    return 0;
}