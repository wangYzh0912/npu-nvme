#ifndef __HELLO_WORLD__KERNEL_FUN_H__
#define __HELLO_WORLD__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define hello_world hello_world_origin
#include "/home/user3/npu-nvme/hello_world.cpp"

#undef hello_world
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_hello_world_kernel(
#ifdef ASCENDC_DUMP
GM_ADDR dumpAddr
#endif
) {
#ifdef ASCENDC_DUMP
    AscendC::InitDump(false, dumpAddr, ONE_CORE_DUMP_SIZE);
#endif

#ifdef ASCENDC_DUMP
    uint64_t __ascendc_tStamp = 0;
    uint64_t __ascendc_version = 0;
     __gm__ char* __ascendc_versionStr = nullptr;
    GetCannVersion(__ascendc_versionStr, __ascendc_version, __ascendc_tStamp);
    if (__ascendc_tStamp == 0) {
        AscendC::printf("[WARNING]: CANN TimeStamp is invalid, CANN TimeStamp is %u\n", __ascendc_tStamp);
    } else {
        AscendC::printf("CANN Version: %s, TimeStamp: %u\n", (__gm__ const char*)(__ascendc_versionStr), __ascendc_tStamp);
    }
#endif
    hello_world_origin();
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
