#ifndef HEADER_ACLRTLAUNCH_HELLO_WORLD_H
#define HEADER_ACLRTLAUNCH_HELLO_WORLD_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_hello_world(uint32_t blockDim, aclrtStream stream);
#endif
