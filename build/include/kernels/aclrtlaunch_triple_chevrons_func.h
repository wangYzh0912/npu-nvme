
#ifndef HEADER_ACLRTLAUNCH_HELLO_WORLD_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_HELLO_WORLD_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_hello_world(uint32_t blockDim, void* stream);

inline uint32_t hello_world(uint32_t blockDim, void* hold, void* stream)
{
    (void)hold;
    return aclrtlaunch_hello_world(blockDim, stream);
}

#endif
