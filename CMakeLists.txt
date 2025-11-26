# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
# NPU to NVMe Transfer Project with SPDK and AscendC

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.16.0)

# project information
project(NPU_NVMe_Transfer LANGUAGES C CXX)

# ==================================================
# Ascend Configuration
# ==================================================
set(SOC_VERSION "Ascend910B3" CACHE STRING "system on chip type")

if(DEFINED ENV{USER} AND "$ENV{USER}" STREQUAL "root")
    set(DEFAULT_ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" 
        CACHE PATH "ASCEND CANN package default installation directory for root user")
else()
    set(DEFAULT_ASCEND_CANN_PACKAGE_PATH "$ENV{HOME}/Ascend/ascend-toolkit/latest" 
        CACHE PATH "ASCEND CANN package default installation directory for other user")
endif()

if(DEFINED ASCEND_CANN_PACKAGE_PATH)
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" 
        CACHE PATH "ASCEND CANN package installation directory" FORCE)
else()
    set(ASCEND_CANN_PACKAGE_PATH "${DEFAULT_ASCEND_CANN_PACKAGE_PATH}" 
        CACHE PATH "ASCEND CANN package installation directory")
endif()

message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

# ==================================================
# SPDK Configuration
# ==================================================
set(SPDK_ROOT_DIR "/home/user3/npu-nvme/spdk" CACHE PATH "SPDK root directory")
if(DEFINED ENV{SPDK_ROOT_DIR})
    set(SPDK_ROOT_DIR "$ENV{SPDK_ROOT_DIR}" CACHE PATH "SPDK root directory" FORCE)
endif()

message(STATUS "SPDK_ROOT_DIR: ${SPDK_ROOT_DIR}")

# ==================================================
# Build Configuration
# ==================================================
set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)" FORCE)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)

# C/C++ Standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

# ==================================================
# Ascend ACL Include and Library Directories
# ==================================================
set(ASCEND_INCLUDE_DIRS
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

# Ascend库搜索路径 - 按优先级排序
set(ASCEND_LIBRARY_DIRS
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/driver/lib64
    /usr/local/Ascend/driver/lib64
)

# Ascend所需的库
set(ASCEND_LIBS
    ascendcl          # ACL主库
    acl_op_compiler   # 算子编译器
    runtime           # 运行时库
    ge_runner         # 图引擎
)

# Ascend HAL和驱动库 - 需要整个链接
set(ASCEND_DRIVER_LIBS
    ascend_hal        # HAL硬件抽象层
    ascend_protobuf   # Protobuf
)

# ==================================================
# SPDK Include and Library Directories
# ==================================================
set(SPDK_INCLUDE_DIRS
    ${SPDK_ROOT_DIR}/include
    ${SPDK_ROOT_DIR}/dpdk/build/include
)

set(SPDK_LIBRARY_DIRS
    ${SPDK_ROOT_DIR}/build/lib
    ${SPDK_ROOT_DIR}/dpdk/build/lib
)

# SPDK核心库 - 必须按依赖顺序
set(SPDK_LIBS
    spdk_nvme
    spdk_vmd
    spdk_env_dpdk
    spdk_util
    spdk_log
    spdk_rpc
    spdk_jsonrpc
    spdk_json
    spdk_thread
    spdk_trace
    spdk_sock
    spdk_notify
    spdk_keyring     # 密钥环管理
)

# DPDK库
set(DPDK_LIBS
    rte_eal
    rte_mempool
    rte_ring
    rte_mbuf
    rte_bus_pci
    rte_pci
    rte_mempool_ring
    rte_telemetry
    rte_kvargs
)

# 系统库 - 包含OpenSSL
set(SYSTEM_LIBS
    pthread
    dl
    rt
    numa
    uuid
    ssl              # OpenSSL SSL库
    crypto           # OpenSSL Crypto库
    z                # zlib压缩库
    stdc++           # C++标准库 (某些Ascend组件需要)
)

# ==================================================
# Main Executable Target
# ==================================================
add_executable(npu_nvme_transfer
    npu_to_nvme_transfer.c
)

# Include directories
target_include_directories(npu_nvme_transfer PRIVATE
    ${SPDK_INCLUDE_DIRS}
    ${ASCEND_INCLUDE_DIRS}
)

# Link directories
target_link_directories(npu_nvme_transfer PRIVATE
    ${SPDK_LIBRARY_DIRS}
    ${ASCEND_LIBRARY_DIRS}
)

# 编译定义
target_compile_definitions(npu_nvme_transfer PRIVATE
    _GNU_SOURCE
)

# Compiler flags
target_compile_options(npu_nvme_transfer PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-missing-field-initializers
    -fno-strict-aliasing
)

# Link libraries - 顺序很重要！
target_link_libraries(npu_nvme_transfer PRIVATE
    # 使用 --start-group 和 --end-group 解决循环依赖
    -Wl,--start-group
    
    # SPDK库
    ${SPDK_LIBS}
    
    # DPDK库
    ${DPDK_LIBS}
    
    # Ascend库
    ${ASCEND_LIBS}
    
    -Wl,--end-group
    
    # Ascend驱动库 - 需要整个链接
    -Wl,--whole-archive
    ${ASCEND_DRIVER_LIBS}
    -Wl,--no-whole-archive
    
    # 系统库
    ${SYSTEM_LIBS}
)

# 链接器选项
target_link_options(npu_nvme_transfer PRIVATE
    -Wl,--no-as-needed
    -Wl,-rpath,${ASCEND_CANN_PACKAGE_PATH}/lib64
    -Wl,-rpath,${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64
    -Wl,-rpath,${ASCEND_CANN_PACKAGE_PATH}/driver/lib64
    -Wl,-rpath,/usr/local/Ascend/driver/lib64
)

# ==================================================
# Installation
# ==================================================
install(TARGETS npu_nvme_transfer
    RUNTIME DESTINATION bin
)

# 创建启动脚本
install(CODE "
    file(WRITE \${CMAKE_INSTALL_PREFIX}/bin/run_npu_nvme_transfer.sh
        \"#!/bin/bash\\n\"
        \"export LD_LIBRARY_PATH=${ASCEND_CANN_PACKAGE_PATH}/lib64:${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64:${ASCEND_CANN_PACKAGE_PATH}/driver/lib64:/usr/local/Ascend/driver/lib64:\\$LD_LIBRARY_PATH\\n\"
        \"sudo \\$(dirname \\$0)/npu_nvme_transfer \\$@\\n\"
    )
    execute_process(COMMAND chmod +x \${CMAKE_INSTALL_PREFIX}/bin/run_npu_nvme_transfer.sh)
")

# ==================================================
# Print Configuration Summary
# ==================================================
message(STATUS "==============================================")
message(STATUS "NPU to NVMe Transfer Configuration Summary")
message(STATUS "==============================================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Ascend Path: ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "SPDK Path: ${SPDK_ROOT_DIR}")
message(STATUS "SOC Version: ${SOC_VERSION}")
message(STATUS "==============================================")