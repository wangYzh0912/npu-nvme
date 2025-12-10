# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved. 

cmake_minimum_required(VERSION 3.16.0)
project(NPU_NVMe_Transfer)

# ==================================================
# Ascend Configuration
# ==================================================
set(SOC_VERSION "Ascend310P3" CACHE STRING "system on chip type")

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

set(RUN_MODE "npu" CACHE STRING "run mode: npu")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--copy-dt-needed-entries")

# Check AscendC CMake
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit/tikcpp/samples/cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# Ascend include and library
set(ASCEND_ACL_INCLUDE_DIRS
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

set(ASCEND_ACL_LIB_DIRS
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)

set(ASCEND_ACL_LIBS
    ascendcl
    acl_op_compiler
    runtime
    ge_runner
)

# ==================================================
# SPDK Configuration
# ==================================================
set(SPDK_ROOT_DIR "$ENV{SPDK_ROOT_DIR}" CACHE PATH "SPDK root directory")
if(NOT SPDK_ROOT_DIR OR NOT EXISTS ${SPDK_ROOT_DIR})
    message(FATAL_ERROR "SPDK_ROOT_DIR is not set or does not exist")
endif()

set(SPDK_INCLUDE_DIRS
    ${SPDK_ROOT_DIR}/include
    ${SPDK_ROOT_DIR}/dpdk/build/include
)

set(SPDK_STATIC_LIBS
    ${SPDK_ROOT_DIR}/build/lib/libspdk_nvme.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_vmd.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_env_dpdk.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_dma.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_util.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_log.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_rpc.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_jsonrpc.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_json.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_thread.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_trace.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_sock.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_notify.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_keyring.a
)

# ISA-L
set(ISAL_LIBS "")
if(EXISTS ${SPDK_ROOT_DIR}/isa-l/.libs/libisal.a)
    list(APPEND ISAL_LIBS ${SPDK_ROOT_DIR}/isa-l/.libs/libisal.a)
    message(STATUS "✓ Found SPDK ISA-L")
elseif(EXISTS ${SPDK_ROOT_DIR}/isa-l/libisal.a)
    list(APPEND ISAL_LIBS ${SPDK_ROOT_DIR}/isa-l/libisal.a)
else()
    find_library(ISAL_SYSTEM_LIB isal)
    if(ISAL_SYSTEM_LIB)
        list(APPEND ISAL_LIBS ${ISAL_SYSTEM_LIB})
        message(STATUS "✓ Using system ISA-L")
    else()
        message(FATAL_ERROR "ISA-L library not found")
    endif()
endif()

file(GLOB DPDK_LIBS_FILES ${SPDK_ROOT_DIR}/dpdk/build/lib/librte_*.a)

set(SYSTEM_LIBS
    pthread dl rt numa uuid ssl crypto m stdc++ fuse3
)

message(STATUS "==============================================")
message(STATUS "Ascend Path: ${ASCEND_CANN_PACKAGE_PATH}")
message(STATUS "SPDK Path: ${SPDK_ROOT_DIR}")
message(STATUS "==============================================")

# ==================================================
# Library: libnpu_nvme (Shared Library)
# ==================================================
add_library(npu_nvme SHARED
    npu_nvme.c
)

target_include_directories(npu_nvme PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${SPDK_INCLUDE_DIRS}
    ${ASCEND_ACL_INCLUDE_DIRS}
)

target_link_directories(npu_nvme PRIVATE
    ${ASCEND_ACL_LIB_DIRS}
)

target_compile_definitions(npu_nvme PRIVATE _GNU_SOURCE)

target_compile_options(npu_nvme PRIVATE
    -Wall -Wno-unused-parameter -Wno-missing-field-initializers
)

target_link_libraries(npu_nvme PRIVATE
    -Wl,--whole-archive
    ${SPDK_STATIC_LIBS}
    -Wl,--no-whole-archive
    
    ${ISAL_LIBS}
    
    -Wl,--start-group
    ${DPDK_LIBS_FILES}
    -Wl,--end-group
    
    -Wl,--start-group
    ${ASCEND_ACL_LIBS}
    -Wl,--end-group
    
    ${SYSTEM_LIBS}
)

set_target_properties(npu_nvme PROPERTIES
    VERSION 1.0
    SOVERSION 1
    BUILD_RPATH "${ASCEND_ACL_LIB_DIRS}"
    INSTALL_RPATH "${ASCEND_ACL_LIB_DIRS}"
)

# ==================================================
# Test Executable
# ==================================================
add_executable(test_npu_nvme
    test_npu_nvme.c
)

target_include_directories(test_npu_nvme PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_ACL_INCLUDE_DIRS}
)

target_link_libraries(test_npu_nvme PRIVATE
    npu_nvme
    ${ASCEND_ACL_INCLUDE_DIRS}
)

# ==================================================
# Installation
# ==================================================
install(TARGETS npu_nvme
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES npu_nvme.h
    DESTINATION include
)

install(TARGETS test_npu_nvme
    RUNTIME DESTINATION bin
)

# Create run script
install(CODE "
set(SCRIPT_CONTENT \"#!/bin/bash
SCRIPT_DIR=\\\$(cd \\\$(dirname \\\${BASH_SOURCE[0]}) && pwd)
INSTALL_ROOT=\\\$(dirname \\\$SCRIPT_DIR)

if [ -f '${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash' ]; then
    source '${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash'
fi

# 永久添加LD_LIBRARY_PATH（支持libnpu_nvme的运行时路径）
export LD_LIBRARY_PATH=\\\$INSTALL_ROOT/lib:${ASCEND_CANN_PACKAGE_PATH}/lib64:\\\$LD_LIBRARY_PATH

# 日志输出（帮助调试）
echo \"[INFO] Executable: \\\$INSTALL_ROOT/bin/test_npu_nvme\"
echo \"[INFO] LD_LIBRARY_PATH: \\\$LD_LIBRARY_PATH\"

# 使用sudo运行目标测试程序且保留环境变量
sudo LD_LIBRARY_PATH=\\\$LD_LIBRARY_PATH \\\$INSTALL_ROOT/bin/test_npu_nvme \\\$@\")
file(WRITE \${CMAKE_INSTALL_PREFIX}/bin/run_test.sh \"\${SCRIPT_CONTENT}\")
execute_process(COMMAND chmod +x \${CMAKE_INSTALL_PREFIX}/bin/run_test.sh)
")


message(STATUS "==============================================")
message(STATUS "Build configured successfully")
message(STATUS "==============================================")