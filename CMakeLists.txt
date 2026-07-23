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
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/install" CACHE STRING "path for install()" FORCE)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--copy-dt-needed-entries")

# Ensure installed binaries can find libnpu_nvme.so in ../lib without LD_LIBRARY_PATH
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_BUILD_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

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
    ${SPDK_ROOT_DIR}/build/lib/libspdk_event.a
    ${SPDK_ROOT_DIR}/build/lib/libspdk_init.a
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

find_program(OBJCOPY_TOOL objcopy REQUIRED)
set(DPDK_MEMPOOL_RING_LIB ${SPDK_ROOT_DIR}/dpdk/build/lib/librte_mempool_ring.a)
set(DPDK_FIXED_DIR ${CMAKE_BINARY_DIR}/dpdk_fix)
set(DPDK_FIXED_MEMPOOL_RING_LIB ${DPDK_FIXED_DIR}/librte_mempool_ring_fixed.a)
add_custom_command(
    OUTPUT ${DPDK_FIXED_MEMPOOL_RING_LIB}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DPDK_FIXED_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${DPDK_MEMPOOL_RING_LIB} ${DPDK_FIXED_MEMPOOL_RING_LIB}
    COMMAND ${OBJCOPY_TOOL}
            --globalize-symbol=common_ring_alloc
            --globalize-symbol=common_ring_free
            --globalize-symbol=common_ring_mp_enqueue
            --globalize-symbol=common_ring_mc_dequeue
            --globalize-symbol=common_ring_get_count
            ${DPDK_FIXED_MEMPOOL_RING_LIB}
    DEPENDS ${DPDK_MEMPOOL_RING_LIB}
    COMMENT "Preparing DPDK ring mempool archive with exported ops symbols"
    VERBATIM
)
add_custom_target(dpdk_mempool_ring_fixed
    DEPENDS ${DPDK_FIXED_MEMPOOL_RING_LIB}
)

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
    src/npu_nvme.c
)
add_dependencies(npu_nvme dpdk_mempool_ring_fixed)

target_include_directories(npu_nvme PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
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

    # DPDK driver libs register via RTE_INIT constructors; --whole-archive
    # prevents the linker from stripping them (e.g. ring_mp_mc mempool ops).
    # The fixed mempool_ring archive has globalized symbols for extern refs.
    -Wl,--whole-archive
    ${DPDK_FIXED_MEMPOOL_RING_LIB}
    ${DPDK_LIBS_FILES}
    -Wl,--no-whole-archive

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
# Test executables
# ==================================================
add_executable(npu_nvme_api_test
    test/npu_nvme_api_test.c
)

target_include_directories(npu_nvme_api_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ASCEND_ACL_INCLUDE_DIRS}
)

target_link_libraries(npu_nvme_api_test PRIVATE
    npu_nvme
)

set_target_properties(npu_nvme_api_test PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

target_link_options(npu_nvme_api_test PRIVATE
    "LINKER:-rpath,$ORIGIN/../lib"
    "LINKER:--enable-new-dtags"
)

# ==================================================
# SPDK thread runtime test
# ==================================================
add_executable(spdk_thread_runtime_test
    test/spdk_thread_runtime_test.c
)
target_include_directories(spdk_thread_runtime_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SPDK_INCLUDE_DIRS}
    ${ASCEND_ACL_INCLUDE_DIRS}
)
target_compile_definitions(spdk_thread_runtime_test PRIVATE _GNU_SOURCE)
target_compile_options(spdk_thread_runtime_test PRIVATE
    -Wall -Wno-unused-parameter
)
target_link_libraries(spdk_thread_runtime_test PRIVATE
    -Wl,--whole-archive ${SPDK_STATIC_LIBS} -Wl,--no-whole-archive
    ${ISAL_LIBS}
    -Wl,--whole-archive ${DPDK_LIBS_FILES} -Wl,--no-whole-archive
    ${SYSTEM_LIBS}
)
set_target_properties(spdk_thread_runtime_test PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib" INSTALL_RPATH "$ORIGIN/../lib"
)
target_link_options(spdk_thread_runtime_test PRIVATE
    "LINKER:-rpath,$ORIGIN/../lib" "LINKER:--enable-new-dtags"
)

# ==================================================
# Reactor host I/O roundtrip test
# ==================================================
add_executable(reactor_host_io_roundtrip_test
    test/reactor_host_io_roundtrip_test.c
)
target_include_directories(reactor_host_io_roundtrip_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ASCEND_ACL_INCLUDE_DIRS}
)
target_link_libraries(reactor_host_io_roundtrip_test PRIVATE
    npu_nvme
)
set_target_properties(reactor_host_io_roundtrip_test PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)
target_link_options(reactor_host_io_roundtrip_test PRIVATE
    "LINKER:-rpath,$ORIGIN/../lib"
    "LINKER:--enable-new-dtags"
)

# ==================================================
# Async request hardware test
# ==================================================
add_executable(npu_nvme_async_hw_test
    test/npu_nvme_async_hw_test.c
)
target_include_directories(npu_nvme_async_hw_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ASCEND_ACL_INCLUDE_DIRS}
)
target_link_libraries(npu_nvme_async_hw_test PRIVATE
    npu_nvme
)
set_target_properties(npu_nvme_async_hw_test PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)
target_link_options(npu_nvme_async_hw_test PRIVATE
    "LINKER:-rpath,$ORIGIN/../lib"
    "LINKER:--enable-new-dtags"
)

add_executable(spdk_reactor_poller_test
    test/spdk_reactor_poller_test.c
)

target_include_directories(spdk_reactor_poller_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${SPDK_INCLUDE_DIRS}
    ${ASCEND_ACL_INCLUDE_DIRS}
)

target_compile_definitions(spdk_reactor_poller_test PRIVATE _GNU_SOURCE)

target_compile_options(spdk_reactor_poller_test PRIVATE
    -Wall -Wno-unused-parameter
)

target_link_libraries(spdk_reactor_poller_test PRIVATE
    -Wl,--whole-archive
    ${SPDK_STATIC_LIBS}
    -Wl,--no-whole-archive

    ${ISAL_LIBS}

    -Wl,--whole-archive
    ${DPDK_LIBS_FILES}
    -Wl,--no-whole-archive

    ${SYSTEM_LIBS}
)

set_target_properties(spdk_reactor_poller_test PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)

target_link_options(spdk_reactor_poller_test PRIVATE
    "LINKER:-rpath,$ORIGIN/../lib"
    "LINKER:--enable-new-dtags"
)

# ==================================================
# Installation
# ==================================================
install(TARGETS npu_nvme
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/npu_nvme.h
    DESTINATION include
)

install(PROGRAMS $<TARGET_FILE:npu_nvme_api_test>
    DESTINATION bin
)

install(PROGRAMS $<TARGET_FILE:npu_nvme_async_hw_test>
    DESTINATION bin
)

install(PROGRAMS scripts/run_npu_nvme_api_test.sh
    DESTINATION bin
    RENAME run_test.sh
)


message(STATUS "==============================================")
message(STATUS "Build configured successfully")
message(STATUS "==============================================")
