aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} op_proto_srcs)

if("x${op_proto_srcs}" STREQUAL "x")
    message(STATUS "op_proto: no source files, skip building op_proto library")
    return(0)
endif()

add_library(cust_op_proto SHARED ${op_proto_srcs})
target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(cust_op_proto PRIVATE -fvisibility=hidden)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_op_proto PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()

target_link_libraries(cust_op_proto PRIVATE
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)

set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME
                      cust_opsproto_rt2.0
)

install(TARGETS cust_op_proto
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
file(GLOB OP_PROTO_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
if(OP_PROTO_HEADERS)
    install(FILES ${OP_PROTO_HEADERS}
            DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
endif()
