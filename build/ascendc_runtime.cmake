add_library(ascendc_runtime_obj OBJECT IMPORTED)
set_target_properties(ascendc_runtime_obj PROPERTIES
    IMPORTED_OBJECTS "/home/user3/npu-nvme/build/elf_tool.c.o;/home/user3/npu-nvme/build/ascendc_runtime.cpp.o"
)
