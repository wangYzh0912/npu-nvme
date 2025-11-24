file(REMOVE_RECURSE
  "lib/libkernels.a"
  "lib/libkernels.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/kernels.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
