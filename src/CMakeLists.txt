include_directories( ${PYTHON_INCLUDE_DIRS} )
include_directories( ${Boost_INCLUDE_DIR} )

# Build our library
add_library( greet SHARED greet.cpp )

# Define the wrapper library that wraps our library
add_library( greet_ext SHARED greet_ext.cpp )
target_link_libraries( greet_ext ${Boost_LIBRARIES} greet )
# don't prepend wrapper library name with lib
set_target_properties( greet_ext PROPERTIES PREFIX "" )
