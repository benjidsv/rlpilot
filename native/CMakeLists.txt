# Out-of-tree build of the rlpilot native pieces against an installed Speed Dreams (./build-macos.sh, ./build-linux.sh).
#   cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release && cmake --build native/build -j && cmake --install native/build
cmake_minimum_required(VERSION 3.20)
project(rlpilot C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
# Binaries are only run from the install prefix: link them with their final rpaths so that
# re-installing never has to edit them (install_name_tool fails on duplicate rpaths).
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

get_filename_component(RLP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(SD_SRC "${RLP_ROOT}/speed-dreams-code/src" CACHE PATH "Speed Dreams src/")
set(SD_BUILD "${RLP_ROOT}/speed-dreams-code/build" CACHE PATH "Speed Dreams build dir (config.h)")
set(SD_PREFIX "${RLP_ROOT}/sd-install" CACHE PATH "Speed Dreams install prefix")
set(SD_LIBDIR "${SD_PREFIX}/lib/games/speed-dreams-2/lib")

# Must match the engine's own definitions: they change struct layouts (tCarElt, ...).
set(SD_DEFINES CLIENT_SERVER GL_GLEXT_PROTOTYPES HAVE_CONFIG_H SHM SPEED_DREAMS THIRD_PARTY_EXPAT
    TRACE_LEVEL=5 TRACE_OUT UNLOAD_SSGGRAPH USE_RANDR_EXT _DEFAULT_SOURCE)
if(APPLE)
  list(APPEND SD_DEFINES USE_MACPORTS)
endif()
set(SD_INCLUDES "${SD_BUILD}" "${RLP_ROOT}/deps/include" "${SD_SRC}/interfaces"
    "${SD_SRC}/libs/portability" "${SD_SRC}/libs/tgf" "${SD_SRC}/libs/tgfdata" "${SD_SRC}/libs/tgfclient"
    "${SD_SRC}/libs/math" "${SD_SRC}/libs/robottools")

foreach(lib tgf tgfdata tgfclient robottools portability)
  find_library(SD_${lib} NAMES ${lib} PATHS "${SD_LIBDIR}" NO_DEFAULT_PATH REQUIRED)
endforeach()

add_executable(rlpilot-worker worker/main.cpp)
target_compile_definitions(rlpilot-worker PRIVATE ${SD_DEFINES})
target_include_directories(rlpilot-worker PRIVATE ${SD_INCLUDES})
if(APPLE)
  target_include_directories(rlpilot-worker PRIVATE /opt/homebrew/include/SDL2)
else()
  target_include_directories(rlpilot-worker PRIVATE /usr/include/SDL2)
endif()
target_link_libraries(rlpilot-worker PRIVATE ${SD_tgfdata} ${SD_tgfclient} ${SD_robottools} ${SD_tgf} ${SD_portability})
# ENABLE_EXPORTS: the robot module finds rlp_host with dlsym(RTLD_DEFAULT), which needs -rdynamic on Linux.
set_target_properties(rlpilot-worker PROPERTIES INSTALL_RPATH "${SD_LIBDIR};${SD_PREFIX}/lib" ENABLE_EXPORTS ON)

add_library(rlpilot MODULE robot/rlpilot.cpp)
set_target_properties(rlpilot PROPERTIES PREFIX "" SUFFIX ".so")
target_compile_definitions(rlpilot PRIVATE ${SD_DEFINES})
target_include_directories(rlpilot PRIVATE ${SD_INCLUDES})
target_link_libraries(rlpilot PRIVATE ${SD_robottools} ${SD_tgf} ${SD_portability})
set_target_properties(rlpilot PROPERTIES INSTALL_RPATH "${SD_LIBDIR}")

add_library(rlpilot_client SHARED client/client.c)
target_compile_options(rlpilot_client PRIVATE -O3)

add_executable(rlpilot-bench bench/bench.c)
target_link_libraries(rlpilot-bench PRIVATE rlpilot_client)
set_target_properties(rlpilot-bench PROPERTIES INSTALL_RPATH "${SD_PREFIX}/lib")

# The engine derives its lib/data dirs from the executable location: install next to speed-dreams-2.
install(TARGETS rlpilot-worker rlpilot-bench RUNTIME DESTINATION "${SD_PREFIX}/games")
install(TARGETS rlpilot_client LIBRARY DESTINATION "${SD_PREFIX}/lib")
install(TARGETS rlpilot LIBRARY DESTINATION "${SD_PREFIX}/lib/games/speed-dreams-2/drivers/rlpilot")
install(FILES "${RLP_ROOT}/data/raceman/rlpilot.xml"
        DESTINATION "${SD_PREFIX}/share/games/speed-dreams-2/config/raceman")
