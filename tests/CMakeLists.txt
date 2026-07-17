# Host-side tests for the 64GPT core engine.
#
#   cmake -B build tests
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#
# Compiles the exact core/*.cpp sources that ship in the ROM, with
# ASan/UBSan enabled. Golden vectors live in tests/vectors/.
cmake_minimum_required(VERSION 3.20)
project(ngpt_tests CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(NGPT_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../core)
file(GLOB NGPT_CORE_SOURCES ${NGPT_CORE_DIR}/*.cpp)

add_library(ngpt_core STATIC ${NGPT_CORE_SOURCES})
target_include_directories(ngpt_core PUBLIC ${NGPT_CORE_DIR})

option(NGPT_SANITIZE "Build tests with ASan/UBSan" ON)
if(NGPT_SANITIZE)
  set(SANITIZE_FLAGS -fsanitize=address,undefined -fno-sanitize-recover=all -g)
else()
  set(SANITIZE_FLAGS -g)
endif()
target_compile_options(ngpt_core PUBLIC ${SANITIZE_FLAGS} -Wall -Wextra -Werror)
target_link_options(ngpt_core PUBLIC ${SANITIZE_FLAGS})

enable_testing()

function(ngpt_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE ngpt_core)
  target_compile_definitions(${name} PRIVATE
    NGPT_VECTOR_DIR="${CMAKE_CURRENT_SOURCE_DIR}/vectors")
  add_test(NAME ${name} COMMAND ${name})
endfunction()

ngpt_test(test_blob_parser)
ngpt_test(test_canned_model)
ngpt_test(test_gru_model)
ngpt_test(test_prompted_model)
ngpt_test(test_sampled_model)
ngpt_test(test_matvec_hook)

# M7 game-side subsystem (EventBus/WorldState/NPCDatabase/ContextBuilder):
# pure portable C++, no libdragon — compiles standalone, not against
# ngpt_core, and not part of the ROM-shipped game/src/user glob (that's
# built by Pyrite64's own generated Makefile).
set(NGPT_GAME_USER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../game/src/user)
add_executable(test_context_builder
  test_context_builder.cpp
  ${NGPT_GAME_USER_DIR}/EventBus.cpp
  ${NGPT_GAME_USER_DIR}/WorldState.cpp
  ${NGPT_GAME_USER_DIR}/NPCDatabase.cpp
  ${NGPT_GAME_USER_DIR}/ContextBuilder.cpp
)
target_include_directories(test_context_builder PRIVATE ${NGPT_GAME_USER_DIR})
target_compile_options(test_context_builder PRIVATE ${SANITIZE_FLAGS} -Wall -Wextra -Werror)
target_link_options(test_context_builder PRIVATE ${SANITIZE_FLAGS})
add_test(NAME test_context_builder COMMAND test_context_builder)

# M9 compositional conditioning (NpcService): same standalone portable-C++
# treatment as test_context_builder above.
add_executable(test_npc_service
  test_npc_service.cpp
  ${NGPT_GAME_USER_DIR}/NPCDatabase.cpp
  ${NGPT_GAME_USER_DIR}/NpcService.cpp
)
target_include_directories(test_npc_service PRIVATE ${NGPT_GAME_USER_DIR})
target_compile_options(test_npc_service PRIVATE ${SANITIZE_FLAGS} -Wall -Wextra -Werror)
target_link_options(test_npc_service PRIVATE ${SANITIZE_FLAGS})
add_test(NAME test_npc_service COMMAND test_npc_service)
