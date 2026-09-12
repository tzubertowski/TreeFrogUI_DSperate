# SPDX-License-Identifier: GPL-3.0-or-later
add_executable(dsperate main.cpp config.cpp display.cpp audio.cpp input.cpp menu.cpp settings.cpp lid.cpp mic_alsa.cpp loader_cart.cpp dmaheap.cpp display_hcge.cpp
  display_disp.cpp display_fbdev.cpp)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "mips")
  target_sources(dsperate PRIVATE display_drm_stub.cpp)
  target_compile_definitions(dsperate PRIVATE DSPERATE_OLD_SDL=1)
else()
  target_sources(dsperate PRIVATE display_drm.cpp)
endif()
target_link_libraries(dsperate PRIVATE dsperate_core ${SDL2_LIBRARIES} ${CMAKE_DL_LIBS})
# RetroAchievements, when it is built at all. The frontend is the only consumer;
# with DSPERATE_CHEEVOS off the macro is undefined and main.cpp compiles as it
# did before any of this existed.
if(DSPERATE_CHEEVOS)
  target_link_libraries(dsperate PRIVATE dsperate_cheevos)
  target_compile_definitions(dsperate PRIVATE DSPERATE_CHEEVOS=1)
endif()
target_include_directories(dsperate PRIVATE ${SDL2_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
if(DSPERATE_WAYLAND)
  target_sources(dsperate PRIVATE display_wl.cpp wl_dyn.cpp wl/wayland-protocol.c wl/linux-dmabuf-v1.c wl/xdg-shell.c)
  target_compile_definitions(dsperate PRIVATE DSPERATE_WAYLAND=1)
  # Vendored protocol code marshals through dlopen'd pointers; force-include the macro layer, no linked wayland.
  set_source_files_properties(wl/wayland-protocol.c wl/linux-dmabuf-v1.c wl/xdg-shell.c PROPERTIES
    COMPILE_OPTIONS "-include;${CMAKE_CURRENT_SOURCE_DIR}/wl_dyn.h;-I${CMAKE_CURRENT_SOURCE_DIR}/wl")
else()
  target_sources(dsperate PRIVATE display_wl_stub.cpp)
endif()
if(DSPERATE_NET)
  target_link_libraries(dsperate PRIVATE dsperate_net)
  target_compile_definitions(dsperate PRIVATE DSPERATE_NET=1)
endif()
target_sources(dsperate PRIVATE gpu_present.cpp video/present_vk_import.cpp video/present_rga.cpp)
if(DSPERATE_VULKAN AND DSPERATE_VULKAN_INCLUDE)
  target_include_directories(dsperate PRIVATE ${DSPERATE_VULKAN_INCLUDE})
  set_property(SOURCE gpu_present.cpp APPEND PROPERTY COMPILE_DEFINITIONS
               DSPERATE_PRESENT_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
  set_property(SOURCE gpu_present.cpp APPEND PROPERTY OBJECT_DEPENDS
               "${CMAKE_CURRENT_SOURCE_DIR}/shaders/present.spv")
endif()
