# GPU passthrough test setup, two independent chains:
#
# 1. venus chain (virglrenderer):
#      guest: mesa vulkan driver "virtio" (venus ICD)  → mesa/build/src/virtio/vulkan/
#      host:  virglrenderer vtest server (--venus)     → virglrenderer/build/vtest/
#      snapshot support: none (vkr has no snapshot)
#      note: `just server` also needs the HOST ICD env below (RADV, not venus!)
#
# 2. gfxstream chain (kumquat) — SNAPSHOT CAPABLE:
#      guest: mesa vulkan driver "gfxstream"           → mesa/build/src/gfxstream/guest/vulkan/
#             needs VIRTGPU_KUMQUAT=1 to select the kumquat transport
#      host:  kumquat_virtio (rutabaga standalone)     → rutabaga_gfx/target/
#             links libgfxstream_backend.so with GFXSTREAM_BUILD_WITH_SNAPSHOT_SUPPORT
#      status: vulkaninfo works, vkcube crashes in wayland WSI (upstream bug)

MESADIR := '/home/jakob/dev/contrib/gpu/mesa'
VIRGLDIR := '/home/jakob/dev/contrib/gpu/virglrenderer'
GFXDIR := '/home/jakob/dev/contrib/gpu/gfxstream'
RUTADIR := '/home/jakob/dev/contrib/gpu/rutabaga_gfx'

VENUS_ICD := MESADIR + '/build/src/virtio/vulkan/virtio_icd.x86_64.json'
GFX_ICD := MESADIR + '/build/src/gfxstream/guest/vulkan/gfxstream_vk_icd.x86_64.json'
RADV_ICD := '/usr/share/vulkan/icd.d/radeon_icd.json'

# ─── venus chain ──────────────────────────────────────────────────────────
# host ICDs (server needs RADV, client needs venus) — set per recipe, not
# globally, so the gfxstream chain can't inherit the wrong one.

server:
    @echo "vtest server listening on /tmp/.virgl_test -- strg+c to stop"
    VK_ICD_FILENAMES={{ RADV_ICD }} VIRGL_LOG_LEVEL=debug \
    VIRGL_LOG_FILE=/tmp/virgl-debug.log \
    cd {{ VIRGLDIR }}/build && ./vtest/virgl_test_server --venus

vkinfo:
    VK_DRIVER_FILES={{ VENUS_ICD }} VN_DEBUG=vtest vulkaninfo --summary

vkcube:
    VK_DRIVER_FILES={{ VENUS_ICD }} VN_DEBUG=vtest vkcube

smoke:
    VK_DRIVER_FILES={{ VENUS_ICD }} VN_DEBUG=vtest \
    vulkaninfo --summary 2>&1 | grep -q "Venus" \
    && echo "venus chain OK" || echo "venus chain BROKEN"

# ─── gfxstream chain (kumquat) ────────────────────────────────────────────

# kumquat host: gfxstream backend behind a userspace virtio-gpu socket.
# GFXSTREAM_PATH_RELEASE links the locally built backend (has snapshot support);
# LD_LIBRARY_PATH makes the binary load it instead of the system /usr/lib copy.
# env -u clears the venus ICD vars — kumquat's host-vulkan must use RADV.
kumquat:
    @echo "kumquat listening on /tmp/kumquat-gpu-0 -- log: /tmp/kumquat.log, strg+c to stop"
    @mv /tmp/kumquat.log /tmp/kumquat.log.1 2>/dev/null || true
    @env -u VK_DRIVER_FILES -u VN_DEBUG \
    GFXSTREAM_PATH_RELEASE={{ GFXDIR }}/build/host \
    LD_LIBRARY_PATH={{ GFXDIR }}/build/host \
    GFXSTREAM_LOG_LEVEL=info \
    cargo run -q --release -p kumquat_virtio --features gfxstream \
        --manifest-path {{ RUTADIR }}/kumquat/server/Cargo.toml -- \
        --capset-names=gfxstream-vulkan --gpu-socket-path=/tmp/kumquat-gpu-0 \
        2>&1 | tee /tmp/kumquat.log

# guest env for the gfxstream chain (vkcube, vulkaninfo, ...):
# VIRTGPU_KUMQUAT=1 selects the kumquat transport inside the gfxstream ICD.
# snapshot/restore the gfxstream context of the running kumquat (SIGUSR1/SIGUSR2)
kumquat-snap:
    rm -rf /tmp/kumquat-snapshot
    pkill -USR1 -x kumquat
    @sleep 1; ls /tmp/kumquat-snapshot/ 2>/dev/null | head -4

kumquat-restore:
    pkill -USR2 -x kumquat

gfxstream-env:
    @echo "export VIRTGPU_KUMQUAT=1"
    @echo "export VK_DRIVER_FILES={{ GFX_ICD }}"

gfxstream-check:
    VIRTGPU_KUMQUAT=1 VK_DRIVER_FILES={{ GFX_ICD }} \
        vulkaninfo --summary 2>&1 | grep -q "GFXStream" \
        && echo "gfxstream chain OK" || echo "gfxstream chain BROKEN"

[positional-arguments]
gfxstream-run *args:
    VIRTGPU_KUMQUAT=1 VK_DRIVER_FILES={{ GFX_ICD }} "$@"

gfxstream-vkcube:
    VIRTGPU_KUMQUAT=1 VK_DRIVER_FILES={{ GFX_ICD }} vkcube

# ─── builds ───────────────────────────────────────────────────────────────

# configure + build all three trees with the options the chains need.
# gfxstream additionally needs the snapshot defines in host/meson.build and
# the generated protobuf files (host/*.pb.cc — see protoc step in git log).
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    [[ -d {{ VIRGLDIR }}/build ]] || meson setup {{ VIRGLDIR }}/build {{ VIRGLDIR }} \
        -Dvenus=true -Drender-server-mode=thread
    meson configure {{ VIRGLDIR }}/build -Dvenus=true -Drender-server-mode=thread >/dev/null
    ninja -C {{ VIRGLDIR }}/build
    [[ -d {{ MESADIR }}/build ]] || meson setup {{ MESADIR }}/build {{ MESADIR }} \
        -Dvulkan-drivers=virtio,gfxstream -Dgallium-drivers= -Dllvm=disabled \
        -Dplatforms=x11,wayland -Dvirtgpu_kumquat=true
    meson configure {{ MESADIR }}/build -Dplatforms=x11,wayland -Dvirtgpu_kumquat=true >/dev/null
    ninja -C {{ MESADIR }}/build
    ninja -C {{ GFXDIR }}/build
    # patch the ICD library paths (meson writes install paths, we need build-tree paths)
    for icd in {{ VENUS_ICD }} {{ GFX_ICD }}; do
        [[ -f "$icd" ]] || continue
        lib=$(basename $(grep -oE '"/usr/local/lib/[^"]+"' "$icd" | tr -d '"'))
        dir=$(dirname "$icd")
        sed -i "s|/usr/local/lib/$lib|$dir/$lib|" "$icd"
        echo "patched: $icd"
    done

build-gfxstream:
    ninja -C {{ GFXDIR }}/build

# gfxstream host snapshot/restore in isolation (no guest, no kumquat, no vulkan app)
gfxstream-snap-test:
    #!/usr/bin/env bash
    set -euo pipefail
    GFX={{ GFXDIR }}
    gcc -o /tmp/gfx-snap-test {{ justfile_directory() }}/gfxstream_snap_test.c \
        -I$GFX/host/include -I$GFX/host -I$GFX/third_party/drm/include \
        -L$GFX/build/host -lgfxstream_backend
    rm -rf /tmp/gfxsnap
    ANDROID_GFXSTREAM_CAPTURE_VK_SNAPSHOT=1 \
    LD_LIBRARY_PATH=$GFX/build/host /tmp/gfx-snap-test /tmp/gfxsnap
    echo "---"
    echo "(shader compile warning from GLES translator init is harmless)"

build-mesa:
    ninja -C {{ MESADIR }}/build

build-virgl:
    ninja -C {{ VIRGLDIR }}/build

# Run a valo harness test binary directly (no harness timeout) — for
# debugging live hangs: start it, then inspect /proc/<pid> or gdb -p.
# kumquat must be running; KUMQUAT_PID is resolved automatically.
run-test-standalone name:
    #!/usr/bin/env bash
    set -euo pipefail
    KP=$(pgrep -x kumquat | head -1)
    [[ -n "$KP" ]] || { echo "kumquat is not running (just kumquat)"; exit 1; }
    cd /home/jakob/dev/rust/valo
    exec env KUMQUAT_PID=$KP \
        VIRTGPU_KUMQUAT=1 \
        VK_DRIVER_FILES=/home/jakob/dev/contrib/gpu/mesa/build/src/gfxstream/guest/vulkan/gfxstream_vk_icd.x86_64.json \
        LD_LIBRARY_PATH=/home/jakob/dev/rust/valo/out \
        GLIBC_TUNABLES=glibc.pthread.rseq=0 \
        VALO_MAX_RESTORES=20 \
        VALO_LOG=/tmp/valo-standalone.log \
        out/tests/"{{ name }}"

