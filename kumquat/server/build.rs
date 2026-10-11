// TODO(ai-review): generated, not yet audited
// The workspace root build.rs links libgfxstream_backend when an explicit
// path override is set, but a link-search directive only applies at build
// time: the binary carries no runtime search path, so running it outside
// the build environment needs LD_LIBRARY_PATH or a wrapper. Bake the
// override path into the binary's RUNPATH instead, so the loader finds the
// backend library without any environment setup.

use std::env;

fn main() {
    if env::var_os("CARGO_FEATURE_GFXSTREAM").is_none() {
        return;
    }

    // Mirrors the override selection of the workspace root build.rs: the
    // PROFILE-based pair takes precedence over the generic GFXSTREAM_PATH.
    let override_path = if env::var("PROFILE").as_deref() == Ok("debug") {
        env::var("GFXSTREAM_PATH_DEBUG")
    } else {
        env::var("GFXSTREAM_PATH_RELEASE")
    }
    .ok()
    .or_else(|| env::var("GFXSTREAM_PATH").ok());

    if let Some(path) = override_path.filter(|path| !path.is_empty()) {
        println!("cargo:rerun-if-env-changed=GFXSTREAM_PATH_RELEASE");
        println!("cargo:rerun-if-env-changed=GFXSTREAM_PATH_DEBUG");
        println!("cargo:rerun-if-env-changed=GFXSTREAM_PATH");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{path}");
    }
}
