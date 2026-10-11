{
  description = "Kumquat with the local gfxstream backend";

  inputs = {
    gfxstream.url = "github:jakobhellermann/gfxstream/wip";
    nixpkgs.follows = "gfxstream/nixpkgs";
  };

  outputs =
    {
      self,
      nixpkgs,
      gfxstream,
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forSystems =
        f:
        nixpkgs.lib.genAttrs systems (
          system: f (import nixpkgs { inherit system; }) gfxstream.packages.${system}.default
        );
    in
    {
      packages = forSystems (
        pkgs: backend: {
          default = pkgs.rustPlatform.buildRustPackage {
            pname = "kumquat-gfxstream-wip";
            version = "0.1.85";
            src = ./.;

            cargoLock.lockFile = ./Cargo.lock;
            cargoBuildFlags = [
              "-p"
              "kumquat_virtio"
              "--features"
              "gfxstream"
            ];
            doCheck = false;

            nativeBuildInputs = with pkgs; [
              makeWrapper
              pkg-config
            ];
            buildInputs = [ backend ];

            # The gfxstream host renderer creates its host Vulkan instance
            # through the Vulkan loader, so a host Vulkan driver must be
            # reachable with absolute paths. On NixOS that is the mesa from
            # /run/opengl-driver; on other distros the nix glibc search paths
            # do not include the host driver or its dependency closure, and
            # exporting the distro library dir via LD_LIBRARY_PATH shadows
            # nix libraries (soname collisions). This package therefore only
            # works on NixOS hosts; elsewhere build kumquat with the host
            # toolchain so it uses the host loader and driver.
            postInstall = ''
              wrapProgram $out/bin/kumquat \
                --prefix LD_LIBRARY_PATH : ${
                  pkgs.lib.makeLibraryPath [
                    backend
                    pkgs.vulkan-loader
                  ]
                }
            '';

            meta.mainProgram = "kumquat";
          };
        }
      );

      devShells = forSystems (
        pkgs: backend: {
          default = pkgs.mkShell {
            inputsFrom = [ self.packages.${pkgs.stdenv.hostPlatform.system}.default ];
            packages = with pkgs; [
              cargo
              rustc
              pkg-config
            ];
            shellHook = ''
              export GFXSTREAM_PATH_RELEASE=${backend}/lib
            '';
          };
        }
      );
    };
}
