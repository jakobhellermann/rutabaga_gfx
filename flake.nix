{
  description = "Kumquat with the local gfxstream backend";

  inputs = {
    gfxstream.url = "path:/home/sipgatejj/.personal/rust/gpu/gfxstream";
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
