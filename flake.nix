{
  description = "deepstrix — V4 Flash heterogeneous inference engine (Phase 0: hardware viability)";

  inputs.nixpkgs.url = "flake:nixpkgs";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.rocmSupport = true;
      };
      rocm = pkgs.rocmPackages;
    in {
      devShells.${system}.default = pkgs.mkShell {
        name = "deepstrix-dev";

        nativeBuildInputs = [
          pkgs.rustc
          pkgs.cargo
          pkgs.rustfmt
          pkgs.clippy
          pkgs.rust-analyzer
          pkgs.pkg-config
          pkgs.cmake

          rocm.clr
          rocm.hipcc
          rocm.rocminfo
          rocm.rocm-smi
          rocm.rocm-bandwidth-test
          rocm.rocm-runtime
          rocm.rocm-device-libs
          rocm.rocm-comgr
          rocm.clang
        ];

        shellHook = ''
          export ROCM_PATH=${rocm.clr}
          export HIP_PATH=${rocm.clr}
          export HIP_CLANG_PATH=${rocm.llvm.clang}/bin
          export HIP_DEVICE_LIB_PATH=${rocm.rocm-device-libs}/amdgcn/bitcode
          export HIPCC=${rocm.hipcc}/bin/hipcc
          export DEEPSTRIX_GFX_TARGETS="gfx1201 gfx1151"
          echo "deepstrix dev shell — ROCm ${rocm.clr.version}, hipcc ${rocm.hipcc.version}, rustc $(rustc --version)"
          echo "Devices:" && rocminfo 2>/dev/null | grep -E '^\s*Name:\s*gfx' | sort -u || echo "  rocminfo failed — check /dev/kfd"
        '';
      };
    };
}
