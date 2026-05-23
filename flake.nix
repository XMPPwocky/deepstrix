{
  description = "deepstrix — V4 Flash heterogeneous inference engine";

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

      # Unified ROCM_PATH tree. ds4's Makefile uses $(ROCM_PATH)/bin/hipcc
      # and -L$(ROCM_PATH)/lib -lhipblas, expecting one /opt/rocm-like prefix.
      # Nix splits these across several derivations, so we symlinkJoin them
      # into a single tree we can point at.
      rocmJoin = pkgs.symlinkJoin {
        name = "rocm-merged-deepstrix";
        paths = [
          rocm.clr
          rocm.hipcc
          rocm.hipblas
          rocm.hipblas-common
          rocm.rocblas
          rocm.rocsolver
          rocm.rocwmma
          rocm.rocm-runtime
          rocm.rocm-device-libs
          rocm.rocm-comgr
          rocm.rocm-core
          rocm.rocminfo
          rocm.rocm-smi
        ];
      };
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
          pkgs.gnumake
          pkgs.git

          # rocmJoin already pulls these into a single tree, but listing
          # them here makes them appear in PATH and pkg-config separately
          # so non-ds4 paths still work.
          rocm.clr
          rocm.hipcc
          rocm.hipblas
          rocm.rocblas
          rocm.rocminfo
          rocm.rocm-smi
          rocm.rocm-bandwidth-test
          rocm.rocm-runtime
          rocm.rocm-device-libs
          rocm.rocm-comgr
          rocm.clang
        ];

        shellHook = ''
          export ROCM_PATH=${rocmJoin}
          export HIP_PATH=${rocmJoin}
          export HIP_CLANG_PATH=${rocm.llvm.clang}/bin
          export HIP_DEVICE_LIB_PATH=${rocm.rocm-device-libs}/amdgcn/bitcode
          export HIPCC=${rocmJoin}/bin/hipcc
          export DEEPSTRIX_GFX_TARGETS="gfx1201 gfx1151 gfx1100"
          export DS4_ROCM_ARCH=gfx1151

          # ds4's Makefile passes no -I flags to hipcc; hipcc by itself
          # doesn't look at HIP_PATH/include for downstream libraries like
          # hipblas. CPATH is honored by the underlying clang, so headers
          # under rocmJoin/include become visible automatically.
          export CPATH=${rocmJoin}/include''${CPATH:+:$CPATH}
          export LIBRARY_PATH=${rocmJoin}/lib''${LIBRARY_PATH:+:$LIBRARY_PATH}
          echo "deepstrix dev shell — ROCm ${rocm.clr.version}, hipcc ${rocm.hipcc.version}, rustc $(rustc --version)"
          echo "Devices:" && rocminfo 2>/dev/null | grep -E '^\s*Name:\s*gfx' | sort -u || echo "  rocminfo failed — check /dev/kfd"
        '';
      };
    };
}
