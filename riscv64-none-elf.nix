# RISC-V cross compiler (GCC 15) with soft-float newlib.
#
# Builds a riscv64-none-elf GCC 15 targeting rv64ima / lp64 (no FPU).

let
  nixpkgsImport = import ./nixpkgs.nix;

  # Cross system with soft-float ABI matching our zkVM targets (rv64ima / lp64).
  # The default riscv64-embedded uses lp64d (double-float) which is incompatible
  # with the bare-metal zkVM environment that has no FPU.
  softFloatCross = {
    config = "riscv64-none-elf";
    libc = "newlib";
    gcc = {
      arch = "rv64ima";
      abi = "lp64";
    };
  };

  # First, import without overlays to discover the bootstrap GCC that
  # newlib will be built with.  We need this concrete reference to break
  # the cycle (an overlay that touches newlib triggers stdenv rebuild).
  basePkgs = nixpkgsImport { crossSystem = softFloatCross; };

  # The bootstrap (nolibc) GCC used to compile newlib — its store path
  # leaks into .a files via ELF .comment sections (~229 MB parasitic dep).
  bootstrapGcc = basePkgs.stdenvNoLibc.cc.cc;

  # Overlay: strip the parasitic reference from newlib to the bootstrap
  # compiler.  We use remove-references-to to nuke the store path string
  # after the build, preventing nix's reference scanner from treating the
  # embedded include-path string as a runtime dependency.
  stripBootstrapRef = final: prev: {
    newlib = prev.newlib.overrideAttrs (old: {
      nativeBuildInputs = (old.nativeBuildInputs or []) ++ [
        final.buildPackages.removeReferencesTo
      ];
      postFixup = (old.postFixup or "") + ''
        echo "Stripping parasitic bootstrap-gcc reference from newlib..."
        find $out -type f -name '*.a' -exec \
          remove-references-to -t ${bootstrapGcc} {} +
      '';
      disallowedReferences = [ bootstrapGcc ];
    });
  };

  softFloatPkgs = nixpkgsImport {
    crossSystem = softFloatCross;
    overlays = [ stripBootstrapRef ];
  };
in softFloatPkgs.buildPackages.gcc15
