let
  pkgs = import (fetchTarball
    "https://github.com/NixOS/nixpkgs/archive/df8e3bd110921621d175fad88c9e67909b7cb3d3.tar.gz"
  ) {};

  futhark-rev = "6df5074681d548d72c5d1bb267a69ca5cab1ff94";
  futhark-src = pkgs.applyPatches {
    name = "futhark-patched";
    src = (fetchTarball "https://github.com/diku-dk/futhark/archive/${futhark-rev}.tar.gz");
    patches = [ ./futhark.patch ];
  };
  futhark-pinned = pkgs.haskellPackages.callPackage futhark-src { suffix = "nightly"; };
in
pkgs.stdenv.mkDerivation {
  name = "shell";
    buildInputs = with pkgs; [
    opencl-headers
    ocl-icd
    futhark-pinned
    gcc
    (python38.withPackages (pypkgs: with pypkgs; [
      numpy
      pyopencl
      statsmodels
    ]))
  ];
} 
