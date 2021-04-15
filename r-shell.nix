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

  unstable = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/870dbb751f4d851a3dfb554835a0c2f528386982.tar.gz") {};

  # TODO: report bug in fetchFromGitHub? hash of other source
  # will evaluate other source
  strucchangeRcpp = unstable.rPackages.buildRPackage rec {
    name = "strucchangeRcpp";
    src = pkgs.fetchFromGitHub {
      owner = "bfast2";
      repo = "strucchangeRcpp";
      rev = "1f0d4b1d5121fbee1fbe9890ee6ed572fd9ea33f";
      sha256 = "1ayrwxwrxwl1f9l87vm252hvizpbrqb39cpyjb4yvkcdfmavmbcg";
    };

    # Remove some stripping attempt and installation of documentation
    # that otherwise fails the build.
    postPatch = ''
      rm src/Makevars
      rm -rf vignettes
    '';

    nativeBuildInputs = with unstable.rPackages; [
      zoo
      sandwich
      Rcpp
      RcppArmadillo
    ];

    propagatedBuildInputs = nativeBuildInputs;
  };

  bfast2 = unstable.rPackages.buildRPackage rec {
    name = "bfast";
    src = pkgs.fetchFromGitHub {
      owner = "bfast2";
      repo = "bfast";
      rev = "def30feada20956c44a7f5b6c29622ae59172122";
      sha256 = "1f6nxi0sm3s5i3prsq0cxb573pndjbc4g1pmr6p4knfiyr6dkixf";
    };

    nativeBuildInputs = with unstable.rPackages; [
      strucchangeRcpp
      zoo
      forecast
      sp
      raster
      stlplus
      Rcpp
    ];

    propagatedBuildInputs = nativeBuildInputs;
  };

  Rpkgs = [
    strucchangeRcpp
    bfast2
  ];
in
pkgs.stdenv.mkDerivation {
  name = "shell";
    buildInputs = with pkgs; [
    opencl-headers
    ocl-icd
    futhark-pinned
    gcc
    (unstable.rWrapper.override { packages = Rpkgs; })
    (unstable.python38.withPackages (pypkgs: with pypkgs; [
      (rpy2.override { extraRPackages = Rpkgs; })
      numpy
      pyopencl
      statsmodels
    ]))
  ];
} 
