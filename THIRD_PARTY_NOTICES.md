# Third-Party Notices

Components bundled in this repository that were not written for LibreNoise.

## tinyexr

- Files: `third_party/tinyexr/tinyexr.h`, `exr_reader.hh`, `streamreader.hh`
- Source: https://github.com/syoyo/tinyexr, tag `v1.0.13` (released 2026-03-31), downloaded 2026-11-07 from `raw.githubusercontent.com/syoyo/tinyexr/v1.0.13/` (the two `.hh` files are headers `tinyexr.h` includes)
- License: BSD-3-Clause (text at the top of `tinyexr.h`). Copyright (c) 2014 - 2021, Syoyo Fujita and many contributors.
- Used by: the command-line harness only (`src/cli`), to read and write OpenEXR test frames. Not linked into the OFX or AE plug-ins.

## miniz

- Files: `third_party/tinyexr/miniz.c`, `third_party/tinyexr/miniz.h`, `third_party/tinyexr/LICENSE.miniz`
- Source: the copy vendored by tinyexr, `deps/miniz/` at tinyexr tag `v1.0.13`, downloaded 2026-11-07
- License: MIT (see `LICENSE.miniz`). Copyright 2013-2014 RAD Game Tools and Valve Software, Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC.
- Used by: tinyexr for ZIP / ZIPS EXR compression (CLI only).

## OpenFX API headers

- Files: `include/openfx/`
- Source: Academy Software Foundation OpenFX
- License: BSD-3-Clause
- Used by: the OFX plug-in.
