#!/bin/sh
# Builds the command-line tools Use As Server runs (xdotool, xclip, wmctrl, playerctl, pactl)
# into the Flatpak, so users don't have to install them. Each source sits in its own folder
# next to this script (see com.jeetstudio.UseAsServer.yml).
#
# Libraries the Flatpak runtime may not have (libXtst, libXinerama, libXt, libXmu) are built
# first, only when missing. A tool that can't be built is skipped, not fatal: the app then
# uses the computer's own copy, if it has one (see tool-wrapper.sh).
#
# Test outside Flatpak:  PREFIX=/tmp/app sh build-tools.sh
set -u
PREFIX="${PREFIX:-/app}"
JOBS="${FLATPAK_BUILDER_N_JOBS:-$(nproc)}"
SRC="$(cd "$(dirname "$0")" && pwd)"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/share/pkgconfig:${PKG_CONFIG_PATH:-}"
export ACLOCAL_PATH="$PREFIX/share/aclocal${ACLOCAL_PATH:+:$ACLOCAL_PATH}"
export LD_LIBRARY_PATH="$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Old C code + new compilers: don't let modern defaults turn old warnings into errors.
export CFLAGS="${CFLAGS:--O2} -std=gnu17 -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=incompatible-pointer-types -Wno-error=int-conversion"
# xclip and wmctrl look for X headers/libraries without pkg-config: point them at $PREFIX too.
export CPPFLAGS="-I$PREFIX/include ${CPPFLAGS:-}"
export LDFLAGS="-L$PREFIX/lib ${LDFLAGS:-}"
SUMMARY="$PREFIX/share/useas/bundled-tools.txt"
mkdir -p "$PREFIX/share/useas" "$PREFIX/bin"
: > "$SUMMARY"

log() { echo "==== $*"; }
ok()   { echo "$1: bundled" >> "$SUMMARY"; log "$1 OK"; }
skip() { echo "$1: NOT bundled (build failed; uses the computer's copy)" >> "$SUMMARY"; log "$1 FAILED"; }

autotools() {   # folder [configure options]
    d="$1"; shift
    ( cd "$SRC/$d" &&
      { [ -x configure ] || autoreconf -fi; } &&
      ./configure --prefix="$PREFIX" --disable-static --disable-docs --disable-specs "$@" &&
      make -j"$JOBS" && make install )
}

lib_if_missing() {   # pkg-config name, folder
    if pkg-config --exists "$1"; then log "$1 already in the runtime"; return 0; fi
    log "building $1"
    autotools "$2"
}

# xorg-macros: only needed to run autoreconf on the X libraries below
pkg-config --exists xorg-macros || autotools macros

lib_if_missing xtst xtst
lib_if_missing xinerama xinerama
lib_if_missing xt xt
lib_if_missing xmu xmu

log xdotool
if ( cd "$SRC/xdotool" &&
     make -j"$JOBS" PREFIX="$PREFIX" WITHOUT_RPATH_FIX=1 xdotool libxdo.so &&
     install -Dm755 xdotool "$PREFIX/bin/xdotool" &&
     install -Dm755 libxdo.so "$PREFIX/lib/libxdo.so.3" &&
     ln -sf libxdo.so.3 "$PREFIX/lib/libxdo.so" ); then ok xdotool; else skip xdotool; fi

log xclip
# xclip's Makefile ignores CPPFLAGS
if ( CFLAGS="$CFLAGS $CPPFLAGS" autotools xclip ); then ok xclip; else skip xclip; fi

log wmctrl
if autotools wmctrl; then ok wmctrl; else skip wmctrl; fi

log playerctl
if ( cd "$SRC/playerctl" &&
     meson setup _build --prefix="$PREFIX" --libdir=lib -Dgtk-doc=false -Dintrospection=false \
         -Dbash-completions=false -Dzsh-completions=false &&
     meson compile -C _build && meson install -C _build ); then ok playerctl; else skip playerctl; fi

log pactl
# PulseAudio's client tools only (no sound server). Only pactl is kept.
if ( cd "$SRC/pulseaudio" &&
     { [ -f .tarball-version ] || echo 17.0 > .tarball-version; } &&
     meson setup _build --prefix="$PREFIX" --libdir=lib -Dauto_features=disabled \
         -Ddaemon=false -Dclient=true -Ddoxygen=false -Dman=false -Dtests=false \
         -Ddatabase=simple -Dbashcompletiondir=no -Dzshcompletiondir=no &&
     meson compile -C _build src/utils/pactl &&
     install -Dm755 _build/src/utils/pactl "$PREFIX/bin/pactl" &&
     cp -L _build/src/pulse/libpulse.so.0 _build/src/libpulsecommon-*.so "$PREFIX/lib/" &&
     "$PREFIX/bin/pactl" --version ); then ok pactl; else skip pactl; fi

echo; echo "Tools in the Flatpak:"; cat "$SUMMARY"
exit 0
