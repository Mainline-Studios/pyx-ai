#!/usr/bin/env bash
# Pyx 1.5 — macOS installer builder.
# Produces:
#   dist/Pyx.app       (the bundle)
#   dist/Pyx-<ver>.dmg (drag-to-Applications disk image)
#   dist/Pyx-<ver>.pkg (standard macOS installer, optional)
#
# Requirements: python3, pyinstaller, create-dmg (brew install create-dmg),
#               and (optional) pkgbuild+productbuild which ship with Xcode CLT.
#
# Run from repo root:   bash packaging/build-macos.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="${PYX_VERSION:-1.5.0}"
APP_NAME="Pyx"
BUNDLE_ID="studio.mainline.pyx"
DIST="$ROOT/dist"
BUILD="$ROOT/build"

echo "==> cleaning"
rm -rf "$DIST" "$BUILD"

echo "==> ensuring venv + deps"
if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r requirements.txt
python -m pip install -r packaging/requirements-desktop.txt
python -m pip install "pyinstaller>=6.3"

echo "==> PyInstaller"
pyinstaller packaging/pyx.spec --noconfirm --log-level=WARN

echo "==> wrapping as .app"
APP_DIR="$DIST/${APP_NAME}.app"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"

# Move the one-folder bundle into Contents/Resources/app
mv "$DIST/Pyx" "$APP_DIR/Contents/Resources/app"

# Launcher shim that just runs the bundled Pyx binary
cat > "$APP_DIR/Contents/MacOS/Pyx" <<'EOF'
#!/bin/sh
DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "$DIR/Resources/app/Pyx" "$@"
EOF
chmod +x "$APP_DIR/Contents/MacOS/Pyx"

cat > "$APP_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>${APP_NAME}</string>
  <key>CFBundleDisplayName</key><string>${APP_NAME} 1.5</string>
  <key>CFBundleIdentifier</key><string>${BUNDLE_ID}</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundleExecutable</key><string>Pyx</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF

echo "==> .app ready: $APP_DIR"

DMG="$DIST/${APP_NAME}-${VERSION}.dmg"
if command -v create-dmg >/dev/null 2>&1; then
  echo "==> building .dmg (create-dmg)"
  create-dmg \
    --volname "${APP_NAME} ${VERSION}" \
    --window-size 540 360 \
    --icon-size 96 \
    --icon "${APP_NAME}.app" 140 180 \
    --app-drop-link 400 180 \
    "$DMG" "$APP_DIR" || true
else
  echo "==> building .dmg (hdiutil fallback)"
  hdiutil create -volname "${APP_NAME} ${VERSION}" \
    -srcfolder "$APP_DIR" -ov -format UDZO "$DMG"
fi
echo "==> .dmg ready: $DMG"

if command -v pkgbuild >/dev/null 2>&1 && command -v productbuild >/dev/null 2>&1; then
  echo "==> building .pkg"
  COMPONENT="$BUILD/${APP_NAME}-component.pkg"
  mkdir -p "$BUILD"
  pkgbuild \
    --identifier "$BUNDLE_ID" \
    --version "$VERSION" \
    --install-location "/Applications" \
    --component "$APP_DIR" \
    "$COMPONENT"
  PKG="$DIST/${APP_NAME}-${VERSION}.pkg"
  productbuild --package "$COMPONENT" "$PKG"
  echo "==> .pkg ready: $PKG"
else
  echo "==> pkgbuild/productbuild not found (install Xcode CLT). Skipping .pkg."
fi

echo
echo "All artifacts in: $DIST"
ls -lh "$DIST" | sed 's/^/  /'
