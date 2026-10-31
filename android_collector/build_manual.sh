#!/usr/bin/env bash
# Manual (Gradle-free) build of EchoCollector.apk.
#
# Why this exists: on the build VM, Gradle's daemon client<->daemon
# socket handshake fails (the client reads a garbage type tag off the
# loopback socket no matter the configuration; plain loopback works
# fine for other runtimes). The Gradle project in this folder is the
# canonical build for a normal machine (`gradle :app:assembleDebug`);
# this script reproduces the same artefact with the raw tools so the
# APK can be produced here, and it runs the JVM tests + the desktop
# harness proof along the way.
#
# Toolchain expected under ~/workspace/android-tools (see README):
#   jdk/ (Temurin 17), sdk/ (platforms/android-34, build-tools/34.0.0),
#   kotlinc/ (Kotlin 2.0.20), jars/ (junit, hamcrest, annotation,
#   core-1.44.0.aar), arcore/ (the AAR, extracted).
set -euo pipefail
cd "$(dirname "$0")"

AT="${ANDROID_TOOLS:-$HOME/workspace/android-tools}"
export JAVA_HOME="$AT/jdk"
export PATH="$JAVA_HOME/bin:$PATH"
export TMPDIR="${TMPDIR:-$HOME/workspace/.tmp}"
KOTLINC="$AT/kotlinc/bin/kotlinc"
ANDROID_JAR="$AT/sdk/platforms/android-34/android.jar"
BT="$AT/sdk/build-tools/34.0.0"
B=build/manual
SIM="${1:-../experiments/sensor_sim/capture_room2_sim}"
mkdir -p "$B"

echo "== 1/7 shared module =="
"$KOTLINC" shared/src/main/kotlin/echo/shared/*.kt -d "$B/shared.jar"

echo "== 2/7 shared unit tests =="
"$KOTLINC" shared/src/test/kotlin/echo/shared/*.kt \
    -cp "$B/shared.jar:$AT/jars/junit-4.13.2.jar:$AT/jars/hamcrest-core-1.3.jar" \
    -d "$B/shared-tests.jar"
java -cp "$B/shared.jar:$B/shared-tests.jar:$AT/jars/junit-4.13.2.jar:$AT/jars/hamcrest-core-1.3.jar:$AT/kotlinc/lib/kotlin-stdlib.jar" \
    org.junit.runner.JUnitCore \
    echo.shared.ConvertTest echo.shared.StampClockTest \
    echo.shared.DepthDownsampleTest echo.shared.FormatTest

echo "== 3/7 desktop harness: synthetic capture folder =="
"$KOTLINC" harness/src/main/kotlin/echo/harness/*.kt \
    -cp "$B/shared.jar" -include-runtime -d "$B/harness.jar"
rm -rf "$B/harness_capture"
java -cp "$B/harness.jar:$B/shared.jar:$AT/kotlinc/lib/kotlin-stdlib.jar" \
    echo.harness.MainKt "$SIM" "$B/harness_capture"

echo "== 4/7 python contract validation =="
"$(dirname "$0")/../.venv/bin/python" validate_capture.py \
    "$B/harness_capture" --sim "$SIM"

echo "== 5/7 app compile =="
"$KOTLINC" app/src/main/kotlin/echo/collector/*.kt \
    -cp "$ANDROID_JAR:$AT/arcore/classes.jar:$AT/jars/annotation-1.3.0.jar:$B/shared.jar" \
    -d "$B/app-classes"

echo "== 6/7 resources, manifest merge, dex =="
mkdir -p "$B/apk"
"$BT/aapt2" compile --dir "$AT/arcore/res" -o "$B/apk/arcore-res.zip"
# AGP injects the package attribute from `namespace`; raw aapt2 needs it
# in the manifest itself, so link against a processed copy.
sed 's|<manifest xmlns:android="http://schemas.android.com/apk/res/android">|<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="echo.collector">|' \
    app/src/main/AndroidManifest.xml > "$B/apk/AndroidManifest.merged.xml"
"$BT/aapt2" link -o "$B/apk/base.apk" -I "$ANDROID_JAR" \
    --manifest "$B/apk/AndroidManifest.merged.xml" \
    --java "$B/gen" --extra-packages com.google.ar.core \
    --min-sdk-version 26 --target-sdk-version 34 \
    --version-code 1 --version-name 0.1 "$B/apk/arcore-res.zip"
mkdir -p "$B/rclasses" "$B/dex"
javac -cp "$ANDROID_JAR" -d "$B/rclasses" \
    "$B/gen/echo/collector/R.java" "$B/gen/com/google/ar/core/R.java"
jar cf "$B/app-classes.jar" -C "$B/app-classes" .
jar cf "$B/rclasses.jar" -C "$B/rclasses" .
"$BT/d8" --lib "$ANDROID_JAR" --min-api 26 --output "$B/dex" \
    "$B/app-classes.jar" "$B/rclasses.jar" "$B/shared.jar" \
    "$AT/arcore/classes.jar" "$AT/jars/annotation-1.3.0.jar" \
    "$AT/kotlinc/lib/kotlin-stdlib.jar"

echo "== 7/7 package, align, sign =="
cp "$B/apk/base.apk" "$B/apk/unsigned.apk"
mkdir -p "$B/apk/lib/arm64-v8a" "$B/apk/lib/armeabi-v7a"
cp "$AT/arcore/jni/arm64-v8a/"*.so "$B/apk/lib/arm64-v8a/"
cp "$AT/arcore/jni/armeabi-v7a/"*.so "$B/apk/lib/armeabi-v7a/"
cp "$B/dex/classes.dex" "$B/apk/classes.dex"
(cd "$B/apk" && zip -q unsigned.apk classes.dex \
    lib/arm64-v8a/*.so lib/armeabi-v7a/*.so)
"$BT/zipalign" -f 4 "$B/apk/unsigned.apk" "$B/apk/aligned.apk"
if [ ! -f "$B/apk/debug.keystore" ]; then
    keytool -genkeypair -keystore "$B/apk/debug.keystore" \
        -storepass android -keypass android -alias androiddebugkey \
        -keyalg RSA -keysize 2048 -validity 10957 \
        -dname "CN=Android Debug,O=Android,C=US"
fi
"$BT/apksigner" sign --ks "$B/apk/debug.keystore" \
    --ks-pass pass:android --key-pass pass:android \
    --out EchoCollector.apk "$B/apk/aligned.apk"
"$BT/apksigner" verify EchoCollector.apk
"$BT/aapt2" dump badging EchoCollector.apk | head -3
echo "BUILT: $(ls -la EchoCollector.apk)"
