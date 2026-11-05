# EchoCollector v2 — ProGuard rules (release builds).
# Minification is disabled in build.gradle.kts; these rules are a no-op
# placeholder kept so the release build type resolves.
-keep class com.echo.collector.** { *; }
-keep class com.google.ar.core.** { *; }
