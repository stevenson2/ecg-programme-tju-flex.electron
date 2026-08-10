allprojects {
    repositories {
        google()
        mavenCentral()
        // 2026-08-10 修复: Flutter 引擎 Maven artifacts (io.flutter:arm64_v8a_release
        // 等 libflutter.so 引擎库) 位于 download.flutter.io 仓库。本环境 Flutter
        // 插件未注入该仓库 (FLUTTER_STORAGE_BASE_URL 指向清华镜像且无此映射),
        // 导致 assembleRelease 报 "Could not find io.flutter:*" (引擎缓存缺
        // libflutter.so)。补充腾讯镜像映射 (已验证 200)。
        maven { url = uri("https://mirrors.cloud.tencent.com/flutter/download.flutter.io") }
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
