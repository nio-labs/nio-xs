const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 1. Standalone Companion Engine Utility: nioxs-core
    const exe_core = b.addExecutable(.{
        .name = "nioxs-core",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe_core);

    // 2. Static C ABI Library for Rust Hybrid Integration: libnioxs_core.a
    const lib = b.addStaticLibrary(.{
        .name = "nioxs_core",
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // Run step
    const run_cmd = b.addRunArtifact(exe_core);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the nioxs-core native engine");
    run_step.dependOn(&run_cmd.step);
}
