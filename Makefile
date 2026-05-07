SCHEME = SwiftTuberia-Package
DESTINATION = platform=macOS,arch=arm64
SWIFT_FORMAT = xcrun swift-format

# MLX owns a process-global Metal GPU stream. Swift Testing parallelizes across
# suites by default, which races on the shared command buffer and trips
# `-[_MTLCommandBuffer addCompletedHandler:] 'Completed handler provided after
# commit call'`. Suite-level `.serialized` traits do not cross suite boundaries,
# so we serialize the whole test run.
TEST_FLAGS = -parallel-testing-enabled NO

.PHONY: build test test-telemetry-overhead lint clean resolve

build:
	xcodebuild build -scheme $(SCHEME) -destination '$(DESTINATION)'

test:
	xcodebuild test -scheme $(SCHEME) -destination '$(DESTINATION)' $(TEST_FLAGS)

# Manually invoked before releases to verify NoopTuberiaTelemetryReporter adds <10%
# wall-clock overhead vs nil reporter over 10 alternating trials.
# Intentionally excluded from `make test` because perf tests are noisy on
# parallel-scheduled CI and a flaky guard would block releases on noise.
# Gate: SWIFT_ACTIVE_COMPILATION_CONDITIONS=TUBERIA_TELEMETRY_OVERHEAD_ENABLED is
# passed as an xcodebuild build setting, enabling the #if gate in
# TelemetryOverheadTests.swift. Without it, the suite compiles as an empty stub
# and Swift Testing discovers no test methods (invisible in `make test`).
# NOTE: macOS SPM test bundles do not propagate host env vars through xcodebuild
# (TEST_RUNNER_ prefix is ineffective for SPM targets on macOS); the build-setting
# approach is used instead.
test-telemetry-overhead:
	xcodebuild test -scheme $(SCHEME) -destination '$(DESTINATION)' \
		-parallel-testing-enabled NO \
		"SWIFT_ACTIVE_COMPILATION_CONDITIONS=TUBERIA_TELEMETRY_OVERHEAD_ENABLED" \
		-only-testing:TuberiaTests/TelemetryOverheadTests

lint:
	$(SWIFT_FORMAT) -i -r .

clean:
	xcodebuild clean -scheme $(SCHEME)

resolve:
	xcodebuild -resolvePackageDependencies -scheme $(SCHEME)
