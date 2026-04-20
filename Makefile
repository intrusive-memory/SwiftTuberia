SCHEME = SwiftTuberia-Package
DESTINATION = platform=macOS,arch=arm64
SWIFT_FORMAT = xcrun swift-format

# MLX owns a process-global Metal GPU stream. Swift Testing parallelizes across
# suites by default, which races on the shared command buffer and trips
# `-[_MTLCommandBuffer addCompletedHandler:] 'Completed handler provided after
# commit call'`. Suite-level `.serialized` traits do not cross suite boundaries,
# so we serialize the whole test run.
TEST_FLAGS = -parallel-testing-enabled NO

.PHONY: build test lint clean resolve

build:
	xcodebuild build -scheme $(SCHEME) -destination '$(DESTINATION)'

test:
	xcodebuild test -scheme $(SCHEME) -destination '$(DESTINATION)' $(TEST_FLAGS)

lint:
	$(SWIFT_FORMAT) -i -r .

clean:
	xcodebuild clean -scheme $(SCHEME)

resolve:
	xcodebuild -resolvePackageDependencies -scheme $(SCHEME)
