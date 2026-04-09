SCHEME = SwiftTuberia-Package
DESTINATION = platform=macOS,arch=arm64
SWIFT_FORMAT = xcrun swift-format

.PHONY: build test lint clean resolve

build:
	xcodebuild build -scheme $(SCHEME) -destination '$(DESTINATION)'

test:
	xcodebuild test -scheme $(SCHEME) -destination '$(DESTINATION)'

lint:
	$(SWIFT_FORMAT) -i -r .

clean:
	xcodebuild clean -scheme $(SCHEME)

resolve:
	xcodebuild -resolvePackageDependencies -scheme $(SCHEME)
