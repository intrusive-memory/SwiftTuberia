// Tools/VerifyComponentManifest/main.swift
//
// VerifyComponentManifest — CDN manifest cross-check tool.
//
// Usage:
//   VerifyComponentManifest <manifest-path>
//
// Reads the manifest.json file at <manifest-path> and cross-checks every file
// entry against the registered ComponentDescriptors in TuberiaCatalog's
// CatalogRegistration. Exits 0 when every file's sha256 and sizeBytes match
// the source-of-truth values in CatalogRegistration.swift. Exits 1 on any
// divergence and prints a diff-style report to stderr naming the component,
// path, and expected vs actual values.
//
// Manifest JSON schema (produced by `acervo manifest create`):
//
//   {
//     "manifestVersion": 1,
//     "modelId": "org/repo",
//     "slug": "org_repo",
//     "updatedAt": "2026-04-20T00:00:00Z",
//     "manifestChecksum": "<sha256-of-sorted-sha256s>",
//     "files": [
//       {
//         "path": "config.json",
//         "sha256": "<64-char lowercase hex>",
//         "sizeBytes": 1234
//       }
//     ]
//   }
//
// The executable does NOT depend on network access or the Acervo download
// runtime. It reads CatalogRegistration directly via TuberiaCatalog's
// public API (`Acervo.registeredComponents()`).

import Foundation
import SwiftAcervo
import TuberiaCatalog

// MARK: - Manifest JSON structures
// Mirrors the CDNManifest / CDNManifestFile types defined in SwiftAcervo but
// decoded here independently so this tool never drifts with CDNManifest API
// changes (it only needs the fields it checks).

private struct ManifestFile: Decodable {
    let path: String
    let sha256: String
    let sizeBytes: Int64
}

private struct Manifest: Decodable {
    let manifestVersion: Int
    let modelId: String
    let slug: String
    let files: [ManifestFile]
}

// MARK: - Entry point

func main() -> Int32 {
    let args = CommandLine.arguments
    guard args.count == 2 else {
        fputs(
            """
            Usage: VerifyComponentManifest <manifest-path>

            Reads a manifest.json produced by `acervo manifest create` and verifies
            that every file entry's sha256 and sizeBytes match the values registered
            in TuberiaCatalog's CatalogRegistration.

            Exit codes:
              0  — All entries match source-of-truth descriptors
              1  — One or more mismatches found (details on stderr)
              2  — Usage error or manifest file cannot be read/decoded

            """,
            stderr
        )
        return 2
    }

    let manifestPath = args[1]
    let manifestURL = URL(fileURLWithPath: manifestPath)

    // Load manifest JSON from disk.
    let manifestData: Data
    do {
        manifestData = try Data(contentsOf: manifestURL)
    } catch {
        fputs("ERROR: Cannot read manifest at '\(manifestPath)': \(error.localizedDescription)\n", stderr)
        return 2
    }

    let manifest: Manifest
    do {
        manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
    } catch {
        fputs("ERROR: Cannot decode manifest JSON at '\(manifestPath)': \(error.localizedDescription)\n", stderr)
        return 2
    }

    // Ensure all catalog components are registered so Acervo.registeredComponents()
    // returns the full set from CatalogRegistration.swift.
    CatalogRegistration.shared.ensureRegistered()

    // Build a lookup: (componentId, relativePath) -> ComponentFile
    // Index all registered descriptors by their repo slug so we can match
    // manifest `slug` to the right component.
    let allDescriptors = Acervo.registeredComponents()

    // A manifest covers exactly one component (one org/repo). Find the
    // matching descriptor by slugifying repoId and comparing to manifest.slug.
    func slugify(_ repoId: String) -> String {
        repoId.replacingOccurrences(of: "/", with: "_")
    }

    let matchingDescriptors = allDescriptors.filter { slugify($0.repoId) == manifest.slug }
    guard !matchingDescriptors.isEmpty else {
        fputs(
            "ERROR: No registered component matches manifest slug '\(manifest.slug)'.\n"
            + "       Registered slugs: \(allDescriptors.map { slugify($0.repoId) }.joined(separator: ", "))\n",
            stderr
        )
        return 2
    }

    // Build file lookup from all matching descriptors.
    // relativePath -> ComponentFile
    var sourceOfTruth: [String: (componentId: String, file: ComponentFile)] = [:]
    for descriptor in matchingDescriptors {
        for componentFile in descriptor.files {
            sourceOfTruth[componentFile.relativePath] = (descriptor.id, componentFile)
        }
    }

    // Compare manifest entries against source-of-truth.
    var mismatches: [(path: String, componentId: String, field: String, expected: String, actual: String)] = []
    var unmatchedPaths: [String] = []

    for manifestFile in manifest.files {
        guard let entry = sourceOfTruth[manifestFile.path] else {
            // The manifest contains a path not in the source-of-truth. This is
            // informational (extra files are not a fatal mismatch) but we track it.
            unmatchedPaths.append(manifestFile.path)
            continue
        }

        let componentId = entry.componentId
        let sourceFile = entry.file

        // Check sha256.
        if let expectedSha = sourceFile.sha256, expectedSha != manifestFile.sha256 {
            mismatches.append((
                path: manifestFile.path,
                componentId: componentId,
                field: "sha256",
                expected: expectedSha,
                actual: manifestFile.sha256
            ))
        }

        // Check sizeBytes.
        if let expectedSize = sourceFile.expectedSizeBytes,
           expectedSize != manifestFile.sizeBytes {
            mismatches.append((
                path: manifestFile.path,
                componentId: componentId,
                field: "sizeBytes",
                expected: "\(expectedSize)",
                actual: "\(manifestFile.sizeBytes)"
            ))
        }
    }

    // Check for source-of-truth files absent from the manifest.
    let manifestPaths = Set(manifest.files.map(\.path))
    for (path, entry) in sourceOfTruth {
        if !manifestPaths.contains(path) {
            mismatches.append((
                path: path,
                componentId: entry.componentId,
                field: "presence",
                expected: "present in manifest",
                actual: "MISSING from manifest"
            ))
        }
    }

    if mismatches.isEmpty && unmatchedPaths.isEmpty {
        // Full match — print a brief confirmation to stdout.
        print("OK: All \(manifest.files.count) files in '\(manifest.slug)' match source-of-truth descriptors.")
        return 0
    }

    // Report mismatches to stderr in a diff-style format.
    fputs("MISMATCH: '\(manifest.slug)' manifest diverges from CatalogRegistration.swift\n", stderr)
    fputs(String(repeating: "-", count: 72) + "\n", stderr)

    for m in mismatches {
        fputs("  component : \(m.componentId)\n", stderr)
        fputs("  path      : \(m.path)\n", stderr)
        fputs("  field     : \(m.field)\n", stderr)
        fputs("- expected  : \(m.expected)\n", stderr)
        fputs("+ actual    : \(m.actual)\n", stderr)
        fputs(String(repeating: "-", count: 72) + "\n", stderr)
    }

    if !unmatchedPaths.isEmpty {
        fputs("NOTE: \(unmatchedPaths.count) manifest path(s) have no source-of-truth entry (extra files):\n", stderr)
        for p in unmatchedPaths {
            fputs("  + \(p)\n", stderr)
        }
    }

    fputs("\(mismatches.count) mismatch(es) detected. Failing.\n", stderr)
    return 1
}

exit(main())
