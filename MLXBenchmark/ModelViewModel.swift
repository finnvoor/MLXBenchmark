import Foundation
import Observation

@MainActor
@Observable class ModelViewModel {
    // MARK: Lifecycle

    init() {
        Task {
            await loadDownloadedModels()
        }
    }

    // MARK: Internal

    var availableModels: [MLXModel] = []
    var collections: [HuggingFaceCollection] = []
    var downloadedModels: [DownloadedModel] = []
    var isLoading = false
    var error: String?
    var selectedCollection: String?

    var groupedModels: [String: [MLXModel]] {
        var grouped: [String: [MLXModel]] = ["All Models": availableModels]

        // Group by each collection
        for collection in collections {
            let modelsInCollection = availableModels.filter { model in
                collection.modelIds.contains(model.id)
            }
            if !modelsInCollection.isEmpty {
                grouped[collection.displayName] = modelsInCollection
            }
        }

        return grouped
    }

    func fetchCollections() async {
        do {
            collections = try await service.fetchCollections()
            print("✅ Loaded \(collections.count) collections")
        } catch {
            print("❌ Failed to load collections: \(error.localizedDescription)")
            self.error = error.localizedDescription
        }
    }

    func fetchAvailableModels(collection: String? = nil) async {
        isLoading = true
        error = nil
        selectedCollection = collection

        print("🔍 ModelViewModel: Starting to fetch models...")

        // Load collections first if not loaded
        if collections.isEmpty {
            await fetchCollections()
        }

        do {
            availableModels = try await service.fetchModels(collection: collection)
            print("✅ ModelViewModel: Loaded \(availableModels.count) models")
        } catch {
            print("❌ ModelViewModel: Error - \(error.localizedDescription)")
            self.error = error.localizedDescription
        }

        isLoading = false
    }

    func downloadModel(_ model: MLXModel) {
        let downloadedModel = DownloadedModel(
            id: model.id,
            model: model,
            localPath: URL(fileURLWithPath: ""),
            status: .downloading(progress: 0)
        )
        downloadedModels.append(downloadedModel)

        let task = Task {
            do {
                print("📥 Starting download task for \(model.id)")

                let path = try await modelManager.downloadModel(model) { progress in
                    Task { @MainActor in
                        let fractionCompleted = progress.fractionCompleted
                        if fractionCompleted == 1.0 {
                            print("📊 Download progress for \(model.id): 100% (complete)")
                        } else if Int(fractionCompleted * 100) % 10 == 0 {
                            print("📊 Download progress for \(model.id): \(Int(fractionCompleted * 100))%")
                        }
                        downloadedModel.status = .downloading(progress: fractionCompleted)
                    }
                }

                // Ensure final progress update is processed
                try? await Task.sleep(for: .milliseconds(100))

                print("✅ Download completed, updating status for \(model.id)")
                downloadedModel.status = .installed
                downloadedModel.localPath = path

            } catch is CancellationError {
                print("🚫 Download cancelled for \(model.id)")
                downloadedModels.removeAll { $0.id == model.id }
            } catch {
                print("❌ Download error for \(model.id): \(error.localizedDescription)")
                downloadedModel.status = .error(error.localizedDescription)
            }

            downloadTasks.removeValue(forKey: model.id)
        }

        downloadTasks[model.id] = task
    }

    func cancelDownload(_ model: DownloadedModel) {
        downloadTasks[model.id]?.cancel()
        downloadTasks.removeValue(forKey: model.id)
        downloadedModels.removeAll { $0.id == model.id }
    }

    func deleteModel(_ model: DownloadedModel) {
        do {
            try modelManager.deleteModel(at: model.localPath)
            downloadedModels.removeAll { $0.id == model.id }
        } catch {
            self.error = error.localizedDescription
        }
    }

    func getStatus(for model: MLXModel) -> ModelStatus {
        if let downloaded = downloadedModels.first(where: { $0.model.id == model.id }) {
            return downloaded.status
        }
        return .available
    }

    // MARK: Private

    private let service = HuggingFaceService.shared
    private let modelManager = ModelManager.shared
    private var downloadTasks: [String: Task<Void, Never>] = [:]

    /// Map models to collections
    private var modelToCollections: [String: [String]] = [:]

    private func loadDownloadedModels() async {
        print("📂 Scanning filesystem for downloaded models")

        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelsPath = documents.appendingPathComponent("huggingface/models")

        guard FileManager.default.fileExists(atPath: modelsPath.path) else {
            print("⚠️ Models directory doesn't exist yet: \(modelsPath.path)")
            return
        }

        do {
            // Get all organization directories (e.g., "mlx-community")
            let orgDirs = try FileManager.default.contentsOfDirectory(
                at: modelsPath,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ).filter { url in
                (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            }

            print("📦 Found \(orgDirs.count) organization directories")

            for orgDir in orgDirs {
                let orgName = orgDir.lastPathComponent

                // Get all model directories within the organization
                let modelDirs = try FileManager.default.contentsOfDirectory(
                    at: orgDir,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: [.skipsHiddenFiles]
                ).filter { url in
                    (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
                }

                for modelDir in modelDirs {
                    let modelName = modelDir.lastPathComponent
                    let modelId = "\(orgName)/\(modelName)"

                    // Check if this directory contains model files (config.json is a good indicator)
                    let configPath = modelDir.appendingPathComponent("config.json")
                    guard FileManager.default.fileExists(atPath: configPath.path) else {
                        print("⚠️ Skipping \(modelId) - no config.json found")
                        continue
                    }

                    print("✅ Found installed model: \(modelId)")

                    // Create a model object for the downloaded model
                    let model = MLXModel(
                        id: modelId,
                        name: modelId,
                        author: orgName,
                        downloads: 0,
                        likes: 0,
                        lastModified: Date(),
                        size: calculateDirectorySize(modelDir),
                        collections: nil,
                        tags: nil
                    )

                    let downloadedModel = DownloadedModel(
                        id: modelId,
                        model: model,
                        localPath: modelDir,
                        status: .installed
                    )

                    downloadedModels.append(downloadedModel)
                }
            }

            print("✅ Loaded \(downloadedModels.count) downloaded models from filesystem")

        } catch {
            print("❌ Error scanning models directory: \(error)")
        }
    }

    private func calculateDirectorySize(_ url: URL) -> Int64? {
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }

        var totalSize: Int64 = 0
        for case let fileURL as URL in enumerator {
            guard let resourceValues = try? fileURL.resourceValues(forKeys: [.fileSizeKey]),
                  let fileSize = resourceValues.fileSize else {
                continue
            }
            totalSize += Int64(fileSize)
        }

        return totalSize
    }
}
