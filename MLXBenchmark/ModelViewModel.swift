import Foundation
import Observation
import os

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
            AppLogger.models.debug("Loaded \(self.collections.count) collections")
        } catch {
            AppLogger.models.error("Failed to load collections: \(error.localizedDescription, privacy: .public)")
            self.error = error.localizedDescription
        }
    }

    func fetchAvailableModels(collection: String? = nil) async {
        isLoading = true
        error = nil
        selectedCollection = collection

        AppLogger.models.debug("Fetching available models for collection \(collection ?? "All", privacy: .public)")

        // Load collections first if not loaded
        if collections.isEmpty {
            await fetchCollections()
        }

        do {
            availableModels = try await service.fetchModels(collection: collection)
            AppLogger.models.debug("Loaded \(self.availableModels.count) models")
        } catch {
            AppLogger.models.error("Error fetching models: \(error.localizedDescription, privacy: .public)")
            self.error = error.localizedDescription
        }

        isLoading = false
    }

    func downloadModel(_ model: MLXModel) {
        if let existing = downloadedModels.first(where: { $0.id == model.id }) {
            switch existing.status {
            case .installed, .downloading:
                AppLogger.models.debug("Ignoring download request for \(model.id, privacy: .public) – already handled")
                return
            case .error:
                existing.status = .downloading(progress: 0)
                startDownloadTask(for: existing)
                return
            default:
                break
            }
        }

        let downloadedModel = DownloadedModel(
            id: model.id,
            model: model,
            localPath: URL(fileURLWithPath: ""),
            status: .downloading(progress: 0)
        )
        downloadedModels.append(downloadedModel)

        startDownloadTask(for: downloadedModel)
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
        AppLogger.models.debug("Scanning filesystem for downloaded models")

        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelsPath = documents.appendingPathComponent("huggingface/models")

        guard FileManager.default.fileExists(atPath: modelsPath.path) else {
            AppLogger.models.debug("Models directory doesn't exist yet: \(modelsPath.path, privacy: .public)")
            return
        }

        downloadedModels.removeAll()

        do {
            // Get all organization directories (e.g., "mlx-community")
            let orgDirs = try FileManager.default.contentsOfDirectory(
                at: modelsPath,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ).filter { url in
                (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            }

            AppLogger.models.debug("Found \(orgDirs.count) organization directories")

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
                        AppLogger.models.debug("Skipping \(modelId, privacy: .public) – missing config.json")
                        continue
                    }

                    AppLogger.models.debug("Found installed model: \(modelId, privacy: .public)")

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

                    self.downloadedModels.append(downloadedModel)
                }
            }

            AppLogger.models.debug("Loaded \(self.downloadedModels.count) downloaded models from filesystem")

        } catch {
            AppLogger.models.error("Error scanning models directory: \(error.localizedDescription, privacy: .public)")
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

    private func startDownloadTask(for downloadedModel: DownloadedModel) {
        let model = downloadedModel.model

        AppLogger.models.debug("Starting download task for \(model.id, privacy: .public)")

        let task = Task {
            do {
                let path = try await modelManager.downloadModel(model) { progress in
                    Task { @MainActor in
                        downloadedModel.status = .downloading(progress: progress.fractionCompleted)
                    }
                }

                try? await Task.sleep(for: .milliseconds(100))

                await MainActor.run {
                    downloadedModel.status = .installed
                    downloadedModel.localPath = path
                }
            } catch is CancellationError {
                await MainActor.run {
                    AppLogger.models.debug("Download cancelled for \(model.id, privacy: .public)")
                    self.downloadedModels.removeAll { $0.id == model.id }
                }
            } catch {
                await MainActor.run {
                    AppLogger.models.error("Download failed for \(model.id, privacy: .public): \(error.localizedDescription, privacy: .public)")
                    downloadedModel.status = .error(error.localizedDescription)
                }
            }

            await MainActor.run {
                self.downloadTasks.removeValue(forKey: model.id)
            }
        }

        downloadTasks[model.id] = task
    }
}
