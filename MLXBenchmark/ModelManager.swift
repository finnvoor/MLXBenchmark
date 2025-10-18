import Foundation
import Hub
import MLXLLM
import MLXLMCommon
import os

@MainActor class ModelManager: @unchecked Sendable {
    // MARK: Lifecycle

    init(fileManager: FileManager = .default) {
        self.fileManager = fileManager
        let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.temporaryDirectory

        modelsDirectory = documentsDirectory.appendingPathComponent("huggingface")
        try? fileManager.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        hub = HubApi(downloadBase: modelsDirectory)
    }

    // MARK: Internal

    static let shared = ModelManager()

    func downloadModel(
        _ model: MLXModel,
        progressHandler: @escaping @Sendable (Progress) -> Void
    ) async throws -> URL {
        let repo = Hub.Repo(id: model.id)

        AppLogger.models.debug("Starting download for \(model.id, privacy: .public)")

        let modelPath = try await hub.snapshot(
            from: repo,
            matching: ["*.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "*.model"]
        ) { progress in
            progressHandler(progress)
        }

        AppLogger.models.debug("Download finished for \(model.id, privacy: .public) at \(modelPath.path, privacy: .public)")

        return modelPath
    }

    func loadModel(modelId: String, from path: URL) async throws -> ModelContainer {
        let modelFactory = LLMModelFactory.shared

        AppLogger.models.debug("Loading model \(modelId, privacy: .public) from \(path.path, privacy: .public)")

        let modelConfig = ModelConfiguration(id: modelId)
        let container = try await modelFactory.loadContainer(
            hub: hub,
            configuration: modelConfig
        )

        AppLogger.models.debug("Model loaded successfully: \(modelId, privacy: .public)")

        return container
    }

    func deleteModel(at path: URL) throws {
        try fileManager.removeItem(at: path)
    }

    // MARK: Private

    private let fileManager: FileManager
    private let modelsDirectory: URL
    private let hub: HubApi
}
