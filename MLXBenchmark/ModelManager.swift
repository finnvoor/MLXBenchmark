import Foundation
import Hub
import MLXLLM
import MLXLMCommon
import Tokenizers

@MainActor class ModelManager: @unchecked Sendable {
    // MARK: Lifecycle

    init() {
        // Use documents directory for model storage
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelsPath = documents.appendingPathComponent("huggingface")
        hub = HubApi(downloadBase: modelsPath)
    }

    // MARK: Internal

    static let shared = ModelManager()

    func downloadModel(
        _ model: MLXModel,
        progressHandler: @escaping @Sendable (Progress) -> Void
    ) async throws -> URL {
        let repo = Hub.Repo(id: model.id)

        print("🔽 Starting download for \(model.id)")

        // Download model using Hub
        let modelPath = try await hub.snapshot(
            from: repo,
            matching: ["*.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "*.model"]
        ) { progress in
            progressHandler(progress)
        }

        print("✅ Download complete for \(model.id) at \(modelPath.path)")

        return modelPath
    }

    func loadModel(modelId: String, from path: URL) async throws -> ModelContainer {
        // Use LLMModelFactory to load the model
        let modelFactory = LLMModelFactory.shared

        print("🔧 Loading model: \(modelId) from \(path.path)")

        // Create a ModelConfiguration with the actual model ID
        let modelConfig = ModelConfiguration(id: modelId)

        // Load the model container
        let container = try await modelFactory.loadContainer(
            hub: hub,
            configuration: modelConfig
        )

        print("✅ Model loaded successfully: \(modelId)")

        return container
    }

    func deleteModel(at path: URL) throws {
        try FileManager.default.removeItem(at: path)
    }

    // MARK: Private

    private let hub: HubApi
}
