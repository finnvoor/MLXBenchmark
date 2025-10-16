import Foundation
import Hub

// MARK: - CollectionResponse

struct CollectionResponse: Codable {
    struct CollectionItem: Codable {
        let id: String
        let type: String
    }

    let slug: String
    let title: String
    let description: String?
    let items: [CollectionItem]?
}

// MARK: - HuggingFaceService

@MainActor class HuggingFaceService {
    // MARK: Internal

    static let shared = HuggingFaceService()

    func fetchCollections() async throws -> [HuggingFaceCollection] {
        var allCollections: [HuggingFaceCollection] = []
        var offset = 0
        let limit = 100

        while true {
            let urlString = "\(baseURL)/collections?owner=\(mlxCommunity)&limit=\(limit)&offset=\(offset)"
            guard let url = URL(string: urlString) else {
                throw URLError(.badURL)
            }

            print("📡 Fetching collections from offset \(offset)")

            let (data, _) = try await URLSession.shared.data(from: url)
            let responses = try JSONDecoder().decode([CollectionResponse].self, from: data)

            if responses.isEmpty {
                break
            }

            let collections = responses.map { response in
                HuggingFaceCollection(
                    slug: response.slug,
                    title: response.title,
                    description: response.description,
                    modelIds: response.items?.filter { $0.type == "model" }.map(\.id) ?? []
                )
            }

            allCollections.append(contentsOf: collections)

            print("✅ Fetched \(collections.count) collections (total: \(allCollections.count))")

            if responses.count < limit {
                break
            }

            offset += limit
        }

        return allCollections
    }

    func fetchModels(collection: String? = nil) async throws -> [MLXModel] {
        var allModels: [MLXModel] = []
        var seenModelIds = Set<String>()
        var nextCursor: String? = nil
        let limit = 500 // Max limit supported by API

        // If filtering by collection, we need to get the collection first
        var collectionModelIds: Set<String>?
        if let collectionSlug = collection {
            let collections = try await fetchCollections()
            if let targetCollection = collections.first(where: { $0.displayName == collectionSlug }) {
                collectionModelIds = Set(targetCollection.modelIds)
            }
        }

        // Fetch all models with cursor-based pagination
        var pageCount = 0
        repeat {
            pageCount += 1
            var urlString = "\(baseURL)/models?author=\(mlxCommunity)&sort=downloads&direction=-1&limit=\(limit)&full=true"

            if let cursor = nextCursor {
                urlString += "&cursor=\(cursor)"
            }

            guard let url = URL(string: urlString) else {
                throw URLError(.badURL)
            }

            print("📡 Fetching models page \(pageCount) (cursor: \(nextCursor != nil ? "yes" : "no"))")

            let (data, response) = try await URLSession.shared.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw URLError(.badServerResponse)
            }

            print("📊 HTTP Status: \(httpResponse.statusCode)")

            // Extract next cursor from Link header
            nextCursor = nil
            if let linkHeader = httpResponse.value(forHTTPHeaderField: "link") {
                // Parse Link header to extract next cursor
                // Format: <url?cursor=XXX>; rel="next"
                if let nextMatch = linkHeader.range(of: "cursor=([^&>]+)", options: .regularExpression) {
                    let cursorString = String(linkHeader[nextMatch])
                    if let cursorValue = cursorString.split(separator: "=").last {
                        nextCursor = String(cursorValue)
                        print("📄 Found next cursor")
                    }
                }
            }

            // Try to decode
            let decoder = JSONDecoder()
            let responses: [HuggingFaceResponse]

            do {
                responses = try decoder.decode([HuggingFaceResponse].self, from: data)
                print("✅ Successfully decoded \(responses.count) models on page \(pageCount)")
            } catch {
                print("❌ Decoding error: \(error)")
                if let jsonString = String(data: data.prefix(500), encoding: .utf8) {
                    print("📄 Response preview: \(jsonString)")
                }
                throw error
            }

            if responses.isEmpty {
                break
            }

            let models = responses.compactMap { response -> MLXModel? in
                let dateFormatter = ISO8601DateFormatter()
                dateFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                guard let date = dateFormatter.date(from: response.lastModified) else {
                    return nil
                }

                let modelId = response.modelId ?? response.id

                // Skip if we've already seen this model
                guard !seenModelIds.contains(modelId) else {
                    return nil
                }

                seenModelIds.insert(modelId)

                return MLXModel(
                    id: modelId,
                    name: modelId,
                    author: response.author,
                    downloads: response.downloads,
                    likes: response.likes,
                    lastModified: date,
                    size: nil,
                    collections: nil,
                    tags: response.tags
                )
            }

            // Filter to only text generation models
            let textGenModels = models.filter(\.isTextGeneration)

            allModels.append(contentsOf: textGenModels)

            print("✅ Total unique models: \(allModels.count) (filtered \(models.count - textGenModels.count) non-text-gen models)")

            // Safety limit
            if pageCount >= 10 {
                print("⚠️ Reached page limit")
                break
            }

        } while nextCursor != nil

        // Filter by collection if needed
        if let modelIds = collectionModelIds {
            allModels = allModels.filter { modelIds.contains($0.id) }
        }

        print("✅ Returning \(allModels.count) models")
        return allModels
    }

    // MARK: Private

    private let baseURL = "https://huggingface.co/api"
    private let mlxCommunity = "mlx-community"
}
