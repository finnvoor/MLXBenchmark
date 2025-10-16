import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Observation
import Tokenizers

@MainActor
@Observable class ChatViewModel {
    // MARK: Lifecycle

    init(model: DownloadedModel) {
        downloadedModel = model
        Task {
            await loadModel()
        }
    }

    // MARK: Internal

    var messages: [ChatMessage] = []
    var currentMessage = ""
    var isGenerating = false
    var metrics = BenchmarkMetrics()
    var isLoading = true
    var loadError: String?

    let downloadedModel: DownloadedModel

    func sendMessage() {
        guard !currentMessage.isEmpty else { return }

        let userMessage = ChatMessage(role: .user, content: currentMessage)
        messages.append(userMessage)

        let prompt = currentMessage
        currentMessage = ""

        generationTask = Task {
            await generateResponse(for: prompt)
        }
    }

    func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
        print("🛑 Generation stopped by user")
    }

    func clearChat() {
        messages.removeAll()
        metrics = BenchmarkMetrics()
    }

    // MARK: Private

    private var modelContainer: ModelContainer?
    private var generationTask: Task<Void, Never>?

    private func loadModel() async {
        isLoading = true
        loadError = nil

        do {
            let container = try await ModelManager.shared.loadModel(
                modelId: downloadedModel.model.id,
                from: downloadedModel.localPath
            )
            modelContainer = container

            isLoading = false
        } catch {
            loadError = "Failed to load model: \(error.localizedDescription)"
            isLoading = false
        }
    }

    private func generateResponse(for prompt: String) async {
        guard let modelContainer else {
            let errorMessage = ChatMessage(role: .assistant, content: "Model not loaded")
            messages.append(errorMessage)
            return
        }

        isGenerating = true
        let startTime = Date()

        do {
            // Create assistant message placeholder
            let assistantMessage = ChatMessage(role: .assistant, content: "")
            messages.append(assistantMessage)
            let messageIndex = messages.count - 1

            print("🤖 Starting generation for prompt: \(prompt.prefix(50))...")

            // Build chat history for context
            var chatHistory: [Chat.Message] = []
            for msg in messages.dropLast() { // Exclude the placeholder we just added
                switch msg.role {
                case .user:
                    chatHistory.append(.user(msg.content))
                case .assistant:
                    chatHistory.append(.assistant(msg.content))
                case .system:
                    chatHistory.append(.system(msg.content))
                }
            }

            // Add current prompt
            chatHistory.append(.user(prompt))

            let userInput = UserInput(chat: chatHistory)

            // Generate with MLXLMCommon
            let (promptTokens, completionTokens, _) = try await modelContainer.perform { (context: ModelContext) in
                // Prepare input with proper chat template
                let lmInput = try await context.processor.prepare(input: userInput)
                let promptTokenCount = lmInput.text.tokens.count

                // Create generation parameters
                let generateParameters = GenerateParameters(
                    temperature: 0.8,
                    topP: 0.95,
                    repetitionPenalty: 1.2,
                    repetitionContextSize: 50
                )

                // Generate tokens
                let stream = try MLXLMCommon.generate(
                    input: lmInput,
                    parameters: generateParameters,
                    context: context
                )

                // Process stream with manual throttling
                var lastUpdateTime = Date()
                var batchedTokens: [Generation] = []
                var isComplete = false
                var firstTokenTime: TimeInterval?
                var localGeneratedText = ""
                var localCompletionTokens = 0

                for await token in stream {
                    // Check for cancellation
                    if Task.isCancelled {
                        print("⚠️ Generation cancelled")
                        break
                    }

                    batchedTokens.append(token)

                    // Check if we got completion info (signals end)
                    if case .info = token {
                        isComplete = true
                    }

                    let now = Date()
                    let shouldUpdate = now.timeIntervalSince(lastUpdateTime) >= 0.1 || isComplete

                    if shouldUpdate, !batchedTokens.isEmpty {
                        // Extract text chunks
                        let output = batchedTokens.compactMap(\.chunk).joined(separator: "")
                        if !output.isEmpty {
                            if firstTokenTime == nil {
                                let ttft = Date().timeIntervalSince(startTime)
                                firstTokenTime = ttft
                                await MainActor.run {
                                    self.metrics.timeToFirstToken = ttft
                                }
                                print("⏱️ Time to first token: \(String(format: "%.2f", ttft))s")
                            }

                            localGeneratedText += output
                            localCompletionTokens += batchedTokens.filter { $0.chunk != nil }.count

                            // Update UI on main actor with a copy
                            let textCopy = localGeneratedText
                            await MainActor.run {
                                let updatedMessage = ChatMessage(role: .assistant, content: textCopy)
                                self.messages[messageIndex] = updatedMessage
                            }
                        }

                        // Extract stats
                        if let completion = batchedTokens.compactMap(\.info).first {
                            await MainActor.run {
                                self.metrics.tokensPerSecond = completion.tokensPerSecond
                            }
                        }

                        lastUpdateTime = now
                        batchedTokens.removeAll()
                    }

                    // Stop on completion
                    if isComplete {
                        print("✅ Generation complete")
                        break
                    }
                }

                // Return final counts
                return (promptTokenCount, localCompletionTokens, localGeneratedText)
            }

            let endTime = Date()

            // Update final metrics
            metrics.promptTokens = promptTokens
            metrics.completionTokens = completionTokens
            metrics.totalTime = endTime.timeIntervalSince(startTime)
            metrics.memoryUsage = getMemoryUsage()

            print("✅ Generation complete: \(completionTokens) tokens in \(String(format: "%.2f", metrics.totalTime))s (\(String(format: "%.1f", metrics.tokensPerSecond)) tok/s)")

        } catch {
            let errorMessage = ChatMessage(role: .assistant, content: "Error: \(error.localizedDescription)")
            let messageIndex = messages.count - 1
            if messageIndex >= 0, messageIndex < messages.count {
                messages[messageIndex] = errorMessage
            } else {
                messages.append(errorMessage)
            }
            print("❌ Generation error: \(error.localizedDescription)")
        }

        isGenerating = false
    }

    private func getMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        return result == KERN_SUCCESS ? info.resident_size : 0
    }
}
