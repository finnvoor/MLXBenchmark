import SwiftUI

// MARK: - ChatView

struct ChatView: View {
    // MARK: Lifecycle

    init(model: DownloadedModel) {
        self.model = model
        _viewModel = State(initialValue: ChatViewModel(model: model))
    }

    // MARK: Internal

    let model: DownloadedModel

    var body: some View {
        VStack(spacing: 0) {
            if viewModel.isLoading {
                VStack(spacing: 16) {
                    ProgressView()
                    Text("Loading model...")
                        .font(.headline)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let error = viewModel.loadError {
                ContentUnavailableView(
                    "Failed to Load Model",
                    systemImage: "exclamationmark.triangle",
                    description: Text(error)
                )
            } else {
                // Chat messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 16) {
                            ForEach(viewModel.messages) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }
                        }
                        .padding()
                    }
                    .onChange(of: viewModel.messages.count) { _, _ in
                        if let lastMessage = viewModel.messages.last {
                            withAnimation {
                                proxy.scrollTo(lastMessage.id, anchor: .bottom)
                            }
                        }
                    }
                }

                Divider()

                // Metrics panel
                MetricsView(metrics: viewModel.metrics)
                    .padding()
                    .background(.ultraThinMaterial)

                Divider()

                // Input area
                HStack(spacing: 12) {
                    TextField("Message", text: $viewModel.currentMessage, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .focused($isInputFocused)
                        .lineLimit(1...5)
                        .onSubmit {
                            viewModel.sendMessage()
                        }
                        .disabled(viewModel.isGenerating)

                    if viewModel.isGenerating {
                        Button {
                            viewModel.stopGeneration()
                        } label: {
                            Image(systemName: "stop.circle.fill")
                                .font(.title2)
                                .foregroundStyle(.red)
                        }
                    } else {
                        Button {
                            viewModel.sendMessage()
                        } label: {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.title2)
                        }
                        .disabled(viewModel.currentMessage.isEmpty)
                        .keyboardShortcut(.return, modifiers: .command)
                    }
                }
                .padding()
                .background(.ultraThinMaterial)
            }
        }
        .navigationTitle(model.model.displayName)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    viewModel.clearChat()
                } label: {
                    Label("Clear", systemImage: "trash")
                }
                .disabled(viewModel.messages.isEmpty)
            }
        }
        .onAppear {
            if !viewModel.isLoading {
                isInputFocused = true
            }
        }
    }

    // MARK: Private

    @State private var viewModel: ChatViewModel
    @FocusState private var isInputFocused: Bool
}

// MARK: - MessageBubble

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user {
                Spacer(minLength: 100)
            }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                Text(message.content)
                    .padding(12)
                    .background(message.role == .user ? Color.blue : Color(.systemGray5))
                    .foregroundStyle(message.role == .user ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: 16))

                Text(message.timestamp, style: .time)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if message.role == .assistant {
                Spacer(minLength: 100)
            }
        }
    }
}

// MARK: - MetricsView

struct MetricsView: View {
    let metrics: BenchmarkMetrics

    var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 20) {
                MetricCard(
                    title: "tok/sec",
                    value: Text(String(format: "%.1f", metrics.tokensPerSecond)),
                    icon: "speedometer"
                )

                MetricCard(
                    title: "ms/tok",
                    value: Text(String(format: "%.1fms", metrics.millisecondsPerToken)),
                    icon: "clock"
                )

                MetricCard(
                    title: "TTFT",
                    value: Text(Duration.milliseconds(metrics.timeToFirstToken * 1000).formatted(.units(allowed: [.seconds, .milliseconds]))),
                    icon: "timer"
                )
            }

            HStack(spacing: 20) {
                MetricCard(
                    title: "Total Tokens",
                    value: Text("\(metrics.promptTokens + metrics.completionTokens)"),
                    icon: "number"
                )

                MetricCard(
                    title: "Total Time",
                    value: Text(String(format: "%.1fs", metrics.totalTime)),
                    icon: "hourglass"
                )

                MetricCard(
                    title: "Memory",
                    value: Text(ByteCountFormatter.string(fromByteCount: Int64(metrics.memoryUsage), countStyle: .memory)),
                    icon: "memorychip"
                )
            }
        }
    }
}

// MARK: - MetricCard

struct MetricCard: View {
    let title: String
    let value: Text
    let icon: String

    var body: some View {
        VStack(spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption)
                Text(title)
                    .font(.caption)
            }
            .foregroundStyle(.secondary)

            value
                .font(.title3.bold())
                .monospacedDigit()
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
    }
}

#Preview {
    NavigationStack {
        ChatView(model: DownloadedModel(
            id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
            model: MLXModel(
                id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
                name: "mlx-community/Llama-3.2-3B-Instruct-4bit",
                author: "mlx-community",
                downloads: 1000,
                likes: 50,
                lastModified: Date(),
                size: 2_000_000_000,
                collections: nil,
                tags: ["text-generation"]
            ),
            localPath: URL(fileURLWithPath: "/tmp/model"),
            status: .installed
        ))
    }
}
