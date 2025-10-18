import SwiftUI
import os

// MARK: - ModelListView

struct ModelListView: View {
    // MARK: Internal

    var selectedModel: DownloadedModel? {
        viewModel.downloadedModels.first { $0.id == selectedModelID }
    }

    var filteredModels: [MLXModel] {
        let models = selectedCollection == nil
            ? viewModel.availableModels
            : viewModel.groupedModels[selectedCollection!] ?? []

        if searchText.isEmpty {
            return models
        }
        return models.filter {
            $0.name.localizedCaseInsensitiveContains(searchText) ||
                $0.author.localizedCaseInsensitiveContains(searchText)
        }
    }

    var collectionNames: [String] {
        ["All Models"] + viewModel.collections.map(\.displayName).sorted()
    }

    var body: some View {
        NavigationSplitView {
            List(selection: $selectedModelID) {
                Section("Downloaded Models") {
                    if viewModel.downloadedModels.isEmpty {
                        ContentUnavailableView(
                            "No Downloaded Models",
                            systemImage: "arrow.down.circle",
                            description: Text("Download models to get started")
                        )
                    } else {
                        ForEach(viewModel.downloadedModels) { model in
                            DownloadedModelRow(model: model, viewModel: viewModel)
                                .tag(model.id)
                        }
                    }
                }

                Section {
                    if !collectionNames.isEmpty {
                        Picker("Collection", selection: $selectedCollection) {
                            ForEach(collectionNames, id: \.self) { collection in
                                Text(collection).tag(collection == "All Models" ? String?.none : String?.some(collection))
                            }
                        }
                        .pickerStyle(.menu)
                    }
                } header: {
                    Text("Filter")
                }

                Section("Available Models") {
                    if viewModel.isLoading {
                        HStack {
                            Spacer()
                            VStack(spacing: 12) {
                                ProgressView()
                                Text("Loading models...")
                                    .foregroundStyle(.secondary)
                            }
                            .padding()
                            Spacer()
                        }
                    } else if let error = viewModel.error {
                        ContentUnavailableView(
                            "Failed to Load Models",
                            systemImage: "exclamationmark.triangle",
                            description: Text(error)
                        )
                    } else if filteredModels.isEmpty, !searchText.isEmpty {
                        ContentUnavailableView.search
                    } else if filteredModels.isEmpty {
                        ContentUnavailableView(
                            "No Models Found",
                            systemImage: "tray",
                            description: Text("Pull to refresh or tap the refresh button")
                        )
                    } else {
                        ForEach(filteredModels) { model in
                            ModelRow(model: model, viewModel: viewModel)
                        }
                    }
                }
            }
            .navigationTitle("MLX Models")
            .searchable(text: $searchText, prompt: "Search models")
            .refreshable {
                await viewModel.fetchAvailableModels()
            }
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button {
                        Task {
                            await viewModel.fetchAvailableModels()
                        }
                    } label: {
                        Label("Refresh", systemImage: "arrow.clockwise")
                    }
                }
            }
            .alert("Error", isPresented: .constant(viewModel.error != nil)) {
                Button("OK") {
                    viewModel.error = nil
                }
            } message: {
                if let error = viewModel.error {
                    Text(error)
                }
            }
            .task {
                if viewModel.availableModels.isEmpty {
                    AppLogger.models.debug("Triggering initial model fetch from view")
                    await viewModel.fetchAvailableModels()
                }
            }
        } detail: {
            if let selected = selectedModel {
                ChatView(model: selected)
            } else {
                ContentUnavailableView(
                    "Select a Model",
                    systemImage: "message.fill",
                    description: Text("Choose a downloaded model to start chatting")
                )
            }
        }
    }

    // MARK: Private

    @State private var viewModel = ModelViewModel()
    @State private var searchText = ""
    @State private var selectedModelID: String?
    @State private var selectedCollection: String?
}

// MARK: - ModelRow

struct ModelRow: View {
    let model: MLXModel
    @Bindable var viewModel: ModelViewModel

    var status: ModelStatus {
        viewModel.getStatus(for: model)
    }

    var modelCollections: [String] {
        viewModel.collections.filter { $0.modelIds.contains(model.id) }.map(\.displayName)
    }

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.displayName)
                    .font(.headline)

                if !modelCollections.isEmpty {
                    Text(modelCollections.prefix(3).joined(separator: ", "))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }

            Spacer()

            switch status {
            case .available:
                Button {
                    viewModel.downloadModel(model)
                } label: {
                    Label("Download", systemImage: "arrow.down.circle")
                }
                .buttonStyle(.bordered)

            case let .downloading(progress):
                VStack(spacing: 4) {
                    ProgressView(value: progress)
                        .frame(width: 100)
                    Text("\(Int(progress * 100))%")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

            case .installed:
                Label("Installed", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)

            case let .error(message):
                Label("Error", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .help(message)
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - DownloadedModelRow

struct DownloadedModelRow: View {
    // MARK: Internal

    @Bindable var model: DownloadedModel
    @Bindable var viewModel: ModelViewModel

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(model.model.displayName)
                    .font(.headline)

                Text(model.model.author)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            switch model.status {
            case let .downloading(progress):
                Button("Cancel") {
                    viewModel.cancelDownload(model)
                }
                .buttonStyle(.bordered)

                ProgressView(value: progress)
                    .frame(width: 100)

            case .installed:
                Button(role: .destructive) {
                    showingDeleteAlert = true
                } label: {
                    Label("Delete", systemImage: "trash")
                }
                .buttonStyle(.bordered)

            case let .error(message):
                Label("Error", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.red)
                    .help(message)

            default:
                EmptyView()
            }
        }
        .padding(.vertical, 4)
        .alert("Delete Model", isPresented: $showingDeleteAlert) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                viewModel.deleteModel(model)
            }
        } message: {
            Text("Are you sure you want to delete \(model.model.displayName)? This will remove all downloaded files.")
        }
    }

    // MARK: Private

    @State private var showingDeleteAlert = false
}

#Preview {
    ModelListView()
}
