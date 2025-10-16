# MLX Benchmark

A modern SwiftUI application for downloading, installing, and benchmarking MLX models from the Hugging Face mlx-community on iPad.

## Features

### Model Management
- **Browse Models**: Discover available MLX models from the mlx-community on Hugging Face
- **Collection Grouping**: Models organized by Hugging Face collections for easy discovery
- **Search & Filter**: Find models by name, author, or collection
- **Real Downloads**: Uses Hugging Face Hub API for authentic model downloads
- **Download Progress**: Real-time download progress via Hub's Progress API
- **Model Information**: View model size, downloads, likes, last modified date, and collections
- **Installation Management**: Install and remove models with confirmation dialogs

### Real MLX Inference
- **MLXLLM Integration**: Uses MLX Swift Examples framework for real on-device inference
- **Interactive Chat**: Send messages to downloaded models with actual MLX generation
- **Streaming Responses**: Token-by-token generation using ChatSession API
- **Real-time Metrics**:
  - Tokens per second (actual generation speed)
  - Time to first token (TTFT)
  - Total tokens (prompt + completion)
  - Memory usage (resident memory)
- **Chat History**: Keep track of conversation with timestamps
- **Session Management**: Reset chat while preserving loaded model

### Modern Design
- **iPad-optimized**: Split-view interface perfect for iPad
- **iOS 26 Features**: Uses modern SwiftUI and Swift Observation framework (@Observable)
- **Material Design**: Frosted glass effects and smooth animations
- **Dark Mode**: Full support for light and dark appearance

## Architecture

### Models
- `MLXModel`: Represents available models from Hugging Face with collections
- `DownloadedModel`: Observable class tracking locally installed models
- `ChatMessage`: Chat conversation messages
- `BenchmarkMetrics`: Performance metrics for inference
- `HuggingFaceCollection`: Collection metadata from HF

### Services
- `HuggingFaceService`: Fetches model lists and collections from Hugging Face API
- `ModelManager`: Uses Hub API for downloads and MLXLLM for model loading
- Handles downloading all required model files via Hub snapshot

### ViewModels
- `ModelViewModel`: Manages model list state, collections, downloads, and installations
- `ChatViewModel`: Handles chat using MLXLLM ChatSession with real streaming inference
- Uses Swift's `@Observable` macro for reactive state management

### Views
- `ModelListView`: 
  - Split-view navigation (iPad optimized)
  - Downloaded models section
  - Available models with collection filtering
  - Download progress indicators
  - Pull-to-refresh and manual refresh
  
- `ChatView`: 
  - Chat interface with message bubbles
  - Real-time metrics display (4 metric cards)
  - Message input with multi-line support
  - Streaming message display
  - Model loading state

## Technical Details

- **Target**: iOS 26.0+
- **Framework**: SwiftUI with Observation
- **ML Framework**: MLX Swift (MLXLLM, MLXLMCommon)
- **API**: Hugging Face Hub API for model discovery and download
- **Storage**: Local file system via Hub with UserDefaults for persistence
- **Concurrency**: Swift async/await for all network and model operations
- **Inference**: Real on-device inference using MLX Swift Examples

## Usage

1. Launch the app on your iPad
2. Browse available MLX models from the mlx-community
3. Filter by collection to find specific model types
4. Tap "Download" on any model to install it locally via Hub API
5. Wait for download to complete (progress shown in real-time)
6. Select a downloaded model to open the chat interface
7. Wait for model to load into memory
8. Send messages and see real MLX inference with streaming responses
9. Monitor actual performance metrics in real-time
10. Clear chat to reset conversation or delete models when no longer needed

## Implementation Notes

This app integrates real MLX Swift inference:

- ✅ Uses MLXLLM from mlx-swift-examples
- ✅ Hub API for model downloads
- ✅ ChatSession for streamlined inference
- ✅ Real token streaming
- ✅ Actual performance metrics
- ✅ On-device generation using Apple Silicon

The app provides a complete benchmarking environment for testing MLX models on iOS/iPadOS devices.

## Dependencies

- **mlx-swift-examples**: Provides MLXLLM, MLXLMCommon, and model implementations
- **swift-transformers**: Provides Hub API for Hugging Face integration
- **mlx-swift**: Core MLX framework for Apple Silicon acceleration

## Requirements

- iOS 26.0 or later
- iPad (optimized for iPad interface)
- Apple Silicon device for MLX acceleration
- Internet connection for downloading models
