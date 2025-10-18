import Foundation
import os

enum AppLogger {
    private static let subsystem = Bundle.main.bundleIdentifier ?? "MLXBenchmark"

    static let network = Logger(subsystem: subsystem, category: "Network")
    static let models = Logger(subsystem: subsystem, category: "Models")
    static let chat = Logger(subsystem: subsystem, category: "Chat")
}

