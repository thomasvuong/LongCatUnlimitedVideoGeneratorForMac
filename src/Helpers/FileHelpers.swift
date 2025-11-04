import Foundation

class FileHelpers {
    
    // Function to read the contents of a file at a given path
    static func readFile(atPath path: String) -> String? {
        do {
            let contents = try String(contentsOfFile: path, encoding: .utf8)
            return contents
        } catch {
            print("Error reading file at \(path): \(error)")
            return nil
        }
    }
    
    // Function to write a string to a file at a given path
    static func writeFile(atPath path: String, contents: String) -> Bool {
        do {
            try contents.write(toFile: path, atomically: true, encoding: .utf8)
            return true
        } catch {
            print("Error writing file at \(path): \(error)")
            return false
        }
    }
    
    // Function to check if a file exists at a given path
    static func fileExists(atPath path: String) -> Bool {
        return FileManager.default.fileExists(atPath: path)
    }
}