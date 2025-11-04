Here are the contents for the file `/LongCatUnlimitedVideoGeneratorForMac/LongCatUnlimitedVideoGeneratorForMac/src/AppDelegate.swift`:

import Cocoa

@main
class AppDelegate: NSObject, NSApplicationDelegate {

    var window: NSWindow!

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        // Create the main application window
        let screenSize = NSScreen.main?.frame.size ?? NSSize(width: 800, height: 600)
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: screenSize.width * 0.8, height: screenSize.height * 0.8),
                          styleMask: [.titled, .closable, .resizable],
                          backing: .buffered, defer: false)
        window.center()
        window.title = "Long Cat Unlimited Video Generator"
        window.makeKeyAndOrderFront(nil)
    }

    func applicationWillTerminate(_ aNotification: Notification) {
        // Insert code here to tear down your application
    }
