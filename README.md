# LongCatUnlimitedVideoGeneratorForMac

## Project Structure

This project is organized into several directories and files, each serving a specific purpose:

- **src/**: Contains the source code for the macOS application.
  - **AppDelegate.swift**: The application delegate that manages the app lifecycle and sets up the main window.
  - **MainViewController.swift**: The main view controller that handles the user interface and interactions.
  - **VideoGenerator.swift**: Contains the logic for generating videos, including processing video data and integrating templates.
  - **Helpers/**: A directory for utility functions.
    - **FileHelpers.swift**: Provides functions for file operations such as reading and writing files.

- **scripts/**: Contains scripts for project setup and maintenance.
  - **prepare_models.sh**: A shell script that prepares necessary models for the application.

- **models/**: Contains documentation related to the models used in the project.
  - **README.md**: Documentation for the models.

- **resources/**: Contains resources used by the application.
  - **templates/**: Directory for video templates.
    - **default_template.mov**: The default video template used for video generation.
  - **config/**: Directory for configuration files.
    - **settings.plist**: Configuration settings for the application.

- **.gitignore**: Specifies files and directories to be ignored by Git.

- **Package.swift**: The Swift package manager configuration file that defines the package and its dependencies.

- **README.md**: This file contains documentation for the project, including setup instructions and usage.

- **LICENSE**: Contains licensing information for the project.

## File Placement

Ensure that the files are placed in the following structure:

```
LongCatUnlimitedVideoGeneratorForMac/
├── src/
│   ├── AppDelegate.swift
│   ├── MainViewController.swift
│   ├── VideoGenerator.swift
│   └── Helpers/
│       └── FileHelpers.swift
├── scripts/
│   └── prepare_models.sh
├── models/
│   └── README.md
├── resources/
│   ├── templates/
│   │   └── default_template.mov
│   └── config/
│       └── settings.plist
├── .gitignore
├── Package.swift
├── README.md
└── LICENSE
```

## Dependencies

This project uses Swift Package Manager for dependency management. To install the required dependencies, run the following command in the terminal from the project root:

```
swift package resolve
```

## Setup Instructions

1. **Clone the Repository**: Clone the repository to your local machine using:
   ```
   git clone https://github.com/thomasvuong/LongCatUnlimitedVideoGeneratorForMac.git
   ```

2. **Navigate to the Project Directory**:
   ```
   cd LongCatUnlimitedVideoGeneratorForMac
   ```

3. **Run the Prepare Models Script**: Execute the `prepare_models.sh` script to set up the necessary models:
   ```
   chmod +x scripts/prepare_models.sh
   ./scripts/prepare_models.sh
   ```

4. **Build the Project**: Use Xcode or the command line to build the project.

5. **Run the Application**: Launch the application from Xcode or the built executable.

By following these instructions, you should be able to set up and run the LongCat Unlimited Video Generator for Mac successfully.