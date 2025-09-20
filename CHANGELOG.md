# 💜 Violet Model Merge - Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### 💜 Coming Soon
- Additional merge algorithms and techniques
- Enhanced performance optimizations

---

## [1.2.1] - 2025-09-19

### 🚀 Performance

- **Major Performance Boost**: Optimized merge execution logic resulting in dramatically faster processing
  - 3-model merges now complete in seconds instead of hours
  - 30% smaller codebase with significantly improved efficiency
  - Memory usage optimizations for large model handling

### 🛡️ Fixed

- **DARE Merge Beta Parameter**: Fixed missing beta argument in DARE merge algorithm
  - Corrected function signature mismatch between 2-model and 3-model merges
  - Updated merge execution logic to properly handle beta parameter for DARE mode
  - All merge modes now work correctly in both CLI and notebook environments
- Improved merge algorithm dispatch logic for better reliability

### 🔧 Changed

- Streamlined merge execution code for better performance and maintainability
- Enhanced merge mode detection and parameter handling

---

## [1.2.0] - 2025-09-19

### 💜 Added

- **Project Rebrand**: Now known as **Violet Model Merge**
- Interactive Jupyter notebook with comprehensive merge documentation
- Artist-friendly interface with guided examples and explanations
- Clean error handling with user-friendly progress reporting
- Enhanced VAE support for `.pt`, `.ckpt`, and `.safetensors` formats
- Automatic CUDA detection and fallback to CPU
- Structured project organization with `lib/` directory
- Comprehensive flag reference and usage examples
- Real-time merge progress tracking with clean output

### 🔧 Changed

- Reorganized project structure for better maintainability
- Moved Python modules to `lib/` directory
- Enhanced README with modern formatting and better organization
- Improved error messages with blocking/non-blocking classification
- Updated all documentation to reflect new project structure

### 🛡️ Fixed

- Corrupted emoji characters in notebook table of contents causing markdown rendering issues
- Unicode decoding issues during subprocess execution
- CUDA compatibility with proper PyTorch installation detection
- Import path resolution for reorganized modules
- Memory handling for large model merges

### 📚 Documentation

- Complete rewrite of README with artist-focused approach
- Added comprehensive merge mode explanations
- Created interactive notebook with all merge techniques
- Enhanced troubleshooting guide
- Added project roadmap and contribution guidelines

---

## [1.0.0] - Initial Fork from Chattiori Model Merger

### 🎉 Initial Release

- Forked from [Chattiori Model Merger](https://github.com/faildes) by Chattiori
- Maintained all original merge algorithms and functionality
- Added foundation for enhanced artist experience
- Preserved compatibility with original CLI interface

---

## 🔗 Links

- **Repository**: [leylahviolet/Violet-Model-Merge](https://github.com/leylahviolet/Violet-Model-Merge)
- **Original Project**: [Chattiori Model Merger](https://github.com/faildes) by Chattiori
- **Issues**: [Report a Bug](https://github.com/leylahviolet/Violet-Model-Merge/issues)

---

## 💜 Legend

- 🎨 **Added** - New features
- 🔧 **Changed** - Changes in existing functionality
- 🛡️ **Fixed** - Bug fixes
- 📚 **Documentation** - Documentation changes
- ⚠️ **Deprecated** - Soon-to-be removed features
- 🗑️ **Removed** - Removed features
- 🛡️ **Security** - Security improvements