# SNN ADMM Optimizer: Known Limitations

This document outlines the current boundaries and technical debt of the SNN ADMM Optimizer. While the core ADMM optimization mechanics are fully functional, the current version has several architectural and technical areas scoped for future improvement.

### 📐 Architecture & State Management

* **Decouple Spiking Logic from the Manager:** Currently, the state of whether a network is "spiking" is tracked redundantly by both the layer objects and the global manager. This duplication of state information introduces a risk of synchronization bugs. The manager should become entirely spiking-agnostic, relying instead on a unified, polymorphic layer interface.
* **Modular Activation Functions:** Activation functions are tightly coupled to the layer definitions. Extracting these into independent, modular components will drastically improve architectural flexibility and simplify state initialization.
* **Refactor Loss Function Module:** While functional, the current loss module's internal logic and API are not as clean or intuitive as they could be. A thorough refactor is needed to improve readability and extensibility.

### 🧮 Tensor Mechanics & Forward Pass

* **Explicit Tensor Shape Handling:** The codebase currently relies on implicit broadcasting utilities (e.g., `broadcast_to_match`, `format_bias`) for rapid development. Replacing these with strict, explicit shape handling at every step will improve code readability and prevent silent dimensional bugs. Additionally, spiking forward tensor shaping should be checked.
* **Dynamic Pooling Dimensions:** All pooling methods currently hardcode the output to a flattened 2D shape to ensure default compatibility with feedforward layers. The pooling mechanism needs to be generalized to support arbitrary tensor dimensions natively.
* **Automatic Membrane Reset Handling (`use_reset`):** The `use_reset=False` parameter currently requires manual configuration for the final output layer to allow spike accumulation. This should be handled automatically by the optimizer. Additionally, this opens a broader discussion for future features regarding non-accumulating SNN output layers.

### ⚙️ Optimization & Training

* **Multibatch Scheduling Support:** The dynamic penalty scheduler currently only supports single-batch environments. It needs to be extended to properly track, average, and balance residuals across multibatch datasets.

### 🧪 Maintenance

* **Comprehensive Test Coverage:** To safely scale the framework and ensure mathematical stability across complex ADMM updates, the testing suite requires significant expansion. Implementing stricter unit and integration tests is necessary for early bug detection and consistent deployment.