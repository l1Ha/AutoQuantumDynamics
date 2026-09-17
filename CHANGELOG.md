# Changelog

## 0.3.1 — NN regression output fix

### Fixed

- FeedForwardNN applies activation functions only to hidden layers; its output layer is linear. Predictions can represent energies outside bounded activation ranges.
- With the linear output, the existing backward pass computes the gradient of half the mean squared error for scalar regression.
- Package metadata and runtime version are both 0.3.1 (previously inconsistent at 0.2.0 and 0.3.0).

### Tests

- Retain the two existing NN tests.
- Add regression tests for an output of 3.0, fitting targets outside tanh's range, and finite-difference checks of every weight and bias gradient.
- Release verification results are recorded in RELEASE_VALIDATION.md.

### Compatibility

Loading existing weights now uses a linear output instead of the previously selected activation. Predictions are therefore not backward-compatible. Revalidate or retrain existing models before using them. Only load trusted pickle model files.

### Scope and limitations

This is a narrowly scoped patch on the existing prototype, not a validated molecular-reaction software release. It excludes uncommitted wavepacket, engine and CLI development changes. The known H3 CLI errors, defective legacy 2D stationary solver, legacy 1D wavepacket bugs and missing ab initio backend are not fixed here. NN early stopping, normalization, multi-dimensional training and force fitting remain incomplete. See README.md for details.

No LEPS barrier heights, reaction thresholds or reaction probabilities are certified by this release.
