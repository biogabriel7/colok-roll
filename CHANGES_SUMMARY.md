# Changes Summary: Architectural Refactor

## Date: 2025-10-07

### Overview
Restructured the `colokroll` package to remove the phase-based architecture and organize code into clearer functional packages. All legacy compatibility shims have been removed for a clean structure.

---

## Final Structure

```
colokroll/
├── config/              # Runtime configuration
│   ├── base.py         # Core config classes
│   ├── presets.py      # Config factories
│   └── templates.py    # Preprocessing templates
├── io/                 # Image I/O
│   ├── converters.py   # Format conversion
│   ├── loaders.py      # ImageLoader
│   └── mip.py          # MIPCreator
├── preprocessing/      # Preprocessing pipeline
│   └── background/     # CUDA background subtraction
├── analysis/           # Analysis routines
│   ├── segmentation/   # Cellpose segmentation
│   ├── colocalization/ # Colocalization metrics
│   └── nuclei_detection.py
├── visualization/      # Plotting
│   ├── plots.py        # Visualizer
│   └── ui.py           # Interactive tools
└── core/              # Shared utilities
```

## Key Changes

- **Phase removal**: All `PhaseXConfig` renamed to descriptive names (`ImageIOConfig`, etc.)
- **No legacy shims**: All compatibility layers removed for clean codebase
- **Proper subpackages**: Segmentation and colocalization now in dedicated subpackages
- **CUDA preprocessing**: Background subtraction consolidated in `preprocessing/background/`

