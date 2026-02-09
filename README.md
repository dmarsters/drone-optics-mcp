# Drone Optics MCP Server

**Aerial photography aesthetics through caustic optics simulation**

Phase 2.6 enhancement with rhythmic presets for temporal composition.
Validated compositional structure through basin analysis with microscopy-aesthetics-mcp.

## Overview

This MCP server implements drone/aerial photography aesthetics using a two-phase caustic spin algorithm that simulates rotating slit filters during long exposure photography. Unlike arbitrary image filters, this is grounded in real optical physics and produces depth-encoded visual transformations.

### Key Features

- **Zero-cost transformations**: Pure Layer 2 NumPy/PIL computation (0 tokens)
- **Validated compositional structure**: CP parameter range [15-270] tested against microscopy domain
- **Emergent attractors**: Discovered Period 8, 10, 12, 14 limit cycles in multi-domain composition
- **Rhythmic presets**: Phase 2.6 temporal sequences for multi-domain limit cycle discovery
- **Depth field extraction**: Z-depth maps derived from phase interference for compositional use

## Compositional Basin Validation

**Tested with**: `microscopy-aesthetics-mcp` focus_sweep preset (brightfield ↔ confocal, Period 24)

### Validated CP Ranges

| CP Range | Density | Total Cycles | Peak Autocorr | Status |
|----------|---------|--------------|---------------|--------|
| 15-45 (Low) | 0.042-0.125 | 33 | **0.96** (P10) | ✓ Strong |
| 60-120 (Med) | 0.167-0.333 | 35 | **0.96** (P8) | ✓✓ **OPTIMAL** |
| 180-270 (High) | 0.5-0.75 | 32 | **0.96** (P10) | ✓ Strong |

**Key Discovery**: No degradation at high CP values - compositional structure remains stable across full tested range.

### Emergent Attractors

When composed with microscopy domain:
- **Period 10** (autocorr 0.96): Universal attractor across all CP ranges
- **Period 8** (autocorr 0.96): Strongest in medium CP range
- **Period 12** (autocorr 0.95): Harmonic of microscopy preset (24÷2)
- **Period 14** (autocorr 0.95): **Novel emergent** - only appears in high CP range

**Reference**: See `cp_basin_analysis.md` for complete compositional testing results.

## Installation

### Local Development

```bash
cd drone-optics-mcp
pip install -e .
```

### FastMCP Cloud Deployment

```bash
fastmcp install drone-optics-mcp
```

**Entrypoint**: `server.py:mcp`

## Layer Architecture

### Layer 1: Taxonomy
- Drone imaging mode profiles (aerial_survey, low_altitude, thermal_imaging)
- Parameter space definition with validated compositional bounds

### Layer 2: Deterministic Transformation
- Caustic spin algorithm (0 tokens)
- Two-phase interference (forward + reverse rotation)
- Z-depth extraction from phase difference
- Rhythmic preset generation

### Layer 3: Claude Synthesis
(Future enhancement for prompt generation)

## Core Tools

### `apply_caustic_spin`
Apply rotating slit filter transformation to image.

```python
result = apply_caustic_spin(
    image_base64="...",
    CP=33.0,              # Color palette density [15, 270]
    frame_offset=0.0,     # Rotation phase [0, 360]
    rotation_speed=1.0,   # Angular velocity multiplier
    output_depth=True     # Return z_depth field
)
```

**Returns**: Transformed image + optional depth field + metadata

**Cost**: 0 tokens (pure Layer 2)

### `extract_depth_field`
Get z-depth map for compositional use.

```python
depth_result = extract_depth_field(
    image_base64="...",
    CP=33.0,
    frame_offset=0.0
)
```

**Output**: Depth field visualization + statistics
**Use case**: Input for multi-domain composition (depth → nuclear blast phase mapping, etc.)

### `map_drone_parameters`
Map imaging mode to parameter space.

```python
params = map_drone_parameters(
    imaging_mode="aerial_survey",
    CP_override=90.0,          # Optional override
    temporal_phase=0.5         # Optional phase override
)
```

**Returns**: Complete parameter set with compositional basin metadata

## Phase 2.6: Rhythmic Presets

### `generate_rhythmic_drone_sequence`
Generate temporal oscillation between imaging modes.

```python
sequence = generate_rhythmic_drone_sequence(
    state_a_mode="aerial_survey",
    state_b_mode="thermal_imaging",
    oscillation_pattern="sinusoidal",  # or "triangular", "square"
    num_cycles=2,
    steps_per_cycle=20,
    phase_offset=0.0
)
```

**Returns**: List of parameter states over time
**Cost**: 0 tokens (pure Layer 2)

### `apply_drone_rhythmic_preset`
Use curated preset configuration.

```python
preset = apply_drone_rhythmic_preset(
    preset_name="aerial_rotation",
    override_params={"num_cycles": 3}
)
```

**Available Presets**:

1. **aerial_rotation** (Medium CP, optimal basin)
   - CP 60→120 over 4 cycles × 20 steps
   - Sinusoidal pattern
   - Expected attractors: Period 10 (0.96), Period 8 (0.96)
   - Use: General aerial cinematography

2. **low_altitude_sweep** (Low CP, fine detail)
   - CP 15→45 over 3 cycles × 24 steps
   - Sinusoidal pattern
   - Expected attractors: Period 10 (0.96), Period 12 (0.90)
   - Use: Close-range inspection

3. **thermal_pulse** (High CP, novel attractor)
   - CP 180→270 over 5 cycles × 16 steps
   - Sinusoidal pattern
   - Expected attractors: **Period 14 (0.95)** - emergent only, Period 10 (0.96)
   - Use: Thermal/IR simulation

4. **orbit_scan** (Triangular, linear zoom)
   - CP 72→144 over 2 cycles × 30 steps
   - Triangular pattern (mechanical rhythm)
   - Use: Satellite imagery with linear progression

### `list_drone_rhythmic_presets`
Get catalog of available presets.

```python
catalog = list_drone_rhythmic_presets()
```

## Multi-Domain Composition

### Integration with composition-graph-mcp

Use rhythmic presets for compositional limit cycle discovery:

```python
# Discover emergent attractors between drone optics and microscopy
result = composition_graph_mcp.discover_compositional_limit_cycles(
    domain_preset_configs={
        "drone_optics": {
            "aerial_rotation": {
                # Preset configuration auto-populated
            }
        },
        "microscopy": {
            "focus_sweep": {
                # Microscopy preset configuration
            }
        }
    },
    interaction_mode="mutual_influence",
    n_samples=50,
    integration_steps=300
)
```

**Expected Emergent Behaviors**:
- Beat frequencies from period differences
- Harmonic phase-locking
- Novel periods (e.g., Period 14 only in high CP + microscopy composition)

### Compositional Depth Mapping

Depth fields can be used as input to other aesthetic domains:

```python
# Extract depth from drone optics
depth = extract_depth_field(image_base64="...", CP=60)

# Map depth to nuclear blast phase evolution
# (future integration with nuclear-aesthetic-mcp)
nuclear_params = map_depth_to_blast_phase(depth_statistics)
```

## Parameter Space

### Color Palette Density (CP)

Normalized to [0, 1] in parameter space:
```
density = CP / 360

Examples:
CP 15  → density 0.042  (low range, fine detail)
CP 90  → density 0.25   (medium range, optimal)
CP 270 → density 0.75   (high range, novel Period 14 attractor)
```

**Compositional Recommendations**:
- Default: CP 90 (density 0.25) - medium range, optimal basin diversity
- Fine detail: CP 15-45 - strong Period 10/12 attractors
- Thermal/IR: CP 180-270 - access novel Period 14 emergent attractor

### Temporal Phase

Rotation angle normalized to [0, 1]:
```
temporal_phase = (frame_offset % 360) / 360
```

### Depth Separation

Interference strength [0, 1]:
- Derived from |forward_phase - reverse_phase|
- Higher values = stronger depth differentiation

## Caustic Spin Algorithm

### Physical Basis

Simulates a spinning slit filter in front of a camera lens during long exposure:

1. **Forward rotation**: `cos(angle - slit_speed × frame_offset)`
2. **Reverse rotation**: `cos(angle + slit_speed × frame_offset)`
3. **Interference**: `|forward - reverse|` → depth field
4. **Hue modulation**: Color shift proportional to depth

### Mathematical Properties

- **Two-phase interference**: Opposing rotations create standing wave patterns
- **Depth encoding**: Phase difference maps to Z-axis information
- **Harmonic structure**: CP determines fundamental frequency of color modulation

**Result**: Motion-encoded visuals with depth information, without physical moving parts.

## Usage Examples

### Basic Transformation

```python
# Apply caustic spin to aerial photo
result = apply_caustic_spin(
    image_base64=aerial_photo_b64,
    CP=90,           # Medium range (optimal)
    frame_offset=45  # 45° rotation
)

# Get transformed image
transformed = result["transformed_image"]
depth_map = result["depth_field"]
```

### Rhythmic Sequence for Video

```python
# Generate 4-cycle rotation sequence
sequence = apply_drone_rhythmic_preset(
    preset_name="aerial_rotation"
)

# Render each frame
for i, params in enumerate(sequence["sequence"]):
    frame = apply_caustic_spin(
        image_base64=base_image,
        CP=params["color_palette_density"] * 360,
        frame_offset=params["temporal_phase"] * 360
    )
    save_frame(f"frame_{i:03d}.png", frame)
```

### Multi-Domain Composition Discovery

```python
# Create preset configs for both domains
drone_config = {
    "drone_optics": {
        "thermal_pulse": apply_drone_rhythmic_preset("thermal_pulse")
    }
}

# Discover emergent attractors
attractors = discover_compositional_limit_cycles(
    domain_preset_configs={**drone_config, **microscopy_config},
    n_samples=50
)

# Expected: Novel Period 14 attractor from high CP thermal preset
```

## Performance

- **Transformation speed**: ~100ms for 400×300 image (Python/NumPy)
- **Token cost**: 0 (all operations are deterministic Layer 2)
- **Memory**: <50MB per transformation
- **Parallelizable**: Each frame independent

## Integration with Lushy

### Workflow Packaging

Drone optics transformations can be packaged as Lushy workflows:

```json
{
  "workflow_name": "Aerial Caustic Animation",
  "description": "Generate rotating slit filter effect sequence",
  "inputs": {
    "base_image": "image",
    "CP_range": "slider[15-270]",
    "num_frames": "integer"
  },
  "steps": [
    {
      "tool": "drone_optics.apply_drone_rhythmic_preset",
      "preset": "aerial_rotation"
    },
    {
      "tool": "render_sequence"
    }
  ]
}
```

### Monetization

- Preset access tiers (basic/professional/research)
- Custom CP range optimization for specific use cases
- Multi-domain composition packages

## Future Enhancements

### Layer 3 Integration
- Prompt synthesis from drone parameters
- Natural language → imaging mode mapping
- Compositional prompt generation

### Extended Presets
- GPS trajectory simulation
- Weather condition presets (fog, haze, rain)
- Time-of-day progression

### Advanced Composition
- Drone + nuclear aesthetics (depth → blast phase)
- Drone + diatom morphology (aerial → microscopic scale bridge)
- Drone + catastrophe theory (terrain topology morphing)

## Testing

### Basin Validation Tests
```bash
pytest tests/test_compositional_basins.py
```

Validates:
- CP range [15-270] produces stable compositions
- Period 10 attractor appears in all ranges
- Period 14 emergence in high CP range
- Medium CP produces optimal basin diversity

### Transformation Tests
```bash
pytest tests/test_caustic_spin.py
```

## License

MIT

## References

- `cp_basin_analysis.md` - Compositional basin testing results
- Spivak, D. - Applied Category Theory (polynomial functors)
- Van Eenwyk, J. - Catastrophe theory in psychology (attractor terminology)

## Author

Dal Marsters
Lushy - AI Workflow Platform
