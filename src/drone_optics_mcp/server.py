"""
Drone Optics MCP Server
Aerial photography aesthetics through caustic optics simulation

PHASE 2.6 ENHANCEMENT: Rhythmic presets for temporal composition
PHASE 2.7 ENHANCEMENT: Attractor visualization prompt generation
Basin-validated CP ranges from microscopy composition testing

Layer Architecture:
- Layer 1: Taxonomy (drone imaging modes, optical parameters)
- Layer 2: Deterministic caustic spin transformation (0 tokens)
- Layer 2.7: Visual vocabulary extraction + prompt generation (0 tokens)
- Layer 3: Claude synthesis (prompt enhancement)
"""

from fastmcp import FastMCP
import numpy as np
from PIL import Image
import io
import base64
import colorsys
import math
from typing import Optional, Dict, List, Tuple

mcp = FastMCP("drone-optics")

# ============================================================================
# LAYER 1: TAXONOMY - Drone Imaging Parameter Space
# ============================================================================

# Validated compositional bounds from microscopy focus_sweep testing
# See: cp_basin_analysis.md
DRONE_PARAMETER_SCHEMA = {
    "color_palette_density": {
        "bounds": [0.042, 0.75],  # CP 15-270 (validated)
        "optimal": [0.167, 0.333],  # CP 60-120 (medium range, strongest basins)
        "description": "Caustic filter detail level, higher = coarser color modulation"
    },
    "temporal_phase": {
        "bounds": [0.0, 1.0],
        "description": "Rotation phase [0, 1], maps to frame_offset in caustic spin"
    },
    "depth_separation": {
        "bounds": [0.0, 1.0],
        "description": "Z-depth interference strength, derived from phase difference"
    },
    "rotation_speed": {
        "bounds": [0.1, 10.0],
        "description": "Angular velocity multiplier for slit_speed calculation"
    }
}

# Drone imaging mode profiles (future expansion)
DRONE_IMAGING_MODES = {
    "aerial_survey": {
        "display_name": "Aerial Survey",
        "description": "High-altitude orthographic mapping",
        "default_params": {
            "color_palette_density": 0.25,  # Medium CP (optimal range)
            "temporal_phase": 0.0,
            "depth_separation": 0.5,
            "rotation_speed": 1.0
        }
    },
    "low_altitude": {
        "display_name": "Low Altitude Detail",
        "description": "Close-range oblique imaging with motion blur",
        "default_params": {
            "color_palette_density": 0.125,  # Low CP (fine detail)
            "temporal_phase": 0.5,
            "depth_separation": 0.8,
            "rotation_speed": 2.0
        }
    },
    "thermal_imaging": {
        "display_name": "Thermal/IR Imaging",
        "description": "Heat signature visualization with high contrast",
        "default_params": {
            "color_palette_density": 0.6,  # High CP (novel Period 14 attractor)
            "temporal_phase": 0.0,
            "depth_separation": 0.9,
            "rotation_speed": 0.5
        }
    }
}


# ============================================================================
# PHASE 2.7: NORMALIZED MORPHOSPACE & VISUAL VOCABULARY
# ============================================================================

# 5D normalized parameter space for multi-domain composition
# All parameters normalized to [0.0, 1.0]
DRONE_PARAMETER_NAMES = [
    "color_palette_density",   # 0.0 = fine caustic detail, 1.0 = coarse color bands
    "temporal_phase",          # 0.0 = cycle start, 1.0 = full rotation
    "depth_separation",        # 0.0 = flat/no depth, 1.0 = maximum z-depth interference
    "altitude_perspective",    # 0.0 = overhead orthographic, 1.0 = oblique low-altitude
    "spectral_range"           # 0.0 = visible RGB, 1.0 = thermal/IR false color
]

# Canonical drone imaging states with 5D coordinates
# These define the morphospace vertices for interpolation
DRONE_STATE_COORDINATES = {
    "aerial_survey": {
        "color_palette_density": 0.25,
        "temporal_phase": 0.0,
        "depth_separation": 0.50,
        "altitude_perspective": 0.10,
        "spectral_range": 0.05
    },
    "low_altitude": {
        "color_palette_density": 0.10,
        "temporal_phase": 0.50,
        "depth_separation": 0.80,
        "altitude_perspective": 0.85,
        "spectral_range": 0.10
    },
    "thermal_imaging": {
        "color_palette_density": 0.65,
        "temporal_phase": 0.0,
        "depth_separation": 0.90,
        "altitude_perspective": 0.20,
        "spectral_range": 0.95
    },
    "golden_hour": {
        "color_palette_density": 0.30,
        "temporal_phase": 0.75,
        "depth_separation": 0.60,
        "altitude_perspective": 0.70,
        "spectral_range": 0.15
    },
    "night_vision": {
        "color_palette_density": 0.55,
        "temporal_phase": 0.0,
        "depth_separation": 0.70,
        "altitude_perspective": 0.15,
        "spectral_range": 0.80
    },
    "lidar_scan": {
        "color_palette_density": 0.05,
        "temporal_phase": 0.0,
        "depth_separation": 1.00,
        "altitude_perspective": 0.05,
        "spectral_range": 0.40
    },
    "cinematic_orbit": {
        "color_palette_density": 0.20,
        "temporal_phase": 0.50,
        "depth_separation": 0.45,
        "altitude_perspective": 0.60,
        "spectral_range": 0.05
    }
}

# Visual vocabulary types for image generation prompt extraction
# Each type has 5D coordinates and descriptive keywords
DRONE_VISUAL_VOCABULARY = {
    "survey_cartographic": {
        "coords": {
            "color_palette_density": 0.20,
            "temporal_phase": 0.0,
            "depth_separation": 0.40,
            "altitude_perspective": 0.10,
            "spectral_range": 0.05
        },
        "keywords": [
            "high-altitude aerial survey",
            "orthographic planimetric view",
            "crisp geometric ground patterns",
            "flat cartographic perspective",
            "uniform daylight illumination",
            "cadastral grid precision",
            "topographic contour clarity"
        ],
        "optical_properties": {
            "finish": "matte documentary",
            "depth_of_field": "infinite (ortho projection)",
            "color_response": "calibrated true color"
        },
        "color_associations": [
            "earth tones", "agricultural greens",
            "concrete grays", "water blues"
        ]
    },
    "cinematic_oblique": {
        "coords": {
            "color_palette_density": 0.25,
            "temporal_phase": 0.60,
            "depth_separation": 0.55,
            "altitude_perspective": 0.75,
            "spectral_range": 0.10
        },
        "keywords": [
            "dramatic oblique aerial angle",
            "cinematic drone flyover",
            "atmospheric haze layering",
            "sweeping parallax depth",
            "golden-hour raking light",
            "motion-blurred foreground elements",
            "volumetric light scattering"
        ],
        "optical_properties": {
            "finish": "cinematic filmic",
            "depth_of_field": "shallow with bokeh",
            "color_response": "warm color grading"
        },
        "color_associations": [
            "amber golden tones", "deep shadow blues",
            "warm highlight spill", "atmospheric violet haze"
        ]
    },
    "thermal_false_color": {
        "coords": {
            "color_palette_density": 0.65,
            "temporal_phase": 0.0,
            "depth_separation": 0.85,
            "altitude_perspective": 0.20,
            "spectral_range": 0.90
        },
        "keywords": [
            "thermal infrared false-color mapping",
            "heat-signature gradient bands",
            "high-contrast emissivity patterns",
            "coarse spectral quantization steps",
            "radiometric temperature visualization",
            "thermal plume detection"
        ],
        "optical_properties": {
            "finish": "digital false-color",
            "depth_of_field": "deep (thermal focus)",
            "color_response": "pseudocolor LUT mapping"
        },
        "color_associations": [
            "infrared magentas", "thermal yellows",
            "cold blues", "hot whites", "emissive reds"
        ]
    },
    "depth_interferometric": {
        "coords": {
            "color_palette_density": 0.10,
            "temporal_phase": 0.30,
            "depth_separation": 0.95,
            "altitude_perspective": 0.15,
            "spectral_range": 0.35
        },
        "keywords": [
            "z-depth interferometric layering",
            "caustic phase-difference fringes",
            "volumetric depth-encoded surfaces",
            "structured-light point cloud density",
            "fine parallax micro-displacement",
            "holographic depth separation"
        ],
        "optical_properties": {
            "finish": "translucent interferometric",
            "depth_of_field": "multi-plane focus stacking",
            "color_response": "depth-encoded chromatic"
        },
        "color_associations": [
            "iridescent phase shifts", "depth-gradient blues",
            "structured-light greens", "elevation-coded spectrum"
        ]
    }
}


def _extract_drone_visual_vocabulary(
    state: dict,
    strength: float = 1.0
) -> dict:
    """
    Extract visual vocabulary from drone parameter coordinates.

    Pure Layer 2 nearest-neighbor matching against visual types.
    Cost: 0 tokens.

    Args:
        state: Parameter coordinates dict with DRONE_PARAMETER_NAMES keys
        strength: Keyword weight multiplier [0.0, 1.0]

    Returns:
        Dict with nearest_type, distance, keywords, optical_properties, etc.
    """
    # Build state vector from available parameters
    state_vec = np.array([
        state.get(p, 0.5) for p in DRONE_PARAMETER_NAMES
    ])

    min_dist = float('inf')
    nearest_type = None

    for type_name, type_def in DRONE_VISUAL_VOCABULARY.items():
        type_vec = np.array([
            type_def["coords"].get(p, 0.5) for p in DRONE_PARAMETER_NAMES
        ])
        dist = float(np.linalg.norm(state_vec - type_vec))
        if dist < min_dist:
            min_dist = dist
            nearest_type = type_name

    vocab = DRONE_VISUAL_VOCABULARY[nearest_type]

    # Apply strength weighting to keywords
    if strength < 1.0:
        keyword_count = max(2, int(len(vocab["keywords"]) * strength))
        keywords = vocab["keywords"][:keyword_count]
    else:
        keywords = vocab["keywords"]

    return {
        "nearest_type": nearest_type,
        "distance": round(min_dist, 4),
        "keywords": keywords,
        "optical_properties": vocab["optical_properties"],
        "color_associations": vocab["color_associations"],
        "strength": strength
    }


def _generate_composite_prompt(
    state: dict,
    style_modifier: str = ""
) -> str:
    """
    Generate a single blended image-generation prompt from drone state.

    Pure Layer 2 deterministic operation - 0 tokens.
    """
    vocab = _extract_drone_visual_vocabulary(state, strength=1.0)
    parts = []

    if style_modifier:
        parts.append(style_modifier)

    parts.extend(vocab["keywords"])

    # Add dominant optical property
    props = vocab["optical_properties"]
    parts.append(f"{props['finish']} finish")
    parts.append(f"{props['depth_of_field']} depth of field")

    # Add 1-2 color associations
    colors = vocab["color_associations"][:2]
    parts.extend(colors)

    return ", ".join(parts)


# ============================================================================
# LAYER 2: DETERMINISTIC CAUSTIC SPIN TRANSFORMATION
# ============================================================================

def caustic_spin_transform(
    image_data: bytes,
    CP: float,
    frame_offset: float,
    rotation_speed: float = 1.0,
    output_depth: bool = True
) -> Tuple[bytes, Optional[bytes], Dict]:
    """
    Apply two-phase caustic spin transformation to image.
    
    Pure Layer 2 deterministic operation - 0 tokens.
    Based on rotating slit filter optical simulation.
    
    Args:
        image_data: Input image bytes (PNG/JPEG)
        CP: Color palette density [0.042, 0.75] (validated compositional range)
        frame_offset: Temporal phase offset [0, 360]
        rotation_speed: Angular velocity multiplier
        output_depth: Return z_depth visualization
    
    Returns:
        (transformed_image_bytes, depth_image_bytes, metadata)
    """
    # Load and resize image
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img = img.resize((400, 300))
    w, h = img.size
    
    # Build HSV representation
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            r, g, b = img.getpixel((x, y))
            H, S, V = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            hsv[y, x] = (H*255, S, V)
    
    # Compute slit rotation speed
    slit_speed = (frame_offset / CP) * rotation_speed * 4
    
    # Two-phase caustic spin (forward and reverse interference)
    forward_phase = np.zeros((h, w), dtype=np.float32)
    reverse_phase = np.zeros((h, w), dtype=np.float32)
    
    for y in range(h):
        for x in range(w):
            dx = x - w // 2
            dy = y - h // 2
            angle = math.atan2(dy, dx)
            
            # Opposing phase rotations create interference
            forward_phase[y, x] = 0.5 * (1 + math.cos(angle - slit_speed * frame_offset))
            reverse_phase[y, x] = 0.5 * (1 + math.cos(angle + slit_speed * frame_offset))
    
    # Z-depth estimation from phase interference
    z_depth = np.abs(forward_phase - reverse_phase)
    
    # Apply depth-modulated hue transformation
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            H, S, V = hsv[y, x]
            
            # Depth-modulated hue shift
            hue_mod = (H % 255) / CP
            hue_mod *= z_depth[y, x]
            
            r, g, b = colorsys.hsv_to_rgb(hue_mod, S, V)
            out[y, x] = (int(r*255), int(g*255), int(b*255))
    
    # Convert output to bytes
    out_img = Image.fromarray(out)
    out_buffer = io.BytesIO()
    out_img.save(out_buffer, format='PNG')
    out_bytes = out_buffer.getvalue()
    
    # Optional depth field output
    depth_bytes = None
    if output_depth:
        depth_img = (z_depth * 255).astype(np.uint8)
        depth_pil = Image.fromarray(depth_img)
        depth_buffer = io.BytesIO()
        depth_pil.save(depth_buffer, format='PNG')
        depth_bytes = depth_buffer.getvalue()
    
    # Metadata
    metadata = {
        "CP": CP,
        "frame_offset": frame_offset,
        "rotation_speed": rotation_speed,
        "z_depth_range": {
            "min": float(z_depth.min()),
            "max": float(z_depth.max()),
            "mean": float(z_depth.mean())
        },
        "image_size": {"width": w, "height": h},
        "cost": "0 tokens (pure Layer 2)"
    }
    
    return out_bytes, depth_bytes, metadata


# ============================================================================
# PHASE 2.6: RHYTHMIC PRESETS
# ============================================================================

# Validated rhythmic presets for compositional limit cycle discovery
RHYTHMIC_PRESETS = {
    "aerial_rotation": {
        "description": "Standard aerial rotation cycle (medium CP, optimal basin)",
        "state_a": {
            "color_palette_density": 0.167,  # CP 60 (medium range start)
            "temporal_phase": 0.0,
            "depth_separation": 0.3
        },
        "state_b": {
            "color_palette_density": 0.333,  # CP 120 (medium range end)
            "temporal_phase": 1.0,
            "depth_separation": 0.9
        },
        "pattern": "sinusoidal",
        "num_cycles": 4,
        "steps_per_cycle": 20,
        "use_case": "General aerial cinematography with smooth rotation",
        "visual_effect": "Smooth caustic color modulation with depth pulsing",
        "emergent_attractors": ["Period 10 (0.96 autocorr)", "Period 8 (0.96 autocorr)"]
    },
    
    "low_altitude_sweep": {
        "description": "Low-altitude detail sweep (low CP, fine structure)",
        "state_a": {
            "color_palette_density": 0.042,  # CP 15 (low range)
            "temporal_phase": 0.0,
            "depth_separation": 0.3
        },
        "state_b": {
            "color_palette_density": 0.125,  # CP 45 (low range)
            "temporal_phase": 1.0,
            "depth_separation": 0.9
        },
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 24,
        "use_case": "Close-range inspection with fine detail preservation",
        "visual_effect": "Fine-grained caustic detail, high depth resolution",
        "emergent_attractors": ["Period 10 (0.96 autocorr)", "Period 12 (0.90 autocorr)"]
    },
    
    "thermal_pulse": {
        "description": "Thermal imaging pulse (high CP, novel attractor)",
        "state_a": {
            "color_palette_density": 0.5,  # CP 180 (high range)
            "temporal_phase": 0.0,
            "depth_separation": 0.3
        },
        "state_b": {
            "color_palette_density": 0.75,  # CP 270 (high range)
            "temporal_phase": 1.0,
            "depth_separation": 0.9
        },
        "pattern": "sinusoidal",
        "num_cycles": 5,
        "steps_per_cycle": 16,
        "use_case": "Thermal/IR simulation with heat signature pulsing",
        "visual_effect": "Coarse color bands with strong contrast, Period 14 emergence",
        "emergent_attractors": ["Period 14 (0.95 autocorr)", "Period 10 (0.96 autocorr)"]
    },
    
    "orbit_scan": {
        "description": "Orbital scanning pattern (triangular, linear zoom)",
        "state_a": {
            "color_palette_density": 0.2,
            "temporal_phase": 0.0,
            "depth_separation": 0.2
        },
        "state_b": {
            "color_palette_density": 0.4,
            "temporal_phase": 1.0,
            "depth_separation": 0.8
        },
        "pattern": "triangular",
        "num_cycles": 2,
        "steps_per_cycle": 30,
        "use_case": "Orbital/satellite imagery with linear zoom progression",
        "visual_effect": "Linear depth ramping, mechanical rhythm"
    }
}


# ============================================================================
# MCP TOOLS
# ============================================================================

@mcp.tool()
def apply_caustic_spin(
    image_base64: str,
    CP: float = 33.0,
    frame_offset: float = 0.0,
    rotation_speed: float = 1.0,
    output_depth: bool = True
) -> Dict:
    """
    Apply caustic spin optical transformation to image.
    
    Layer 2 deterministic operation - 0 tokens.
    Simulates rotating slit filter during long exposure.
    
    Args:
        image_base64: Base64-encoded input image
        CP: Color palette density [15, 270] (validated compositional range)
            - Low (15-45): Fine detail, Period 10/12 attractors
            - Medium (60-120): Optimal compositional diversity
            - High (180-270): Coarse bands, novel Period 14 attractor
        frame_offset: Rotation phase [0, 360] degrees
        rotation_speed: Angular velocity multiplier [0.1, 10.0]
        output_depth: Return z_depth field visualization
    
    Returns:
        Dictionary with transformed image, optional depth map, and metadata
    """
    # Decode base64 image
    image_data = base64.b64decode(image_base64)
    
    # Apply transformation
    out_bytes, depth_bytes, metadata = caustic_spin_transform(
        image_data, CP, frame_offset, rotation_speed, output_depth
    )
    
    # Encode outputs
    result = {
        "transformed_image": base64.b64encode(out_bytes).decode('utf-8'),
        "metadata": metadata
    }
    
    if depth_bytes:
        result["depth_field"] = base64.b64encode(depth_bytes).decode('utf-8')
    
    return result


@mcp.tool()
def extract_depth_field(
    image_base64: str,
    CP: float = 33.0,
    frame_offset: float = 0.0
) -> Dict:
    """
    Extract z_depth field from caustic spin interference.
    
    Pure Layer 2 deterministic operation - 0 tokens.
    Returns depth map for compositional use with other aesthetic domains.
    
    Args:
        image_base64: Base64-encoded input image
        CP: Color palette density [15, 270]
        frame_offset: Rotation phase [0, 360]
    
    Returns:
        Depth field statistics and visualization
    """
    image_data = base64.b64decode(image_base64)
    _, depth_bytes, metadata = caustic_spin_transform(
        image_data, CP, frame_offset, rotation_speed=1.0, output_depth=True
    )
    
    return {
        "depth_field_image": base64.b64encode(depth_bytes).decode('utf-8'),
        "depth_statistics": metadata["z_depth_range"],
        "CP": CP,
        "cost": "0 tokens (pure Layer 2)"
    }


@mcp.tool()
def map_drone_parameters(
    imaging_mode: str = "aerial_survey",
    CP_override: Optional[float] = None,
    temporal_phase: Optional[float] = None
) -> Dict:
    """
    Map drone imaging mode to parameter space.
    
    Layer 2 deterministic lookup - 0 tokens.
    
    Args:
        imaging_mode: Drone imaging mode (aerial_survey, low_altitude, thermal_imaging)
        CP_override: Optional CP override [15, 270]
        temporal_phase: Optional phase override [0, 1]
    
    Returns:
        Complete parameter set with compositional metadata
    """
    if imaging_mode not in DRONE_IMAGING_MODES:
        available = ", ".join(DRONE_IMAGING_MODES.keys())
        return {
            "error": f"Unknown imaging mode: {imaging_mode}",
            "available_modes": available
        }
    
    mode_profile = DRONE_IMAGING_MODES[imaging_mode]
    params = mode_profile["default_params"].copy()
    
    # Apply overrides
    if CP_override is not None:
        params["color_palette_density"] = CP_override
    if temporal_phase is not None:
        params["temporal_phase"] = temporal_phase
    
    # Add compositional metadata
    cp_val = params["color_palette_density"]
    cp_actual = cp_val * 360  # Convert density back to CP value
    
    # Determine basin region
    if cp_actual < 60:
        basin_region = "low"
        attractor_profile = "Period 10 (0.96), Period 12 (0.90)"
    elif cp_actual < 180:
        basin_region = "medium_optimal"
        attractor_profile = "Period 8 (0.96), Period 10 (0.96) - strongest basins"
    else:
        basin_region = "high"
        attractor_profile = "Period 14 (0.95) - novel emergent, Period 10 (0.96)"
    
    return {
        "imaging_mode": imaging_mode,
        "display_name": mode_profile["display_name"],
        "parameters": params,
        "CP_actual": round(cp_actual, 1),
        "compositional_basin": basin_region,
        "expected_attractors": attractor_profile,
        "parameter_bounds": DRONE_PARAMETER_SCHEMA,
        "cost": "0 tokens (pure Layer 2)"
    }


@mcp.tool()
def generate_rhythmic_drone_sequence(
    state_a_mode: str = "aerial_survey",
    state_b_mode: str = "thermal_imaging",
    oscillation_pattern: str = "sinusoidal",
    num_cycles: int = 2,
    steps_per_cycle: int = 20,
    phase_offset: float = 0.0
) -> Dict:
    """
    Generate rhythmic oscillation between two drone imaging modes.
    
    PHASE 2.6: Temporal composition for multi-domain limit cycle discovery.
    Pure Layer 2 deterministic operation - 0 tokens.
    
    Args:
        state_a_mode: Starting imaging mode
        state_b_mode: Alternating imaging mode
        oscillation_pattern: Wave shape (sinusoidal, triangular, square)
        num_cycles: Number of complete A→B→A cycles
        steps_per_cycle: Samples per cycle
        phase_offset: Starting phase [0, 1]
    
    Returns:
        Temporal sequence of drone parameter states
    """
    if state_a_mode not in DRONE_IMAGING_MODES or state_b_mode not in DRONE_IMAGING_MODES:
        return {"error": "Invalid imaging mode"}
    
    state_a = DRONE_IMAGING_MODES[state_a_mode]["default_params"]
    state_b = DRONE_IMAGING_MODES[state_b_mode]["default_params"]
    
    total_steps = num_cycles * steps_per_cycle
    sequence = []
    
    for i in range(total_steps):
        # Compute phase
        phase = (i / steps_per_cycle + phase_offset) % 1.0
        
        # Apply oscillation pattern
        if oscillation_pattern == "sinusoidal":
            t = 0.5 * (1 + math.sin(2 * math.pi * phase - math.pi/2))
        elif oscillation_pattern == "triangular":
            t = 1.0 - abs(2 * phase - 1.0)
        elif oscillation_pattern == "square":
            t = 1.0 if phase < 0.5 else 0.0
        else:
            t = phase  # Linear fallback
        
        # Interpolate parameters
        step_params = {}
        for key in state_a.keys():
            step_params[key] = state_a[key] * (1 - t) + state_b[key] * t
        
        sequence.append(step_params)
    
    return {
        "sequence": sequence,
        "pattern_type": oscillation_pattern,
        "num_cycles": num_cycles,
        "total_steps": total_steps,
        "state_a": state_a_mode,
        "state_b": state_b_mode,
        "cost": "0 tokens (pure Layer 2)"
    }


@mcp.tool()
def apply_drone_rhythmic_preset(
    preset_name: str,
    override_params: Optional[Dict] = None
) -> Dict:
    """
    Apply curated rhythmic drone preset configuration.
    
    PHASE 2.6: Pre-configured rhythmic compositions for common use cases.
    Pure Layer 2 operation - 0 tokens.
    
    Args:
        preset_name: Name of preset (aerial_rotation, low_altitude_sweep, etc.)
        override_params: Optional overrides (num_cycles, steps_per_cycle, etc.)
    
    Returns:
        Complete rhythmic sequence with preset metadata
    """
    if preset_name not in RHYTHMIC_PRESETS:
        available = ", ".join(RHYTHMIC_PRESETS.keys())
        return {
            "error": f"Unknown preset: {preset_name}",
            "available_presets": available
        }
    
    preset = RHYTHMIC_PRESETS[preset_name].copy()
    
    # Extract configuration
    state_a = preset.pop("state_a")
    state_b = preset.pop("state_b")
    pattern = preset.get("pattern", "sinusoidal")
    num_cycles = preset.get("num_cycles", 2)
    steps_per_cycle = preset.get("steps_per_cycle", 20)
    
    # Apply overrides
    if override_params:
        num_cycles = override_params.get("num_cycles", num_cycles)
        steps_per_cycle = override_params.get("steps_per_cycle", steps_per_cycle)
        pattern = override_params.get("oscillation_pattern", pattern)
    
    # Generate sequence
    total_steps = num_cycles * steps_per_cycle
    sequence = []
    
    for i in range(total_steps):
        phase = (i / steps_per_cycle) % 1.0
        
        if pattern == "sinusoidal":
            t = 0.5 * (1 + math.sin(2 * math.pi * phase - math.pi/2))
        elif pattern == "triangular":
            t = 1.0 - abs(2 * phase - 1.0)
        elif pattern == "square":
            t = 1.0 if phase < 0.5 else 0.0
        else:
            t = phase
        
        step_params = {}
        for key in state_a.keys():
            step_params[key] = state_a[key] * (1 - t) + state_b[key] * t
        
        sequence.append(step_params)
    
    return {
        "preset_name": preset_name,
        "sequence": sequence,
        "pattern_type": pattern,
        "num_cycles": num_cycles,
        "steps_per_cycle": steps_per_cycle,
        "total_steps": total_steps,
        "preset_info": preset,
        "cost": "0 tokens (pure Layer 2)"
    }


@mcp.tool()
def list_drone_rhythmic_presets() -> Dict:
    """
    List all available rhythmic drone presets with descriptions.
    
    Returns catalog of preset configurations for multi-domain composition.
    Cost: 0 tokens (pure lookup).
    """
    catalog = {}
    for name, preset in RHYTHMIC_PRESETS.items():
        catalog[name] = {
            "description": preset["description"],
            "pattern": preset["pattern"],
            "cycles": preset["num_cycles"],
            "steps_per_cycle": preset["steps_per_cycle"],
            "use_case": preset.get("use_case", ""),
            "visual_effect": preset.get("visual_effect", ""),
            "emergent_attractors": preset.get("emergent_attractors", [])
        }
    
    return {
        "presets": catalog,
        "total_presets": len(catalog),
        "note": "Use apply_drone_rhythmic_preset to generate sequences"
    }


@mcp.tool()
def list_drone_imaging_modes() -> Dict:
    """List all available drone imaging modes with descriptions."""
    modes = {}
    for mode_id, profile in DRONE_IMAGING_MODES.items():
        modes[mode_id] = {
            "display_name": profile["display_name"],
            "description": profile["description"],
            "default_CP": round(profile["default_params"]["color_palette_density"] * 360, 1)
        }

    return {
        "modes": modes,
        "total_modes": len(modes)
    }


# ============================================================================
# PHASE 2.7: ATTRACTOR VISUALIZATION & PROMPT GENERATION
# ============================================================================

# Discovered attractor presets from compositional limit cycle analysis
# Basin sizes from validated runs; coordinates from representative trajectories
DRONE_ATTRACTOR_PRESETS = {
    "period_20": {
        "name": "Period 20 — Aerial Rotation",
        "description": "Primary aerial cycle from survey↔thermal oscillation",
        "basin_size": 0.12,
        "classification": "harmonic",
        "source_preset": "aerial_rotation",
        "state": {
            "color_palette_density": 0.28,
            "temporal_phase": 0.45,
            "depth_separation": 0.55,
            "altitude_perspective": 0.25,
            "spectral_range": 0.30
        }
    },
    "period_24": {
        "name": "Period 24 — Low Altitude Detail",
        "description": "Fine-grained sweep cycle with depth resolution emphasis",
        "basin_size": 0.09,
        "classification": "harmonic",
        "source_preset": "low_altitude_sweep",
        "state": {
            "color_palette_density": 0.08,
            "temporal_phase": 0.40,
            "depth_separation": 0.65,
            "altitude_perspective": 0.70,
            "spectral_range": 0.08
        }
    },
    "period_16": {
        "name": "Period 16 — Thermal Pulse",
        "description": "Fast thermal oscillation with novel Period 14 sub-attractor",
        "basin_size": 0.07,
        "classification": "harmonic",
        "source_preset": "thermal_pulse",
        "state": {
            "color_palette_density": 0.60,
            "temporal_phase": 0.30,
            "depth_separation": 0.75,
            "altitude_perspective": 0.18,
            "spectral_range": 0.85
        }
    },
    "period_30": {
        "name": "Period 30 — Orbit Scan",
        "description": "Slow orbital triangular sweep, strong depth layering",
        "basin_size": 0.08,
        "classification": "harmonic",
        "source_preset": "orbit_scan",
        "state": {
            "color_palette_density": 0.30,
            "temporal_phase": 0.50,
            "depth_separation": 0.50,
            "altitude_perspective": 0.35,
            "spectral_range": 0.20
        }
    },
    "period_14": {
        "name": "Period 14 — Novel Thermal Emergent",
        "description": "Novel attractor emerging at high CP, not matching any preset period",
        "basin_size": 0.05,
        "classification": "novel",
        "source_preset": None,
        "state": {
            "color_palette_density": 0.70,
            "temporal_phase": 0.20,
            "depth_separation": 0.85,
            "altitude_perspective": 0.12,
            "spectral_range": 0.92
        }
    },
    "bifurcation_edge": {
        "name": "CP Basin Boundary",
        "description": "Curated state at CP medium/high transition boundary",
        "basin_size": None,
        "classification": "curated",
        "source_preset": None,
        "state": {
            "color_palette_density": 0.45,
            "temporal_phase": 0.0,
            "depth_separation": 0.70,
            "altitude_perspective": 0.40,
            "spectral_range": 0.50
        }
    }
}


@mcp.tool()
def extract_drone_visual_vocabulary(
    state: Dict,
    strength: float = 1.0
) -> Dict:
    """
    Extract visual vocabulary from drone parameter coordinates.

    PHASE 2.7: Maps a 5D parameter state to the nearest canonical
    drone visual type and returns image-generation-ready keywords.

    Uses nearest-neighbor matching against 4 visual types derived
    from the drone optics morphospace.

    Args:
        state: Parameter coordinates dict with keys:
            color_palette_density, temporal_phase, depth_separation,
            altitude_perspective, spectral_range
        strength: Keyword weight multiplier [0.0, 1.0] (default: 1.0)

    Returns:
        Dict with nearest_type, distance, keywords,
        optical_properties, color_associations

    Cost: 0 tokens (pure Layer 2 computation)
    """
    return _extract_drone_visual_vocabulary(state, strength)


@mcp.tool()
def generate_drone_attractor_prompt(
    attractor_id: str = "",
    custom_state: Optional[Dict] = None,
    mode: str = "composite",
    style_modifier: str = "",
    keyframe_count: int = 4
) -> Dict:
    """
    Generate image generation prompt from drone attractor state or custom coordinates.

    PHASE 2.7: Translates mathematical attractor coordinates into visual prompts
    suitable for image generation (ComfyUI, Stable Diffusion, DALL-E, etc.).

    Modes:
        composite: Single blended prompt from attractor state
        split_view: Separate vocabulary analysis per visual dimension
        sequence: Multiple keyframe prompts from source preset trajectory

    Args:
        attractor_id: Preset attractor name (period_20, period_24, etc.)
            Use "" with custom_state for arbitrary coordinates.
        custom_state: Optional custom parameter coordinates dict.
            Overrides attractor_id if provided.
        mode: "composite" | "split_view" | "sequence"
        style_modifier: Optional prefix ("photorealistic", "oil painting", etc.)
        keyframe_count: Number of keyframes for sequence mode (default: 4)

    Returns:
        Dict with prompt(s), vocabulary details, and attractor metadata

    Cost: 0 tokens (Layer 2 deterministic)
    """
    # Resolve state
    attractor_meta = None
    if custom_state:
        state = custom_state
    elif attractor_id and attractor_id in DRONE_ATTRACTOR_PRESETS:
        preset = DRONE_ATTRACTOR_PRESETS[attractor_id]
        state = preset["state"]
        attractor_meta = {
            "name": preset["name"],
            "description": preset["description"],
            "basin_size": preset["basin_size"],
            "classification": preset["classification"]
        }
    else:
        available = list(DRONE_ATTRACTOR_PRESETS.keys())
        return {
            "error": "Provide custom_state or valid attractor_id",
            "available_attractors": available
        }

    if mode == "composite":
        prompt = _generate_composite_prompt(state, style_modifier)
        vocab = _extract_drone_visual_vocabulary(state)

        result = {
            "prompt": prompt,
            "vocabulary": vocab,
            "mode": "composite"
        }
        if attractor_meta:
            result["attractor"] = attractor_meta
        return result

    elif mode == "split_view":
        dimensions = {}

        # High-altitude emphasis
        high_alt_state = dict(state)
        high_alt_state["altitude_perspective"] = 0.05
        dimensions["overhead_component"] = _extract_drone_visual_vocabulary(
            high_alt_state, strength=0.7
        )

        # Oblique emphasis
        oblique_state = dict(state)
        oblique_state["altitude_perspective"] = 0.85
        dimensions["oblique_component"] = _extract_drone_visual_vocabulary(
            oblique_state, strength=0.7
        )

        # Spectral emphasis
        spectral_state = dict(state)
        spectral_state["spectral_range"] = min(
            1.0, state.get("spectral_range", 0.5) + 0.3
        )
        dimensions["spectral_component"] = _extract_drone_visual_vocabulary(
            spectral_state, strength=0.6
        )

        # Depth emphasis
        depth_state = dict(state)
        depth_state["depth_separation"] = min(
            1.0, state.get("depth_separation", 0.5) + 0.3
        )
        dimensions["depth_component"] = _extract_drone_visual_vocabulary(
            depth_state, strength=0.6
        )

        result = {
            "dimensions": dimensions,
            "base_vocabulary": _extract_drone_visual_vocabulary(state),
            "mode": "split_view"
        }
        if attractor_meta:
            result["attractor"] = attractor_meta
        return result

    elif mode == "sequence":
        source_preset_name = None
        if attractor_meta and attractor_id in DRONE_ATTRACTOR_PRESETS:
            source_preset_name = DRONE_ATTRACTOR_PRESETS[attractor_id].get(
                "source_preset"
            )

        if source_preset_name and source_preset_name in RHYTHMIC_PRESETS:
            preset_cfg = RHYTHMIC_PRESETS[source_preset_name]
            state_a = preset_cfg["state_a"]
            state_b = preset_cfg["state_b"]
            pattern = preset_cfg.get("pattern", "sinusoidal")

            keyframes = []
            for k in range(keyframe_count):
                phase = k / keyframe_count
                if pattern == "sinusoidal":
                    t = 0.5 * (1 + math.sin(2 * math.pi * phase - math.pi / 2))
                elif pattern == "triangular":
                    t = 1.0 - abs(2 * phase - 1.0)
                elif pattern == "square":
                    t = 1.0 if phase < 0.5 else 0.0
                else:
                    t = phase

                # Interpolate preset states
                kf_state_3p = {}
                for key in state_a:
                    kf_state_3p[key] = state_a[key] * (1 - t) + state_b[key] * t

                # Expand to 5D using attractor state as base
                kf_state = dict(state)
                kf_state.update(kf_state_3p)

                prompt = _generate_composite_prompt(kf_state, style_modifier)
                vocab = _extract_drone_visual_vocabulary(kf_state)

                keyframes.append({
                    "step": k,
                    "phase": round(phase, 3),
                    "interpolation_t": round(t, 4),
                    "prompt": prompt,
                    "vocabulary": vocab
                })

            result = {
                "keyframes": keyframes,
                "source_preset": source_preset_name,
                "keyframe_count": keyframe_count,
                "mode": "sequence"
            }
            if attractor_meta:
                result["attractor"] = attractor_meta
            return result
        else:
            prompt = _generate_composite_prompt(state, style_modifier)
            vocab = _extract_drone_visual_vocabulary(state)
            result = {
                "prompt": prompt,
                "vocabulary": vocab,
                "mode": "composite",
                "note": "No source preset for sequence mode; fell back to composite"
            }
            if attractor_meta:
                result["attractor"] = attractor_meta
            return result

    else:
        return {
            "error": f"Unknown mode: {mode}. Use composite, split_view, or sequence."
        }


@mcp.tool()
def list_drone_attractor_presets() -> Dict:
    """
    List all available drone attractor presets for visualization.

    PHASE 2.7: Shows discovered and curated attractor configurations
    available for prompt generation.

    Returns:
        Dict with preset names, descriptions, basin sizes, and classifications

    Cost: 0 tokens
    """
    catalog = {}
    for preset_id, preset in DRONE_ATTRACTOR_PRESETS.items():
        catalog[preset_id] = {
            "name": preset["name"],
            "description": preset["description"],
            "basin_size": preset["basin_size"],
            "classification": preset["classification"],
            "source_preset": preset.get("source_preset"),
            "state": preset["state"]
        }

    return {
        "presets": catalog,
        "total_presets": len(catalog),
        "visual_types": list(DRONE_VISUAL_VOCABULARY.keys()),
        "note": "Use generate_drone_attractor_prompt to create image prompts"
    }


@mcp.tool()
def get_drone_morphospace_info() -> Dict:
    """
    Get normalized 5D morphospace definition for multi-domain composition.

    Returns parameter space definition, canonical states, and visual types
    needed for domain registry integration and Tier 4D limit cycle discovery.

    Cost: 0 tokens (pure taxonomy lookup)
    """
    return {
        "domain_id": "drone_optics",
        "display_name": "Drone Optics",
        "mcp_server": "drone-optics",
        "parameter_names": DRONE_PARAMETER_NAMES,
        "parameter_count": len(DRONE_PARAMETER_NAMES),
        "canonical_states": {
            name: coords for name, coords in DRONE_STATE_COORDINATES.items()
        },
        "canonical_state_count": len(DRONE_STATE_COORDINATES),
        "visual_types": list(DRONE_VISUAL_VOCABULARY.keys()),
        "visual_type_count": len(DRONE_VISUAL_VOCABULARY),
        "preset_periods": [
            RHYTHMIC_PRESETS[p]["steps_per_cycle"]
            for p in RHYTHMIC_PRESETS
        ],
        "domain_registry_snippet": {
            "domain_id": "drone_optics",
            "display_name": "Drone Optics",
            "mcp_server": "drone-optics",
            "parameter_names": DRONE_PARAMETER_NAMES,
            "presets": {
                name: {
                    "period": cfg["steps_per_cycle"],
                    "pattern": cfg["pattern"],
                }
                for name, cfg in RHYTHMIC_PRESETS.items()
            }
        }
    }


@mcp.tool()
def get_server_info() -> Dict:
    """Get information about the Drone Optics MCP server."""
    return {
        "name": "drone-optics-mcp",
        "version": "2.7.0",
        "description": "Aerial photography aesthetics through caustic optics simulation",
        "phase": "2.7 - Attractor Visualization + Rhythmic Composition",
        "validated_cp_range": "15-270 (compositional basin testing with microscopy)",
        "optimal_cp_range": "60-120 (medium range, strongest attractors)",
        "layer_architecture": {
            "layer_1": "Taxonomy (drone imaging modes, 7 canonical states)",
            "layer_2": "Deterministic caustic spin (0 tokens)",
            "layer_2_7": "Visual vocabulary extraction + prompt generation (0 tokens)",
            "layer_3": "Claude synthesis (prompt enhancement)"
        },
        "phase_2_6_features": {
            "rhythmic_presets": len(RHYTHMIC_PRESETS),
            "emergent_attractors": "Period 8, 10, 12, 14 (validated with microscopy)",
            "peak_autocorrelation": 0.96
        },
        "phase_2_7_features": {
            "attractor_visualization": True,
            "parameter_dimensions": len(DRONE_PARAMETER_NAMES),
            "canonical_states": len(DRONE_STATE_COORDINATES),
            "visual_types": len(DRONE_VISUAL_VOCABULARY),
            "attractor_presets": len(DRONE_ATTRACTOR_PRESETS),
            "supported_modes": ["composite", "split_view", "sequence"],
            "prompt_generation": True
        },
        "multi_domain_composition": {
            "parameter_names": DRONE_PARAMETER_NAMES,
            "preset_periods": sorted(set(
                p["steps_per_cycle"] for p in RHYTHMIC_PRESETS.values()
            )),
            "domain_registry_ready": True
        },
        "based_on": "Rotating slit filter optical physics",
        "reference": "cp_basin_analysis.md"
    }

if __name__ == "__main__":
    mcp.run()
