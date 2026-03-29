"""Global configuration for the avatar generation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the avatar generation pipeline."""

    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    # Data directories
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def flame_dir(self) -> Path:
        return self.data_dir / "flame"

    @property
    def deca_dir(self) -> Path:
        return self.data_dir / "deca"

    @property
    def arkit_reference_dir(self) -> Path:
        return self.data_dir / "arkit_reference"

    @property
    def bust_template_dir(self) -> Path:
        return self.data_dir / "bust_template"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "output"

    @property
    def vendor_dir(self) -> Path:
        return self.project_root / "vendor"

    # FLAME model settings (FLAME 2023 Open: 300 shape + 100 expression)
    flame_model_path: str = "generic_model.pkl"
    flame_num_shape_params: int = 300
    flame_num_expression_params: int = 100

    # Texture settings
    texture_resolution: int = 1024

    # Processing
    device: str = "cpu"  # cpu or mps

    # Image preprocessing
    input_image_size: int = 224  # DECA input size
    face_crop_margin: float = 0.3  # Extra margin around detected face

    # Export
    export_format: str = "GLB"


# Singleton config
config = PipelineConfig()
