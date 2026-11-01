"""Ocean noise generation at the Pi sample rate."""
from src.ambient_sounds import AmbientSoundGenerator


class AmbientSoundGeneratorRPiZ2W(AmbientSoundGenerator):
    """Shares the continuous filter and avoids sub-audible output."""
