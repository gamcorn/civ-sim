from enum import Enum


class ResourceType(Enum):
    FOOD = "food"
    WATER = "water"
    WOOD = "wood"
    MINERALS = "minerals"


# Noise scale per resource — lower = broader patches
NOISE_SCALE = {
    ResourceType.FOOD: 0.06,
    ResourceType.WATER: 0.04,
    ResourceType.WOOD: 0.05,
    ResourceType.MINERALS: 0.08,
}
