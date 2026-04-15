# ------------------------
# TOOLS DEFINITIONS
# ------------------------

def analyze_material(material):
    db = {
        "xenonite": "Ultra-strong fictional material. High heat resistance. Likely ideal for spacecraft hulls.",
        "steel": "Strong structural metal. Heavy. Good for mechanical frames.",
        "aluminum": "Lightweight metal. Good balance of strength and mass.",
        "copper": "Excellent conductor. Useful for wiring and heat transfer.",
        "ice": "Poor structural material in warm environments. Useful for radiation shielding in space."
    }

    key = material.lower()

    return db.get(
        key,
        f"No clue about the material {material}. Likely unknown alien material."
    )


# ------------------------
# TOOLS REGISTRY
# ------------------------
TOOLS_REGISTRY = {
    "analyze_material": {
        "function": analyze_material,
        "description": "Retrieve detailed properties and use cases for a specific material.",
        "parameters": {
            "material": "The name of the material to analyze (e.g., 'xenonite', 'steel')"
        }
    }
}
