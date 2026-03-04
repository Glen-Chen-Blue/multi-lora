import json

# Read the original metadata file
with open('lora_metadata.json', 'r') as f:
    metadata = json.load(f)

# Remove substitutes from all entries
for key in metadata:
    if isinstance(metadata[key], dict) and 'substitutes' in metadata[key]:
        metadata[key]['substitutes'] = []

# Write to new file
with open('lora_metadata_without_substitutes.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Created lora_metadata_without_substitutes.json")