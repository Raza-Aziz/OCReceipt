import json
import os

# Save to JSON file
def save_to_json(data, filename="transactions.json"):
    try:
        # Load existing data if file exists
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        # Append new data
        existing_data.append(data)
        
        # Save back to file
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        return True, filename
    except Exception as e:
        return False, str(e)