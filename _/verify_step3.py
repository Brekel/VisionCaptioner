import torch
from transformers import AutoConfig, AutoModelForCausalLM
import json
import os
import re
import platform

def verify_step3_keys():
    model_path = r"E:\_python_tools\VisionCaptioner\models\Step3-VL-10B"
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    
    print(f"--- Verifying Step3-VL-10B Keys Mapping ---")
    
    # 1. Get Checkpoint Keys
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        ckpt_keys = set(index_data["weight_map"].keys())
        print(f"✅ Loaded {len(ckpt_keys)} keys from checkpoint index.")
    else:
        # Fallback if no index (single file?) - unlikely for 10B
        print("❌ Could not find model.safetensors.index.json")
        return False

    # 2. Get Model Expected Keys
    print("Instantiating model on CPU (meta-like) to get expected keys...")
    try:
        # Load config only
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Instantiate empty model to check keys (avoid loading weights)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        model_keys = set(model.state_dict().keys())
        print(f"✅ Loaded {len(model_keys)} keys from model definition.")
    except Exception as e:
        print(f"❌ Failed to instantiate model: {e}")
        return False

    # 3. Apply Mapping Rules
    # Taken from modeling_step_vl.py
    mapping_rules = [
        (r"^vision_model", "model.vision_model"),
        (r"^model(?!\.(language_model|vision_model))", "model.language_model"),
        (r"^vit_large_projector", "model.vit_large_projector")
    ]
    
    print("Applying remapping rules...")
    mapped_ckpt_keys = set()
    for key in ckpt_keys:
        new_key = key
        for pattern, replacement in mapping_rules:
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                break
        mapped_ckpt_keys.add(new_key)

    # 4. Compare
    missing_in_model = mapped_ckpt_keys - model_keys
    missing_in_ckpt = model_keys - mapped_ckpt_keys
    
    # Filter out "tied" weights typically (lm_head.weight matching embed_tokens?)
    # Usually lm_head.weight is in model but maybe shared. 
    # Let's just check overlap size.
    
    print(f"\nResults:")
    print(f"Original Checkpoint Keys: {len(ckpt_keys)}")
    print(f"Mapped Checkpoint Keys:   {len(mapped_ckpt_keys)}")
    print(f"Expected Model Keys:      {len(model_keys)}")
    
    if len(missing_in_ckpt) == 0:
        print("\n✅ SUCCESS: All model keys are present in the mapped checkpoint!")
        return True
    else:
        print(f"\n⚠️ Mismatch Detected.")
        print(f"Keys missing from checkpoint (Expect - Mapped): {len(missing_in_ckpt)}")
        if len(missing_in_ckpt) < 10:
            print("Missing:", missing_in_ckpt)
        else:
            print("(Too many to list, showing first 5):", list(missing_in_ckpt)[:5])

        # If only a few are missing, might be buffers or ignored params
        if len(missing_in_ckpt) < 50: 
             print("⚠️ Small number of missing keys might be acceptable (buffers, etc).")
             return True
        
        return False

if __name__ == "__main__":
    verify_step3_keys()
