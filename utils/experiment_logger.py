import json
from datetime import datetime
import os

def log_experiment(name, goal, model, temperature, token_limit, top_k, top_p, prompt, output, save_dir="experiments"):
    os.makedirs(save_dir, exist_ok=True)
    datetime_now = datetime.now()
    timestamp = datetime_now.isoformat()

    log = {
        "name": name,
        "goal": goal,
        "model": model,
        "temperature": temperature,
        "token_limit": token_limit,
        "top_k": top_k,
        "top_p": top_p,
        "prompt": prompt,
        "output": output if isinstance(output, list) else [output],
        "timestamp": timestamp
    }

    filename = f"{save_dir}/exp_{datetime_now.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(log, f, indent=4)

    print(f"[✔] Logged experiment to {filename}")