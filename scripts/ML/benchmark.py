import os
import sys
import sqlite3
import json
import tempfile
import spider.evaluation as sp
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ROOT_PATH = os.environ.get("ROOT_PATH")

if not ROOT_PATH:
    raise ValueError("ROOT_PATH environment variable not set. Please set it in your .env file.")


def main(model, strategy, template, predict_file):
    BRONZE_DB = f"{ROOT_PATH}/database/bronze/bronze.sqlite"

    conn_bronze = sqlite3.connect(BRONZE_DB)
    conn_bronze.row_factory = sqlite3.Row

    cursor_bronze = conn_bronze.cursor()
    cursor_bronze.execute("SELECT * FROM spider_tables")
    rows = cursor_bronze.fetchall()

    json_fields = ['table_names', 'table_names_original', 'column_names', 
                'column_names_original', 'column_types', 'primary_keys', 'foreign_keys']

    template_folder = template.removesuffix('.j2')
    
    json_data = []
    for row in rows:
        row_dict = dict(row)
        
        for field in json_fields:
            if field in row_dict and row_dict[field]:
                try:
                    row_dict[field] = json.loads(row_dict[field])
                except json.JSONDecodeError:
                    pass
        
        json_data.append(row_dict)

    json_string = json.dumps(json_data, indent=2)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
        temp_file.write(json_string)
        temp_file_path = temp_file.name

    KMAPS = sp.build_foreign_key_map_from_json(temp_file_path)
    os.unlink(temp_file_path)

    predict_file = predict_file.removesuffix('.sql')
    gold_file, finetuned = predict_file.split('_predictions_')
    DB_DIR = f"{ROOT_PATH}/database/spider"
    GOLD = f"{ROOT_PATH}/data/training/{strategy}/{template_folder}/{gold_file+'.sql'}"
    PREDICT = f"{ROOT_PATH}/data/predictions/{strategy}/{template_folder}/{model.removeprefix('mlx-community/')}/{predict_file+'.sql'}"
    ETYPE = "all" # all, easy, medium, hard

    output_file=f"{ROOT_PATH}/data/benchmark/{strategy}/{template_folder}/{model.removeprefix('mlx-community/')}/{gold_file+'_benchmark_'+finetuned+'.txt'}"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        sp.evaluate(gold=GOLD , predict=PREDICT, db_dir=DB_DIR, etype=ETYPE, kmaps=KMAPS)
        sys.stdout = original_stdout
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark nl2SQL or nl2NatSQL models')
    parser.add_argument('--model', type=str, required=True,
                       help='Model to fine-tune')
    parser.add_argument('--strategy', type=str, required=True, choices=['nl2SQL', 'nl2NatSQL'],
                       help='Type of model to create dataset for')
    parser.add_argument('--template', type=str, required=True,
                       help='Name of the template file without')
    parser.add_argument('--prediction-file', type=str, required=True,
                       help='Input file for predicted queries. MUST be a sql file')
    args = parser.parse_args()
    main(args.model, args.strategy, args.template, args.prediction_file)