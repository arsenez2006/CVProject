import os
import urllib.request
import zipfile
import argparse

def download_dataset(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Пример для PennFudanPed или вашего датасета
    # (можете заменить URL на ваш источник или оставить логику загрузки архива)
    dataset_url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    zip_path = os.path.join(output_dir, "PennFudanPed.zip")
    
    print(f"Downloading dataset from {dataset_url}...")
    urllib.request.urlretrieve(dataset_url, zip_path)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        
    # Удаляем архив после распаковки, чтобы не занимать лишнее место
    os.remove(zip_path)
    print(f"Dataset successfully downloaded and extracted to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw", help="Output directory for raw data")
    args = parser.parse_args()
    
    download_dataset(args.output)
