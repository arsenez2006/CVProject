import os
import urllib.request
import zipfile
import argparse
import certifi
import ssl

def download_dataset(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    zip_path = os.path.join(output_dir, "PennFudanPed.zip")

    context = ssl.create_default_context(cafile=certifi.where())
    
    print(f"Downloading dataset from {dataset_url}...")
    # urllib.request.urlretrieve(dataset_url, zip_path)
    with urllib.request.urlopen(dataset_url, context=context) as response:
      with open(zip_path, "wb") as out_file:
        out_file.write(response.read())
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        
    os.remove(zip_path)
    print(f"Dataset successfully downloaded and extracted to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw", help="Output directory for raw data")
    args = parser.parse_args()
    
    download_dataset(args.output)
