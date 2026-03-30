"""Download PubChem CID-SMILES and CID-IUPAC files."""
import urllib.request
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/"

FILES = ["CID-SMILES.gz", "CID-IUPAC.gz"]


def download():
    os.makedirs(RAW_DIR, exist_ok=True)
    for fname in FILES:
        dest = os.path.join(RAW_DIR, fname)
        if os.path.exists(dest):
            print(f"Already exists: {dest}")
            continue
        url = BASE_URL + fname
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")


if __name__ == "__main__":
    download()
