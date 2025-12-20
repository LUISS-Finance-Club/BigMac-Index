import requests

URL = "https://raw.githubusercontent.com/TheEconomist/big-mac-data/master/output-data/big-mac-full-index.csv"
OUT = "big-mac-source-data-v2.csv"  # path used by your Streamlit app

def main():
    r = requests.get(URL)
    r.raise_for_status()
    with open(OUT, "wb") as f:
        f.write(r.content)
    print("Updated Big Mac data:", OUT)

if __name__ == "__main__":
    main()
