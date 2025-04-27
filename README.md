# MoonBit Doc VectorDB

A Vector DB for MoonBit Official Document based on Milvus Lite.

## Method

![](./method.svg)

## Usage

```bash
git clone https://github.com/MoonBit-Dev/MoonBit-Doc-VectorDB.git
cd MoonBit-Doc-VectorDB
pip install -r requirements.txt
npm install
git submodule update --init --recursive
python scripts/unity.py
node scripts/extract.js processed
python scripts/complement.py
docker compose up -d
python scripts/create-vecdb.py
python scripts/search-vecdb.py # Just for test
```

## Contact Us

- sunyixua@kean.edu
- liboy@kean.edu
- dengzih@kean.edu
