# hyperspectral-potassium-deficiency-detection
# 🌿 FAU Farm Owls SpectraSense

Hyperspectral potassium deficiency detector for mango trees.

---

## For Farmers — Quick Start

1. Download `FAU Farm Owls SpectraSense.exe` from the releases page
2. Double-click it to open the app — no installation needed
3. Click **Select Leaf Files** or **Select Folder** to load your captures
4. Click **Run Analysis**
5. Results are colour coded:
   - 🟢 **Healthy** — leaf has sufficient potassium
   - 🔴 **Deficient** — leaf is potassium deficient
   - 🟡 **Uncertain** — borderline, re-capture recommended
6. Double-click any result to see its spectral chart
7. Click **Open Results CSV** to view the full results log in Excel

---

## Before Each Field Session

- Point your hyperspectral camera at your **white spectralon panel**
- Save that capture as `white_reference.txt` and load it in the app
- Capture your leaf spectra as normal
- The app will calibrate automatically

---

## For Developers — Setup

### Requirements
- Python 3.12+
- Dependencies listed in `requirements.txt`

### Install
```bash
git clone https://github.com/yourusername/mango-leaf-analyzer.git
cd mango-leaf-analyzer
pip install -r requirements.txt
```

### Run the GUI
```bash
python gui.py
```

### Run the pipeline via terminal
```bash
python main.py --input ./camera_data --output ./results/field_results.csv
```

### Run tests
```bash
pytest tests/ -v
```

### Build the exe
```bash
pyinstaller "FAU Farm Owls SpectraSense.spec"
```


## How It Works

Every leaf has a unique spectral fingerprint — a curve showing how much light it reflects at each wavelength. Potassium deficient leaves look different from healthy ones because K-deficiency changes the leaf's chlorophyll and cell structure, altering how it absorbs and reflects light especially in the visible range (400–700 nm).

The classifier compares each sample against both baselines across three spectral windows using Spectral Angle Mapper (SAM) and Pearson correlation, weighted by how discriminative each band is for potassium deficiency.

| Band | Range | Weight | Reason |
|------|-------|--------|--------|
| Visible | 400–700 nm | 55% | Largest difference between baselines |
| Red Edge | 680–750 nm | 30% | Strong K-deficiency signal |
| NIR | 750–1010 nm | 15% | Less discriminative |

---

## Built By

FAU Farm Owls — Florida Atlantic University



