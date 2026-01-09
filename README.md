# 🔬 Colon Length Measurement Tool

> A simple, accurate tool to measure colon length from images using computer vision and AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [What You Need](#-what-you-need)
- [Installation](#-installation)
  - [Step 1: Download the Code](#step-1-download-the-code)
  - [Step 2: Set Up Python Environment](#step-2-set-up-python-environment)
  - [Step 3: Install Required Libraries](#step-3-install-required-libraries)
  - [Step 4: Download SAM Model](#step-4-download-sam-model-only-for-sampy)
- [How to Use](#-how-to-use)
  - [Method 1: Simple Method (no_sam.py)](#method-1-simple-method-no_sampy)
  - [Method 2: AI-Powered Method (sam.py)](#method-2-ai-powered-method-sampy)
- [Understanding Your Results](#-understanding-your-results)
- [Which Method Should I Use?](#-which-method-should-i-use)
- [Tips for Best Results](#-tips-for-best-results)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)

---

## 🎯 Overview

This tool helps researchers and veterinarians measure colon length accurately from images. Simply take a photo of a colon with a ruler, click a few points, and get precise measurements in millimeters and centimeters.

**Two methods available:**
- **Classic CV** (`no_sam.py`) - Fast, simple, no AI required
- **AI-Powered** (`sam.py`) - More robust, handles complex backgrounds

---

## 💻 What You Need

- 🐍 **Python** (version 3.8, 3.9, 3.10, or 3.11)
- 💾 **Operating System:** Windows, Mac, or Linux
- 🖼️ **An image** of a colon with a visible ruler/scale
- ⚡ **Optional:** GPU for faster processing (CPU works fine too!)

---

## 🚀 Installation

### Step 1: Download the Code

```bash
git clone https://github.com/siusiubeom/colon_length_calculator.git
cd colon_length
git submodule update --init --recursive
```

### Step 2: Set Up Python Environment

#### 🪟 Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### 🍎 Mac / 🐧 Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Libraries

```bash
pip install numpy opencv-python matplotlib scikit-image networkx torch torchvision
```

> ⏱️ This may take a few minutes depending on your internet connection

### Step 4: Download SAM Model (Only for sam.py)

> ⚠️ **Only needed if you plan to use the AI-powered method (sam.py)**

1. **Download one of these model files:**
   - ✅ **[sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)** (recommended - ~375 MB)
   - 📦 [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) (~1.2 GB)
   - 📦 [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (~2.4 GB)

2. **Create a folder and save the file:**
   ```bash
   mkdir checkpoints
   # Move your downloaded .pth file here
   ```

   Final path should be: `checkpoints/sam_vit_b_01ec64.pth`

---

## 📖 How to Use

### Method 1: Simple Method (no_sam.py)

**✨ Best for:** Clean images with simple backgrounds

#### Basic Usage

```bash
python no_sam.py --image your_image.jpg
```

**What happens:**
1. 🖱️ Click 2 points on the ruler/scale in your image
2. ⌨️ Enter the real distance between those points (in millimeters)
3. ✅ Get your measurement!

ASIDE FOR QUICK CHECKS, NO USAGE OF THIS IS RECOMMENDED

#### Recommended Usage (Higher Accuracy)

```bash
python no_sam.py --image your_image.jpg --pick-ends
```

This lets you also click the start and end points of the colon for more accurate measurements.

#### Save Results

```bash
python no_sam.py --image your_image.jpg --pick-ends --save-dir results/
```

Creates a `results/` folder with:
- 🖼️ Mask image
- 🦴 Skeleton image
- 🎨 Overlay visualization
- 📄 JSON file with measurements

---

### Method 2: AI-Powered Method (sam.py)

**🤖 Best for:** Complex images with cluttered backgrounds or challenging lighting

#### Basic Usage

```bash
python sam.py --image your_image.jpg --known-mm 100 --checkpoint checkpoints/sam_vit_b_01ec64.pth
```

**What happens:**
You'll click **exactly 4 points** in this order:
1. 📏 First point on the ruler/scale
2. 📏 Second point on the ruler/scale
3. 🔵 Start of the colon (cecum side)
4. 🔴 End of the colon (rectum side)

> 💡 **Tip:** The distance in `--known-mm` is the real-world distance between your two ruler points

#### With Debug Visualization

```bash
python sam.py --image your_image.jpg --known-mm 100 --checkpoint checkpoints/sam_vit_b_01ec64.pth --debug
```

Shows you a visual breakdown of the segmentation and measurement process.

#### Save Results

```bash
python sam.py --image your_image.jpg --known-mm 100 --checkpoint checkpoints/sam_vit_b_01ec64.pth --save-dir results/
```

---

## 📊 Understanding Your Results

Both methods create a **JSON file** with your measurements:

```json
{
  "image": "your_image.jpg",
  "mm_per_pixel": 0.0214,
  "length_px": 5234.8,
  "length_mm": 112.3,
  "length_cm": 11.23,
  "method": "sam_mask_skeleton_between_clicked_ends"
}
```

### Key Values:
- 📏 **length_mm**: Colon length in millimeters
- 📐 **length_cm**: Colon length in centimeters  
- 🔍 **mm_per_pixel**: Scale factor from your ruler
- 🖼️ **length_px**: Length in pixels (raw measurement)

### Output Files (when using --save-dir):
- 🎭 **`*_mask.png`** - Shows what the tool identified as the colon
- 🦴 **`*_skel.png`** - Shows the centerline path used for measurement
- 🎨 **`*_overlay.png`** - Original image with measurements overlaid
- 📄 **`*_result.json`** - All measurement data

---

## 🤔 Which Method Should I Use?

| Your Situation | Recommended Method | Why? |
|----------------|-------------------|------|
| Clean, simple background | 🔵 `no_sam.py` | Faster, simpler |
| Cluttered/complex background | 🤖 `sam.py` | AI handles complexity better |
| Need highest accuracy | 🤖 `sam.py` | More robust segmentation |
| Don't want to download AI models | 🔵 `no_sam.py` | No large downloads needed |
| Processing many images | 🔵 `no_sam.py` | Faster processing |

---

## 💡 Tips for Best Results

1. ✅ **Always include a ruler or scale** in your image
2. 🎯 **Click precisely** on the exact endpoints of your scale
3. 📍 **For sam.py:** Click the colon endpoints directly on the tissue, not on the background
4. 📸 **Take clear photos** with good, even lighting
5. 📏 **Keep the colon as straight as possible** in the image (though curves are handled)
6. 🔍 **Higher resolution images** generally give more accurate results
7. 📐 **Use a known scale** - metric rulers work best (cm/mm)

---

## 🔧 Troubleshooting

### ❌ "Could not read image"
- ✔️ Check that your image path is correct
- ✔️ Make sure the image file isn't corrupted
- ✔️ Try using the full file path instead of relative path

### ❌ "Skeleton too small" 
- ✔️ Try clicking the colon endpoints more precisely on the tissue
- ✔️ Make sure your image shows the entire colon clearly
- ✔️ Increase image resolution if possible

### ❌ Sam.py not working
- ✔️ Verify you downloaded the correct `.pth` file
- ✔️ Check that the checkpoint path matches where you saved the file
- ✔️ Make sure the `--model-type` matches your downloaded file:
  - `sam_vit_b_01ec64.pth` → use `--model-type vit_b`
  - `sam_vit_l_0b3195.pth` → use `--model-type vit_l`
  - `sam_vit_h_4b8939.pth` → use `--model-type vit_h`

### ❌ Import errors
- ✔️ Make sure you've activated your virtual environment
- ✔️ Re-run: `pip install numpy opencv-python matplotlib scikit-image networkx torch torchvision`
- ✔️ Check your Python version is 3.8-3.11

### ❌ Point selection not working
- ✔️ Make sure matplotlib window is in focus
- ✔️ Use left-click only
- ✔️ Press Enter when done, 'r' to reset, or Esc to cancel

---

## 📚 Citation

```bibtex
@article{kirillov2023segment,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 📧 Support

If you encounter issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Make sure you've followed all installation steps
3. Verify your Python version is 3.8-3.11
4. Open an issue on GitHub with your error message and setup details

---

**Made with ❤️ for research and veterinary science**