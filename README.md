# Map Text Extractor

A Python tool for extracting text from topographic maps using OCR (Optical Character Recognition).

## Features

- Extracts region numbers (typically in red) and place names (typically in black) from map images
- Uses multiple image processing techniques to enhance OCR accuracy
- Saves results to CSV files for easy analysis
- Creates visualizations of detected text
- Organizes results in separate directories for each processed map

## Requirements

- Python 3.6+
- OpenCV (cv2)
- NumPy
- EasyOCR
- Other dependencies listed in requirements.txt

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python OCR/extract_map_text.py --image path/to/your/map.png
```

This will:
1. Process the specified map image
2. Create a timestamped output directory
3. Save CSV files with extracted region numbers and place names
4. Generate visualization images
5. Create a summary file

### Command Line Arguments

```
--image, -i       Path to the input map image (default: "OCR/map.png")
--gpu             Use GPU for OCR processing if available
--language, -l    Primary language for OCR (default: "en")
--output, -o      Custom output directory name (optional)
--min-confidence, -c  Minimum confidence threshold for OCR results (0.0-1.0, default: 0.4)
```

### Examples

Process a map with custom output directory:
```bash
python OCR/extract_map_text.py --image maps/topo_map_1.jpg --output results/topo_map_1
```

Process a map with higher confidence threshold:
```bash
python OCR/extract_map_text.py --image maps/topo_map_2.png --min-confidence 0.6
```

Use GPU acceleration (if available):
```bash
python OCR/extract_map_text.py --image maps/topo_map_3.tif --gpu
```

## Output

For each processed map, the tool creates:

1. **region_numbers.csv** - CSV file containing:
   - Number value
   - X/Y coordinates
   - Confidence score
   - Source method

2. **place_names.csv** - CSV file containing:
   - Place name
   - X/Y coordinates
   - Confidence score
   - Source method

3. **Visualization images**:
   - extraction_results.jpg - Combined visualization of all detected text
   - regions_only.jpg - Visualization of region numbers only
   - places_only.jpg - Visualization of place names only
   - original_[filename] - Copy of the original image

4. **summary.txt** - Text file with processing details and summary of results

## How It Works

The tool uses a multi-stage approach to extract text from maps:

1. **Red text extraction** - Isolates red elements (typically region numbers) using color filtering
2. **Black text extraction** - Isolates dark elements (typically place names) using color filtering
3. **Enhanced OCR fallback** - Applies multiple image processing techniques for improved text detection
4. **Result combination** - Combines and deduplicates results from all methods
5. **Classification** - Classifies results into region numbers and place names
6. **Visualization** - Creates visual representations of the detected text

## License

This project is licensed under the MIT License - see the LICENSE file
