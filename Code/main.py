# File: OCR/main.py
import cv2
import numpy as np
import easyocr
import re
import os
import csv
import datetime
from collections import defaultdict
from pathlib import Path

class MapTextExtractor:
    def __init__(self, gpu=False, language='en'):
        """
        Initialize the MapTextExtractor
        
        Args:
            gpu (bool): Whether to use GPU for OCR processing
            language (str): Primary language for OCR
        """
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader([language], gpu=gpu)
        print(f"Initialized EasyOCR with language: {language}, GPU: {gpu}")
        
    def extract_map_text(self, image_path):
        """Main extraction pipeline optimized for topographic maps"""
        
        # Load image
        img = cv2.imread(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Step 1: Extract red numbers (region numbers)
        red_numbers = self._extract_red_numbers(img, hsv)
        
        # Step 2: Extract black text (place names)
        black_names = self._extract_black_text(img, hsv)
        
        # Step 3: Fallback - general OCR on enhanced image
        enhanced_results = self._enhanced_ocr_fallback(img)
        
        # Step 4: Combine and deduplicate results
        all_results = self._combine_results(red_numbers, black_names, enhanced_results)
        
        # Step 5: Classify final results
        region_numbers, place_names = self._classify_results(all_results)
        
        return region_numbers, place_names
    
    def _extract_red_numbers(self, img, hsv):
        """Extract red region numbers - highest priority"""
        print("Extracting red numbers...")
        
        # Define red color ranges (handles both bright and dark reds)
        red_ranges = [
            (np.array([0, 50, 50]), np.array([10, 255, 255])),    # Lower red
            (np.array([170, 50, 50]), np.array([180, 255, 255])) # Upper red
        ]
        
        # Create combined red mask
        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in red_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            red_mask = cv2.bitwise_or(red_mask, mask)
        
        # Clean up the mask
        kernel = np.ones((2,2), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to original image
        red_text_img = cv2.bitwise_and(img, img, mask=red_mask)
        
        # Convert to format suitable for EasyOCR
        red_text_rgb = cv2.cvtColor(red_text_img, cv2.COLOR_BGR2RGB)
        
        # OCR with high confidence for numbers
        results = self.reader.readtext(red_text_rgb, 
                                      allowlist='0123456789',
                                      width_ths=0.7,
                                      height_ths=0.7)
        
        return self._process_ocr_results(results, 'red_number')
    
    def _extract_black_text(self, img, hsv):
        """Extract black place names"""
        print("Extracting black text...")
        
        # Create mask for dark text (black/dark gray)
        # Adjust these values based on your map's text darkness
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 50, 100])  # Catches dark grays too
        
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        
        # Clean up mask
        kernel = np.ones((1,1), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask
        black_text_img = cv2.bitwise_and(img, img, mask=black_mask)
        black_text_rgb = cv2.cvtColor(black_text_img, cv2.COLOR_BGR2RGB)
        
        # OCR for text (no character restrictions)
        results = self.reader.readtext(black_text_rgb,
                                      width_ths=0.8,
                                      height_ths=0.8)
        
        return self._process_ocr_results(results, 'black_text')
    
    def _enhanced_ocr_fallback(self, img):
        """Fallback OCR on enhanced full image with multiple processing techniques"""
        print("Running enhanced OCR fallback with multiple image processing techniques...")
        
        all_results = []
        
        # Method 1: CLAHE enhancement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_clahe = clahe.apply(gray)
        enhanced_clahe_rgb = cv2.cvtColor(enhanced_clahe, cv2.COLOR_GRAY2RGB)
        
        # OCR on CLAHE enhanced image
        results_clahe = self.reader.readtext(enhanced_clahe_rgb,
                                           width_ths=0.6,
                                           height_ths=0.6)
        all_results.extend(self._process_ocr_results(results_clahe, 'enhanced_clahe'))
        
        # Method 2: Bilateral filtering for noise reduction while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        bilateral_rgb = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2RGB)
        
        # OCR on bilateral filtered image
        results_bilateral = self.reader.readtext(bilateral_rgb,
                                               width_ths=0.6,
                                               height_ths=0.6)
        all_results.extend(self._process_ocr_results(results_bilateral, 'enhanced_bilateral'))
        
        # Method 3: Thresholding for better text extraction
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        
        # OCR on thresholded image
        results_thresh = self.reader.readtext(thresh_rgb,
                                            width_ths=0.6,
                                            height_ths=0.6)
        all_results.extend(self._process_ocr_results(results_thresh, 'enhanced_thresh'))
        
        return all_results
    
    def _process_ocr_results(self, results, source_type):
        """Process raw OCR results into standardized format"""
        processed = []
        
        for (bbox, text, confidence) in results:
            # Clean text
            cleaned_text = text.strip()
            
            # Skip very short or low confidence results
            if len(cleaned_text) < 1 or confidence < 0.4:
                continue
                
            # Calculate center point of bounding box
            center_x = int(np.mean([point[0] for point in bbox]))
            center_y = int(np.mean([point[1] for point in bbox]))
            
            processed.append({
                'text': cleaned_text,
                'bbox': bbox,  # Keep the bounding box
                'center': (center_x, center_y),
                'confidence': confidence,
                'source': source_type
            })
        
        return processed
    
    def _combine_results(self, red_results, black_results, enhanced_results):
        """Combine results from different methods and remove duplicates"""
        print("Combining and deduplicating results...")
        
        all_results = red_results + black_results + enhanced_results
        
        # Sort by confidence (highest first)
        all_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove duplicates based on proximity and text similarity
        final_results = []
        proximity_threshold = 30  # pixels
        
        for result in all_results:
            is_duplicate = False
            
            for existing in final_results:
                # Check if texts are similar and locations are close
                if (self._text_similarity(result['text'], existing['text']) > 0.8 and
                    self._distance(result['center'], existing['center']) < proximity_threshold):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_results.append(result)
        
        return final_results
    
    def _classify_results(self, results):
        """Classify results into region numbers and place names"""
        print("Classifying results...")
        
        region_numbers = []
        place_names = []
        
        for result in results:
            text = result['text']
            
            # Region numbers: 1-3 digits, often from red text
            if (re.match(r'^\d{1,3}$', text) and 
                1 <= int(text) <= 999):
                
                region_numbers.append({
                    'number': int(text),
                    'location': result['center'],
                    'bbox': result['bbox'], # Keep bbox
                    'confidence': result['confidence'],
                    'source': result['source']
                })
            
            # Place names: contain letters, reasonable length
            elif (re.search(r'[A-Za-z]', text) and 
                  2 <= len(text) <= 30 and
                  not re.match(r'^\d+$', text)):  # Not just numbers
                
                # Clean up common OCR errors in place names
                cleaned_name = self._clean_place_name(text)
                
                place_names.append({
                    'name': cleaned_name,
                    'location': result['center'],
                    'bbox': result['bbox'], # Keep bbox
                    'confidence': result['confidence'],
                    'source': result['source']
                })
        
        return region_numbers, place_names
    
    def _clean_place_name(self, text):
        """
        Clean up common OCR errors in place names with enhanced correction
        
        This method applies multiple cleaning steps to improve OCR text quality:
        1. Removes special characters that are likely OCR artifacts
        2. Normalizes whitespace
        3. Corrects common OCR character substitutions
        4. Applies basic capitalization rules for place names
        """
        # Step 1: Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s\-\']', '', text)  # Keep hyphens and apostrophes for place names
        
        # Step 2: Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Step 3: Correct common OCR character substitutions
        ocr_corrections = {
            '0': 'O',  # Zero to letter O
            '1': 'I',  # One to letter I
            '5': 'S',  # Five to letter S
            '8': 'B',  # Eight to letter B
            '|': 'I',  # Pipe to letter I
            '}': 'J',  # Right brace to letter J
            '{': 'C',  # Left brace to letter C
            '[': 'L',  # Left bracket to letter L
            ']': 'I',  # Right bracket to letter I
            '<': 'C',  # Less than to letter C
            '>': 'G',  # Greater than to letter G
        }
        
        # Only apply corrections if they make sense in context
        for char, correction in ocr_corrections.items():
            # Only replace if the character is surrounded by letters
            # This avoids changing actual numbers in place names
            cleaned = re.sub(f'(?<=[A-Za-z]){re.escape(char)}(?=[A-Za-z])', correction, cleaned)
        
        # Step 4: Apply basic capitalization rules for place names
        # Capitalize first letter of each word for place names
        if re.search(r'[A-Za-z]{2,}', cleaned):  # Only if it contains at least 2 letters
            words = cleaned.split()
            cleaned = ' '.join(word.capitalize() for word in words)
        
        return cleaned.strip()
    
    def _text_similarity(self, text1, text2):
        """
        Calculate similarity between two texts using multiple metrics
        
        This improved method combines character-level and word-level similarity
        to better handle OCR variations and partial matches.
        """
        # Convert to lowercase for comparison
        t1 = text1.lower()
        t2 = text2.lower()
        
        # Handle empty strings
        if len(t1) == 0 and len(t2) == 0:
            return 1.0
        if len(t1) == 0 or len(t2) == 0:
            return 0.0
            
        # Method 1: Character-level Jaccard similarity
        set1 = set(t1)
        set2 = set(t2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        char_similarity = intersection / union if union > 0 else 0
        
        # Method 2: Length ratio similarity
        # Penalize big differences in length
        len_ratio = min(len(t1), len(t2)) / max(len(t1), len(t2))
        
        # Method 3: Sequence similarity
        # Calculate longest common subsequence length
        def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        # Normalize LCS by the length of the longer string
        lcs = lcs_length(t1, t2)
        seq_similarity = lcs / max(len(t1), len(t2))
        
        # Method 4: Word-level similarity for multi-word texts
        # Only apply if both texts contain spaces
        if ' ' in t1 and ' ' in t2:
            words1 = set(t1.split())
            words2 = set(t2.split())
            
            word_intersection = len(words1.intersection(words2))
            word_union = len(words1.union(words2))
            
            word_similarity = word_intersection / word_union if word_union > 0 else 0
            
            # Combine all similarity metrics with word similarity
            return 0.3 * char_similarity + 0.2 * len_ratio + 0.3 * seq_similarity + 0.2 * word_similarity
        
        # Combine similarity metrics for single-word texts
        return 0.4 * char_similarity + 0.2 * len_ratio + 0.4 * seq_similarity
    
    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def visualize_results(self, image_path, region_numbers, place_names, output_dir=None):
        """
        Visualize the extraction results on the original image with enhanced annotations
        
        Args:
            image_path: Path to the original image
            region_numbers: List of detected region numbers
            place_names: List of detected place names
            output_dir: Optional directory to save intermediate visualizations
            
        Returns:
            Annotated image with all detections
        """
        # Load original image
        img = cv2.imread(image_path)
        
        # Create a copy for each visualization type
        region_img = img.copy()
        place_img = img.copy()
        combined_img = img.copy()
        
        # Draw region numbers in red with bounding boxes and confidence indicator
        for num_data in region_numbers:
            bbox = num_data['bbox']
            confidence = num_data['confidence']
            number = num_data['number']
            
            # Get top-left and bottom-right corners
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            
            # Draw rectangle on region-specific image
            cv2.rectangle(region_img, top_left, bottom_right, (0, 0, 255), 2)
            cv2.putText(region_img, f"{number} ({confidence:.2f})", 
                       (top_left[0], top_left[1] - 10), # Position text above bbox
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw on combined image
            cv2.rectangle(combined_img, top_left, bottom_right, (0, 0, 255), 2)
            cv2.putText(combined_img, str(number), 
                       (top_left[0], top_left[1] - 10), # Position text above bbox
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw place names in blue with bounding boxes and confidence indicator
        for name_data in place_names:
            bbox = name_data['bbox']
            confidence = name_data['confidence']
            name = name_data['name']
            
            # Get top-left and bottom-right corners
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            
            # Draw rectangle on place-specific image
            cv2.rectangle(place_img, top_left, bottom_right, (255, 0, 0), 2)
            
            # Truncate long names for display
            display_name = name[:20] + '...' if len(name) > 20 else name
            
            cv2.putText(place_img, f"{display_name} ({confidence:.2f})", 
                       (top_left[0], bottom_right[1] + 15), # Position text below bbox
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Draw on combined image
            cv2.rectangle(combined_img, top_left, bottom_right, (255, 0, 0), 2)
            cv2.putText(combined_img, display_name, 
                       (top_left[0], bottom_right[1] + 15), # Position text below bbox
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Add legend to combined image
        cv2.putText(combined_img, "Red Box: Region Numbers", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(combined_img, "Blue Box: Place Names", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Save intermediate visualizations if output directory is provided
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, "regions_only.jpg"), region_img)
            cv2.imwrite(os.path.join(output_dir, "places_only.jpg"), place_img)
        
        return combined_img

def create_output_directory(image_path):
    """Create a timestamped output directory for the processed map"""
    # Get base filename without extension
    base_name = Path(image_path).stem
    
    # Create a timestamp for unique directory naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory name
    output_dir = f"OCR_Results_{base_name}_{timestamp}"
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def save_to_csv(data, csv_path, headers):
    """Save data to a CSV file"""
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for item in data:
            writer.writerow(item)

def parse_arguments():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract text from topographic maps using OCR')
    parser.add_argument('--image', '-i', type=str, default="OCR/map.png",
                        help='Path to the input map image (default: OCR/map.png)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for OCR processing (if available)')
    parser.add_argument('--language', '-l', type=str, default='en',
                        help='Primary language for OCR (default: en)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Custom output directory name (optional)')
    parser.add_argument('--min-confidence', '-c', type=float, default=0.4,
                        help='Minimum confidence threshold for OCR results (0.0-1.0)')
    
    return parser.parse_args()

# Main function with improved functionality
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate input image path
    image_path = args.image
    
    # If no command-line arguments provided, use default path
    if not os.path.exists(image_path):
        # Try with and without the project root directory
        alternative_paths = [
            "OCR/map.png",  # Relative to current directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.png"),  # Same directory as script
            "map.png"  # Just the filename
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"Input image not found at {image_path}, using {alt_path} instead")
                image_path = alt_path
                break
        else:
            print(f"Error: Input image not found at {image_path} or any alternative locations")
            print(f"Please provide a valid image path with --image argument")
            print(f"\nExample usage:")
            print(f"  python OCR/extract_map_text.py --image path/to/your/map.png")
            print(f"\nNo sample map images were found in the project directory.")
            print(f"You need to provide your own map image to process.")
            return
    
    # Create output directory
    if args.output:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = create_output_directory(image_path)
    
    print(f"Creating output directory: {output_dir}")
    
    # Initialize extractor with specified parameters
    extractor = MapTextExtractor(gpu=args.gpu, language=args.language)
    
    # Process the map
    print(f"Processing map image: {image_path}")
    region_numbers, place_names = extractor.extract_map_text(image_path)
    
    # Filter results by confidence threshold if specified
    if args.min_confidence > 0.4:  # Only if different from default
        region_numbers = [r for r in region_numbers if r['confidence'] >= args.min_confidence]
        place_names = [p for p in place_names if p['confidence'] >= args.min_confidence]
        print(f"Filtered results with minimum confidence: {args.min_confidence}")
    
    # Prepare data for CSV export
    region_csv_data = []
    for num_data in sorted(region_numbers, key=lambda x: x['number']):
        region_csv_data.append({
            'number': num_data['number'],
            'x_coordinate': num_data['location'][0],
            'y_coordinate': num_data['location'][1],
            'confidence': num_data['confidence'],
            'source': num_data['source']
        })
    
    place_csv_data = []
    for name_data in sorted(place_names, key=lambda x: x['confidence'], reverse=True):
        place_csv_data.append({
            'name': name_data['name'],
            'x_coordinate': name_data['location'][0],
            'y_coordinate': name_data['location'][1],
            'confidence': name_data['confidence'],
            'source': name_data['source']
        })
    
    # Save results to CSV files
    region_csv_path = os.path.join(output_dir, "region_numbers.csv")
    place_csv_path = os.path.join(output_dir, "place_names.csv")
    
    save_to_csv(region_csv_data, region_csv_path, 
                ['number', 'x_coordinate', 'y_coordinate', 'confidence', 'source'])
    save_to_csv(place_csv_data, place_csv_path, 
                ['name', 'x_coordinate', 'y_coordinate', 'confidence', 'source'])
    
    # Print summary of results
    print(f"\n=== REGION NUMBERS FOUND ({len(region_numbers)}) ===")
    print(f"Results saved to: {region_csv_path}")
    
    print(f"\n=== PLACE NAMES FOUND ({len(place_names)}) ===")
    print(f"Results saved to: {place_csv_path}")
    
    # Create visualization with separate images for regions and places
    result_img = extractor.visualize_results(image_path, region_numbers, place_names, output_dir)
    
    # Save visualization to output directory
    result_img_path = os.path.join(output_dir, "extraction_results.jpg")
    cv2.imwrite(result_img_path, result_img)
    print(f"\nVisualization saved as: {result_img_path}")
    
    # Also save a copy of the original image for reference
    original_copy_path = os.path.join(output_dir, f"original_{Path(image_path).name}")
    original_img = cv2.imread(image_path)
    cv2.imwrite(original_copy_path, original_img)
    
    # Create a summary file
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"OCR Map Text Extraction Summary\n")
        f.write(f"==============================\n\n")
        f.write(f"Input image: {image_path}\n")
        f.write(f"Processed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Language: {args.language}\n")
        f.write(f"GPU enabled: {args.gpu}\n\n")
        f.write(f"Region numbers found: {len(region_numbers)}\n")
        f.write(f"Place names found: {len(place_names)}\n\n")
        f.write(f"Files in this directory:\n")
        f.write(f"- region_numbers.csv: CSV file with all region numbers\n")
        f.write(f"- place_names.csv: CSV file with all place names\n")
        f.write(f"- extraction_results.jpg: Visualization of all detected text\n")
        f.write(f"- regions_only.jpg: Visualization of region numbers only\n")
        f.write(f"- places_only.jpg: Visualization of place names only\n")
        f.write(f"- original_{Path(image_path).name}: Copy of the original image\n")
    
    print(f"\nProcessing complete. All results saved to: {output_dir}")
    print(f"Summary file created: {summary_path}")

if __name__ == "__main__":
    main()
