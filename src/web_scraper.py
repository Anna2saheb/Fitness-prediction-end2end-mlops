"""
Web scraper module for extracting menu items from restaurant websites using OCR.
Specifically designed to handle image-based menus using PaddleOCR.
"""

import re
import requests
from typing import List, Dict, Tuple, Optional
import pandas as pd
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    OCR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PaddleOCR not available: {e}. Installing it is required for menu image processing.")
    ocr = None
    OCR_AVAILABLE = False
except Exception as e:
    logger.warning(f"Error initializing PaddleOCR: {e}. Will attempt to initialize on first use.")
    ocr = None
    OCR_AVAILABLE = True

class MenuScraper:
    """
    A web scraper for extracting menu information from restaurant websites
    that display menus as images using OCR technology.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse webpage content.
        
        Args:
            url: Website URL to scrape
            
        Returns:
            BeautifulSoup object or None if failed
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to fetch page content from {url}: {e}")
            return None
    
    def find_menu_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Find all image URLs that likely contain menu information.
        Only processes images with 'menu', 'kaart', or 'speisen' in their URL.
        
        Args:
            soup: BeautifulSoup object of the webpage
            base_url: Base URL for resolving relative links
            
        Returns:
            List of menu image URLs
        """
        menu_images = []
        menu_keywords = ['menu', 'kaart', 'speisen']
        
        # Find all img tags
        img_tags = soup.find_all('img')
        
        for img in img_tags:
            img_url = img.get('src') or img.get('data-src')
            if not img_url:
                continue
                
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, img_url)
            
            # Check if image URL contains menu-related keywords
            url_lower = full_url.lower()
            if any(keyword in url_lower for keyword in menu_keywords):
                # Skip logo images
                if 'logo' not in url_lower:
                    menu_images.append(full_url)
                    logger.info(f"Found menu image: {full_url}")
        
        logger.info(f"Found {len(menu_images)} menu images")
        return menu_images
    
    def download_image(self, image_url: str) -> Optional[bytes]:
        """
        Download image from URL.
        
        Args:
            image_url: URL of the image to download
            
        Returns:
            Image bytes or None if failed
        """
        try:
            response = self.session.get(image_url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download image from {image_url}: {e}")
            return None
    
    def extract_text_from_image(self, image_bytes: bytes) -> List[Dict]:
        """
        Extract text from image using PaddleOCR.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            List of OCR results with text and bounding boxes
        """
        global ocr
        
        if not OCR_AVAILABLE:
            logger.error("PaddleOCR not available. Cannot extract text from images.")
            return []
        
        # Try to initialize OCR if not already done
        if ocr is None:
            try:
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                return []
        
        try:
            # Save image temporarily
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
            
            # Use PaddleOCR to extract text
            results = ocr.ocr(tmp_path, cls=True)
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            if not results or not results[0]:
                return []
            
            # Process OCR results
            text_data = []
            for line in results[0]:
                if line:
                    bbox, (text, confidence) = line
                    if confidence > 0.5:  # Filter low confidence results
                        text_data.append({
                            'text': text.strip(),
                            'confidence': confidence,
                            'bbox': bbox,
                            'y_position': bbox[0][1]  # Use top-left y coordinate for grouping
                        })
            
            logger.info(f"Extracted {len(text_data)} text elements from image")
            return text_data
            
        except Exception as e:
            logger.error(f"Failed to extract text from image: {e}")
            return []
    
    def group_text_by_lines(self, text_data: List[Dict]) -> List[str]:
        """
        Group OCR text results by Y-position to reconstruct lines of text.
        
        Args:
            text_data: List of OCR text results
            
        Returns:
            List of reconstructed text lines
        """
        if not text_data:
            return []
        
        # Sort by Y position
        text_data.sort(key=lambda x: x['y_position'])
        
        lines = []
        current_line = []
        current_y = None
        y_threshold = 20  # Pixels tolerance for same line
        
        for item in text_data:
            y_pos = item['y_position']
            
            if current_y is None or abs(y_pos - current_y) <= y_threshold:
                # Same line
                current_line.append(item)
                current_y = y_pos if current_y is None else (current_y + y_pos) / 2
            else:
                # New line
                if current_line:
                    # Sort current line by X position and join text
                    current_line.sort(key=lambda x: x['bbox'][0][0])
                    line_text = ' '.join(item['text'] for item in current_line)
                    if line_text.strip():
                        lines.append(line_text.strip())
                
                current_line = [item]
                current_y = y_pos
        
        # Add the last line
        if current_line:
            current_line.sort(key=lambda x: x['bbox'][0][0])
            line_text = ' '.join(item['text'] for item in current_line)
            if line_text.strip():
                lines.append(line_text.strip())
        
        logger.info(f"Grouped text into {len(lines)} lines")
        return lines
    
    def extract_menu_items(self, text_lines: List[str]) -> List[Dict[str, str]]:
        """
        Extract dish names and prices from text lines using regex patterns.
        
        Args:
            text_lines: List of text lines from OCR
            
        Returns:
            List of dictionaries with dish names and prices
        """
        menu_items = []
        
        # Price patterns (€, EUR, numbers with decimal points)
        price_patterns = [
            r'€\s*(\d+[.,]\d{2})',  # €12.50 or € 12,50
            r'(\d+[.,]\d{2})\s*€',  # 12.50€ or 12,50 €
            r'(\d+[.,]\d{2})\s*EUR',  # 12.50 EUR
            r'(\d+[.,]\d{2})(?=\s*$)',  # Price at end of line
        ]
        
        for line in text_lines:
            if not line.strip():
                continue
                
            print(f"DEBUG: Processing line: '{line}'")  # Debug output
            
            # Look for price patterns
            price_found = None
            dish_name = line
            
            for pattern in price_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    # Extract the price value
                    if match.group(1) if len(match.groups()) > 0 else match.group(0):
                        price_str = match.group(1) if len(match.groups()) > 0 else match.group(0)
                        price_found = price_str.replace(',', '.')
                        
                        # Remove price from dish name
                        dish_name = re.sub(pattern, '', line, flags=re.IGNORECASE).strip()
                        break
                
                if price_found:
                    break
            
            # Clean up dish name
            dish_name = re.sub(r'\s+', ' ', dish_name).strip()
            dish_name = re.sub(r'^[-•·\s]+|[-•·\s]+$', '', dish_name).strip()
            
            # Only add if we have both dish name and price, or just a meaningful dish name
            if dish_name and len(dish_name) > 2:
                if price_found:
                    try:
                        price_float = float(price_found)
                        menu_items.append({
                            'dish_name': dish_name,
                            'price': price_float
                        })
                        print(f"DEBUG: Added item: {dish_name} - €{price_float}")
                    except ValueError:
                        pass
                elif len(dish_name) > 5:  # Add items without prices if they look like dish names
                    menu_items.append({
                        'dish_name': dish_name,
                        'price': None
                    })
                    print(f"DEBUG: Added item without price: {dish_name}")
        
        logger.info(f"Extracted {len(menu_items)} menu items")
        return menu_items
    
    def process_menu_images(self, menu_images: List[str]) -> List[Dict[str, str]]:
        """
        Process all menu images and extract menu items.
        
        Args:
            menu_images: List of menu image URLs
            
        Returns:
            List of menu items with dishes and prices
        """
        all_menu_items = []
        
        for img_url in menu_images:
            logger.info(f"Processing image: {img_url}")
            
            # Download image
            image_bytes = self.download_image(img_url)
            if not image_bytes:
                continue
            
            # Extract text using OCR
            text_data = self.extract_text_from_image(image_bytes)
            if not text_data:
                continue
            
            # Print all OCR text for debugging
            print(f"\nDEBUG: OCR results for {img_url}:")
            for item in text_data:
                print(f"  Text: '{item['text']}' (confidence: {item['confidence']:.2f})")
            
            # Group text by lines
            text_lines = self.group_text_by_lines(text_data)
            
            print(f"\nDEBUG: Grouped lines:")
            for i, line in enumerate(text_lines):
                print(f"  Line {i+1}: '{line}'")
            
            # Extract menu items
            menu_items = self.extract_menu_items(text_lines)
            all_menu_items.extend(menu_items)
        
        return all_menu_items


def full_scrape_website(url: str) -> pd.DataFrame:
    """
    Main function to scrape menu items from a restaurant website.
    
    Args:
        url: Website URL to scrape (e.g., 'https://www.tvijverhof.be')
        
    Returns:
        DataFrame with columns: dish_name, price
    """
    logger.info(f"Starting to scrape website: {url}")
    
    scraper = MenuScraper()
    
    # Get webpage content
    soup = scraper.get_page_content(url)
    if not soup:
        logger.error("Failed to get webpage content")
        return pd.DataFrame(columns=['dish_name', 'price'])
    
    # Find menu images
    menu_images = scraper.find_menu_images(soup, url)
    if not menu_images:
        logger.warning("No menu images found")
        return pd.DataFrame(columns=['dish_name', 'price'])
    
    # Process menu images
    menu_items = scraper.process_menu_images(menu_images)
    
    # Create DataFrame
    if menu_items:
        df = pd.DataFrame(menu_items)
        logger.info(f"Successfully extracted {len(df)} menu items")
        return df
    else:
        logger.warning("No menu items extracted")
        return pd.DataFrame(columns=['dish_name', 'price'])


if __name__ == "__main__":
    # Test the scraper
    test_url = 'https://www.tvijverhof.be'
    df_result = full_scrape_website(test_url)
    print("\nFinal Results:")
    print(df_result)