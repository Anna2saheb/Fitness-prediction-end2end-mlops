# Web Scraping Module - Menu Extraction with OCR

This module provides functionality to extract menu items from restaurant websites that display menus as images using Optical Character Recognition (OCR).

## Features

- **Image-based menu extraction** using PaddleOCR
- **Smart image filtering** (only processes images with 'menu', 'kaart', or 'speisen' in URL)
- **Text line reconstruction** by grouping OCR results by Y-position
- **Robust price detection** with multiple regex patterns
- **Debug output** for troubleshooting OCR results
- **Error handling** for network issues and OCR failures

## Installation

Install the required dependencies:

```bash
pip install pandas requests beautifulsoup4 paddlepaddle paddleocr
```

## Usage

### Basic Usage

```python
from src.web_scraper import full_scrape_website
import pandas as pd

# Extract menu items from a restaurant website
web_in = 'https://www.tvijverhof.be'
df_test = full_scrape_website(web_in)

# Display results
print(df_test)
```

### Expected Output

The function returns a pandas DataFrame with the following columns:
- `dish_name`: Name of the dish extracted from the menu
- `price`: Price of the dish (float) or None if not found

Example output:
```
                    dish_name  price
0           Schnitzel Wiener Art  18.50
1             Pasta Carbonara  14.90
2               Caesar Salad  12.00
3    Beef Steak with Fries  24.50
4            Soup of the Day   None
```

## How It Works

1. **Web Page Fetching**: Downloads the HTML content from the restaurant website
2. **Image Discovery**: Finds all images and filters for menu-related images using keywords:
   - 'menu' (English)
   - 'kaart' (Dutch)
   - 'speisen' (German)
3. **Image Processing**: Downloads each menu image and processes it with PaddleOCR
4. **Text Extraction**: Extracts text with confidence scores and bounding boxes
5. **Line Reconstruction**: Groups text elements by Y-position to rebuild menu lines
6. **Menu Parsing**: Uses regex patterns to identify dish names and prices
7. **DataFrame Creation**: Returns structured data ready for analysis

## Supported Price Formats

The scraper recognizes various price formats:
- `€12.50` or `€ 12,50`
- `12.50€` or `12,50 €`
- `12.50 EUR`
- Numbers at the end of lines

## Debug Features

The scraper provides detailed debug output including:
- All OCR text extracted from images with confidence scores
- Line grouping results showing how text is reconstructed
- Menu item parsing steps with regex matches
- Network and processing error messages

## Error Handling

- **Graceful degradation** when PaddleOCR is not available
- **Network timeout handling** for slow websites
- **Temporary file cleanup** after OCR processing
- **Comprehensive logging** for debugging issues

## Technical Requirements

- **Python 3.7+**
- **PaddlePaddle**: Deep learning framework for OCR
- **PaddleOCR**: OCR library for text extraction
- **BeautifulSoup4**: HTML parsing
- **Requests**: HTTP client
- **Pandas**: Data manipulation

## Limitations

- Requires menu information to be displayed as images
- OCR accuracy depends on image quality and text clarity
- Processing time depends on number and size of menu images
- Network connectivity required for downloading images

## Testing

Run the unit tests to verify functionality:

```bash
python -m unittest tests.test_web_scraper -v
```

## Troubleshooting

### Common Issues

1. **No menu items found**:
   - Check if website has images with menu-related keywords
   - Verify image quality is sufficient for OCR
   - Enable debug output to see OCR results

2. **PaddleOCR installation issues**:
   - Install PaddlePaddle first: `pip install paddlepaddle`
   - Then install PaddleOCR: `pip install paddleocr`
   - Check system requirements for GPU acceleration (optional)

3. **Network connectivity**:
   - Verify internet connection
   - Check if website blocks automated requests
   - Consider adding delays between requests

### Debug Mode

Enable detailed logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run scraper with debug output
df = full_scrape_website('https://example.com')
```

## Example Integration

The web scraper can be integrated into existing workflows:

```python
# Example: Analyze menu pricing across multiple restaurants
restaurants = [
    'https://www.restaurant1.com',
    'https://www.restaurant2.com',
    'https://www.restaurant3.com'
]

all_menus = []
for restaurant in restaurants:
    try:
        menu_df = full_scrape_website(restaurant)
        menu_df['restaurant'] = restaurant
        all_menus.append(menu_df)
    except Exception as e:
        print(f"Failed to scrape {restaurant}: {e}")

# Combine all menu data
combined_df = pd.concat(all_menus, ignore_index=True)
print(f"Total menu items collected: {len(combined_df)}")
```

## Performance Considerations

- **Processing time**: Depends on number of images and OCR complexity
- **Memory usage**: Images are processed one at a time to minimize memory footprint
- **Network requests**: Sequential processing to avoid overwhelming servers
- **Temporary files**: Automatically cleaned up after processing

## Future Enhancements

Potential improvements for the web scraper:
- Support for additional languages in OCR
- Machine learning models for better menu item classification
- Parallel processing of multiple images
- Caching mechanisms for repeated requests
- Integration with restaurant databases for validation