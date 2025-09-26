#!/usr/bin/env python3
"""
Test script for the web scraper functionality.
This script demonstrates how to use the full_scrape_website function
to extract menu items from https://www.tvijverhof.be
"""

import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from web_scraper import full_scrape_website

def main():
    """
    Test the web scraper with the specified website.
    """
    print("=" * 60)
    print("Web Scraper Test - Menu Extraction")
    print("=" * 60)
    
    # Test URL as specified in the problem statement
    web_in = 'https://www.tvijverhof.be'
    
    print(f"Scraping website: {web_in}")
    print("-" * 40)
    
    try:
        # Run the scraper
        df_test = full_scrape_website(web_in)
        
        print("\nScraping completed!")
        print(f"Found {len(df_test)} menu items")
        print("-" * 40)
        
        # Display results
        if not df_test.empty:
            print("\nExtracted Menu Items:")
            print("=" * 60)
            for idx, row in df_test.iterrows():
                dish_name = row['dish_name']
                price = row['price']
                if price is not None:
                    print(f"{idx+1:2d}. {dish_name} - €{price:.2f}")
                else:
                    print(f"{idx+1:2d}. {dish_name} - Price not found")
            
            print("\nDataFrame Summary:")
            print(df_test.info())
            print("\nDataFrame Head:")
            print(df_test.head(10))
        else:
            print("No menu items found. This could be due to:")
            print("- No menu images found on the website")
            print("- OCR unable to extract text from images")
            print("- Images don't contain recognizable menu items")
            print("- Network connectivity issues")
        
    except Exception as e:
        print(f"Error occurred during scraping: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()