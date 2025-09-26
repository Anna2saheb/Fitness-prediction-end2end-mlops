"""
Unit tests for the web scraper module.
"""

import unittest
import sys
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from web_scraper import MenuScraper, full_scrape_website


class TestMenuScraper(unittest.TestCase):
    """Test cases for MenuScraper class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = MenuScraper()
    
    def test_scraper_initialization(self):
        """Test that MenuScraper initializes correctly."""
        self.assertIsNotNone(self.scraper.session)
        self.assertIn('User-Agent', self.scraper.session.headers)
    
    def test_find_menu_images_filters_correctly(self):
        """Test that menu image filtering works correctly."""
        # Mock HTML content
        html_content = """
        <html>
            <body>
                <img src="/logo.png" alt="logo">
                <img src="/images/menu_main.jpg" alt="main menu">
                <img src="/gallery/food.jpg" alt="food">
                <img src="/kaart/drinks.png" alt="drinks menu">
                <img src="/speisen/desserts.jpg" alt="desserts">
            </body>
        </html>
        """
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        base_url = "https://example.com"
        
        menu_images = self.scraper.find_menu_images(soup, base_url)
        
        # Should find 3 menu images (menu_main.jpg, drinks.png, desserts.jpg)
        # but not logo.png or food.jpg
        self.assertEqual(len(menu_images), 3)
        self.assertTrue(any("menu_main.jpg" in img for img in menu_images))
        self.assertTrue(any("drinks.png" in img for img in menu_images))
        self.assertTrue(any("desserts.jpg" in img for img in menu_images))
        self.assertFalse(any("logo.png" in img for img in menu_images))
        self.assertFalse(any("food.jpg" in img for img in menu_images))
    
    def test_group_text_by_lines(self):
        """Test text grouping by Y-position."""
        # Mock OCR data
        text_data = [
            {'text': 'Dish1', 'y_position': 100, 'bbox': [[10, 100], [50, 100], [50, 120], [10, 120]]},
            {'text': '€12.50', 'y_position': 105, 'bbox': [[60, 105], [100, 105], [100, 125], [60, 125]]},
            {'text': 'Dish2', 'y_position': 150, 'bbox': [[10, 150], [50, 150], [50, 170], [10, 170]]},
            {'text': '€15.00', 'y_position': 155, 'bbox': [[60, 155], [100, 155], [100, 175], [60, 175]]},
        ]
        
        lines = self.scraper.group_text_by_lines(text_data)
        
        # Should group into 2 lines
        self.assertEqual(len(lines), 2)
        self.assertIn('Dish1', lines[0])
        self.assertIn('€12.50', lines[0])
        self.assertIn('Dish2', lines[1])
        self.assertIn('€15.00', lines[1])
    
    def test_extract_menu_items(self):
        """Test menu item extraction from text lines."""
        text_lines = [
            'Schnitzel Wiener Art €18.50',
            'Pasta Carbonara 14,90€',
            'Caesar Salad 12.00 EUR',
            'Beef Steak with Fries 24.50',
            'Soup of the Day',  # No price
            'Coffee €3.50'
        ]
        
        menu_items = self.scraper.extract_menu_items(text_lines)
        
        # Should extract menu items with various price formats
        self.assertGreater(len(menu_items), 0)
        
        # Check specific extractions
        item_names = [item['dish_name'] for item in menu_items]
        self.assertIn('Schnitzel Wiener Art', item_names)
        self.assertIn('Pasta Carbonara', item_names)
        
        # Check prices
        for item in menu_items:
            if item['dish_name'] == 'Schnitzel Wiener Art':
                self.assertEqual(item['price'], 18.50)
            elif item['dish_name'] == 'Pasta Carbonara':
                self.assertEqual(item['price'], 14.90)
    
    @patch('web_scraper.requests.Session.get')
    def test_get_page_content_success(self, mock_get):
        """Test successful page content retrieval."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b'<html><body>Test</body></html>'
        mock_get.return_value = mock_response
        
        soup = self.scraper.get_page_content('https://example.com')
        
        self.assertIsNotNone(soup)
        self.assertEqual(soup.body.text, 'Test')
    
    @patch('web_scraper.requests.Session.get')
    def test_get_page_content_failure(self, mock_get):
        """Test page content retrieval failure."""
        # Mock failed response
        mock_get.side_effect = Exception("Network error")
        
        soup = self.scraper.get_page_content('https://example.com')
        
        self.assertIsNone(soup)


class TestFullScrapeWebsite(unittest.TestCase):
    """Test cases for full_scrape_website function."""
    
    @patch('web_scraper.MenuScraper')
    def test_full_scrape_website_returns_dataframe(self, mock_scraper_class):
        """Test that full_scrape_website returns a DataFrame."""
        # Mock the scraper
        mock_scraper = Mock()
        mock_scraper.get_page_content.return_value = Mock()
        mock_scraper.find_menu_images.return_value = []
        mock_scraper.process_menu_images.return_value = []
        mock_scraper_class.return_value = mock_scraper
        
        result = full_scrape_website('https://example.com')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ['dish_name', 'price'])
    
    @patch('web_scraper.MenuScraper')
    def test_full_scrape_website_with_menu_items(self, mock_scraper_class):
        """Test full_scrape_website with mock menu items."""
        # Mock menu items
        mock_menu_items = [
            {'dish_name': 'Test Dish 1', 'price': 12.50},
            {'dish_name': 'Test Dish 2', 'price': 15.00}
        ]
        
        # Mock the scraper
        mock_scraper = Mock()
        mock_scraper.get_page_content.return_value = Mock()
        mock_scraper.find_menu_images.return_value = ['menu1.jpg']
        mock_scraper.process_menu_images.return_value = mock_menu_items
        mock_scraper_class.return_value = mock_scraper
        
        result = full_scrape_website('https://example.com')
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['dish_name'], 'Test Dish 1')
        self.assertEqual(result.iloc[0]['price'], 12.50)


if __name__ == '__main__':
    unittest.main()