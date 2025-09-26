"""
Source code for fitness prediction ML pipeline.
Includes data preprocessing, model training, and web scraping functionality.
"""

from .web_scraper import full_scrape_website, MenuScraper

__all__ = ["full_scrape_website", "MenuScraper"]
