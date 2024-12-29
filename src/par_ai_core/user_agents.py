"""
User Agent Generator Module

This module provides functionality to generate random user agent strings that mimic
common web browsers. It's useful for applications that need to simulate different
user agents, such as web scraping tools or testing frameworks.

The module includes:
- A function to generate random user agent strings
- Lists of operating systems and browsers to create realistic combinations
- Random version number generation for various browser components

Usage:
    from par_ai_core.user_agents import get_random_user_agent

    user_agent = get_random_user_agent()
    print(user_agent)

Note:
    The generated user agents are designed to be realistic but may not cover all
    possible real-world combinations. They should be used responsibly and in
    compliance with the terms of service of any websites or services you interact with.
"""

import random


def get_random_user_agent() -> str:
    """Generate a random user agent string.

    Returns:
        str: A randomly generated user agent string that mimics common web browsers.
        The string includes randomized versions, OS info, and browser-specific details.
    """
    os_list = [
        ("Windows NT 10.0", "Win64; x64"),
        ("Windows NT 11.0", "Win64; x64"),
        ("Macintosh; Intel Mac OS X 10_15_7", "Intel Mac OS X"),
        ("Macintosh; Apple M1 Mac OS X 13_5_1", "arm64"),
        ("Macintosh; Apple M2 Mac OS X 14_2_1", "arm64"),
    ]
    browser_list = ["Chrome", "Firefox", "Safari", "Edge"]
    webkit_version = f"{random.randint(537, 615)}.{random.randint(36, 50)}"
    chrome_version = f"{random.randint(120, 122)}.0.{random.randint(6000, 6500)}.{random.randint(100, 200)}"
    edge_version = f"{random.randint(120, 122)}.0.{random.randint(2000, 2500)}.{random.randint(100, 200)}"
    firefox_version = f"{random.randint(121, 123)}.0"
    os, platform = random.choice(os_list)
    browser = random.choice(browser_list)

    webkit = f" AppleWebKit/{webkit_version}"
    gecko = " (KHTML, like Gecko)"
    if browser == "Safari":
        safari_version = f"{random.randint(16, 17)}.{random.randint(2, 4)}"
        version = f"Version/{safari_version} Safari/{webkit_version}"
    elif browser == "Firefox":
        version = f"Gecko/20100101 Firefox/{firefox_version}"
        gecko = ""
        webkit = ""
    elif browser == "Edge":
        version = f"Edg/{edge_version}"
    else:  # Chrome
        version = f"Chrome/{chrome_version} Mobile Safari/{webkit_version}"

    return f"Mozilla/5.0 ({os.split('; ')[0]}; {platform}){webkit}{gecko} {version}"
