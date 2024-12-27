"""Generate a random user agent."""

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
