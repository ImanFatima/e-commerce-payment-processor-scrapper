import csv
import json
import os
# import os
import shutil
import re
import time
import contextlib
from typing import Tuple,List

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

import io
import requests
from urllib.parse import urljoin
from PIL import Image, ImageOps
import pytesseract
import glob



# ============================
# Config
# ============================

# Base directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Find all CSV files inside "output" and its subfolders
csv_files = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.csv"), recursive=True)

if not csv_files:
    raise FileNotFoundError("No CSV files found in output/")

# Example: pick the first CSV file
csv_file = csv_files[0]
print(f"Found CSV file: {csv_file}")

# Define output file name in results folder
base_name = os.path.basename(csv_file)
output_file = os.path.join(RESULTS_DIR, base_name)
print(f"Result path: {output_file}")


checkpoint_file = os.path.join(RESULTS_DIR, "checkpoint_for_payment_methods.json")

CHUNK_SIZE = 40
PAGE_LOAD_TIMEOUT = 25
IMPLICIT_WAIT = 0
GO_DEEP_ON_ORDER_CTA = True   # try clicking "Start Order"/"Order Now"/"Buy Now" and rescan


# If tesseract isn't on PATH, set its full path here, e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = None  # or "/usr/bin/tesseract"
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# Image filename/class/id hints to consider for OCR
PAYMENT_IMAGE_HINT_PAT = re.compile(
    r"\b(payment[-_\s]*methods?|payments?|pay[-_\s]*options?|accepted[-_\s]*payments?)\b",
    re.IGNORECASE
)

EXTRA_PAYMENT_METHODS = [
    ("Visa",       [re.compile(r"\bvisa\b", re.IGNORECASE)]),
    ("Mastercard", [re.compile(r"\bmaster\s*card\b|\bmastercard\b", re.IGNORECASE)]),
    ("Discover",   [re.compile(r"\bdiscover\b", re.IGNORECASE)]),
    ("Diners Club",[re.compile(r"\bdiners?\s*club\b", re.IGNORECASE)]),
    ("JCB",        [re.compile(r"\bjcb\b", re.IGNORECASE)]),
    ("UnionPay",   [re.compile(r"\bunion\s*pay\b|\bcup\b", re.IGNORECASE)]),
    ("PayPal",     [re.compile(r"\bpaypal\b", re.IGNORECASE)]),
    ("Klarna",     [re.compile(r"\bklarna\b", re.IGNORECASE)]),
    ("Afterpay",   [re.compile(r"\bafter\s*pay\b|\bafterpay\b", re.IGNORECASE)]),
    ("Affirm",     [re.compile(r"\baffirm\b", re.IGNORECASE)]),
    ("Shop Pay",   [re.compile(r"\bshop\s*pay\b", re.IGNORECASE)]),
]

def _image_candidate(el) -> bool:
    """
    True if an <img> (or any tag with background?) looks like a 'payment methods' banner.
    """
    if el.name != "img":
        return False
    texts = [
        el.get("src", ""), el.get("alt", ""), el.get("title", ""),
        el.get("id", "")
    ]
    classes = el.get("class", []) or []
    texts.extend([str(c) for c in classes])
    blob = " ".join(texts)
    return bool(PAYMENT_IMAGE_HINT_PAT.search(blob))

def _download_image_bytes(page_url: str, img_src: str) -> bytes:
    try:
        url = urljoin(page_url, img_src)
        resp = requests.get(url, timeout=12)
        if resp.ok and resp.content:
            return resp.content
    except Exception:
        pass
    return b""

def _preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    # Convert to grayscale and auto-contrast; simple and robust
    g = ImageOps.grayscale(pil_img)
    g = ImageOps.autocontrast(g)
    return g

def _ocr_image_bytes(data: bytes) -> str:
    if not data:
        return ""
    try:
        img = Image.open(io.BytesIO(data))
        img = _preprocess_for_ocr(img)
        # PSM 6 (Assume a block of text) generally works well for banners.
        return pytesseract.image_to_string(img, config="--psm 6") or ""
    except Exception:
        return ""

def _detect_payment_methods_from_text(text: str) -> List[str]:
    found = set()
    if not text:
        return []
    for name, patterns in PAYMENT_SIGNATURES + EXTRA_PAYMENT_METHODS:
        if any(p.search(text) for p in patterns):
            found.add(name)
    # Preserve consistent ordering: show in the order defined above
    ordered = [name for name, _ in PAYMENT_SIGNATURES if name in found]
    # Insert card wallets & BNPLs in their declared order
    ordered += [name for name, _ in EXTRA_PAYMENT_METHODS if name in found and name not in ordered]
    return ordered


def detect_payment_methods_from_images(driver: webdriver.Chrome, soup: BeautifulSoup) -> Tuple[List[str], List[str]]:
    """
    Returns (methods, evidences). Evidences are strings.
    """
    page_url = driver.current_url
    texts_per_img = []  # [(img_url, text)]
    for img in soup.find_all("img"):
        if _image_candidate(img):
            img_src = img.get("src", "")
            resolved = urljoin(page_url, img_src)
            data = _download_image_bytes(page_url, img_src)
            txt = _ocr_image_bytes(data)
            if txt:
                texts_per_img.append((resolved, txt))


    found = {}
    evidences = []
    for img_url, txt in texts_per_img:
        for name, patterns in PAYMENT_SIGNATURES + EXTRA_PAYMENT_METHODS:
            for p in patterns:
                m = p.search(txt)
                if m:
                    found[name] = True
                    snippet = _clip_snippet(txt, m.start(), m.end())
                    evidences.append(f"{name} — OCR img={img_url}: \"{snippet}\"")
                    break


    ordered = [n for n, _ in PAYMENT_SIGNATURES if n in found]
    ordered += [n for n, _ in EXTRA_PAYMENT_METHODS if n in found and n not in ordered]
    return ordered, evidences




def _clip_snippet(s: str, start: int, end: int, max_len: int = 140) -> str:
    if start < 0 or end < 0:
        return (s[:max_len] + "…") if len(s) > max_len else s
    pad = max_len // 2
    a = max(0, start - pad)
    b = min(len(s), end + pad)
    snippet = s[a:b].strip()
    snippet = re.sub(r"\s+", " ", snippet)
    if a > 0:
        snippet = "…" + snippet
    if b < len(s):
        snippet = snippet + "…"
    return snippet


def _short_tag(el) -> str:
    tid = el.get("id")
    classes = el.get("class")
    parts = [el.name]
    if tid:
        parts.append(f"#{tid}")
    if classes:
        parts.append("." + ".".join(c for c in classes if isinstance(c, str)))
    return "".join(parts)




def detect_payment_methods_with_evidence(raw_html: str, soup: BeautifulSoup, driver: webdriver.Chrome) -> Tuple[List[str], List[str]]:
    """
    Returns (ordered_methods, evidences).
    Evidences are short human-readable strings (~140 chars).
    """
    found = {}
    evidences = []


    # 1) Visible TEXT
    visible_text = soup.get_text(" ", strip=True)[:200000]
    for name, patterns in PAYMENT_SIGNATURES + EXTRA_PAYMENT_METHODS:
        for p in patterns:
            m = p.search(visible_text)
            if m:
                found[name] = True
                evidences.append(f"{name} — TEXT: \"{_clip_snippet(visible_text, m.start(), m.end())}\"")
                break


    # 2) ATTRs per element for better provenance
    for el in soup.find_all(True):
        for attr in ("alt", "title", "aria-label", "href", "src", "content", "value"):
            val = el.get(attr)
            if not isinstance(val, str) or not val:
                continue
            for name, patterns in PAYMENT_SIGNATURES + EXTRA_PAYMENT_METHODS:
                if name in found:
                    continue
                for p in patterns:
                    m = p.search(val)
                    if m:
                        found[name] = True
                        tagloc = _short_tag(el)
                        evidences.append(f"{name} — ATTR {tagloc}[{attr}]: \"{_clip_snippet(val, m.start(), m.end())}\"")
                        break


    # 3) Raw HTML URLs / tokens (script/link/etc.)
    # This helps catch js.stripe.com, checkout.stripe.com, pay.google.com, etc.
    for name, patterns in PAYMENT_SIGNATURES + EXTRA_PAYMENT_METHODS:
        if name in found:
            continue
        for p in patterns:
            m = p.search(raw_html or "")
            if m:
                found[name] = True
                evidences.append(f"{name} — HTML: \"{_clip_snippet(raw_html, m.start(), m.end())}\"")
                break


    # 4) OCR of payment-method images
    ocr_methods, ocr_evidences = detect_payment_methods_from_images(driver, soup)
    for m in ocr_methods:
        found[m] = True
    evidences.extend(ocr_evidences)


    # Ordered output (stable)
    ordered = [n for n, _ in PAYMENT_SIGNATURES if n in found]
    ordered += [n for n, _ in EXTRA_PAYMENT_METHODS if n in ordered or n not in found and False]  # no-op line to keep format
    # Correctly add remaining extras:
    ordered += [n for n, _ in EXTRA_PAYMENT_METHODS if n in found and n not in ordered]
    return ordered, evidences


# ============================
# Heuristics
# ============================
# Cart/checkout
CHECKOUT_TEXT_PAT = re.compile(
    r"\b(checkout|add\s*to\s*cart|view\s*cart|cart|buy\s*now|proceed\s*to\s*checkout|my\s*cart|shopping\s*cart|basket|bag|pay\s*now)\b",
    re.IGNORECASE
)
CHECKOUT_HREF_PAT = re.compile(r"/(checkout|cart|basket|bag)(/|$|\?)", re.IGNORECASE)
CHECKOUT_ATTR_PAT = re.compile(r"(checkout|cart|basket|bag)", re.IGNORECASE)


# Direct order CTA (no cart)
ORDER_CTA_PAT = re.compile(
    r"\b(start\s*order|start\s*your\s*order|order\s*now|place\s*order|order\s*online|buy\s*now|start\s*order\s*now|start\s*shopping)\b",
    re.IGNORECASE
)
ORDER_CTA_TOKENS = [
    "start order", "start your order", "order now", "place order",
    "order online", "buy now", "start shopping"
]


# Delivery / Pickup toggles
DELIVERY_PAT = re.compile(r"\b(delivery|deliver)\b", re.IGNORECASE)
PICKUP_PAT = re.compile(r"\b(pick\s*up|pickup|pick-up|click\s*&\s*collect|click\s*and\s*collect)\b", re.IGNORECASE)


# Payment processors / methods
PAYMENT_SIGNATURES: List[Tuple[str, List[re.Pattern]]] = [
    ("USAePay", [re.compile(r"\busae?pay\b", re.IGNORECASE), re.compile(r"usaepay\.com", re.IGNORECASE)]),
    ("UsePay", [re.compile(r"\buse\s*pay\b", re.IGNORECASE), re.compile(r"\busepay\b", re.IGNORECASE), re.compile(r"usepay\.com", re.IGNORECASE)]),
    ("Clover", [re.compile(r"\bclover(\s+payments?)?\b", re.IGNORECASE), re.compile(r"clover\.com", re.IGNORECASE)]),
    ("Stripe", [re.compile(r"\bstripe\b", re.IGNORECASE), re.compile(r"js\.stripe\.com", re.IGNORECASE), re.compile(r"checkout\.stripe\.com", re.IGNORECASE)]),
    ("Authorize.net", [re.compile(r"authorize\.?\s*\.?net", re.IGNORECASE), re.compile(r"accept(?:[.-])js", re.IGNORECASE), re.compile(r"secure\.authorize\.net", re.IGNORECASE)]),
    ("Chase", [re.compile(r"paymentech", re.IGNORECASE), re.compile(r"\borbital\b", re.IGNORECASE), re.compile(r"chase\s+(commerce|merchant|pay)", re.IGNORECASE), re.compile(r"chasepaymentech", re.IGNORECASE)]),
    ("Fiserv", [re.compile(r"\bfiserv\b", re.IGNORECASE), re.compile(r"payeezy", re.IGNORECASE)]),
    ("CardConnect", [re.compile(r"card\s*connect", re.IGNORECASE), re.compile(r"cardconnect", re.IGNORECASE), re.compile(r"\bbolt\s*pay\b", re.IGNORECASE)]),
    ("QuickBooks", [re.compile(r"quickbooks\s*payments?", re.IGNORECASE), re.compile(r"intuit\s*payments?", re.IGNORECASE), re.compile(r"payments\.intuit\.com", re.IGNORECASE), re.compile(r"quickbooks\.intuit\.com", re.IGNORECASE)]),
    ("First Data", [re.compile(r"\bfirst\s*data\b", re.IGNORECASE), re.compile(r"firstdata", re.IGNORECASE), re.compile(r"payeezy", re.IGNORECASE)]),
    ("Google Pay", [re.compile(r"google\s*pay", re.IGNORECASE), re.compile(r"\bgpay\b", re.IGNORECASE), re.compile(r"pay\.google\.com", re.IGNORECASE)]),
    ("Apple Pay", [re.compile(r"apple\s*pay", re.IGNORECASE), re.compile(r"applepaysession", re.IGNORECASE)]),
    ("American Express", [re.compile(r"american\s*express", re.IGNORECASE), re.compile(r"\bamex\b", re.IGNORECASE)]),
]


# ============================
# Utilities
# ============================
def safe_open_append(path):
    try:
        return open(path, "a", newline="", encoding="utf-8"), path
    except PermissionError:
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        alt = os.path.join(os.path.dirname(path), f"{os.path.splitext(os.path.basename(path))[0]}_{ts}.csv")
        print(f"[warn] '{path}' is locked/not writable. Using '{alt}' instead.")
        return open(alt, "a", newline="", encoding="utf-8"), alt


def make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1365,900")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--incognito")
    d = webdriver.Chrome(options=opts)
    d.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    d.implicitly_wait(IMPLICIT_WAIT)
    return d


def _normalize_url(url: str) -> str:
    return url if re.match(r"^https?://", url, re.IGNORECASE) else "https://" + url


def _safe_get(driver: webdriver.Chrome, url: str) -> bool:
    url = _normalize_url(url)
    try:
        driver.get(url)
        WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        return True
    except Exception as e:
        print(f"[warn] Load failed: {url} -> {e}")
        return False


# ============================
# Load websites
# ============================
websites = []          # list of (row_dict, url)
input_fieldnames = []  # original CSV columns


with open(csv_file, "r", encoding="utf-8", newline="") as infile:
    reader = csv.DictReader(infile)
    input_fieldnames = reader.fieldnames or []
    for row in reader:
        if row.get("Accessibility", "").strip().lower() == "accessible":
            url = row.get("track-visit-website href", "").strip()
            if url:
                websites.append((row, url))


print(f"Loaded {len(websites)} accessible websites to scan.")


# Build output columns: all input columns + our extras
extra_fields = ["isCart", "Payment methods", "Evidence Snippet"]
output_fieldnames = input_fieldnames + [f for f in extra_fields if f not in input_fieldnames]




def _section_hint(el) -> str:
    # Best-effort section label
    try:
        cur = el
        while cur and getattr(cur, "name", None):
            name = cur.name or ""
            classes = " ".join(cur.get("class", []) or [])
            ident = f"{name} {classes}".lower()
            if "footer" in ident:
                return "Footer"
            if "header" in ident or "navbar" in ident:
                return "Header"
            cur = cur.parent
    except Exception:
        pass
    return "Page"


def _friendly_evidence(kind: str, method: str, snippet: str, where: str = "", el=None) -> str:
    # kind: "text" | "attr" | "html" | "ocr"
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if kind == "text":
        loc = _section_hint(el) if el is not None else "Page"
        return f"{method} — On-page text ({loc}): “{snippet}”"
    if kind == "attr":
        loc = _section_hint(el) if el is not None else "Page"
        return f"{method} — Link/Script reference ({loc}): “{snippet}”"
    if kind == "html":
        return f"{method} — Link/Script reference: “{snippet}”"
    if kind == "ocr":
        return f"{method} — Image banner: “{snippet}”"
    return f"{method}: “{snippet}”"


# ============================
# Checkpoint (rich format, auto-upgrade)
# ============================
from datetime import datetime, timezone

COLUMN_NAME = "track-visit-website href"  # which column we are processing

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _ensure_checkpoint() -> dict:
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ck = json.load(f)
            #auto-upgrade: if old file, discard it
            if not isinstance(ck, dict) or "files" not in ck:
                raise ValueError("old checkpoint format")
            return ck
        except Exception:
            pass
    # fallback: create new skeleton
    now = _now_iso()
    return {
        "version": 1,
        "created_at": now,
        "updated_at": now,
        "files": []
    }

def _find_or_create_entry(ck: dict) -> dict:
    input_file = csv_file
    output_folder = os.path.dirname(output_file)
    current_output_file = output_file
    total_rows = len(websites)

    # try find existing
    for f in ck["files"]:
        if f.get("input_file") == input_file and f.get("output_file") == current_output_file:
            return f

    # not found → create new entry
    now = _now_iso()
    entry = {
        "input_file": input_file,
        "output_folder": output_folder,
        "output_file": current_output_file,
        "column_name": COLUMN_NAME,
        "total_rows": total_rows,
        "last_processed_index": -1,
        "processed_rows": 0,
        "status": "in_progress",
        "started_at": now,
        "updated_at": now,
        "completed_at": None
    }
    ck["files"].append(entry)
    return entry

def load_checkpoint() -> int:
    ck = _ensure_checkpoint()
    entry = _find_or_create_entry(ck)
    ck["updated_at"] = _now_iso()
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(ck, f, ensure_ascii=False, indent=2)
    return entry.get("last_processed_index", -1)

def save_checkpoint(idx: int) -> None:
    ck = _ensure_checkpoint()
    entry = _find_or_create_entry(ck)
    entry["last_processed_index"] = idx
    entry["processed_rows"] = idx + 1
    entry["updated_at"] = _now_iso()
    entry["total_rows"] = len(websites)
    ck["updated_at"] = entry["updated_at"]
    with open(checkpoint_file + ".tmp", "w", encoding="utf-8") as f:
        json.dump(ck, f, ensure_ascii=False, indent=2)
    os.replace(checkpoint_file + ".tmp", checkpoint_file)

# ============================
# Resume from checkpoint
# ============================
last_index = load_checkpoint()

# derive where to start from
start_index = last_index + 1
if start_index >= len(websites):
    print("[info] All websites already processed.")
    raise SystemExit(0)

# plan batch size
end_index = min(start_index + CHUNK_SIZE, len(websites)) if CHUNK_SIZE and CHUNK_SIZE > 0 else len(websites)
planned = end_index - start_index

print(f"[info] Resuming at index {start_index}. Planning to process {planned} site(s) this run.")


# ============================
# Parsers / Detectors
# ============================
def soupify(html: str) -> BeautifulSoup:
    s = BeautifulSoup(html or "", "html.parser")
    for t in s(["script", "style", "noscript"]):
        t.extract()
    return s


def has_checkout_or_order(soup: BeautifulSoup) -> bool:
    # checkout/cart patterns
    for el in soup.find_all(["a", "button"]):
        txt = (el.get_text(" ", strip=True) or "")[:200]
        if CHECKOUT_TEXT_PAT.search(txt):
            return True
    for a in soup.find_all("a", href=True):
        if CHECKOUT_HREF_PAT.search(a["href"]):
            return True
    for el in soup.find_all(True):
        el_id = el.get("id", "")
        if el_id and CHECKOUT_ATTR_PAT.search(el_id):
            return True
        classes = el.get("class", []) or []
        for c in classes:
            if CHECKOUT_ATTR_PAT.search(str(c)):
                return True


    # direct order CTAs
    for el in soup.find_all(["a", "button"]):
        txt = (el.get_text(" ", strip=True) or "")[:200]
        if ORDER_CTA_PAT.search(txt):
            return True


    return False


def detect_order_options(soup: BeautifulSoup) -> Tuple[bool, bool]:
    """Return (delivery_available, pickup_available). Looks at clickable/label-ish elements."""
    delivery, pickup = False, False
    for el in soup.find_all(["button", "a", "label", "option", "div", "span", "input"]):
        txt = (el.get_text(" ", strip=True) or "")
        if not txt and el.name in ("input",):
            # try attributes for inputs
            for attr in ("value", "aria-label", "title", "placeholder", "id", "name"):
                v = el.get(attr)
                if isinstance(v, str) and v:
                    txt += " " + v
        if not delivery and DELIVERY_PAT.search(txt):
            delivery = True
        if not pickup and PICKUP_PAT.search(txt):
            pickup = True
        if delivery and pickup:
            break
    return delivery, pickup


def detect_payment_methods(raw_html: str, soup: BeautifulSoup) -> List[str]:
    found = set()
    visible_text = soup.get_text(" ", strip=True)[:200000]
    attr_bits = []
    for el in soup.find_all(True):
        for attr in ("alt", "title", "aria-label", "href", "src", "content"):
            v = el.get(attr)
            if isinstance(v, str) and v:
                attr_bits.append(v)
    searchable = "\n".join([visible_text, " ".join(attr_bits), raw_html or ""])


    # Text/DOM based detection
    for name, patterns in PAYMENT_SIGNATURES + EXTRA_PAYMENT_METHODS:
        if any(p.search(searchable) for p in patterns):
            found.add(name)


    # OCR-based detection for 'payment methods' images
    ocr_methods = detect_payment_methods_from_images(driver, soup)
    for m in ocr_methods:
        found.add(m)


    # Keep stable, readable order
    ordered = [name for name, _ in PAYMENT_SIGNATURES if name in found]
    ordered += [name for name, _ in EXTRA_PAYMENT_METHODS if name in found and name not in ordered]
    return ordered




def click_first_order_cta_and_wait(driver: webdriver.Chrome) -> bool:
    """Try to click a Start Order / Order Now button and wait for navigation or DOM change."""
    try:
        cur_url = driver.current_url
        for token in ORDER_CTA_TOKENS:
            xp = f"//*[self::a or self::button][contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{token}')]"
            elems = driver.find_elements(By.XPATH, xp)
            if elems:
                el = elems[0]
                driver.execute_script("arguments[0].click();", el)
                # wait for URL change or new content
                try:
                    WebDriverWait(driver, 8).until(lambda d: d.current_url != cur_url or d.execute_script("return document.readyState") == "complete")
                except Exception:
                    pass
                # small settle
                time.sleep(1.0)
                return True
    except Exception:
        pass
    return False


def scan_page(driver: webdriver.Chrome) -> Tuple[str, List[str], Tuple[bool, bool], List[str]]:
    """Return (status, methods, (delivery, pickup), evidences)."""
    html = driver.page_source or ""
    soup = soupify(html)
    status = "available" if has_checkout_or_order(soup) else "not available"
    delivery, pickup = detect_order_options(soup)
    methods, evidences = detect_payment_methods_with_evidence(html, soup, driver)
    return status, methods, (delivery, pickup), evidences




def detect_site(url: str) -> Tuple[str, List[str], bool, bool, str, List[str]]:
    if not _safe_get(driver, url):
        return "not available", [], False, False, "page load failed", []


    status, methods, (delivery, pickup), evidences = scan_page(driver)


    if status == "not available" and GO_DEEP_ON_ORDER_CTA:
        if click_first_order_cta_and_wait(driver):
            status2, methods2, (delivery2, pickup2), evidences2 = scan_page(driver)
            status = status2 if status2 == "available" else status
            # merge methods and evidences
            methods = list(dict.fromkeys(methods + methods2))
            evidences = evidences + [e for e in evidences2 if e not in evidences]
            delivery = delivery or delivery2
            pickup = pickup or pickup2
            return status, methods, delivery, pickup, "deep-scan", evidences


    return status, methods, delivery, pickup, "ok", evidences


# ============================
# Open output first
# ============================
write_header = not os.path.exists(output_file)
out, actual_output_file = safe_open_append(output_file)
writer = csv.DictWriter(out, fieldnames=output_fieldnames)




if write_header or actual_output_file != output_file:
    writer.writeheader()
print(f"[info] Writing results to: {actual_output_file}")


# ============================
# Run
# ============================
driver = make_driver()
try:
    for idx in range(start_index, end_index):
        row, url = websites[idx]
        status, methods, delivery, pickup, note, evidences = detect_site(url)


        output_row = dict(row)
        output_row.update({
            "isCart": status,
            "Payment methods": "; ".join(methods),
            "Evidence Snippet": " | ".join(evidences) if evidences else "",
        })
        writer.writerow(output_row)
        out.flush()


        save_checkpoint(idx)
        done = idx - start_index + 1
        print(f"[{done}/{planned}] idx={idx} {url} -> {status} | methods: {', '.join(methods) or 'none'} | "
              f"delivery: {'yes' if delivery else 'no'} | pickup: {'yes' if pickup else 'no'} ({note})")
except KeyboardInterrupt:
    print("\n[info] Interrupted by user, checkpoint saved.")
finally:
    with contextlib.suppress(Exception):
        driver.quit()
    out.close()
    print(f"[done] Processed {planned if planned>0 else 0} site(s). Checkpoint updated.")



