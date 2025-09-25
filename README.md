# e-commerce-payment-processor-scrapper

This project is designed to **scrape payment processor details** and their **evidence snippets** from e-commerce websites.
It works in two major steps:

---

## 🚀 Workflow

### **Step 1 – Website Accessibility Check**

* Input: a **CSV file** containing website URLs.
* The script checks if each website is accessible by sending a request with a **25–30s timeout**.
* Output:

  * A **copy of the input file** with an additional column `Accessibility` (`Accessible` / `Not Accessible`).
  * A **checkpoint.json** file that records processing progress.

Example of `checkpoint.json`:

```json
{
  "input_file": "input/Mountain Pacific AK_HI Shop data scrape_09-23-2025_14k.csv",
  "output_folder": "output/Mountain Pacific AK_HI Shop data scrape_09-23-2025_14k",
  "output_file": "output/Mountain Pacific AK_HI Shop data scrape_09-23-2025_14k/Mountain Pacific AK_HI Shop data scrape_09-23-2025_14k.csv",
  "column_name": "track-visit-website href",
  "total_rows": 6448,
  "last_processed_index": 6447,
  "processed_rows": 6448,
  "status": "completed",
  "started_at": "2025-09-23T09:11:41Z",
  "updated_at": "2025-09-24T14:05:34Z",
  "completed_at": "2025-09-23T12:27:31Z",
  "duration_minutes": 195.83
}
```

✅ The checkpoint system ensures that if the process is interrupted (due to errors or large file size), the script can resume from the last processed index.

---

### **Step 2 – Scraping Payment Processors**

* Input: the **Accessible websites** identified in Step 1.

* For each site:

  1. Detect if a **shopping cart** exists.
  2. If a cart is available, **scrape payment processors** used by the site.

     * **HTML scraping**: extracting processor names from the site.
     * **Image-to-text scraping**: handling cases where payment processor logos are shown as images.

* Output:

  * Extended CSV file with additional columns:

    * `Cart Available` (Yes/No)
    * `Payment Processors` (list of detected processors)
    * `Evidence Snippet` (supporting snippet or OCR result)

---

## 🏦 Payment Processors Covered

The scraper currently detects the following payment processors:

* USAePay
* Clover
* Stripe
* Authorize.net
* Chase
* Fiserv
* CardConnect
* QuickBooks
* First Data
* Google Pay
* Apple Pay
* American Express

---

## 📂 Example Output

| URL              | Accessibility  | Cart Available | Payment Processors | Evidence Snippet              |
| ---------------- | -------------- | -------------- | ------------------ | ----------------------------- |
| example.com      | Accessible     | Yes            | Stripe, PayPal     | `<img src="stripe-logo.png">` |
| sample-store.com | Accessible     | No             | —                  | —                             |
| offline.com      | Not Accessible | —              | —                  | —                             |

---

## ⚡ Features

* Handles **large CSVs** with checkpointing for safe recovery.
* Supports **HTML and OCR-based scraping**.
* Produces **detailed evidence snippets** for validation.
* Extensible for adding more payment processors.

---

## 🔧 Usage

1. Place your input CSV file in the `input/` directory.
2. Run Step 1 script (accessibility check).
3. Run Step 2 script (cart + payment processor scraping).
4. Collect output from the `output/` folder.

---

## 📜 License

MIT License – free to use, modify, and distribute.

**installation + run commands** (Python dependencies, CLI usage examples)
python version : 3.12
for other dependencires insatll requirement.txt
