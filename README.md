# TEDPara: Download & Preprocessing

This script processes TED Talk transcripts by:
- Downloading transcripts via TED's public GraphQL API
- Splitting transcript text into sentences
- Annotating sentence-level paragraph boundaries as a binary string

The result is a structured dataset for each of the `train`, `val`, and `test` partitions, ready for use in sequence modeling, segmentation, or NLP tasks.

---

## 📁 Input

The following JSON files are in the directory, containting the talk IDs that are associated with each partition.

```bash
talk_ids.train.json
talk_ids.val.json
talk_ids.test.json
```

## 🚀 Usage

Run the script:

```bash
python download_and_process.py
```

Make sure you have the required libraries installed:

```bash
pip install pandas requests tqdm nltk
```

## 🧠 Output

After running the script, you'll get three JSON files:

```bash
transcripts.train.json
transcripts.val.json
transcripts.test.json
```

Each entry in the file is a dictionary with:


| Key      | Type                  | Description                                                                 |
|----------|-----------------------|-----------------------------------------------------------------------------|
| talk_id  | `string`              | The TED talk ID                                                             |
| text     | `list[string]`        | A list of tokenized sentences from the talk                                |
| targets  | `string` (prefixed with `\|=`) | Binary string marking paragraph starts (e.g., `\|=010100` means paragraph boundaries are before sentences 1 and 3) |

## 🛠 Notes
- Sentences are tokenized using NLTK's `sent_tokenize`.
- Paragraph boundaries are encoded with a 1 at the start of each paragraph, 0 otherwise.

- The binary string is prepended with `|=` to enforce string formatting when saving/loading.
