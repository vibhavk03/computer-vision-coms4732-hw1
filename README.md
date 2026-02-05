# COMS4732 — Homework 1

This folder contains `index.html` that documents the alignment results for Homework 1 and points to the generated aligned images saved by the notebook.

Files added
- `index.html` — the webpage for your submission (open in a browser or convert to PDF).

How to regenerate outputs
1. Open and run the `starter.ipynb` notebook in this folder. Ensure you run the helper and processing cells so the `process_JPG()` and `process_TIF()` functions are defined.
2. Choose which images to process by editing the lists `images_JPGs`, `images_TIFs`, or `images_mine_TIFs` inside the notebook and run the processing cells. The notebook code saves aligned outputs to:
   - `outputJPGs/` (JPEG-aligned outputs)
   - `outputTIFs/` (aligned TIF outputs)
   - `myOutputTIFs/` (your selected Prokudin-Gorskii images)

Notes about filenames
- The notebook saves files with base names like `cathedral_aligned.png`, `monastery_aligned.png`, and `tobolsk_aligned.png` into `outputJPGs/`.
- For TIFs the notebook creates `outputTIFs/<base>_aligned.png`.
- For your own Prokudin-Gorskii images the notebook writes into `myOutputTIFs/` using the original basenames.

Generating a PDF of the webpage
Option A — Use wkhtmltopdf (non-interactive):
```
wkhtmltopdf index.html index.pdf
```
Option B — Use headless Chrome / Chromium:
```
google-chrome --headless --disable-gpu --print-to-pdf=index.pdf index.html
```
Option C — Serve locally and print from browser:
```
python -m http.server 8000
# then open http://localhost:8000/index.html in your browser and Print -> Save as PDF
```

What to edit in `index.html`
- Replace the placeholder offset text in `<figcaption>` elements with the actual offsets printed by the notebook when each image was processed.

Submission reminder
- Do NOT upload image files to Gradescope; instead submit your code, `index.html`, `README.md`, and the generated `index.pdf` (you can generate the PDF locally using one of the commands above).

If you want, I can also automatically fill the captions with the offsets from a notebook run (requires running the notebook here). Ask me to run the notebook and I'll run it and update `index.html` captions with the computed offsets.
