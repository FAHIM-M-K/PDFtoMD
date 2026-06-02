# Vision PDF to Markdown Extractor

A robust, high-fidelity PDF text extraction tool powered by Mistral's **Pixtral-12B** Vision model. 

This tool is specifically designed to handle complex document layouts—such as side-by-side columns, tables, and checklists—that traditional OCR or simple text extractors fail on. It is programmed to output pristine Markdown while automatically stripping out watermarks, timestamps, and confidentiality banners.

## Features

- **True Layout Understanding**: Uses vision models to physically "look" at the page, ensuring multi-column layouts aren't accidentally merged together.
- **Strict Markdown Output**: Returns 100% clean markdown without conversational filler, making it perfect for feeding into other LLM pipelines.
- **Dual-Pass Verification**: Every page undergoes a two-step prompt (Extract → Validate) to guarantee no headers, tables, or bullets are dropped.
- **Zero Local GPU Required**: Offloads all the heavy deep-learning visual inference to Mistral's cloud servers.

## Prerequisites

You will need a **Mistral API Key**. Mistral offers a highly capable free tier that handles 1 request per second.
1. Go to [console.mistral.ai](https://console.mistral.ai) and sign up.
2. Navigate to **API Keys** and create a new key.

## Setup

1. **Clone the repository and install dependencies:**
   ```bash
   git clone <your-repo-url>
   cd pdf_text_extraction
   pip install -r requirements.txt
   ```

2. **Configure your Environment:**
   Create a `.env` file in the root directory and add your Mistral API key:
   ```env
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

3. **Add your PDFs:**
   Place any PDF files you want to convert into the `pdfs/` directory.

## Usage

Run the extractor script:
```bash
python extractor.py
```

### How it Works
1. The script automatically reads all PDFs in the `pdfs/` folder.
2. It loops through every page, rendering them as high-quality images.
3. The images are sent to Mistral API (`pixtral-12b-2409`) where they are extracted into pure Markdown.
4. A built-in regex parser strictly strips any AI conversational filler.
5. The final Markdown is saved in the `md/` folder, completely ready for RAG pipelines or direct LLM ingestion.

## Example 

### Input (PDF Page)
- You can edit the prompt to your requirement, if you have any water mark you want to remove or specific formatting to do

### Output (Generated Markdown)
- Pure Markdown tables `| Column 1 | Column 2 |`
- Valid checklist items `- [ ]`
- Watermarks totally ignored.
