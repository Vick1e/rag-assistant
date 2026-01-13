import os
import nest_asyncio
from llama_parse import LlamaParse

# API Key
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-HhE6e6u0T8RaUmX1vAUchjxxFcChKCrf3wMeHlNcrzo9QCNS"

nest_asyncio.apply()

print("⚙️ Initializing parser...")
parser = LlamaParse(
    result_type="markdown",
    verbose=True,
    language="en",
    num_workers=4
)

pdf_name = "manual.pdf"

if not os.path.exists(pdf_name):
    print(f"❌ Error: File '{pdf_name}' not found.")
else:
    print(f"🚀 Uploading and parsing '{pdf_name}'...")
    
    # 开始解析
    documents = parser.load_data(pdf_name)

    output_file = "manual_parsed.md"
    with open(output_file, "w", encoding="utf-8") as f:
        # ✅ 修正了这里：是 for doc in documents
        full_text = "\n\n".join([doc.text for doc in documents])
        f.write(full_text)

    print(f"✅ Success! Parsed content saved to: {output_file}")
