import json
import glob
import time
import shutil
from pathlib import Path
from typing import List
import aiohttp
import asyncio

class ContextGenerator:
    def __init__(
        self,
        api_key: str,
        model: str,
        output_dir: str,
        temperature: float = 0.7,
        max_tokens: int = 400,
        batch_size: int = 5,
        delay: float = 1.0,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.delay = delay
        self.api_base = "https://api.siliconflow.cn/v1"
        self.samples_generated = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / "progress.json"
        self.processed = set()
        self.load_progress()

    def load_progress(self):
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    progress = json.load(f)
                    self.processed = set(progress.get("processed", []))
                print(f"Loaded existing progress, {len(self.processed)} files processed")
            except Exception as e:
                print(f"Failed to load progress file: {e}")
                self.processed = set()
        else:
            print("No progress file found, starting from scratch")

    def save_progress(self, total: int):
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "processed": list(self.processed),
                    "total": total,
                    "lastUpdate": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    async def generate_context(self, section_file: str) -> bool:
        if section_file in self.processed:
            print(f"Skipping {Path(section_file).name}")
            return True

        uuid = Path(section_file).name.split("_")[0]
        context_pattern = f"splited_files/{uuid}_context_*.md"
        context_files = glob.glob(context_pattern)

        if not context_files:
            print(f"No context file found for {uuid}")
            return False

        context_file = context_files[0]

        try:
            with open(section_file, "r", encoding="utf-8") as f:
                section_text = f.read()

            with open(context_file, "r", encoding="utf-8") as f:
                context_text = f.read()

            prompt = f"""
下面我将会给出一个语言文档当中的小节和它的前后文，请根据前后文为这个小节编写一个"定位段"，用来标识这一段是说明了什么内容，在一个什么样的文档上下文中。
请注意重点是内容本身，而非对内容的评价。
不要提及"文档结构"等元描述，也不要评论其"作用"、"承上启下"之类的关系。

使用英语，而且不要加入一些 markdown 标题成分，你要做的就是输出一个纯粹的段落。

# 上下文：
{context_text}

# 小节正文
{section_text}
"""

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位客观的内容整合助手，不做评价，只做内容融合与补充。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/chat/completions", headers=headers, json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"❌ API request failed: {response.status} - {error_text}")
                        return False

                    result = await response.json()
                    generated = result["choices"][0]["message"]["content"].strip()

            output_file = self.output_dir / f"{uuid}_generated_context.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"{generated}\n")

            print(f"✅ Generated and saved: {output_file.name}")

            self.samples_generated += 1

            self.processed.add(section_file)
            self.save_progress(total=len(glob.glob("splited_files/*_section_*.md")))
            return True

        except Exception as e:
            print(f"❌ Failed to process file {section_file}: {str(e)}")
            return False

    async def process_batch(self, batch: List[str]) -> List[bool]:
        results = []

        for file in batch:

            print(f"Processing file: {Path(file).name}")

            result = await self.generate_context(file)
            results.append(result)

            await asyncio.sleep(self.delay)

        return results

    async def run(self):
        section_files = glob.glob("splited_files/*_section_*.md")
        print(f"Found {len(section_files)} section files, starting processing...")

        remaining_files = [f for f in section_files if f not in self.processed]
        print(f"{len(remaining_files)} files need to be processed")

        total_batches = len(remaining_files) // self.batch_size

        for i in range(0, len(remaining_files), self.batch_size):
            batch = remaining_files[i : i + self.batch_size]
            current_batch = i // self.batch_size + 1

            print(f"Processing batch {current_batch}/{total_batches}, {len(batch)} files")
            await self.process_batch(batch)

            self.save_progress(total=len(section_files))
            print(f"Batch completed, progress: {len(self.processed)}/{len(section_files)}")

        print(f"Samples generated, total: {self.samples_generated}")


def concat_contexts():
    contexts = glob.glob("generated_contexts/*_generated_context.md")
    final_section_dir = Path("final_sections")
    if final_section_dir.exists():
        shutil.rmtree(final_section_dir)
    final_section_dir.mkdir()
    for context in contexts:
        uuid = Path(context).name.split("_")[0]
        section_files = glob.glob(f"splited_files/{uuid}_section_*.md")
        with open(context, "r", encoding="utf-8") as f:
            context_text = f.read()
        for section_file in section_files:
            section_name = Path(section_file).name
            with open(section_file, "r", encoding="utf-8") as f:
                section_text = f.read()
            with open(f"final_sections/{section_name}", "w", encoding="utf-8") as f:
                f.write(f"{context_text}\n{section_text}")


def main():

    generator = ContextGenerator(
        api_key="Protected",
        model="deepseek-ai/DeepSeek-V3",
        temperature=0.7,
        max_tokens=400,
        output_dir="generated_contexts",
        batch_size=5,
        delay=1.0,
    )

    try:
        asyncio.run(generator.run())
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, saving progress and exiting...")
        section_files = glob.glob("splited_files/*_section_*.md")
        generator.save_progress(total=len(section_files))
        
    concat_contexts()

if __name__ == "__main__":
    main()
