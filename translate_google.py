import argparse
import asyncio
from googletrans import Translator

async def main(lang: str):
    async with Translator() as translator:
        with open('data/texts.tsv', 'r', encoding='utf-8') as file:
            for line in file:
                sentences = line.strip().split('\t')
                results = await translator.translate(sentences, dest=lang)
                translations = [t.text for t in results]
                print("\t".join(translations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text using Google Translate API.")
    parser.add_argument('lang', type=str, help='Target language code (e.g., "en" for English, "es" for Spanish).')
    args = parser.parse_args()
    asyncio.run(main(args.lang))
